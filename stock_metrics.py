"""
Shared data-fetching and metric helpers for scraper.py and
stock_filter.py.

Every number comes from the usstock library:
    - stockanalysis.com pages (statistics, statements, ratios,
      and the full stock list) via usstock
    - Yahoo Finance (sector/industry, quotes, daily stock, ETF and
      S&P 500 prices) via usstock
    - US Treasury (risk-free rate) via usstock
"""

import csv
import datetime as dt
import math
import re
import statistics

import usstock
from usstock import _http
from usstock import _stockanalysis


# ============================================================
# SETTINGS
# ============================================================

TRADING_DAYS = 252

# Years of daily prices used for Sharpe ratio and volatility.
RISK_WINDOW_YEARS = 5

# How close (in days) the first available price must be to the
# start of a CAGR window for the CAGR to count.
CAGR_START_TOLERANCE_DAYS = 10

# Yahoo sector -> SPDR sector ETF used as the sector benchmark.
SECTOR_ETFS = {
    "Technology": "XLK",
    "Financial Services": "XLF",
    "Healthcare": "XLV",
    "Energy": "XLE",
    "Industrials": "XLI",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Utilities": "XLU",
    "Basic Materials": "XLB",
    "Real Estate": "XLRE",
    "Communication Services": "XLC",
}

# Trend labels: slopes smaller than these per year count as "stable".
DEBT_TO_EQUITY_STABLE_SLOPE = 0.02
LIABILITIES_TO_ASSETS_STABLE_SLOPE = 0.005
MARGIN_STABLE_SLOPE = 0.5  # percentage points per year

STOCK_LIST_URL = "https://stockanalysis.com/stocks/?page={page}"


# ============================================================
# BASIC HELPERS
# ============================================================

def parse_number(value):
    """'39.40' -> 39.4, '5.01T' -> 5.01e12, '12.5%' -> 12.5, 'n/a' -> None."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return _stockanalysis._num(value)


def yahoo_symbol(symbol):
    """stockanalysis uses BRK.B; Yahoo uses BRK-B."""
    return symbol.upper().replace(".", "-")


def fiscal_series(series):
    """[(year, value), ...] oldest first from {'FY 2025': v, ...}."""
    points = []
    for period, value in (series or {}).items():
        match = re.fullmatch(r"FY (\d{4})", period)
        if match and value is not None:
            points.append((int(match.group(1)), value))
    return sorted(points)


def latest_value(series):
    """TTM / Current value, falling back to the latest fiscal year."""
    series = series or {}
    for key in ("TTM", "Current"):
        if series.get(key) is not None:
            return series[key]
    points = fiscal_series(series)
    return points[-1][1] if points else None


def slope_per_year(points):
    """Least-squares slope of [(year, value), ...]."""
    if len(points) < 3:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_mean = statistics.mean(xs)
    y_mean = statistics.mean(ys)
    denominator = sum((x - x_mean) ** 2 for x in xs)
    if denominator == 0:
        return None
    return sum(
        (x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)
    ) / denominator


def trend_label(slope, stable_threshold):
    if slope is None:
        return None
    if abs(slope) < stable_threshold:
        return "stable"
    return "rising" if slope > 0 else "falling"


def format_history(points, digits=2):
    """'2021:0.55;2022:0.61;...' for compact CSV storage."""
    return ";".join(f"{year}:{value:.{digits}f}" for year, value in points)


def ratio(numerator, denominator):
    if numerator is None or not denominator:
        return None
    return numerator / denominator


def as_percent(value):
    return None if value is None else value * 100


def format_cell(value):
    """CSV formatting: whole numbers for large values, 4 decimals otherwise."""
    if isinstance(value, float):
        if abs(value) >= 1e6:
            return f"{value:.0f}"
        return f"{value:.4f}"
    return "" if value is None else value


def write_csv(path, columns, rows):
    """
    Write rows to a CSV and return the path used. If the file is open
    in Excel (locked on Windows), save to a timestamped copy instead of
    losing the results.
    """
    def write(target):
        with open(target, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({c: format_cell(row.get(c)) for c in columns})

    try:
        write(path)
        return path
    except PermissionError:
        stem, _, extension = path.rpartition(".")
        fallback = f"{stem}_{dt.datetime.now():%Y%m%d_%H%M%S}.{extension}"
        print(f"{path} is open in another program; saving to {fallback}")
        write(fallback)
        return fallback


# ============================================================
# STOCKANALYSIS.COM DATA
# ============================================================

def statement(symbol, kind):
    """
    Full statement as {label: {period: number}}.

    usstock's own income_statement()/balance_sheet()/ratios() only read
    the first table on each page (e.g. the balance sheet stops at Total
    Assets), so this merges every table using usstock's parser.

    kind: "income", "balance", "cashflow" or "ratios".
    """
    html = _stockanalysis._fetch(_stockanalysis._PAGES[kind], symbol)
    data = {}

    for table in _stockanalysis._parse_tables(html):
        periods = [
            header for header in table["header"]
            if header and header not in ("Fiscal Year", "Period Ending")
        ]

        for row in table["rows"]:
            label = _stockanalysis._clean_label(row[0])
            series = {
                periods[i]: parse_number(value)
                for i, value in enumerate(row[1:])
                if i < len(periods)
            }
            data.setdefault(label, series)

    return data


def key_statistics(symbol):
    """Current snapshot from the stockanalysis statistics page."""
    return usstock.statistics(symbol)


def stock_universe():
    """
    Every stock listed on https://stockanalysis.com/stocks/ as
    [{"symbol", "name", "industry", "market_cap"}, ...].
    """
    stocks = []
    expected_count = None
    page = 1

    while True:
        url = STOCK_LIST_URL.format(page=page)
        response = _http.get(url, ttl=3600)
        if response.status_code != 200:
            raise RuntimeError(f"{url} -> HTTP {response.status_code}")
        html = response.text

        if expected_count is None:
            match = re.search(r"resultsCount:(\d+)", html)
            expected_count = int(match.group(1)) if match else None

        page_stocks = []

        # Each stock is an object literal: {s:"AAPL",n:"Apple Inc.",
        # industry:"Consumer Electronics",marketCap:5012517000000}
        for chunk in html.split('{s:"')[1:]:
            entry = chunk.split("}", 1)[0]
            symbol = entry.split('"', 1)[0]
            name = re.search(r'n:"((?:[^"\\]|\\.)*)"', entry)
            industry = re.search(r'industry:"((?:[^"\\]|\\.)*)"', entry)
            market_cap = re.search(r'marketCap:([0-9.eE+]+)', entry)

            page_stocks.append({
                "symbol": symbol,
                "name": name.group(1) if name else None,
                "industry": industry.group(1) if industry else None,
                "market_cap": float(market_cap.group(1)) if market_cap else None,
            })

        stocks.extend(page_stocks)

        if len(page_stocks) < 500:
            break

        page += 1

    # A blocked or changed page would otherwise silently shrink the
    # universe and skip part of the market.
    if expected_count is not None and len(stocks) < expected_count:
        raise RuntimeError(
            f"Only downloaded {len(stocks):,} of {expected_count:,} "
            f"stocks from stockanalysis.com; try again later."
        )

    return stocks


def industry_market_cap_shares(universe):
    """{symbol: share of its industry's total market cap}."""
    industry_totals = {}
    for stock in universe:
        if stock["industry"] and stock["market_cap"]:
            industry_totals[stock["industry"]] = (
                industry_totals.get(stock["industry"], 0.0)
                + stock["market_cap"]
            )

    return {
        stock["symbol"]: ratio(
            stock["market_cap"],
            industry_totals.get(stock["industry"])
        )
        for stock in universe
        if stock["industry"] and stock["market_cap"]
    }


# ============================================================
# YAHOO / CBOE / TREASURY DATA
# ============================================================

def company_profile(symbol):
    """(sector, industry) from Yahoo's asset profile."""
    profile = usstock.overview(yahoo_symbol(symbol)).get("assetProfile", {})
    return profile.get("sector"), profile.get("industry")


def latest_prices(symbols, batch_size=20):
    """{symbol: price} using batched Yahoo quotes."""
    prices = {}

    for start in range(0, len(symbols), batch_size):
        batch = symbols[start:start + batch_size]
        by_yahoo = {yahoo_symbol(s): s for s in batch}

        try:
            quotes = usstock.quotes(list(by_yahoo))
        except Exception:
            # One bad symbol can fail the whole batch; retry singly.
            quotes = {}
            for yahoo in by_yahoo:
                try:
                    quotes.update(usstock.quotes([yahoo]))
                except Exception:
                    pass

        for yahoo, quote in quotes.items():
            if yahoo in by_yahoo and quote.get("price") is not None:
                prices[by_yahoo[yahoo]] = quote["price"]

    return prices


def price_history(symbol):
    """[(date, close), ...] oldest first: 10 years of daily Yahoo closes."""
    rows = usstock.chart(yahoo_symbol(symbol), "10y", "1d")["rows"]
    return [
        (dt.date.fromisoformat(row["date"][:10]), row["close"])
        for row in rows
        if row["close"]
    ]


def sp500_history():
    """
    [(date, close), ...] oldest first for the S&P 500 index. Uses the
    same Yahoo source as the stock prices so trading days line up.
    """
    return price_history("^GSPC")


def risk_free_rate():
    """Current 3-month Treasury yield as a decimal."""
    try:
        curve = usstock.yield_curve()
    except RuntimeError:
        # Early January: no rates published for the new year yet.
        curve = usstock.yield_curve(dt.date.today().year - 1)
    return curve["3 Mo"] / 100


# ============================================================
# PRICE-BASED METRICS
# ============================================================

def cagr(prices, years):
    """Annualized price return over the last `years` years."""
    if not prices:
        return None

    end_date, end_price = prices[-1]
    start_target = end_date - dt.timedelta(days=round(years * 365.25))

    for date, price in prices:
        if date >= start_target:
            if (date - start_target).days > CAGR_START_TOLERANCE_DAYS:
                return None  # history does not go back far enough
            if price <= 0:
                return None
            return (end_price / price) ** (1 / years) - 1

    return None


def calendar_year_returns(prices):
    """{year: return} for each complete calendar year."""
    year_end = {}
    for date, price in prices:
        year_end[date.year] = price

    current_year = prices[-1][0].year if prices else None

    return {
        year: year_end[year] / year_end[year - 1] - 1
        for year in sorted(year_end)
        if year - 1 in year_end and year != current_year
    }


def covers_years(prices, years):
    """True if the price history reaches back the full `years` years."""
    if not prices:
        return False
    start_target = prices[-1][0] - dt.timedelta(days=round(years * 365.25))
    return (prices[0][0] - start_target).days <= CAGR_START_TOLERANCE_DAYS


def recent_daily_returns(prices, years=RISK_WINDOW_YEARS):
    """{date: daily return} over the last `years` years."""
    if not prices:
        return {}

    start = prices[-1][0] - dt.timedelta(days=round(years * 365.25))
    returns = {}

    for (_, previous), (date, price) in zip(prices, prices[1:]):
        if date >= start and previous > 0:
            returns[date] = price / previous - 1

    return returns


def sharpe_ratio(daily_returns, annual_risk_free):
    """Annualized Sharpe ratio from daily returns."""
    values = list(daily_returns.values())
    if len(values) < TRADING_DAYS:
        return None

    daily_risk_free = annual_risk_free / TRADING_DAYS
    excess = [r - daily_risk_free for r in values]
    std = statistics.stdev(excess)
    if std == 0:
        return None

    return statistics.mean(excess) / std * math.sqrt(TRADING_DAYS)


def downside_specific_volatility(stock_returns, market_returns):
    """
    Annualized downside deviation of the stock's idiosyncratic
    (market-model residual) daily returns: only the stock-specific
    moves that went against the holder count.
    """
    dates = sorted(set(stock_returns) & set(market_returns))
    if len(dates) < TRADING_DAYS:
        return None

    stock = [stock_returns[d] for d in dates]
    market = [market_returns[d] for d in dates]

    market_variance = statistics.variance(market)
    if market_variance == 0:
        return None

    beta = statistics.covariance(stock, market) / market_variance
    alpha = statistics.mean(stock) - beta * statistics.mean(market)

    residuals = [s - (alpha + beta * m) for s, m in zip(stock, market)]
    downside = math.sqrt(
        sum(min(e, 0.0) ** 2 for e in residuals) / len(residuals)
    )

    return downside * math.sqrt(TRADING_DAYS)


# ============================================================
# FUNDAMENTAL METRICS
# ============================================================

def earnings_volatility(net_income_points):
    """
    Standard deviation of year-over-year net income changes divided
    by average absolute net income. Scale-free and works when
    earnings turn negative. Higher = less predictable earnings.
    """
    values = [value for _, value in net_income_points]
    if len(values) < 3:
        return None

    changes = [b - a for a, b in zip(values, values[1:])]
    average_size = statistics.mean(abs(v) for v in values)
    if average_size == 0 or len(changes) < 2:
        return None

    return statistics.stdev(changes) / average_size


def growth_cagr(points):
    """CAGR between the first and last fiscal-year values."""
    if len(points) < 2:
        return None
    (start_year, start), (end_year, end) = points[0], points[-1]
    years = end_year - start_year
    if years <= 0 or start <= 0 or end <= 0:
        return None
    return (end / start) ** (1 / years) - 1


# ============================================================
# SHARED MARKET CONTEXT
# ============================================================

class MarketContext:
    """Data shared by every ticker, downloaded once and cached."""

    def __init__(self):
        self._universe = None
        self._industry_shares = None
        self._sp500_returns = None
        self._risk_free = None
        self._sector_cagr = {}

    @property
    def universe(self):
        if self._universe is None:
            self._universe = stock_universe()
        return self._universe

    @property
    def industry_shares(self):
        if self._industry_shares is None:
            self._industry_shares = industry_market_cap_shares(self.universe)
        return self._industry_shares

    @property
    def sp500_returns(self):
        if self._sp500_returns is None:
            self._sp500_returns = recent_daily_returns(sp500_history())
        return self._sp500_returns

    @property
    def risk_free(self):
        if self._risk_free is None:
            self._risk_free = risk_free_rate()
        return self._risk_free

    def sector_cagr(self, sector, years=5):
        """(ETF, CAGR) for the sector's benchmark ETF."""
        etf = SECTOR_ETFS.get(sector)
        if etf is None:
            return None, None
        key = (etf, years)
        if key not in self._sector_cagr:
            self._sector_cagr[key] = cagr(price_history(etf), years)
        return etf, self._sector_cagr[key]


def load_shared_data(context, include_universe=False):
    """
    Download the data every ticker depends on before any worker
    threads start, and stop with a clear message if it is missing.

    This also creates the Yahoo session and crumb once: worker threads
    creating them at the same time can pair a crumb with the wrong
    session's cookies, which makes profile lookups fail.
    """
    try:
        _http.yahoo_crumb()
        if len(context.sp500_returns) < TRADING_DAYS * RISK_WINDOW_YEARS * 0.9:
            raise RuntimeError("S&P 500 price history is incomplete")
        _ = context.risk_free
        if include_universe:
            _ = context.industry_shares
    except Exception as error:
        raise SystemExit(
            f"Could not download shared market data ({error}). "
            f"Check your connection and try again."
        )


# ============================================================
# METRIC GROUPS
# ============================================================

def valuation_and_health_metrics(stats):
    """Metrics available from the statistics page alone."""
    return {
        "market_cap": parse_number(stats.get("Market Cap")),
        "pe_ratio": parse_number(stats.get("PE Ratio")),
        "ev_ebitda": parse_number(stats.get("EV / EBITDA")),
        "debt_to_equity": parse_number(stats.get("Debt / Equity")),
        "beta_5y": parse_number(stats.get("Beta (5Y)")),
        "ebitda_margin_pct": parse_number(stats.get("EBITDA Margin")),
        "altman_z": parse_number(stats.get("Altman Z-Score")),
        "piotroski_f": parse_number(stats.get("Piotroski F-Score")),
        "interest_coverage": parse_number(stats.get("Interest Coverage")),
        "book_equity": parse_number(stats.get("Equity (Book Value)")),
        "operating_cash_flow_ttm": parse_number(stats.get("Operating Cash Flow")),
        "net_cash": parse_number(stats.get("Net Cash")),
    }


def ratio_trend_metrics(ratios_data):
    """Debt/equity trend and EV/EBITDA change from the ratios page."""
    row = {}

    de_points = fiscal_series(ratios_data.get("Debt / Equity Ratio"))
    de_slope = slope_per_year(de_points)
    row["debt_to_equity_history"] = format_history(de_points)
    row["debt_to_equity_slope_per_year"] = de_slope
    row["debt_to_equity_trend"] = trend_label(
        de_slope, DEBT_TO_EQUITY_STABLE_SLOPE
    )
    row["debt_to_equity_change_over_period"] = (
        de_points[-1][1] - de_points[0][1] if len(de_points) >= 2 else None
    )

    ev_series = ratios_data.get("EV/EBITDA Ratio") or {}
    ev_points = fiscal_series(ev_series)
    current = ev_series.get("Current")
    last_fy = ev_points[-1][1] if ev_points else None
    row["ev_ebitda_last_fy"] = last_fy
    row["ev_ebitda_change_vs_last_fy_pct"] = as_percent(
        ratio(current - last_fy, abs(last_fy))
        if current is not None and last_fy else None
    )

    return row


def balance_metrics(balance_data):
    """Liabilities-to-assets ratio, current and trend."""
    liabilities = balance_data.get("Total Liabilities") or {}
    assets = balance_data.get("Total Assets") or {}

    points = []
    for year, total_assets in fiscal_series(assets):
        total_liabilities = liabilities.get(f"FY {year}")
        if total_liabilities is not None and total_assets:
            points.append((year, total_liabilities / total_assets))

    slope = slope_per_year(points)

    # Use TTM for both if available, otherwise the latest fiscal year
    # that has both, so liabilities and assets come from one period.
    if liabilities.get("TTM") is not None and assets.get("TTM"):
        current = liabilities["TTM"] / assets["TTM"]
    else:
        current = points[-1][1] if points else None

    return {
        "liabilities_to_assets": current,
        "liabilities_to_assets_history": format_history(points, 3),
        "liabilities_to_assets_slope_per_year": slope,
        "liabilities_to_assets_trend": trend_label(
            slope, LIABILITIES_TO_ASSETS_STABLE_SLOPE
        ),
    }


def income_metrics(income_data, cashflow_data):
    """YoY TTM changes, revenue CAGR, margins, earnings volatility."""
    row = {
        "revenue_yoy_ttm_pct": (income_data.get("Revenue Growth") or {}).get("TTM"),
        "eps_yoy_ttm_pct": (income_data.get("EPS Growth") or {}).get("TTM"),
        "fcf_yoy_ttm_pct": (income_data.get("Free Cash Flow Growth") or {}).get("TTM"),
        "operating_cash_flow_yoy_ttm_pct": (
            cashflow_data.get("Operating Cash Flow Growth") or {}
        ).get("TTM"),
    }

    revenue_points = fiscal_series(income_data.get("Revenue"))
    row["revenue_cagr_pct"] = as_percent(growth_cagr(revenue_points))
    row["revenue_cagr_years"] = (
        revenue_points[-1][0] - revenue_points[0][0]
        if len(revenue_points) >= 2 else None
    )

    for label, column in (
        ("Gross Margin", "gross_margin"),
        ("Operating Margin", "operating_margin"),
        ("Profit Margin", "profit_margin"),
        ("FCF Margin", "fcf_margin"),
    ):
        series = income_data.get(label) or {}
        points = fiscal_series(series)
        slope = slope_per_year(points)
        row[f"{column}_pct"] = latest_value(series)
        row[f"{column}_slope_pp_per_year"] = slope
        row[f"{column}_trend"] = trend_label(slope, MARGIN_STABLE_SLOPE)

    row["earnings_volatility"] = earnings_volatility(
        fiscal_series(income_data.get("Net Income"))
    )

    return row


def price_metrics(prices, context):
    """Historical returns, CAGR, Sharpe and downside specific volatility."""
    row = {
        "return_1y_pct": as_percent(cagr(prices, 1)),
        "price_cagr_3y_pct": as_percent(cagr(prices, 3)),
        "price_cagr_5y_pct": as_percent(cagr(prices, 5)),
        "price_cagr_10y_pct": as_percent(cagr(prices, 10)),
    }

    yearly = calendar_year_returns(prices)
    for year in sorted(yearly)[-5:]:
        row[f"return_{year}_pct"] = as_percent(yearly[year])

    # A stock listed for less than the full window would otherwise get
    # a "5-year" Sharpe ratio measured over a shorter period.
    if covers_years(prices, RISK_WINDOW_YEARS):
        daily = recent_daily_returns(prices)
        row["sharpe_5y"] = sharpe_ratio(daily, context.risk_free)
        row["downside_specific_volatility_pct"] = as_percent(
            downside_specific_volatility(daily, context.sp500_returns)
        )
    else:
        row["sharpe_5y"] = None
        row["downside_specific_volatility_pct"] = None

    return row


def sector_metrics(symbol, sector, industry, stock_cagr_5y_pct, context):
    """Sector benchmark growth, relative performance and industry share."""
    etf, sector_cagr = context.sector_cagr(sector)
    sector_cagr_pct = as_percent(sector_cagr)

    return {
        "sector_etf": etf,
        "sector_cagr_5y_pct": sector_cagr_pct,
        "relative_sector_cagr_5y_pct": (
            stock_cagr_5y_pct - sector_cagr_pct
            if stock_cagr_5y_pct is not None and sector_cagr_pct is not None
            else None
        ),
        "industry_market_cap_share_pct": as_percent(
            context.industry_shares.get(symbol.upper())
        ),
        "is_conglomerate": (
            None if industry is None else industry.lower() == "conglomerates"
        ),
    }
