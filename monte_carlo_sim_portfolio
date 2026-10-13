from scipy.stats import t
try:
    import usstock
except ModuleNotFoundError:
    # usstock is only required when individual stock tickers are configured.
    usstock = None
import statistics
import random
import math
from collections import defaultdict


# ============================================================
# PORTFOLIO ALLOCATIONS
# ============================================================

INITIAL_PERCENT_S_AND_P = 0.0
INITIAL_PERCENT_BONDS = 0.34
INITIAL_PERCENT_ASSETS = 0.0

LATER_PERCENT_S_AND_P = 0.45
LATER_PERCENT_BONDS = 0.55
LATER_PERCENT_ASSETS = 0.0

INITIAL_STOCKS_LIST = [("XLK", 0.15 * 0.36), ("VGT", 0.15 * 0.36), ("XLE", 0.125 * 0.36), ("XOP", 0.125 * 0.36), ("XLB", 0.125 * 0.36), ("VAW", 0.125 * 0.36), ("XLF", 0.1 * 0.36), ("KRE", 0.1 * 0.36), ("GRMN", 0.0225), ("KO", 0.0225), ("NYT", 0.0225), ("DECK", 0.0225), ("LDOS", 0.022125), ("ACN", 0.022125), ("FSLR", 0.022125), ("INTU", 0.022125), ("CF", 0.022), ("GD", 0.022), ("MLI", 0.022), ("AMP", 0.015), ("PYPL", 0.015), ("SD", 0.021), ("NEM", 0.003), ("ASND", 0.0015)]
LATER_STOCKS_LIST = []


# ============================================================
# CONTRIBUTIONS / WITHDRAWALS
# ============================================================

INITIAL_CONTRIBUTION = 150000
ANNUAL_WITHDRAWAL = 50000

# The 2033 variable withdrawal is determined in two stages:
# 1. At the beginning of 2031, project the beginning-of-2033
#    portfolio and establish the allowed withdrawal range.
# 2. At the beginning of 2033, calculate the withdrawal again
#    using the actual portfolio, then clamp it to that range.
# Exception: if the actual beginning-of-2033 portfolio is above
# WITHDRAWAL_RETAINED_VALUE, everything above it is withdrawn and the
# retained balance is moved entirely into bonds for 2033-2042. If that
# excess is smaller than the regular $50k payment, everything above
# WITHDRAWAL_RETAINED_FLOOR_VALUE is withdrawn instead.
#
# The variable withdrawal is separate from the regular $50k payment,
# which is always taken in 2033 as well. Amounts below are the
# variable portion only.
WITHDRAWAL_RETAINED_VALUE = 500000
WITHDRAWAL_RETAINED_FLOOR_VALUE = 450000
WITHDRAWAL_TARGET_VALUE = 500000
WITHDRAWAL_LARGE_AMOUNT = 200000
WITHDRAWAL_STANDARD_AMOUNT = 100000
WITHDRAWAL_HEALTHY_AMOUNT = 125000
WITHDRAWAL_VERY_HEALTHY_AMOUNT = 150000
WITHDRAWAL_MINIMUM_AMOUNT = 0

# The minimum (no variable withdrawal, only the regular $50k) is
# reserved for portfolios that cannot safely support the normal $100k
# variable withdrawal while still funding the remaining $50k annual
# payments at the central T-bill return.
# Regular $50k payments still owed AFTER 2033: 2034-2042 = 9.
# (The 2033 payment is counted separately as part of the 2033 draw.)
WITHDRAWAL_REMAINING_PAYMENT_COUNT = 9
WITHDRAWAL_VERY_HEALTHY_VALUE = 700000


# ============================================================
# MODEL PARAMETERS
# ============================================================

# Long-run return toward which individual-stock historical
# means are shrunk.
STOCK_TARGET_RETURN = 0.08

# 50% historical mean / 50% long-run target.
STOCK_MEAN_SHRINKAGE = 0.50

# Student-t degrees of freedom.
# Lower = fatter tails.
STOCK_T_DF = 5

# Winsorization limits for historical stock returns.
STOCK_LOWER_PERCENTILE = 0.05
STOCK_UPPER_PERCENTILE = 0.95

# ------------------------------------------------------------
# S&P 500 model
# ------------------------------------------------------------
# Returns are generated directly from a Student-t distribution.
# These are arithmetic annual-return targets.
SP_MEAN = 0.11
SP_STD = 0.194
SP_T_DF = 5


# ------------------------------------------------------------
# Treasury bill / CIR model
# ------------------------------------------------------------
# The T-bill rate is modeled as a mean-reverting CIR process:
#
#   dr = kappa * (theta - r) * dt
#        + sigma * sqrt(max(r, 0)) * dW
#
# The process stays non-negative and tends toward 4.25%.
TBILL_INITIAL_RATE = 0.0425
TBILL_LONG_RUN_RATE = 0.0425
TBILL_KAPPA = 0.75
TBILL_SIGMA = 0.12


# ------------------------------------------------------------
# Gold model
# ------------------------------------------------------------
# Gold gets a Student-t distribution with substantially heavier
# tails and higher volatility than the S&P 500.
GOLD_MEAN = 0.08
GOLD_STD = 0.25
GOLD_T_DF = 4


# ------------------------------------------------------------
# Individual-stock model
# ------------------------------------------------------------
STOCK_TARGET_RETURN = 0.08
STOCK_MEAN_SHRINKAGE = 0.50
STOCK_T_DF = 5

# Fraction of each stock's non-market variance driven by its sector
# factor. 0.30 means stocks in the same sector have meaningful extra
# positive correlation without moving identically.
STOCK_SECTOR_CORRELATION = 0.30

# Correlation between individual stocks and the S&P 500. This is
# replaced at startup by the average historical correlation of the
# configured stocks with the S&P 500, capped so stocks never move in
# lockstep with the index.
STOCK_SP_CORRELATION = 0.0
STOCK_MAX_SP_CORRELATION = 0.80

# Minimum overlapping years needed to use a stock's correlation.
STOCK_MIN_CORRELATION_YEARS = 5

# If a stock tuple only has (ticker, weight), it uses this sector.
DEFAULT_STOCK_SECTOR = "Other"

# Optional fallback sector map. Add tickers here if you do not
# want to put the sector directly into each stock tuple.
STOCK_SECTORS = {
    # "AAPL": "Technology",
    # "MSFT": "Technology",
    # "NVDA": "Technology",
    # "JPM": "Financials",
    # "XOM": "Energy",
    # "JNJ": "Healthcare",
}

# Winsorization limits for historical stock returns.
STOCK_LOWER_PERCENTILE = 0.05
STOCK_UPPER_PERCENTILE = 0.95


# ============================================================
# TREASURY BILL / CIR MODEL
# ============================================================

def simulated_tbill_return(current_rate):
    """
    Generate the next year's T-bill rate using a Cox-Ingersoll-Ross
    (CIR) mean-reverting process.

    Returns:
        next_rate: decimal annual T-bill return/rate
    """

    dt = 1.0
    shock = random.gauss(0.0, 1.0)

    next_rate = (
        current_rate
        + TBILL_KAPPA
        * (TBILL_LONG_RUN_RATE - current_rate)
        * dt
        + TBILL_SIGMA
        * math.sqrt(max(current_rate, 0.0))
        * math.sqrt(dt)
        * shock
    )

    # Euler discretization can very occasionally step below zero.
    # T-bill rates cannot be negative in this model.
    return max(next_rate, 0.0)


# ============================================================
# HISTORICAL DATA
# ============================================================

# Annual returns keyed by year, cached per ticker so the S&P
# correlation calculation does not download the same data twice.
_returns_by_year_cache = {}
SP500_CACHE_KEY = "__SP500_INDEX__"


def stock_data(ticker_list):
    """Annual-return lists (oldest first) for each configured stock."""

    annual_returns = []

    for ticker in ticker_list:

        returns_by_year = stock_returns_by_year(ticker[0])

        annual_returns.append([
            returns_by_year[year]
            for year in sorted(returns_by_year)
        ])

    return annual_returns


def stock_returns_by_year(ticker):
    """Return {year: annual return} for one stock ticker."""

    ticker = ticker.upper()

    if ticker in _returns_by_year_cache:
        return _returns_by_year_cache[ticker]

    if usstock is None:
        raise ModuleNotFoundError(
            "usstock is required when individual stock tickers are configured."
        )

    rows, source, errors = fetch_price_rows(ticker)

    if rows:
        print(f"{ticker}: {len(rows)} price rows from {source}")
    else:
        # Don't crash the whole run on one bad ticker. An empty history
        # makes stock_distribution_parameters fall back to the
        # STOCK_TARGET_RETURN / 25% volatility defaults.
        print(
            f"WARNING: no price data for {ticker}; using default "
            f"return/volatility. Errors: {' | '.join(errors)}"
        )

    returns_by_year = annual_returns_from_rows(rows)

    _returns_by_year_cache[ticker] = returns_by_year

    return returns_by_year


def _yahoo_rows_with_adjclose(ticker, interval):
    """
    Yahoo chart rows that keep the dividend-adjusted close.

    usstock.chart() drops Yahoo's "adjclose" series, which turns ETF and
    dividend-stock returns into price-only returns (understating them by
    the dividend yield). This calls the same endpoint but keeps adjclose,
    and falls back to usstock.chart() if the library's internals change.
    """

    import datetime as dt

    yahoo = getattr(usstock, "_yahoo", None)

    if yahoo is None or not hasattr(yahoo, "_chart"):
        return usstock.chart(
            ticker, period="max", interval=interval
        ).get("rows", [])

    result = yahoo._chart(ticker, "max", interval, "div,split")

    timestamps = result.get("timestamp") or []
    indicators = result.get("indicators") or {}
    quote = (indicators.get("quote") or [{}])[0] or {}
    closes = quote.get("close") or []
    adj = (indicators.get("adjclose") or [{}])[0] or {}
    adjcloses = adj.get("adjclose") or []

    rows = []

    for i, ts in enumerate(timestamps):
        close = closes[i] if i < len(closes) else None
        adjclose = adjcloses[i] if i < len(adjcloses) else None

        if close is None and adjclose is None:
            continue

        rows.append({
            "date": dt.datetime.fromtimestamp(
                ts, dt.timezone.utc
            ).strftime("%Y-%m-%d"),
            "close": close,
            "adjclose": adjclose,
        })

    return rows


def fetch_price_rows(ticker):
    """
    Try several sources, returning (rows, source_name, errors).

    1. Yahoo daily, max range (with dividend-adjusted closes)
    2. Yahoo monthly, max range
    3. CBOE delayed-quotes CDN (usstock.ohlcv_history), which also
       carries ETFs; price-only, no dividend adjustment
    """

    attempts = [
        ("Yahoo daily", lambda: _yahoo_rows_with_adjclose(ticker, "1d")),
        ("Yahoo monthly", lambda: _yahoo_rows_with_adjclose(ticker, "1mo")),
        ("CBOE", lambda: usstock.ohlcv_history(ticker).get("rows", [])),
    ]

    errors = []

    for name, fetch in attempts:
        try:
            rows = fetch()
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {str(exc)[:150]}")
            continue

        usable = [row for row in rows if _row_price(row) is not None]

        if len(usable) >= 2:
            return usable, name, errors

        errors.append(f"{name}: returned no usable rows")

    return [], None, errors


def sp500_returns_by_year():
    """Return {year: annual return} for the S&P 500 index."""

    if SP500_CACHE_KEY in _returns_by_year_cache:
        return _returns_by_year_cache[SP500_CACHE_KEY]

    if usstock is None:
        raise ModuleNotFoundError(
            "usstock is required when individual stock tickers are configured."
        )

    try:
        rows = usstock.ohlcv_history("SPX").get("rows", [])
    except Exception as exc:
        print(f"CBOE SPX failed ({type(exc).__name__}); trying Yahoo ^GSPC")
        rows = []

    if len(rows) < 2:
        rows, _, errors = fetch_price_rows("^GSPC")
        if not rows:
            print(f"WARNING: no S&P 500 data. Errors: {' | '.join(errors)}")

    returns_by_year = annual_returns_from_rows(rows)

    _returns_by_year_cache[SP500_CACHE_KEY] = returns_by_year

    return returns_by_year


def _row_price(row):
    """Dividend-adjusted close when available, otherwise the raw close."""

    for key in ("adjusted_close", "adjclose", "close"):
        value = row.get(key)
        if value is not None:
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if value > 0:
                return value

    return None


def annual_returns_from_rows(rows):
    """
    Convert price rows into {year: year-end to year-end return}.

    - Uses the last row of each year that actually has a price, so a
      missing final-day price doesn't throw away the whole year.
    - Only produces a return when the previous calendar year also has a
      year-end price, so a data gap can't turn a two-year move into one
      "annual" return.
    - Skips the first (partial) year and any year that isn't finished
      yet (e.g. the current year to date).
    - Uses the same price field (adjusted vs. raw) for both ends of
      each return so the two are comparable.
    """

    import datetime as dt

    rows = sorted(rows, key=lambda row: str(row["date"]))

    year_end = {}

    for row in rows:
        price = _row_price(row)

        if price is None:
            continue

        date_str = str(row["date"])[:10]
        year = int(date_str[:4])

        # Later rows overwrite earlier ones, leaving the last priced row.
        year_end[year] = (date_str, row)

    today = dt.date.today()

    returns_by_year = {}

    for year in sorted(year_end):

        if year - 1 not in year_end:
            continue

        end_date, end_row = year_end[year]
        _, start_row = year_end[year - 1]

        # Treat a year as complete only if it has data from mid-December
        # on (or it's a past year). The current year to date isn't a full
        # annual return and would understate volatility.
        if year >= today.year and end_date[5:] < "12-15":
            continue

        # Pick one price field that both endpoints have.
        start_price = end_price = None
        for key in ("adjusted_close", "adjclose", "close"):
            s = start_row.get(key)
            e = end_row.get(key)
            if s is not None and e is not None:
                start_price, end_price = float(s), float(e)
                break

        if not start_price or not end_price or start_price <= 0:
            continue

        returns_by_year[year] = end_price / start_price - 1

    return returns_by_year


# ============================================================
# S&P 500 MODEL
# ============================================================

def _student_t_scale(std, df):
    """Convert a desired standard deviation into scipy t scale."""
    if df <= 2:
        raise ValueError("Student-t df must be greater than 2.")
    return std * math.sqrt((df - 2) / df)


def simulated_sp_return():
    """
    Generate an S&P 500 annual return from a Student-t distribution
    centered at 11% with a 19.4% standard deviation.
    """

    return max(
        t.rvs(
            df=SP_T_DF,
            loc=SP_MEAN,
            scale=_student_t_scale(SP_STD, SP_T_DF)
        ),
        -1.0
    )


# ============================================================
# GOLD MODEL
# ============================================================

def simulated_asset_return():
    """
    Generate a gold annual return from a heavy-tailed Student-t
    distribution centered at 8%.

    Gold is intentionally more volatile and has heavier tails than
    the S&P 500.
    """

    return max(
        t.rvs(
            df=GOLD_T_DF,
            loc=GOLD_MEAN,
            scale=_student_t_scale(GOLD_STD, GOLD_T_DF)
        ),
        -1.0
    )


# ============================================================
# INDIVIDUAL STOCK MODEL
# ============================================================

def percentile_value(values, percentile):

    values = sorted(values)

    if not values:
        return 0

    position = percentile * (len(values) - 1)

    lower = int(position)
    upper = min(lower + 1, len(values) - 1)

    fraction = position - lower

    return (
        values[lower]
        + (values[upper] - values[lower]) * fraction
    )


def get_stock_sector(stock_entry):
    """
    Accept either:
        ("AAPL", 0.05)
    or:
        ("AAPL", 0.05, "Technology")

    The third element is preferred because it makes sector
    classification explicit.
    """

    ticker = stock_entry[0].upper()

    if len(stock_entry) >= 3 and stock_entry[2]:
        return str(stock_entry[2])

    return STOCK_SECTORS.get(
        ticker,
        DEFAULT_STOCK_SECTOR
    )


def stock_distribution_parameters(historical_returns):
    """
    Calculate the stock's historical mean/std after winsorization
    and shrink its mean toward the long-run equity target.
    """

    if not historical_returns:
        return STOCK_TARGET_RETURN, 0.25

    if len(historical_returns) < 5:
        return STOCK_TARGET_RETURN, 0.25

    lower_bound = percentile_value(
        historical_returns,
        STOCK_LOWER_PERCENTILE
    )

    upper_bound = percentile_value(
        historical_returns,
        STOCK_UPPER_PERCENTILE
    )

    winsorized = [
        min(max(value, lower_bound), upper_bound)
        for value in historical_returns
    ]

    historical_mean = statistics.mean(winsorized)
    historical_std = statistics.stdev(winsorized)

    adjusted_mean = (
        historical_mean * (1 - STOCK_MEAN_SHRINKAGE)
        + STOCK_TARGET_RETURN * STOCK_MEAN_SHRINKAGE
    )

    return adjusted_mean, max(historical_std, 0.01)


def simulated_individual_stock_return(
    historical_returns,
    sector_shock=0.0,
    market_shock=0.0
):
    """
    Generate one stock return.

    The stock retains its own historical mean/volatility. A shared
    S&P 500 shock gives every stock a correlation of
    STOCK_SP_CORRELATION with the index, and a common sector shock
    adds extra correlation among stocks in the same sector.

    sector_shock should be a standard-normal draw shared by every
    stock in the same sector for that simulated year.
    market_shock should be that year's standardized S&P 500 return.
    """

    mean, historical_std = stock_distribution_parameters(
        historical_returns
    )

    beta = max(
        -STOCK_MAX_SP_CORRELATION,
        min(STOCK_SP_CORRELATION, STOCK_MAX_SP_CORRELATION)
    )

    rho = max(
        0.0,
        min(STOCK_SECTOR_CORRELATION, 0.95)
    )

    # The market component accounts for beta^2 of variance.
    market_component = (
        beta
        * historical_std
        * market_shock
    )

    # The sector and idiosyncratic components split the rest.
    non_market_variance = 1.0 - beta ** 2

    # The common sector component accounts for rho of that remainder.
    sector_component = (
        math.sqrt(rho * non_market_variance)
        * historical_std
        * sector_shock
    )

    # The idiosyncratic component gets the remaining variance.
    idio_std = historical_std * math.sqrt(
        (1.0 - rho) * non_market_variance
    )

    idio_scale = _student_t_scale(
        idio_std,
        STOCK_T_DF
    )

    idio_component = t.rvs(
        df=STOCK_T_DF,
        loc=0.0,
        scale=idio_scale
    )

    simulated_return = (
        mean
        + market_component
        + sector_component
        + idio_component
    )

    return max(simulated_return, -1.0)


# ============================================================
# PORTFOLIO RETURN
# ============================================================

def calculate_return(
    annual_returns,
    ticker_list,
    stocks,
    bonds,
    assets,
    tbill_rate,
    show_breakdown=False
):
    """
    Calculate one year's portfolio return.

    Returns:
        (total_return, next_tbill_rate)
    """

    individual_stock_return = 0.0

    # The S&P 500 return is drawn first so individual stocks can
    # share its shock.
    sp_return = simulated_sp_return()
    market_shock = (sp_return - SP_MEAN) / SP_STD

    # One common shock per sector per simulated year.
    sector_shocks = {}

    for i in range(len(ticker_list)):

        sector = get_stock_sector(ticker_list[i])

        if sector not in sector_shocks:
            sector_shocks[sector] = random.gauss(0.0, 1.0)

        simulated_return = simulated_individual_stock_return(
            annual_returns[i],
            sector_shocks[sector],
            market_shock
        )

        individual_stock_return += (
            simulated_return * ticker_list[i][1]
        )

    sp_contribution = stocks * sp_return

    # T-bills earn the rate in effect at the start of the year (that's
    # the rate you lock in when you buy them). The rate then evolves
    # for next year.
    bond_contribution = bonds * tbill_rate

    next_tbill_rate = simulated_tbill_return(
        tbill_rate
    )

    asset_return = simulated_asset_return()
    asset_contribution = assets * asset_return

    total_return = (
        individual_stock_return
        + sp_contribution
        + bond_contribution
        + asset_contribution
    )

    if show_breakdown:

        print("\n--- RETURN BREAKDOWN ---")

        print(
            f"Individual stocks: "
            f"{individual_stock_return * 100:.2f}%"
        )

        print(
            f"S&P 500:           "
            f"{sp_contribution * 100:.2f}%"
        )

        print(
            f"T-bills:            "
            f"{bond_contribution * 100:.2f}%"
        )

        print(
            f"Gold/other assets:  "
            f"{asset_contribution * 100:.2f}%"
        )

        print(
            f"T-bill rate:        "
            f"{tbill_rate * 100:.2f}%"
        )

        print(
            f"TOTAL:              "
            f"{total_return * 100:.2f}%"
        )

        print("------------------------\n")

    return total_return, next_tbill_rate


# ============================================================
# ALLOCATION CHECK
# ============================================================

def check_allocation(
    stocks,
    bonds,
    assets,
    ticker_list
):

    stock_weight = sum(
        weight
        for _, weight, *rest in ticker_list
    )

    total = (
        stocks
        + bonds
        + assets
        + stock_weight
    )

    print("\n--- ALLOCATION CHECK ---")

    print(f"S&P 500:       {stocks:.2%}")
    print(f"T-bills:       {bonds:.2%}")
    print(f"Gold/other:     {assets:.2%}")
    print(f"Individual:    {stock_weight:.2%}")
    print(f"TOTAL:         {total:.2%}")

    print("------------------------")

    if abs(total - 1.0) > 0.001:

        print("!" * 70)
        print(
            f"WARNING: Portfolio allocation is {total:.2%}, not 100%.\n"
            f"Every return is multiplied by these weights, so the "
            f"simulation behaves as if {total:.2f}x invested (borrowing "
            f"for free). Results will be distorted."
        )
        print("!" * 70)


# ============================================================
# GET DATA
# ============================================================

print(
    "Be patient. Getting ur data takes time..."
)

initial_annual_returns = stock_data(
    INITIAL_STOCKS_LIST
)

later_annual_returns = stock_data(
    LATER_STOCKS_LIST
)


# ============================================================
# STOCK / S&P 500 CORRELATION
# ============================================================

def average_sp_correlation(ticker_lists):
    """
    Average historical correlation between each configured stock's
    annual returns and the S&P 500's annual returns over the years
    they overlap. Returns 0.0 when no usable stock is configured.
    """

    tickers = sorted({
        stock_entry[0].upper()
        for ticker_list in ticker_lists
        for stock_entry in ticker_list
    })

    if not tickers:
        return 0.0

    sp_by_year = sp500_returns_by_year()

    correlations = []

    for ticker in tickers:

        stock_by_year = stock_returns_by_year(ticker)

        shared_years = sorted(
            set(stock_by_year) & set(sp_by_year)
        )

        if len(shared_years) < STOCK_MIN_CORRELATION_YEARS:
            print(
                f"{ticker}: only {len(shared_years)} years overlap "
                f"with the S&P 500; excluded from correlation."
            )
            continue

        stock_returns = [stock_by_year[year] for year in shared_years]
        sp_returns = [sp_by_year[year] for year in shared_years]

        try:
            correlation = statistics.correlation(
                stock_returns,
                sp_returns
            )
        except statistics.StatisticsError:
            # One of the series is constant.
            continue

        print(
            f"{ticker}: S&P 500 correlation {correlation:.3f} "
            f"({len(shared_years)} years)"
        )

        correlations.append(correlation)

    if not correlations:
        return 0.0

    return statistics.mean(correlations)


historical_sp_correlation = average_sp_correlation(
    [INITIAL_STOCKS_LIST, LATER_STOCKS_LIST]
)

STOCK_SP_CORRELATION = max(
    -STOCK_MAX_SP_CORRELATION,
    min(historical_sp_correlation, STOCK_MAX_SP_CORRELATION)
)

if INITIAL_STOCKS_LIST or LATER_STOCKS_LIST:
    print(
        f"Average stock/S&P 500 correlation: "
        f"{historical_sp_correlation:.3f} "
        f"(using {STOCK_SP_CORRELATION:.3f})"
    )


# ============================================================
# CHECK ALLOCATIONS
# ============================================================

check_allocation(
    INITIAL_PERCENT_S_AND_P,
    INITIAL_PERCENT_BONDS,
    INITIAL_PERCENT_ASSETS,
    INITIAL_STOCKS_LIST
)

check_allocation(
    LATER_PERCENT_S_AND_P,
    LATER_PERCENT_BONDS,
    LATER_PERCENT_ASSETS,
    LATER_STOCKS_LIST
)


# ============================================================
# WITHDRAWAL / ALLOCATION POLICY
# ============================================================

def expected_initial_portfolio_return():
    """Return the central expected nominal return for 2027-2032."""

    individual_stock_weight = sum(
        weight
        for _, weight, *rest in INITIAL_STOCKS_LIST
    )

    individual_expected = 0.0

    for i, stock_entry in enumerate(INITIAL_STOCKS_LIST):
        if i < len(initial_annual_returns):
            mean, _ = stock_distribution_parameters(
                initial_annual_returns[i]
            )
        else:
            mean = STOCK_TARGET_RETURN

        individual_expected += stock_entry[1] * mean

    return (
        INITIAL_PERCENT_S_AND_P * SP_MEAN
        + INITIAL_PERCENT_BONDS * TBILL_LONG_RUN_RATE
        + INITIAL_PERCENT_ASSETS * GOLD_MEAN
        + individual_expected
    )


def later_equity_baseline():
    """Total S&P + individual-stock weight in the default later mix."""

    return (
        LATER_PERCENT_S_AND_P
        + sum(
            weight
            for _, weight, *rest in LATER_STOCKS_LIST
        )
    )


def expected_later_equity_return():
    """
    Central expected nominal return for the stock/equity sleeve: the
    S&P 500 and the default later-phase individual stocks, blended by
    their share of the sleeve.
    """

    total_equity_baseline = later_equity_baseline()

    if total_equity_baseline <= 0:
        return SP_MEAN

    expected = LATER_PERCENT_S_AND_P * SP_MEAN

    for i, stock_entry in enumerate(LATER_STOCKS_LIST):
        if i < len(later_annual_returns):
            mean, _ = stock_distribution_parameters(
                later_annual_returns[i]
            )
        else:
            mean = STOCK_TARGET_RETURN

        expected += stock_entry[1] * mean

    return expected / total_equity_baseline


def project_initial_portfolio_to_2033(portfolio_value):
    """
    Project beginning-2031 value to beginning-2033 using central
    nominal return assumptions and the fixed 2027-2032 allocation.
    """

    annual_return = expected_initial_portfolio_return()

    # Beginning 2031 -> beginning 2032 -> beginning 2033.
    return portfolio_value * (1 + annual_return) ** 2


def calculate_variable_withdrawal(portfolio_before_withdrawal):
    """Choose the variable withdrawal based on actual funding health.

    The minimum amount is reserved for a portfolio that cannot safely
    fund the variable withdrawal plus the ten scheduled $50k payments
    at the central T-bill return. Otherwise, healthier portfolios get
    a larger variable withdrawal, with tiers based on the balance.
    The returned amount excludes the regular $50k 2033 payment.
    """

    # Danger threshold: enough to cover the STANDARD 2033 withdrawal
    # (the $100k variable draw + the regular $50k) and the remaining
    # $50k payments, assuming the safe T-bill rate. Below this, taking
    # the $100k variable draw could leave the later payments unfunded.
    danger_required_balance = required_starting_balance(
        [WITHDRAWAL_STANDARD_AMOUNT + ANNUAL_WITHDRAWAL]
        + [ANNUAL_WITHDRAWAL] * WITHDRAWAL_REMAINING_PAYMENT_COUNT,
        TBILL_LONG_RUN_RATE
    )

    if portfolio_before_withdrawal < danger_required_balance:
        return WITHDRAWAL_MINIMUM_AMOUNT

    if portfolio_before_withdrawal < WITHDRAWAL_TARGET_VALUE:
        return WITHDRAWAL_STANDARD_AMOUNT

    if portfolio_before_withdrawal < WITHDRAWAL_VERY_HEALTHY_VALUE:
        return WITHDRAWAL_HEALTHY_AMOUNT

    return WITHDRAWAL_VERY_HEALTHY_AMOUNT

def establish_variable_withdrawal_range(portfolio_at_beginning_2031):
    """
    Set the allowable 2033 variable-withdrawal range using only
    information available at the beginning of 2031.
    """

    projected_2033 = project_initial_portfolio_to_2033(
        portfolio_at_beginning_2031
    )

    projected_withdrawal = calculate_variable_withdrawal(
        projected_2033
    )

    return (
        WITHDRAWAL_MINIMUM_AMOUNT,
        max(
            WITHDRAWAL_MINIMUM_AMOUNT,
            projected_withdrawal
        ),
        projected_2033
    )


def clamp_variable_withdrawal(
    actual_portfolio,
    lower_bound,
    upper_bound
):
    """Recalculate the 2033 withdrawal and keep it inside its 2031 range."""

    candidate = calculate_variable_withdrawal(
        actual_portfolio
    )

    return max(
        lower_bound,
        min(candidate, upper_bound)
    )


def required_starting_balance(
    withdrawals,
    annual_return
):
    """
    Balance needed today to fund the supplied withdrawals if the
    portfolio compounds at a constant nominal annual return.

    The first withdrawal occurs immediately, matching the simulation.
    The simulation's future withdrawals are level, so the remaining
    payments can be valued as a geometric series.
    """

    if not withdrawals:
        return 0.0

    first_withdrawal = withdrawals[0]

    if len(withdrawals) == 1:
        return first_withdrawal

    future_withdrawal = withdrawals[1]
    future_count = len(withdrawals) - 1

    discount = 1.0 / (1.0 + annual_return)

    if abs(1.0 - discount) < 1e-12:
        future_value = future_withdrawal * future_count
    else:
        future_value = (
            future_withdrawal
            * discount
            * (1.0 - discount ** future_count)
            / (1.0 - discount)
        )

    return first_withdrawal + future_value


def solve_required_return(
    portfolio_value,
    withdrawals
):
    """
    Solve for the constant nominal annual return needed to fund all
    remaining withdrawals, using the same beginning-of-year timing
    as the simulation.
    """

    if portfolio_value <= 0:
        return float("inf")

    if not withdrawals:
        return 0.0

    # For the allocation decision we only need to distinguish rates
    # between the safe T-bill return and the central stock return.
    # Outside that interval the caller will use 0% or 100% stocks.
    low = TBILL_LONG_RUN_RATE
    high = expected_later_equity_return()

    low_balance = required_starting_balance(withdrawals, low)
    high_balance = required_starting_balance(withdrawals, high)

    if portfolio_value >= low_balance:
        return low

    if portfolio_value <= high_balance:
        return high

    for _ in range(45):
        mid = (low + high) / 2.0
        required = required_starting_balance(
            withdrawals,
            mid
        )

        if required > portfolio_value:
            low = mid
        else:
            high = mid

    return (low + high) / 2.0


def determine_later_allocation(
    portfolio_value,
    withdrawals
):
    """
    Determine the 2033-2042 stock/bond allocation from projected
    funding needs.

    The stock sleeve is the S&P 500 plus the individual stocks in
    the default later-phase allocation (LATER_STOCKS_LIST); no other
    stocks are ever added. Its expected return blends all of them.

    - If T-bills alone can fund the remaining withdrawals: 100% bonds.
    - If stocks are not enough on a central-return basis: 100% stocks.
    - Otherwise interpolate the stock weight between those two cases.

    The stock weight is split between the S&P 500 and each later
    stock in proportion to their default later-phase weights.

    Later assets are retained only as a configured baseline; because
    LATER_PERCENT_ASSETS is currently 0%, the current implementation
    reduces to a stock/bond decision.

    Returns:
        (sp_weight, bond_weight, asset_weight, later_stocks), where
        later_stocks is LATER_STOCKS_LIST with rescaled weights.
    """

    total_equity_baseline = later_equity_baseline()

    if total_equity_baseline <= 0:
        # No stock sleeve is configured, so the only feasible allocation
        # is bonds plus any explicitly configured later assets.
        return (
            0.0,
            1.0 - LATER_PERCENT_ASSETS,
            LATER_PERCENT_ASSETS,
            []
        )

    bond_return = TBILL_LONG_RUN_RATE
    equity_return = expected_later_equity_return()

    # Stocks are not expected to beat T-bills, so taking stock risk
    # cannot help fund the withdrawals.
    if equity_return <= bond_return:
        return 0.0, 1.0, 0.0, []

    # Compare the current portfolio with the amount that would be
    # required if the remaining payments were funded entirely from
    # T-bills versus the central stock-return assumption.
    bond_required = required_starting_balance(
        withdrawals,
        bond_return
    )

    equity_required = required_starting_balance(
        withdrawals,
        equity_return
    )

    # The portfolio already generates enough growth with T-bills.
    if portfolio_value >= bond_required:
        stock_weight = 0.0

    # Even the central stock-return assumption is insufficient.
    elif portfolio_value <= equity_required:
        stock_weight = 1.0

    # Otherwise interpolate between the all-bond and all-stock funding
    # cases. A portfolio only slightly short of the all-bond requirement
    # gets a small stock allocation; a much larger shortfall gets a
    # correspondingly stronger stock concentration.
    else:
        stock_weight = (
            bond_required - portfolio_value
        ) / (
            bond_required - equity_required
        )

        stock_weight = max(0.0, min(1.0, stock_weight))

    # Keep the configured internal equity composition: every holding
    # in the stock sleeve is scaled by the same factor.
    scale = stock_weight / total_equity_baseline

    sp_weight = LATER_PERCENT_S_AND_P * scale

    later_stocks = [
        (
            stock_entry[0],
            stock_entry[1] * scale,
            *stock_entry[2:]
        )
        for stock_entry in LATER_STOCKS_LIST
    ]

    asset_weight = 0.0
    bond_weight = 1.0 - stock_weight

    return (
        sp_weight,
        bond_weight,
        asset_weight,
        later_stocks
    )


# ============================================================
# MONTE CARLO SETTINGS
# ============================================================

NUM_SIMULATIONS = 1000

num_worked = 0
num_times = 0

failure_count_by_year = defaultdict(int)

tracked_years = [
    2027, 2028, 2029, 2030,
    2031, 2032, 2033, 2034,
    2035, 2036, 2037, 2038,
    2039, 2040, 2041, 2042
]

year_value_totals = {
    year: 0.0
    for year in tracked_years
}

final_values = []
successful_final_values = []
variable_withdrawal_values = []


# ============================================================
# RUN MONTE CARLO
# ============================================================

for simulation in range(NUM_SIMULATIONS):

    portfolio_value = 300000
    success = True

    # Each Monte Carlo path gets its own evolving T-bill rate.
    tbill_rate = TBILL_INITIAL_RATE

    simulation_values = {
        year: 0.0
        for year in tracked_years
    }

    # The 2033 variable withdrawal range is not allowed to use
    # information from 2031-2032 future outcomes. It is established
    # exactly at the beginning of 2031.
    variable_withdrawal_low = None
    variable_withdrawal_high = None
    projected_2033_value = None
    actual_variable_withdrawal = None

    # --------------------------------------------------------
    # 2027
    # --------------------------------------------------------

    annual_return, tbill_rate = calculate_return(
        initial_annual_returns,
        INITIAL_STOCKS_LIST,
        INITIAL_PERCENT_S_AND_P,
        INITIAL_PERCENT_BONDS,
        INITIAL_PERCENT_ASSETS,
        tbill_rate
    )

    portfolio_value *= (1 + annual_return)

    portfolio_value += INITIAL_CONTRIBUTION

    simulation_values[2027] = portfolio_value

    # --------------------------------------------------------
    # 2028-2030
    # --------------------------------------------------------

    current_year = 2028

    for _ in range(3):

        annual_return, tbill_rate = calculate_return(
            initial_annual_returns,
            INITIAL_STOCKS_LIST,
            INITIAL_PERCENT_S_AND_P,
            INITIAL_PERCENT_BONDS,
            INITIAL_PERCENT_ASSETS,
            tbill_rate
        )

        portfolio_value *= (1 + annual_return)

        simulation_values[current_year] = portfolio_value
        current_year += 1

    # --------------------------------------------------------
    # Beginning of 2031: establish variable-withdrawal range
    # --------------------------------------------------------

    (
        variable_withdrawal_low,
        variable_withdrawal_high,
        projected_2033_value
    ) = establish_variable_withdrawal_range(
        portfolio_value
    )

    # --------------------------------------------------------
    # 2031-2032
    # --------------------------------------------------------

    current_year = 2031

    for _ in range(2):

        annual_return, tbill_rate = calculate_return(
            initial_annual_returns,
            INITIAL_STOCKS_LIST,
            INITIAL_PERCENT_S_AND_P,
            INITIAL_PERCENT_BONDS,
            INITIAL_PERCENT_ASSETS,
            tbill_rate
        )

        portfolio_value *= (1 + annual_return)

        simulation_values[current_year] = portfolio_value
        current_year += 1

    # --------------------------------------------------------
    # Beginning of 2033: calculate variable withdrawal
    # --------------------------------------------------------

    # Above the retained value: withdraw the excess and hold the
    # remaining balance entirely in bonds. The regular 2033 $50k
    # payment comes out of that excess, so the variable portion is
    # whatever is left over. If the excess is under $50k, withdraw
    # everything above the lower floor instead. Otherwise use the
    # tiered policy clamped to the range set in 2031.
    all_bonds_after_2033 = portfolio_value > WITHDRAWAL_RETAINED_VALUE

    if all_bonds_after_2033:
        if (
            portfolio_value - WITHDRAWAL_RETAINED_VALUE
            >= ANNUAL_WITHDRAWAL
        ):
            retained_value = WITHDRAWAL_RETAINED_VALUE
        else:
            retained_value = WITHDRAWAL_RETAINED_FLOOR_VALUE

        actual_variable_withdrawal = (
            portfolio_value
            - retained_value
            - ANNUAL_WITHDRAWAL
        )
    else:
        actual_variable_withdrawal = clamp_variable_withdrawal(
            portfolio_value,
            variable_withdrawal_low,
            variable_withdrawal_high
        )

    variable_withdrawal_values.append(actual_variable_withdrawal)

    # --------------------------------------------------------
    # 2033-2042
    # --------------------------------------------------------

    current_year = 2033

    for j in range(10):

        # Current-year withdrawal happens at the beginning of the year.
        # 2033 takes the variable withdrawal plus the regular $50k.
        if current_year == 2033:
            current_withdrawal = (
                actual_variable_withdrawal + ANNUAL_WITHDRAWAL
            )
        else:
            current_withdrawal = ANNUAL_WITHDRAWAL

        if portfolio_value <= current_withdrawal:

            portfolio_value = 0
            success = False

            failure_count_by_year[current_year] += 1

            for year in tracked_years:
                if year >= current_year:
                    simulation_values[year] = 0.0

            break

        # Determine how much return is needed from this point forward.
        remaining_withdrawals = [
            current_withdrawal
        ] + [
            ANNUAL_WITHDRAWAL
            for _ in range(9 - j)
        ]

        if all_bonds_after_2033:
            later_sp, later_bonds, later_assets, later_stocks = (
                0.0, 1.0, 0.0, []
            )
        else:
            (
                later_sp,
                later_bonds,
                later_assets,
                later_stocks
            ) = determine_later_allocation(
                portfolio_value,
                remaining_withdrawals
            )

        portfolio_value -= current_withdrawal

        annual_return, tbill_rate = calculate_return(
            later_annual_returns,
            later_stocks,
            later_sp,
            later_bonds,
            later_assets,
            tbill_rate
        )

        portfolio_value *= (1 + annual_return)

        if portfolio_value <= 0:

            portfolio_value = 0
            success = False

            failure_count_by_year[current_year] += 1

            for year in tracked_years:
                if year >= current_year:
                    simulation_values[year] = 0.0

            break

        simulation_values[current_year] = portfolio_value
        current_year += 1

    # --------------------------------------------------------
    # Yearly totals
    # --------------------------------------------------------

    for year in tracked_years:
        year_value_totals[year] += simulation_values[year]

    # --------------------------------------------------------
    # Final values
    # --------------------------------------------------------

    final_value = simulation_values[2042]

    final_values.append(final_value)

    if success:

        successful_final_values.append(final_value)
        num_worked += 1

    num_times += 1

    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    if num_times % 100 == 0 or num_times == NUM_SIMULATIONS:
        print(
            f"\rSimulation {num_times:,}/{NUM_SIMULATIONS:,} | "
            f"success rate so far: {num_worked / num_times:.2%}",
            end="",
            flush=True
        )


# ============================================================
# FINAL RESULTS
# ============================================================

print()
print()

print("=" * 70)
print("FINAL MONTE CARLO RESULTS")
print("=" * 70)
print(
    f"Average 2033 variable withdrawal: "
    f"${statistics.mean(variable_withdrawal_values):,.0f}"
)
print(
    f"Median 2033 variable withdrawal:  "
    f"${statistics.median(variable_withdrawal_values):,.0f}"
)

success_rate = (
    num_worked
    / num_times
    * 100
)

print(
    f"Simulations:       {num_times:,}"
)

print(
    f"Successful:        {num_worked:,}"
)

print(
    f"Failed:            "
    f"{num_times - num_worked:,}"
)

print(
    f"Success Rate:      "
    f"{success_rate:.2f}%"
)


# ============================================================
# 2042 STATISTICS
# ============================================================

print()
print("=" * 70)
print("2042 PORTFOLIO STATISTICS")
print("=" * 70)

average_final = statistics.mean(
    final_values
)

median_final = statistics.median(
    final_values
)

sorted_values = sorted(
    final_values
)

p10 = sorted_values[
    int(len(sorted_values) * 0.10)
]

p25 = sorted_values[
    int(len(sorted_values) * 0.25)
]

p75 = sorted_values[
    int(len(sorted_values) * 0.75)
]

p90 = sorted_values[
    int(len(sorted_values) * 0.90)
]

maximum_final = max(
    final_values
)

if successful_final_values:

    average_successful = statistics.mean(
        successful_final_values
    )

else:

    average_successful = 0


print(
    f"Average (all):        "
    f"${average_final:,.2f}"
)

print(
    f"Average (successful): "
    f"${average_successful:,.2f}"
)

print(
    f"Median:               "
    f"${median_final:,.2f}"
)

print(
    f"10th percentile:      "
    f"${p10:,.2f}"
)

print(
    f"25th percentile:      "
    f"${p25:,.2f}"
)

print(
    f"75th percentile:      "
    f"${p75:,.2f}"
)

print(
    f"90th percentile:      "
    f"${p90:,.2f}"
)

print(
    f"Maximum:              "
    f"${maximum_final:,.2f}"
)


# ============================================================
# FAILURE YEARS
# ============================================================

print()
print("=" * 70)
print("FAILURES BY YEAR")
print("=" * 70)

if failure_count_by_year:

    for year in sorted(
        failure_count_by_year
    ):

        count = failure_count_by_year[
            year
        ]

        percentage = (
            count
            / num_times
            * 100
        )

        print(
            f"{year}: "
            f"{count:,} failures "
            f"({percentage:.2f}%)"
        )

else:

    print("No failures.")


# ============================================================
# AVERAGE PORTFOLIO VALUE BY YEAR
# ============================================================

print()
print("=" * 70)
print("AVERAGE PORTFOLIO VALUE BY YEAR")
print("=" * 70)

for year in tracked_years:

    average_value = (
        year_value_totals[year]
        / num_times
    )

    print(
        f"{year}: "
        f"${average_value:,.2f}"
    )

print("=" * 70)
