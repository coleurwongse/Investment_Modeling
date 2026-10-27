"""
Scrape fundamentals, risk and return metrics for a list of tickers
and save them to scraped_data.csv.

Usage:
    python scraper.py                 # uses TICKERS below
    python scraper.py AAPL MSFT JPM   # tickers from the command line

All data comes from the usstock library (see stock_metrics.py).

The DCF is an estimate, not reported data: it extrapolates historical
growth, decaying toward 3% terminal growth (settings in
stock_metrics.py). Sector terminal growth isn't scraped.
"""

import sys
from concurrent.futures import ThreadPoolExecutor

from stock_metrics import (
    MarketContext,
    balance_metrics,
    company_profile,
    dcf_metrics,
    income_metrics,
    key_statistics,
    latest_prices,
    load_shared_data,
    price_histories,
    price_metrics,
    ratio_trend_metrics,
    sector_metrics,
    statement,
    valuation_and_health_metrics,
    write_csv,
)


# ============================================================
# SETTINGS
# ============================================================

TICKERS = ["AAPL", "MSFT", "JPM"]

OUTPUT_FILE = "scraped_data.csv"

MAX_WORKERS = 4


# ============================================================
# COLUMN ORDER
# ============================================================

COLUMNS = [
    "ticker", "name", "price", "sector", "industry", "is_conglomerate",
    "market_cap",

    # Valuation
    "pe_ratio", "ev_ebitda", "ev_ebitda_last_fy",
    "ev_ebitda_change_vs_last_fy_pct",

    # DCF (estimate)
    "dcf_value_per_share", "dcf_upside_pct", "dcf_base_fcf",
    "dcf_base_fcf_periods", "dcf_start_growth_pct",
    "dcf_growth_basis", "dcf_discount_rate_pct",
    "dcf_terminal_value_share_pct", "dcf_note",

    # Leverage
    "debt_to_equity", "debt_to_equity_trend",
    "debt_to_equity_slope_per_year", "debt_to_equity_change_over_period",
    "debt_to_equity_history",
    "liabilities_to_assets", "liabilities_to_assets_trend",
    "liabilities_to_assets_slope_per_year", "liabilities_to_assets_history",

    # Year-over-year TTM changes
    "revenue_yoy_ttm_pct", "eps_yoy_ttm_pct", "fcf_yoy_ttm_pct",
    "operating_cash_flow_yoy_ttm_pct",

    # Growth. Return-based columns come in pairs: _no_div is price
    # change only, _with_div includes reinvested dividends.
    "revenue_cagr_pct", "revenue_cagr_years",
    "cagr_3y_pct_no_div", "cagr_3y_pct_with_div",
    "cagr_5y_pct_no_div", "cagr_5y_pct_with_div",
    "cagr_10y_pct_no_div", "cagr_10y_pct_with_div",

    # Sector
    "sector_etf",
    "sector_cagr_5y_pct_no_div", "sector_cagr_5y_pct_with_div",
    "relative_sector_cagr_5y_pct_no_div",
    "relative_sector_cagr_5y_pct_with_div",
    "industry_market_cap_share_pct",

    # Margins
    "gross_margin_pct", "gross_margin_trend", "gross_margin_slope_pp_per_year",
    "operating_margin_pct", "operating_margin_trend",
    "operating_margin_slope_pp_per_year",
    "profit_margin_pct", "profit_margin_trend",
    "profit_margin_slope_pp_per_year",
    "fcf_margin_pct", "fcf_margin_trend", "fcf_margin_slope_pp_per_year",
    "ebitda_margin_pct",

    # Risk / return
    "beta_5y", "sharpe_5y_no_div", "sharpe_5y_with_div",
    "downside_specific_volatility_pct_no_div",
    "downside_specific_volatility_pct_with_div",
    "earnings_volatility", "return_1y_pct_no_div", "return_1y_pct_with_div",
    # return_<year>_pct_<no_div/with_div> columns are inserted here

    # Financial health
    "altman_z", "piotroski_f", "interest_coverage",

    "errors",
]


# ============================================================
# SCRAPING
# ============================================================

def scrape_ticker(symbol, context, prices_now):
    """Collect every metric for one ticker. Failures are noted, not fatal."""

    symbol = symbol.upper()
    row = {"ticker": symbol, "price": prices_now.get(symbol)}
    errors = []

    def attempt(label, function):
        try:
            return function()
        except Exception as error:
            errors.append(f"{label}: {error}")
            return None

    # Each source and each calculation is attempted separately so one
    # bad page or unexpected value only blanks its own columns.
    stats = attempt("statistics", lambda: key_statistics(symbol))
    if stats:
        row.update(attempt(
            "statistics", lambda: valuation_and_health_metrics(stats)
        ) or {})

    ratios_data = attempt("ratios", lambda: statement(symbol, "ratios"))
    if ratios_data:
        row.update(attempt(
            "ratios", lambda: ratio_trend_metrics(ratios_data)
        ) or {})

    balance_data = attempt("balance sheet", lambda: statement(symbol, "balance"))
    if balance_data:
        row.update(attempt(
            "balance sheet", lambda: balance_metrics(balance_data)
        ) or {})

    income_data = attempt("income statement", lambda: statement(symbol, "income"))
    cashflow_data = attempt("cash flow", lambda: statement(symbol, "cashflow"))
    if income_data:
        row.update(attempt(
            "income statement",
            lambda: income_metrics(income_data, cashflow_data or {})
        ) or {})

    if stats:
        row.update(attempt(
            "dcf",
            lambda: dcf_metrics(
                stats, income_data, cashflow_data,
                row.get("price"), context.risk_free
            )
        ) or {})

    profile = attempt("profile", lambda: company_profile(symbol))
    sector, industry = profile if profile else (None, None)
    row["sector"] = sector
    row["industry"] = industry

    histories = attempt("price history", lambda: price_histories(symbol))
    if histories:
        if histories[1] is None:
            errors.append(
                "dividends: Yahoo's dividend-adjusted prices are missing "
                "or invalid, so the _with_div columns are blank"
            )
        row.update(attempt(
            "price history", lambda: price_metrics(histories, context)
        ) or {})

    sector_row = attempt(
        "sector",
        lambda: sector_metrics(symbol, sector, industry, row, context)
    )
    if sector_row:
        row.update(sector_row)

    name = next(
        (s["name"] for s in context.universe if s["symbol"] == symbol),
        None
    )
    row["name"] = name

    row["errors"] = " | ".join(errors)
    return row


def ordered_columns(rows):
    """COLUMNS plus the calendar-year return columns that appeared."""
    year_columns = sorted({
        key for row in rows for key in row
        if key.startswith("return_") and key[7:11].isdigit()
    })
    insert_at = COLUMNS.index("return_1y_pct_with_div") + 1
    return COLUMNS[:insert_at] + year_columns + COLUMNS[insert_at:]


def main():
    # stockanalysis writes class shares as BRK.B; accept BRK-B too.
    # dict.fromkeys drops duplicates but keeps the original order.
    tickers = list(dict.fromkeys(
        t.strip().upper().replace("-", ".")
        for t in (sys.argv[1:] or TICKERS)
        if t.strip()
    ))
    if not tickers:
        raise SystemExit("No tickers given.")
    print(f"Scraping {len(tickers)} tickers...")

    context = MarketContext()

    # Shared data first so worker threads only read it.
    load_shared_data(context, include_universe=True)
    prices_now = latest_prices(tickers)

    with ThreadPoolExecutor(MAX_WORKERS) as executor:
        rows = list(executor.map(
            lambda symbol: scrape_ticker(symbol, context, prices_now),
            tickers
        ))

    for row in rows:
        if not row["errors"]:
            status = "ok"
        elif row.get("price") is None and row.get("market_cap") is None:
            status = "not found"
        else:
            failed = [e.split(":", 1)[0] for e in row["errors"].split(" | ")]
            status = f"partial (missing {', '.join(failed)}; see errors column)"
        print(f"  {row['ticker']}: {status}")

    columns = ordered_columns(rows)

    saved_to = write_csv(OUTPUT_FILE, columns, rows)
    print(f"Saved {len(rows)} rows to {saved_to}")


if __name__ == "__main__":
    main()
