"""
Screen every stock on https://stockanalysis.com/stocks/ and save the
ones that pass all filters to filtered_stocks.csv.

Filters run cheapest-first so expensive downloads are only made for
stocks that are still in the running:
    1. Stock list (industry)         - 12 page downloads
    2. Price                         - batched quotes
    3. P/E, EV/EBITDA, beta, health  - one statistics page per stock
    4. Sector                        - one profile lookup per stock
    5. CAGR, Sharpe, vs. sector      - 10 years of daily prices per stock

A stock that is missing the data a filter needs fails that filter,
because it can't be shown to pass. Banks and insurers usually have no
EV/EBITDA, so set EV_EBITDA_RANGE = None if you want to keep them.

All data comes from the usstock library (see stock_metrics.py).
"""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from stock_metrics import (
    MarketContext,
    company_profile,
    key_statistics,
    load_shared_data,
    latest_prices,
    price_histories,
    price_metrics,
    sector_metrics,
    valuation_and_health_metrics,
    write_csv,
)


# ============================================================
# FILTER SETTINGS
# ============================================================
# Ranges are (minimum, maximum); use None for an open side, or set
# the whole setting to None to switch that filter off.

MIN_PRICE = 5.0

# Return-based filters (CAGR, Sharpe, vs. sector) use returns without
# dividends when False, or with reinvested dividends when True. The
# output file shows both either way.
FILTER_WITH_DIVIDENDS = True

# 5-year annualized return, in percent.
MIN_PRICE_CAGR_5Y_PCT = 8.0

# 5-year CAGR minus the sector ETF's 5-year CAGR, in
# percentage points. 0 = must at least match its sector.
MIN_RELATIVE_SECTOR_CAGR_PCT = 0.0

BETA_RANGE = (0.5, 1.5)

MIN_SHARPE_5Y = 0.5

# Lower bound of 0 excludes negative EBITDA / negative earnings.
EV_EBITDA_RANGE = (0.0, 20.0)
PE_RANGE = (0.0, 30.0)

# Yahoo sector names, e.g. "Technology", "Healthcare",
# "Financial Services", "Consumer Defensive", "Energy", ...
# Empty INCLUDE_SECTORS = every sector.
INCLUDE_SECTORS = []
EXCLUDE_SECTORS = []


# ------------------------------------------------------------
# No cause for existential crisis
# ------------------------------------------------------------
# Altman Z below 1.81 is the "distress zone". Banks and insurers
# have no Altman Z; for them this check is skipped and the other
# checks still apply.
MIN_ALTMAN_Z = 1.81

REQUIRE_POSITIVE_BOOK_EQUITY = True
REQUIRE_POSITIVE_OPERATING_CASH_FLOW = True

# Operating income / interest expense. Skipped when not reported
# (usually because the company has no meaningful interest expense).
MIN_INTEREST_COVERAGE = 1.5

# SPACs have no operating business to evaluate.
EXCLUDED_INDUSTRIES = {"Shell Companies"}


# ============================================================
# OTHER SETTINGS
# ============================================================

OUTPUT_FILE = "filtered_stocks.csv"

MAX_WORKERS = 6

OUTPUT_COLUMNS = [
    "ticker", "name", "price", "sector", "industry", "market_cap",
    "pe_ratio", "ev_ebitda", "beta_5y",
    "sharpe_5y_no_div", "sharpe_5y_with_div",
    "cagr_5y_pct_no_div", "cagr_5y_pct_with_div", "sector_etf",
    "sector_cagr_5y_pct_no_div", "sector_cagr_5y_pct_with_div",
    "relative_sector_cagr_5y_pct_no_div",
    "relative_sector_cagr_5y_pct_with_div",
    "altman_z", "interest_coverage", "debt_to_equity",
]

# Column suffix the return-based filters read.
FILTER_SUFFIX = "with_div" if FILTER_WITH_DIVIDENDS else "no_div"


# ============================================================
# FILTER CHECKS
# ============================================================

def in_range(value, bounds):
    """True if the filter is off, or value is present and inside bounds."""
    if bounds is None:
        return True
    if value is None:
        return False
    low, high = bounds
    return (low is None or value >= low) and (high is None or value <= high)


def at_least(value, minimum):
    if minimum is None:
        return True
    return value is not None and value >= minimum


def statistics_failure(row):
    """First statistics-page filter the stock fails, or None."""

    if not in_range(row["pe_ratio"], PE_RANGE):
        return "P/E"
    if not in_range(row["ev_ebitda"], EV_EBITDA_RANGE):
        return "EV/EBITDA"
    if not in_range(row["beta_5y"], BETA_RANGE):
        return "beta"

    # Existential checks.
    if MIN_ALTMAN_Z is not None and row["altman_z"] is not None:
        if row["altman_z"] < MIN_ALTMAN_Z:
            return "existential: Altman Z"
    if REQUIRE_POSITIVE_BOOK_EQUITY and not (row["book_equity"] or 0) > 0:
        return "existential: book equity"
    if REQUIRE_POSITIVE_OPERATING_CASH_FLOW and not (
        (row["operating_cash_flow_ttm"] or 0) > 0
    ):
        return "existential: operating cash flow"
    if MIN_INTEREST_COVERAGE is not None and row["interest_coverage"] is not None:
        if row["interest_coverage"] < MIN_INTEREST_COVERAGE:
            return "existential: interest coverage"

    return None


def sector_failure(sector):
    if INCLUDE_SECTORS and sector not in INCLUDE_SECTORS:
        return "sector"
    if sector in EXCLUDE_SECTORS:
        return "sector"
    return None


def performance_failure(row):
    if not at_least(
        row.get(f"cagr_5y_pct_{FILTER_SUFFIX}"), MIN_PRICE_CAGR_5Y_PCT
    ):
        return "CAGR"
    if not at_least(row.get(f"sharpe_5y_{FILTER_SUFFIX}"), MIN_SHARPE_5Y):
        return "Sharpe"
    if not at_least(
        row.get(f"relative_sector_cagr_5y_pct_{FILTER_SUFFIX}"),
        MIN_RELATIVE_SECTOR_CAGR_PCT
    ):
        return "relative sector performance"
    return None


# ============================================================
# STAGES
# ============================================================

def run_parallel(function, items, label):
    """Map function over items with a thread pool, printing progress."""
    results = []
    with ThreadPoolExecutor(MAX_WORKERS) as executor:
        for i, result in enumerate(executor.map(function, items), start=1):
            results.append(result)
            if i % 250 == 0 or i == len(items):
                print(f"    {label}: {i:,}/{len(items):,}")
    return results


def check_statistics(stock):
    try:
        row = valuation_and_health_metrics(key_statistics(stock["symbol"]))
    except Exception:
        return stock, None, "no statistics data"
    return stock, row, statistics_failure(row)


def check_sector(item):
    stock, row = item
    try:
        sector, industry = company_profile(stock["symbol"])
    except Exception:
        sector, industry = None, None
    row["sector"] = sector
    row["industry"] = industry or stock["industry"]

    # Without a sector the sector filters can't be checked.
    if sector is None and (
        INCLUDE_SECTORS or EXCLUDE_SECTORS
        or MIN_RELATIVE_SECTOR_CAGR_PCT is not None
    ):
        return stock, row, "no sector data"

    return stock, row, sector_failure(sector)


def check_performance(item, context):
    stock, row = item
    try:
        histories = price_histories(stock["symbol"])
    except Exception:
        return stock, row, "no price history"

    try:
        row.update(price_metrics(histories, context))
        row.update(sector_metrics(
            stock["symbol"], row["sector"], row["industry"], row, context
        ))
    except Exception:
        # e.g. the sector ETF's prices failed to download.
        return stock, row, "performance data error"

    return stock, row, performance_failure(row)


def main():
    context = MarketContext()
    rejected = Counter()

    # Shared data is fetched up front so a failure stops the run
    # immediately instead of after the slow stages.
    print("1. Downloading the stock list, S&P 500 history and risk-free rate...")
    load_shared_data(context, include_universe=True)
    universe = context.universe
    candidates = [
        stock for stock in universe
        if stock["industry"] not in EXCLUDED_INDUSTRIES
    ]
    rejected["excluded industry"] += len(universe) - len(candidates)
    print(f"   {len(universe):,} stocks listed, {len(candidates):,} remain")

    print("2. Checking prices...")
    prices = latest_prices([stock["symbol"] for stock in candidates])
    remaining = []
    for stock in candidates:
        price = prices.get(stock["symbol"])
        if price is None:
            rejected["no price"] += 1
        elif MIN_PRICE is not None and price < MIN_PRICE:
            rejected["price"] += 1
        else:
            stock["price"] = price
            remaining.append(stock)
    candidates = remaining
    print(f"   {len(candidates):,} remain")

    print("3. Checking P/E, EV/EBITDA, beta and financial health...")
    survivors = []
    for stock, row, failure in run_parallel(
        check_statistics, candidates, "statistics"
    ):
        if failure:
            rejected[failure] += 1
        else:
            survivors.append((stock, row))
    print(f"   {len(survivors):,} remain")

    print("4. Checking sectors...")
    remaining = []
    for stock, row, failure in run_parallel(check_sector, survivors, "sectors"):
        if failure:
            rejected[failure] += 1
        else:
            remaining.append((stock, row))
    survivors = remaining
    print(f"   {len(survivors):,} remain")

    print("5. Checking CAGR, Sharpe ratio and performance vs. sector...")
    passed = []
    for stock, row, failure in run_parallel(
        lambda item: check_performance(item, context), survivors, "prices"
    ):
        if failure:
            rejected[failure] += 1
        else:
            row["ticker"] = stock["symbol"]
            row["name"] = stock["name"]
            row["price"] = stock["price"]
            passed.append(row)
    print(f"   {len(passed):,} remain")

    passed.sort(
        key=lambda row: row.get(f"sharpe_5y_{FILTER_SUFFIX}") or 0,
        reverse=True
    )

    saved_to = write_csv(OUTPUT_FILE, OUTPUT_COLUMNS, passed)

    print()
    print("=" * 70)
    print("REJECTED BY")
    print("=" * 70)
    for reason, count in rejected.most_common():
        print(f"{reason:40s} {count:,}")

    print()
    print("=" * 70)
    dividends = "with" if FILTER_WITH_DIVIDENDS else "without"
    print(f"{len(passed)} STOCKS PASSED (sorted by Sharpe ratio, "
          f"returns {dividends} dividends)")
    print("=" * 70)

    def show(value, digits=1):
        return "n/a" if value is None else f"{value:.{digits}f}"

    for row in passed:
        print(
            f"{row['ticker']:8s} {(row['sector'] or ''):24s} "
            f"CAGR {show(row.get(f'cagr_5y_pct_{FILTER_SUFFIX}')):>6s}%  "
            f"Sharpe {show(row.get(f'sharpe_5y_{FILTER_SUFFIX}'), 2):>5s}  "
            f"P/E {show(row['pe_ratio']):>6s}  "
            f"EV/EBITDA {show(row['ev_ebitda']):>6s}"
        )

    print()
    print(f"Saved to {saved_to}. Ticker list for scraper.py:")
    print([row["ticker"] for row in passed])


if __name__ == "__main__":
    main()
