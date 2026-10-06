from scipy.stats import t
import usstock
import statistics
import random
import math
from collections import defaultdict


# ============================================================
# PORTFOLIO ALLOCATIONS
# ============================================================

INITIAL_PERCENT_S_AND_P = 0.64
INITIAL_PERCENT_BONDS = 0.34
INITIAL_PERCENT_ASSETS = 0.0

LATER_PERCENT_S_AND_P = 0.45
LATER_PERCENT_BONDS = 0.55
LATER_PERCENT_ASSETS = 0.0

INITIAL_STOCKS_LIST = []
LATER_STOCKS_LIST = []


# ============================================================
# CONTRIBUTIONS / WITHDRAWALS
# ============================================================

INITIAL_CONTRIBUTION = 150000
VARIABLE_WITHDRAWAL = 100000
ANNUAL_WITHDRAWAL = 50000


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

# Fraction of each stock's variance driven by its sector factor.
# 0.30 means stocks in the same sector have meaningful positive
# correlation without moving identically.
STOCK_SECTOR_CORRELATION = 0.30

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

def stock_data(ticker_list):

    annual_returns = []

    for ticker in ticker_list:

        rows = usstock.chart(
            ticker[0],
            period="max",
            interval="1d"
        ).get("rows", [])

        rows.sort(key=lambda row: row["date"])

        by_year = defaultdict(list)

        for row in rows:
            year = int(str(row["date"])[:4])
            by_year[year].append(row)

        years = sorted(by_year)

        temp_stock_returns = []

        previous_year_end = None

        for year in years:

            year_rows = by_year[year]

            if not year_rows:
                continue

            first_row = year_rows[0]
            last_row = year_rows[-1]

            start_price = (
                first_row.get("adjusted_close")
                or first_row.get("adjclose")
                or first_row.get("close")
            )

            end_price = (
                last_row.get("adjusted_close")
                or last_row.get("adjclose")
                or last_row.get("close")
            )

            if start_price is None or end_price is None:
                continue

            start_price = float(start_price)
            end_price = float(end_price)

            if end_price <= 0:
                continue

            if previous_year_end is not None and previous_year_end > 0:

                annual_return = (
                    end_price / previous_year_end
                ) - 1

                temp_stock_returns.append(annual_return)

            previous_year_end = end_price

        annual_returns.append(temp_stock_returns)

    return annual_returns


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

    return t.rvs(
        df=SP_T_DF,
        loc=SP_MEAN,
        scale=_student_t_scale(SP_STD, SP_T_DF)
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

    return t.rvs(
        df=GOLD_T_DF,
        loc=GOLD_MEAN,
        scale=_student_t_scale(GOLD_STD, GOLD_T_DF)
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
    sector_shock=0.0
):
    """
    Generate one stock return.

    The stock retains its own historical mean/volatility, while a
    common sector shock creates positive correlation among stocks
    in the same sector.

    sector_shock should be a standard-normal draw shared by every
    stock in the same sector for that simulated year.
    """

    mean, historical_std = stock_distribution_parameters(
        historical_returns
    )

    rho = max(
        0.0,
        min(STOCK_SECTOR_CORRELATION, 0.95)
    )

    # The common sector component accounts for rho of variance.
    sector_component = (
        math.sqrt(rho)
        * historical_std
        * sector_shock
    )

    # The idiosyncratic component gets the remaining variance.
    idio_std = historical_std * math.sqrt(1.0 - rho)

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

    # One common shock per sector per simulated year.
    sector_shocks = {}

    for i in range(len(ticker_list)):

        sector = get_stock_sector(ticker_list[i])

        if sector not in sector_shocks:
            sector_shocks[sector] = random.gauss(0.0, 1.0)

        simulated_return = simulated_individual_stock_return(
            annual_returns[i],
            sector_shocks[sector]
        )

        individual_stock_return += (
            simulated_return * ticker_list[i][1]
        )

    sp_return = simulated_sp_return()
    sp_contribution = stocks * sp_return

    next_tbill_rate = simulated_tbill_return(
        tbill_rate
    )
    bond_contribution = bonds * next_tbill_rate

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
            f"{next_tbill_rate * 100:.2f}%"
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

        print(
            "WARNING: Portfolio allocation "
            "does NOT equal 100%."
        )


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
# MONTE CARLO SETTINGS
# ============================================================

NUM_SIMULATIONS = 10000

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

    portfolio_value *= (
        1 + annual_return
    )

    portfolio_value += INITIAL_CONTRIBUTION

    simulation_values[2027] = portfolio_value

    # --------------------------------------------------------
    # 2028-2032
    # --------------------------------------------------------

    current_year = 2028

    for j in range(5):

        annual_return, tbill_rate = calculate_return(
            initial_annual_returns,
            INITIAL_STOCKS_LIST,
            INITIAL_PERCENT_S_AND_P,
            INITIAL_PERCENT_BONDS,
            INITIAL_PERCENT_ASSETS,
            tbill_rate
        )

        portfolio_value *= (
            1 + annual_return
        )

        simulation_values[current_year] = (
            portfolio_value
        )

        current_year += 1

    # --------------------------------------------------------
    # Variable withdrawal
    # --------------------------------------------------------

    portfolio_value -= VARIABLE_WITHDRAWAL

    if portfolio_value <= 0:

        portfolio_value = 0
        success = False

        failure_count_by_year[2032] += 1

        for year in tracked_years:

            if year >= 2032:
                simulation_values[year] = 0.0

    # --------------------------------------------------------
    # 2033-2042
    # --------------------------------------------------------

    if success:

        current_year = 2033

        for j in range(10):

            portfolio_value -= ANNUAL_WITHDRAWAL

            if portfolio_value <= 0:

                portfolio_value = 0
                success = False

                failure_count_by_year[current_year] += 1

                for year in tracked_years:

                    if year >= current_year:
                        simulation_values[year] = 0.0

                break

            annual_return, tbill_rate = calculate_return(
                later_annual_returns,
                LATER_STOCKS_LIST,
                LATER_PERCENT_S_AND_P,
                LATER_PERCENT_BONDS,
                LATER_PERCENT_ASSETS,
                tbill_rate
            )

            portfolio_value *= (
                1 + annual_return
            )

            if portfolio_value <= 0:

                portfolio_value = 0
                success = False

                failure_count_by_year[current_year] += 1

                for year in tracked_years:

                    if year >= current_year:
                        simulation_values[year] = 0.0

                break

            simulation_values[current_year] = (
                portfolio_value
            )

            current_year += 1

    # --------------------------------------------------------
    # Yearly totals
    # --------------------------------------------------------

    for year in tracked_years:

        year_value_totals[year] += (
            simulation_values[year]
        )

    # --------------------------------------------------------
    # Final values
    # --------------------------------------------------------

    final_value = simulation_values[2042]

    final_values.append(final_value)

    if success:

        successful_final_values.append(
            final_value
        )

        num_worked += 1

    num_times += 1

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    success_rate = (
        num_worked
        / num_times
        * 100
    )

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

        average_successful = (
            statistics.mean(
                successful_final_values
            )
        )

    else:

        average_successful = 0

    # --------------------------------------------------------
    # Dashboard
    # --------------------------------------------------------

    dashboard = []

    dashboard.append(
        "=" * 70
    )

    dashboard.append(
        f"SIMULATION: "
        f"{num_times:,}/{NUM_SIMULATIONS:,}"
    )

    dashboard.append(
        f"SUCCESS RATE: "
        f"{success_rate:.2f}%"
    )

    dashboard.append("")

    dashboard.append(
        "2042 PORTFOLIO STATISTICS"
    )

    dashboard.append(
        "-" * 70
    )

    dashboard.append(
        f"Average (all):       "
        f"${average_final:,.0f}"
    )

    dashboard.append(
        f"Average (successful):"
        f" ${average_successful:,.0f}"
    )

    dashboard.append(
        f"Median:              "
        f"${median_final:,.0f}"
    )

    dashboard.append(
        f"10th percentile:     "
        f"${p10:,.0f}"
    )

    dashboard.append(
        f"25th percentile:     "
        f"${p25:,.0f}"
    )

    dashboard.append(
        f"75th percentile:     "
        f"${p75:,.0f}"
    )

    dashboard.append(
        f"90th percentile:     "
        f"${p90:,.0f}"
    )

    dashboard.append(
        f"Maximum:             "
        f"${maximum_final:,.0f}"
    )

    dashboard.append("")

    dashboard.append(
        "AVERAGE PORTFOLIO VALUE BY YEAR"
    )

    dashboard.append(
        "-" * 70
    )

    for year in tracked_years:

        average_value = (
            year_value_totals[year]
            / num_times
        )

        dashboard.append(
            f"{year}: ${average_value:,.0f}"
        )

    dashboard.append("")

    dashboard.append(
        "FAILURES BY YEAR"
    )

    dashboard.append(
        "-" * 70
    )

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

            dashboard.append(
                f"{year}: "
                f"{count:,} "
                f"({percentage:.2f}%)"
            )

    else:

        dashboard.append(
            "No failures yet."
        )

    dashboard.append(
        "=" * 70
    )

    print(
        "\r"
        + " | ".join(dashboard),
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
