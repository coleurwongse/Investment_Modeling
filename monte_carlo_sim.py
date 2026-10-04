from scipy.stats import truncnorm
import usstock
import statistics
import random
import sys
from collections import defaultdict

# ============================================================
# PORTFOLIO ALLOCATIONS
# ============================================================

INITIAL_PERCENT_S_AND_P = 0.64
INITIAL_PERCENT_BONDS = 0.34
INITIAL_PERCENT_ASSETS = 0.02

LATER_PERCENT_S_AND_P = 0.45
LATER_PERCENT_BONDS = 0.55
LATER_PERCENT_ASSETS = 0.0

INITIAL_STOCKS_LIST = []
LATER_STOCKS_LIST = []

VARIABLE_CONTRIBUTION = 200000


# Example:
# INITIAL_STOCKS_LIST = [
#     ("NVDA", 0.1),
#     ("MSFT", 0.2),
#     ("WMT", 0.2),
#     ("GOOGL", 0.1),
#     ("COST", 0.1)
# ]

# LATER_STOCKS_LIST = [
#     ("NVDA", 0.03),
#     ("WMT", 0.1),
#     ("GOOGL", 0.07),
#     ("COST", 0.1)
# ]


# ============================================================
# HISTORICAL DATA
# ============================================================

def stock_data(ticker_list):
    """
    Returns historical calendar-year returns.

    Returns are stored as DECIMALS:
        0.10  = +10%
       -0.20  = -20%

    Uses adjusted close when available.
    """

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

        # ----------------------------------------------------
        # Calculate each calendar year's return.
        #
        # We use the previous year's final price as the
        # starting price for the current year.
        # ----------------------------------------------------

        previous_year_end = None

        for year in years:

            year_rows = by_year[year]

            if not year_rows:
                continue

            # Prefer adjusted close if the API supplies it.
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

            # For the first available year, we can't calculate
            # a true calendar-year return without the previous
            # year's closing price.
            if previous_year_end is not None and previous_year_end > 0:
                annual_return = (end_price / previous_year_end) - 1

                temp_stock_returns.append(annual_return)

            previous_year_end = end_price

        annual_returns.append(temp_stock_returns)

    return annual_returns


# ============================================================
# TRUNCATED NORMAL DISTRIBUTIONS
# ============================================================

def simulated_sp_return():
    """
    Returns a randomly selected historical S&P 500 annual
    total return.

    Returns are decimals:
        0.10  = +10%
       -0.20  = -20%
    """

    # Historical S&P 500 annual total returns, 1928-2025.
    # Values are decimals.
    historical_returns = [
        0.4381, -0.0830, -0.2512, -0.4384, -0.0864,
        0.4959, -0.4770, -0.4284, -0.5309, 0.3655,
        0.0555, 0.0578, 0.0562, 0.1840, 0.0551,
        0.1813, 0.3169, 0.1048, 0.1276, 0.0160,
        0.1915, 0.3582, 0.0577, 0.0159, 0.1337,
        0.1405, 0.0000, 0.0873, 0.2261, 0.1676,
        0.2280, 0.1422, -0.0698, 0.1631, 0.1221,
        0.0199, 0.1648, 0.3250, 0.1697, 0.1588,
        0.0550, 0.1812, 0.3172, 0.1897, 0.2140,
        0.3364, 0.2274, 0.3300, 0.2965, 0.2210,
        0.3718, 0.2390, 0.0911, 0.0744, 0.1953,
        0.1814, 0.2689, 0.0989, 0.1230, 0.2296,
        0.2123, 0.3370, 0.2268, 0.3720, 0.3147,
        0.0170, 0.1210, 0.2027, 0.1947, 0.3101,
        0.2589, -0.0897, 0.0492, 0.1659, 0.3192,
        0.0549, -0.3700, 0.2646, 0.1558, 0.1088,
        0.2836, -0.0910, -0.1189, -0.2210, 0.2104,
        0.2858, 0.3336, 0.2296, 0.3758, 0.0132,
        0.1008, 0.0762, 0.3047, -0.0310, 0.3169,
        0.0525, 0.1867, 0.3173, 0.0627, 0.2155,
        0.3250, 0.1861, 0.0657, 0.2370, 0.3723,
        0.3240, 0.0683, 0.2256, 0.2155, 0.1840,
        0.1641, 0.1224, 0.2580, 0.2388, 0.2871,
        0.1830, 0.1254, 0.1840, 0.3150, 0.1761,
        0.2502, 0.2629, -0.1811, 0.2871, 0.1544
    ]

    return random.choice(historical_returns)

def simulated_asset_return():
    """
    Simulated asset/gold return.

    Distribution:
        mean = 8%
        std dev = 7%

    Bounds:
        -5.43 standard deviations
        +7.42 standard deviations
    """

    return truncnorm.rvs(
        -5.43,
        7.42,
        loc=8,
        scale=7
    ) / 100


# ============================================================
# PORTFOLIO RETURN
# ============================================================

def calculate_return(annual_returns, ticker_list, stocks, bonds, assets, show_breakdown=False ):
    """
    Calculates one year's portfolio return.

    ALL RETURNS ARE DECIMALS.

    Example:
        0.10 = +10%
       -0.05 = -5%
    """

    individual_stock_return = 0

    # --------------------------------------------------------
    # Individual stocks
    # --------------------------------------------------------

    for i in range(len(ticker_list)):

        if not annual_returns[i]:
            continue

        average_return = statistics.mean(annual_returns[i])

        if len(annual_returns[i]) >= 2:
            standard_deviation = statistics.stdev(
                annual_returns[i]
            )
        else:
            standard_deviation = 0

        simulated_return = random.gauss(
            average_return,
            standard_deviation
        )

        # A stock cannot lose more than 100%.
        simulated_return = max(simulated_return, -1.0)

        # ticker_list weight is already a fraction of
        # the total portfolio, so DO NOT divide by
        # number of stocks.
        individual_stock_return += (simulated_return * ticker_list[i][1])

    # --------------------------------------------------------
    # S&P 500
    # --------------------------------------------------------

    sp_return = simulated_sp_return()

    sp_contribution = stocks * sp_return

    # --------------------------------------------------------
    # Bonds
    # --------------------------------------------------------

    bond_return = 0.045

    bond_contribution = bonds * bond_return

    # --------------------------------------------------------
    # Other assets
    # --------------------------------------------------------

    asset_return = simulated_asset_return()

    asset_contribution = assets * asset_return

    # --------------------------------------------------------
    # Total
    # --------------------------------------------------------

    total_return = (
            individual_stock_return
            + sp_contribution
            + bond_contribution
            + asset_contribution
    )

    if show_breakdown:
        print("\n--- RETURN BREAKDOWN ---")
        print(
            f"Individual stocks: {individual_stock_return * 100:.2f}%"
        )
        print(
            f"S&P 500:           {sp_contribution * 100:.2f}%"
        )
        print(
            f"Bonds:              {bond_contribution * 100:.2f}%"
        )
        print(
            f"Other assets:       {asset_contribution * 100:.2f}%"
        )
        print(
            f"TOTAL:              {total_return * 100:.2f}%"
        )
        print("------------------------\n")

    return total_return


# ============================================================
# VALIDATE ALLOCATIONS
# ============================================================

def check_allocation(stocks, bonds, assets, ticker_list):
    stock_weight = sum(weight for _, weight in ticker_list)

    total = (
            stocks
            + bonds
            + assets
            + stock_weight
    )

    print("\n--- ALLOCATION CHECK ---")
    print(f"S&P 500:       {stocks:.2%}")
    print(f"Bonds:         {bonds:.2%}")
    print(f"Other assets:  {assets:.2%}")
    print(f"Individual:    {stock_weight:.2%}")
    print(f"TOTAL:         {total:.2%}")
    print("------------------------")

    if abs(total - 1.0) > 0.001:
        print("WARNING: Portfolio allocation does NOT equal 100%.")


# ============================================================
# GET DATA
# ============================================================

print("Be patient. Getting ur data takes time...")

initial_annual_returns = stock_data(INITIAL_STOCKS_LIST)
later_annual_returns = stock_data(LATER_STOCKS_LIST)

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
# MONTE CARLO SIMULATION
# ============================================================

NUM_SIMULATIONS = 10000

num_worked = 0
num_times = 0

# ------------------------------------------------------------
# FAILURE TRACKING
# ------------------------------------------------------------

failure_count_by_year = defaultdict(int)

# ------------------------------------------------------------
# YEARLY PORTFOLIO VALUE TRACKING
# ------------------------------------------------------------

tracked_years = [2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035, 2036, 2037, 2038, 2039, 2040, 2041, 2042]

year_value_totals = {
    year: 0.0
    for year in tracked_years
}

# ------------------------------------------------------------
# FINAL PORTFOLIO VALUES
# ------------------------------------------------------------
#
# final_values contains EVERY simulation's final value.
#
# successful_final_values contains only simulations that
# successfully made it through 2042.
# ------------------------------------------------------------

final_values = []
successful_final_values = []


# ============================================================
# RUN SIMULATIONS
# ============================================================

for simulation in range(NUM_SIMULATIONS):

    # --------------------------------------------------------
    # RESET SIMULATION
    # --------------------------------------------------------

    portfolio_value = 300000
    current_year = 2027
    success = True

    # Every simulation gets exactly one value for every year.
    simulation_values = {
        year: 0.0
        for year in tracked_years
    }

    # --------------------------------------------------------
    # 2027
    # --------------------------------------------------------

    annual_return = calculate_return(
        initial_annual_returns,
        INITIAL_STOCKS_LIST,
        INITIAL_PERCENT_S_AND_P,
        INITIAL_PERCENT_BONDS,
        INITIAL_PERCENT_ASSETS
    )

    portfolio_value = portfolio_value * (1 + annual_return)

    simulation_values[2027] = portfolio_value

    # --------------------------------------------------------
    # 2028-2032
    # --------------------------------------------------------

    current_year = 2028
    portfolio_value += 150000  # Second contribution

    for j in range(5):
        annual_return = calculate_return(
            initial_annual_returns,
            INITIAL_STOCKS_LIST,
            INITIAL_PERCENT_S_AND_P,
            INITIAL_PERCENT_BONDS,
            INITIAL_PERCENT_ASSETS
        )

        portfolio_value *= (1 + annual_return)

        simulation_values[current_year] = portfolio_value

        current_year += 1

    portfolio_value -= VARIABLE_CONTRIBUTION

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

            # Beginning-of-year withdrawal
            portfolio_value -= 50000

            if portfolio_value <= 0:

                portfolio_value = 0
                success = False

                failure_count_by_year[current_year] += 1

                for year in tracked_years:

                    if year >= current_year:
                        simulation_values[year] = 0.0

                break

            # Investment return
            annual_return = calculate_return(
                later_annual_returns,
                LATER_STOCKS_LIST,
                LATER_PERCENT_S_AND_P,
                LATER_PERCENT_BONDS,
                LATER_PERCENT_ASSETS
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

            # Record end-of-year value
            simulation_values[current_year] = portfolio_value

            current_year += 1

    # --------------------------------------------------------
    # ADD THIS SIMULATION TO YEARLY TOTALS
    # --------------------------------------------------------

    for year in tracked_years:
        year_value_totals[year] += simulation_values[year]

    # --------------------------------------------------------
    # RECORD FINAL VALUE
    # --------------------------------------------------------

    final_value = simulation_values[2042]

    final_values.append(final_value)

    # Only successful simulations go into this list
    if success:
        successful_final_values.append(final_value)
        num_worked += 1

    num_times += 1

    # ========================================================
    # CALCULATE LIVE STATISTICS
    # ========================================================

    success_rate = (num_worked / num_times) * 100

    # --------------------------------------------------------
    # Average final value
    # --------------------------------------------------------

    average_final = statistics.mean(final_values)

    # --------------------------------------------------------
    # Median final value
    # --------------------------------------------------------

    median_final = statistics.median(final_values)

    # --------------------------------------------------------
    # Percentiles
    # --------------------------------------------------------

    sorted_values = sorted(final_values)
    p10 = sorted_values[int(len(sorted_values) * 0.10)]
    p25 = sorted_values[int(len(sorted_values) * 0.25)]
    p75 = sorted_values[int(len(sorted_values) * 0.75)]
    p90 = sorted_values[int(len(sorted_values) * 0.90)]

    maximum_final = max(final_values)

    # Average among successful simulations
    if successful_final_values:
        average_successful = statistics.mean(successful_final_values)
    else:
        average_successful = 0

    # ========================================================
    # LIVE DASHBOARD
    # ========================================================

    dashboard = []
    dashboard.append("=" * 70)
    dashboard.append(f"SIMULATION: "f"{num_times:,}/{NUM_SIMULATIONS:,}")
    dashboard.append(f"SUCCESS RATE: "f"{success_rate:.2f}%")
    dashboard.append("")
    dashboard.append("2042 PORTFOLIO STATISTICS")
    dashboard.append("-" * 70)
    dashboard.append(f"Average (all):       "f"${average_final:,.0f}")
    dashboard.append(f"Average (successful):"f" ${average_successful:,.0f}")
    dashboard.append(f"Median:              "f"${median_final:,.0f}")
    dashboard.append(f"10th percentile:     "f"${p10:,.0f}")
    dashboard.append(f"25th percentile:     " f"${p25:,.0f}")
    dashboard.append(f"75th percentile:     " f"${p75:,.0f}")
    dashboard.append(f"90th percentile:     "f"${p90:,.0f}" )
    dashboard.append(f"Maximum:             "f"${maximum_final:,.0f}")
    dashboard.append("")
    dashboard.append("AVERAGE PORTFOLIO VALUE BY YEAR")
    dashboard.append("-" * 70)

    for year in tracked_years:

        average_value = (year_value_totals[year] / num_times)

        dashboard.append(f"{year}: ${average_value:,.0f}")

    dashboard.append("")
    dashboard.append("FAILURES BY YEAR")
    dashboard.append("-" * 70)

    if failure_count_by_year:

        for year in sorted(failure_count_by_year):

            count = failure_count_by_year[year]
            percentage = (count / num_times) * 100
            dashboard.append(f"{year}: "f"{count:,} "f"({percentage:.2f}%)")

    else:
        dashboard.append("No failures yet.")

    dashboard.append("=" * 70)

    # --------------------------------------------------------
    # LIVE OUTPUT
    # --------------------------------------------------------
    #
    # \r returns to the beginning of the current line.
    #
    # Because some consoles don't support ANSI escape codes,
    # we keep the dashboard on one physical line.
    # --------------------------------------------------------

    one_line_dashboard = " | ".join(dashboard)

    sys.stdout.write(
        "\r" + one_line_dashboard + " " * 50
    )

    sys.stdout.flush()


# ============================================================
# FINAL RESULTS
# ============================================================

print()
print()
print("=" * 70)
print("FINAL MONTE CARLO RESULTS")
print("=" * 70)

success_rate = (num_worked / num_times) * 100

print(f"Simulations:       {num_times:,}")
print(f"Successful:        {num_worked:,}")
print(f"Failed:            "f"{num_times - num_worked:,}")
print(f"Success Rate:      "f"{success_rate:.2f}%")

# ------------------------------------------------------------
# 2042 STATISTICS
# ------------------------------------------------------------

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

sorted_values = sorted(final_values)

p10 = sorted_values[int(len(sorted_values) * 0.10)]
p25 = sorted_values[int(len(sorted_values) * 0.25)]
p75 = sorted_values[int(len(sorted_values) * 0.75)]
p90 = sorted_values[int(len(sorted_values) * 0.90)]

maximum_final = max(final_values)

if successful_final_values:
    average_successful = statistics.mean(successful_final_values)
else:
    average_successful = 0

print(f"Average (all):        "f"${average_final:,.2f}")
print(f"Average (successful): "f"${average_successful:,.2f}")
print(f"Median:               "f"${median_final:,.2f}")
print(f"10th percentile:      "f"${p10:,.2f}")
print(f"25th percentile:      "f"${p25:,.2f}")
print(f"75th percentile:      "f"${p75:,.2f}")
print(f"90th percentile:      "f"${p90:,.2f}")
print( f"Maximum:              "f"${maximum_final:,.2f}")

# ============================================================
# FAILURE YEARS
# ============================================================

print()
print("=" * 70)
print("FAILURES BY YEAR")
print("=" * 70)

if failure_count_by_year:

    for year in sorted(failure_count_by_year):

        count = failure_count_by_year[year]
        percentage = (count / num_times) * 100
        print(f"{year}: " f"{count:,} failures "f"({percentage:.2f}%)")

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

    average_value = (year_value_totals[year] / num_times)

    print(f"{year}: ${average_value:,.2f}")

print("=" * 70)
