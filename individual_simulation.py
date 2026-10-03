from scipy.stats import truncnorm
import usstock
import statistics
import random
from collections import defaultdict


# ============================================================
# PORTFOLIO ALLOCATIONS
# ============================================================

INITIAL_PERCENT_S_AND_P = 0.7
INITIAL_PERCENT_BONDS = 0.2
INITIAL_PERCENT_ASSETS = 0.1

LATER_PERCENT_S_AND_P = 0.3
LATER_PERCENT_BONDS = 0.7
LATER_PERCENT_ASSETS = 0.0

INITIAL_STOCKS_LIST = []
LATER_STOCKS_LIST = []

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
                annual_return = (
                    end_price / previous_year_end
                ) - 1

                temp_stock_returns.append(annual_return)

            previous_year_end = end_price

        annual_returns.append(temp_stock_returns)

    return annual_returns


# ============================================================
# TRUNCATED NORMAL DISTRIBUTIONS
# ============================================================

def simulated_sp_return():
    """
    Simulated S&P 500 return.

    Distribution:
        mean = 11%
        std dev = 19.39%

    Bounds:
        -1.6 standard deviations
        +0.9 standard deviations
    """

    return truncnorm.rvs(
        -1.6,
        0.9,
        loc=11,
        scale=19.39
    ) / 100


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

def calculate_return(
    annual_returns,
    ticker_list,
    stocks,
    bonds,
    assets,
    show_breakdown=False
):
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
        individual_stock_return += (
            simulated_return * ticker_list[i][1]
        )

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
# INITIAL PORTFOLIO
# ============================================================

portfolio_value = 300000
current_year = 2027

print(
    f"\nPortfolio Value: ${portfolio_value:,.2f}, "
    f"Beginning of {current_year}"
)


# ============================================================
# 2027
# ============================================================
#
# Current interpretation:
# $150,000 is contributed AFTER the 2027 investment return.
#
# If the contribution happens at the beginning of 2027,
# move the +150000 inside the parentheses instead.
# ============================================================

return_2027 = calculate_return(
    initial_annual_returns,
    INITIAL_STOCKS_LIST,
    INITIAL_PERCENT_S_AND_P,
    INITIAL_PERCENT_BONDS,
    INITIAL_PERCENT_ASSETS,
    show_breakdown=True
)

portfolio_value = (
    portfolio_value * (1 + return_2027)
    + 150000
)

current_year += 1

print(
    f"Portfolio Value: ${portfolio_value:,.2f}, "
    f"Beginning of {current_year}"
)


# ============================================================
# 2029-2032
# ============================================================

for j in range(4):

    annual_return = calculate_return(
        initial_annual_returns,
        INITIAL_STOCKS_LIST,
        INITIAL_PERCENT_S_AND_P,
        INITIAL_PERCENT_BONDS,
        INITIAL_PERCENT_ASSETS
    )

    portfolio_value *= (1 + annual_return)

    current_year += 1

    print(
        f"Portfolio Value: ${portfolio_value:,.2f}, "
        f"Beginning of {current_year}"
    )


# ============================================================
# $250,000 VARIABLE WITHDRAWAL
# ============================================================

portfolio_value -= 200000

print(
    f"\nAfter $250,000 withdrawal: "
    f"${portfolio_value:,.2f}"
)


# ============================================================
# 2033-2042
# ============================================================
#
# $50,000 is withdrawn at the BEGINNING of each year,
# then the remaining portfolio earns that year's return.
# ============================================================

success = True

for j in range(10):

    portfolio_value -= 50000

    if portfolio_value <= 0:
        portfolio_value = 0
        print(
            f"Portfolio depleted in {current_year}"
        )
        break

    annual_return = calculate_return(
        later_annual_returns,
        LATER_STOCKS_LIST,
        LATER_PERCENT_S_AND_P,
        LATER_PERCENT_BONDS,
        LATER_PERCENT_ASSETS
    )

    portfolio_value *= (1 + annual_return)

    current_year += 1

    print(
        f"Portfolio Value: ${portfolio_value:,.2f}, "
        f"Beginning of {current_year}"
    )