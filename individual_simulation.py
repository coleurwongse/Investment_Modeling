from scipy.stats import truncnorm, t
import usstock
import statistics
import random
from collections import defaultdict


# ============================================================
# PORTFOLIO ALLOCATIONS
# ============================================================

INITIAL_PERCENT_S_AND_P = 0.64
INITIAL_PERCENT_BONDS = 0.34
INITIAL_PERCENT_REITS = 0.0
INITIAL_PERCENT_ASSETS = 0.02

LATER_PERCENT_S_AND_P = 0.45
LATER_PERCENT_BONDS = 0.55
LATER_PERCENT_REITS = 0.0
LATER_PERCENT_ASSETS = 0.0

INITIAL_STOCKS_LIST = []
LATER_STOCKS_LIST = []


# ============================================================
# CONTRIBUTIONS / WITHDRAWALS
# ============================================================

INITIAL_CONTRIBUTION = 150000
VARIABLE_WITHDRAWAL = 200000
ANNUAL_WITHDRAWAL = 50000


# ============================================================
# MODEL PARAMETERS
# ============================================================

STOCK_TARGET_RETURN = 0.08
STOCK_MEAN_SHRINKAGE = 0.50
STOCK_T_DF = 5

STOCK_LOWER_PERCENTILE = 0.05
STOCK_UPPER_PERCENTILE = 0.95

REIT_MEAN = 8.24
REIT_STD = 19.07


# ============================================================
# HISTORICAL 1-YEAR TREASURY BILL RATES
# ============================================================

TREASURY_BILL_RATES = [
    4.64, 3.42, 2.81, 3.01, 3.30, 3.75, 4.06, 5.07,
    4.70, 5.46, 6.79, 6.49, 4.67, 4.76, 7.02, 7.72,
    6.30, 5.52, 5.70, 7.74, 9.73, 10.85, 13.16, 11.07,
    8.80, 9.94, 7.81, 6.07, 6.33, 7.13, 7.92, 7.35,
    5.52, 3.71, 3.29, 5.02, 5.60, 5.22, 5.32, 4.80,
    4.81, 5.78, 3.84, 1.67,

    # 2003-2007
    1.05, 1.57, 3.39, 4.81, 4.45,

    # 2008-2025
    1.63, 0.45, 0.30, 0.17, 0.17, 0.13, 0.11,
    0.30, 0.60, 1.17, 2.25, 1.99, 0.36, 0.10,
    2.68, 4.84, 4.48, 3.76
]


def simulated_tbill_return():

    return random.choice(
        TREASURY_BILL_RATES
    ) / 100


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

        rows.sort(
            key=lambda row: row["date"]
        )

        by_year = defaultdict(list)

        for row in rows:

            year = int(
                str(row["date"])[:4]
            )

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
                first_row.get(
                    "adjusted_close"
                )
                or first_row.get(
                    "adjclose"
                )
                or first_row.get(
                    "close"
                )
            )

            end_price = (
                last_row.get(
                    "adjusted_close"
                )
                or last_row.get(
                    "adjclose"
                )
                or last_row.get(
                    "close"
                )
            )

            if (
                start_price is None
                or end_price is None
            ):
                continue

            start_price = float(
                start_price
            )

            end_price = float(
                end_price
            )

            if end_price <= 0:
                continue

            if (
                previous_year_end is not None
                and previous_year_end > 0
            ):

                annual_return = (
                    end_price
                    / previous_year_end
                ) - 1

                temp_stock_returns.append(
                    annual_return
                )

            previous_year_end = end_price

        annual_returns.append(
            temp_stock_returns
        )

    return annual_returns


# ============================================================
# S&P 500 MODEL
# ============================================================

def simulated_sp_return():

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

    return random.choice(
        historical_returns
    )


# ============================================================
# REIT MODEL
# ============================================================

def simulated_reit_return():

    return truncnorm.rvs(
        -5.43,
        7.42,
        loc=REIT_MEAN,
        scale=REIT_STD
    ) / 100


# ============================================================
# OTHER ASSETS / GOLD
# ============================================================

def simulated_asset_return():

    return truncnorm.rvs(
        -5.43,
        7.42,
        loc=8,
        scale=7
    ) / 100


# ============================================================
# INDIVIDUAL STOCK MODEL
# ============================================================

def percentile_value(
    values,
    percentile
):

    values = sorted(values)

    if not values:
        return 0

    position = (
        percentile
        * (len(values) - 1)
    )

    lower = int(position)

    upper = min(
        lower + 1,
        len(values) - 1
    )

    fraction = (
        position - lower
    )

    return (
        values[lower]
        + (
            values[upper]
            - values[lower]
        )
        * fraction
    )


def simulated_individual_stock_return(
    historical_returns
):

    if not historical_returns:

        return 0.0

    if len(historical_returns) < 5:

        return random.gauss(
            STOCK_TARGET_RETURN,
            0.25
        )

    lower_bound = percentile_value(
        historical_returns,
        STOCK_LOWER_PERCENTILE
    )

    upper_bound = percentile_value(
        historical_returns,
        STOCK_UPPER_PERCENTILE
    )

    winsorized = [
        min(
            max(
                value,
                lower_bound
            ),
            upper_bound
        )
        for value in historical_returns
    ]

    historical_mean = statistics.mean(
        winsorized
    )

    historical_std = statistics.stdev(
        winsorized
    )

    adjusted_mean = (
        historical_mean
        * (1 - STOCK_MEAN_SHRINKAGE)
        + STOCK_TARGET_RETURN
        * STOCK_MEAN_SHRINKAGE
    )

    scale = (
        historical_std
        * (
            (STOCK_T_DF - 2)
            / STOCK_T_DF
        ) ** 0.5
    )

    if scale <= 0:

        return adjusted_mean

    simulated_return = t.rvs(
        df=STOCK_T_DF,
        loc=adjusted_mean,
        scale=scale
    )

    return max(
        simulated_return,
        -1.0
    )


# ============================================================
# PORTFOLIO RETURN
# ============================================================

def calculate_return(
    annual_returns,
    ticker_list,
    stocks,
    bonds,
    reits,
    assets,
    show_breakdown=False
):

    individual_stock_return = 0

    for i in range(
        len(ticker_list)
    ):

        simulated_return = (
            simulated_individual_stock_return(
                annual_returns[i]
            )
        )

        individual_stock_return += (
            simulated_return
            * ticker_list[i][1]
        )

    sp_return = simulated_sp_return()

    sp_contribution = (
        stocks * sp_return
    )

    bond_return = (
        simulated_tbill_return()
    )

    bond_contribution = (
        bonds * bond_return
    )

    reit_return = (
        simulated_reit_return()
    )

    reit_contribution = (
        reits * reit_return
    )

    asset_return = (
        simulated_asset_return()
    )

    asset_contribution = (
        assets * asset_return
    )

    total_return = (
        individual_stock_return
        + sp_contribution
        + bond_contribution
        + reit_contribution
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
            f"REITs:              "
            f"{reit_contribution * 100:.2f}%"
        )

        print(
            f"Other assets:       "
            f"{asset_contribution * 100:.2f}%"
        )

        print(
            f"TOTAL:              "
            f"{total_return * 100:.2f}%"
        )

        print("------------------------\n")

    return total_return


# ============================================================
# ALLOCATION CHECK
# ============================================================

def check_allocation(
    stocks,
    bonds,
    reits,
    assets,
    ticker_list
):

    stock_weight = sum(
        weight
        for _, weight in ticker_list
    )

    total = (
        stocks
        + bonds
        + reits
        + assets
        + stock_weight
    )

    print("\n--- ALLOCATION CHECK ---")

    print(
        f"S&P 500:       {stocks:.2%}"
    )

    print(
        f"T-bills:       {bonds:.2%}"
    )

    print(
        f"REITs:         {reits:.2%}"
    )

    print(
        f"Other assets:  {assets:.2%}"
    )

    print(
        f"Individual:    {stock_weight:.2%}"
    )

    print(
        f"TOTAL:         {total:.2%}"
    )

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
    INITIAL_PERCENT_REITS,
    INITIAL_PERCENT_ASSETS,
    INITIAL_STOCKS_LIST
)

check_allocation(
    LATER_PERCENT_S_AND_P,
    LATER_PERCENT_BONDS,
    LATER_PERCENT_REITS,
    LATER_PERCENT_ASSETS,
    LATER_STOCKS_LIST
)


# ============================================================
# INITIAL PORTFOLIO
# ============================================================

portfolio_value = 300000
current_year = 2027
success = True

print(
    f"\nPortfolio Value: "
    f"${portfolio_value:,.2f}, "
    f"Beginning of {current_year}"
)


# ============================================================
# 2027
# ============================================================

return_2027 = calculate_return(
    initial_annual_returns,
    INITIAL_STOCKS_LIST,
    INITIAL_PERCENT_S_AND_P,
    INITIAL_PERCENT_BONDS,
    INITIAL_PERCENT_REITS,
    INITIAL_PERCENT_ASSETS,
    show_breakdown=True
)

portfolio_value = (
    portfolio_value
    * (1 + return_2027)
    + INITIAL_CONTRIBUTION
)

current_year += 1

print(
    f"Portfolio Value: "
    f"${portfolio_value:,.2f}, "
    f"Beginning of {current_year}"
)


# ============================================================
# 2028-2032
# ============================================================

for j in range(5):

    annual_return = calculate_return(
        initial_annual_returns,
        INITIAL_STOCKS_LIST,
        INITIAL_PERCENT_S_AND_P,
        INITIAL_PERCENT_BONDS,
        INITIAL_PERCENT_REITS,
        INITIAL_PERCENT_ASSETS
    )

    portfolio_value *= (
        1 + annual_return
    )

    current_year += 1

    print(
        f"Portfolio Value: "
        f"${portfolio_value:,.2f}, "
        f"Beginning of {current_year}"
    )


# ============================================================
# $200,000 VARIABLE WITHDRAWAL
# ============================================================

portfolio_value -= VARIABLE_WITHDRAWAL

print(
    f"\nAfter ${VARIABLE_WITHDRAWAL:,.0f} "
    f"withdrawal: "
    f"${portfolio_value:,.2f}"
)

if portfolio_value <= 0:

    portfolio_value = 0
    success = False

    print(
        "\nPortfolio depleted before "
        "the 2033-2042 period."
    )


# ============================================================
# 2033-2042
# ============================================================

if success:

    for j in range(10):

        portfolio_value -= ANNUAL_WITHDRAWAL

        print(
            f"\nBeginning of {current_year}: "
            f"${portfolio_value:,.2f} "
            f"after "
            f"${ANNUAL_WITHDRAWAL:,.0f} "
            f"withdrawal"
        )

        if portfolio_value <= 0:

            portfolio_value = 0
            success = False

            print(
                f"Portfolio depleted in "
                f"{current_year}"
            )

            break

        annual_return = calculate_return(
            later_annual_returns,
            LATER_STOCKS_LIST,
            LATER_PERCENT_S_AND_P,
            LATER_PERCENT_BONDS,
            LATER_PERCENT_REITS,
            LATER_PERCENT_ASSETS,
            show_breakdown=False
        )

        portfolio_value *= (
            1 + annual_return
        )

        if portfolio_value <= 0:

            portfolio_value = 0
            success = False

            print(
                f"Portfolio depleted in "
                f"{current_year}"
            )

            break

        print(
            f"End of {current_year}: "
            f"${portfolio_value:,.2f}"
        )

        current_year += 1


# ============================================================
# FINAL RESULT
# ============================================================

print()
print("=" * 60)

if success:

    print(
        "SIMULATION SUCCESSFUL"
    )

else:

    print(
        "SIMULATION FAILED"
    )

print(
    f"Final Portfolio Value: "
    f"${portfolio_value:,.2f}"
)

print("=" * 60)
