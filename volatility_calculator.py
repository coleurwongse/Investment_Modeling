import math
import numpy as np
import usstock


# ============================================================
# PORTFOLIO
# ============================================================

STOCKS_LIST = [
    ("NVDA", 0.10),
    ("MSFT", 0.20),
    ("WMT", 0.10),
    ("GOOGL", 0.10),
    ("COST", 0.10),
]

# These are separate from STOCKS_LIST.
PERCENT_S_AND_P = 0.00
PERCENT_BONDS = 0.20
PERCENT_GOLD = 0.10
PERCENT_REITS = 0.10


# ============================================================
# ASSET PROXIES
# ============================================================

S_AND_P_TICKER = "SPX"
BONDS_TICKER = "BND"
GOLD_TICKER = "GLD"
REITS_TICKER = "VNQ"


# ============================================================
# SETTINGS
# ============================================================

# How much historical data to use.
HISTORY_PERIOD = "5y"

# There are approximately 252 trading days per year.
TRADING_DAYS_PER_YEAR = 252


# ============================================================
# GET PORTFOLIO
# ============================================================

def build_portfolio():
    portfolio = []

    # Individual stocks
    for ticker, weight in STOCKS_LIST:
        portfolio.append((ticker, weight))

    # S&P 500
    if PERCENT_S_AND_P > 0:
        portfolio.append(
            (S_AND_P_TICKER, PERCENT_S_AND_P)
        )

    # Bonds
    if PERCENT_BONDS > 0:
        portfolio.append(
            (BONDS_TICKER, PERCENT_BONDS)
        )

    # Gold
    if PERCENT_GOLD > 0:
        portfolio.append(
            (GOLD_TICKER, PERCENT_GOLD)
        )

    # REITs
    if PERCENT_REITS > 0:
        portfolio.append(
            (REITS_TICKER, PERCENT_REITS)
        )

    return portfolio


# ============================================================
# VALIDATE WEIGHTS
# ============================================================

def validate_weights(portfolio):

    total = sum(weight for _, weight in portfolio)

    if total > 1.0:
        raise ValueError(
            f"Portfolio weights add up to {total:.2%}. "
            f"They cannot exceed 100%."
        )

    if total < 1.0:
        print(
            f"WARNING: Portfolio weights add up to "
            f"{total:.2%}, not 100%."
        )

    for ticker, weight in portfolio:
        if weight < 0:
            raise ValueError(
                f"{ticker} has a negative weight."
            )

    return total


# ============================================================
# GET HISTORICAL PRICES
# ============================================================

def get_prices(ticker):
    data = usstock.chart(
        ticker,
        period=HISTORY_PERIOD,
        interval="1d"
    )

    rows = data["rows"]

    prices = []

    for row in rows:
        prices.append(float(row["close"]))

    if len(prices) < 2:
        raise ValueError(
            f"Not enough historical data for {ticker}"
        )

    return prices




# ============================================================
# CALCULATE DAILY RETURNS
# ============================================================

def calculate_returns(prices):

    returns = []

    for i in range(1, len(prices)):
        previous_price = prices[i - 1]
        current_price = prices[i]

        daily_return = (
            current_price / previous_price
        ) - 1

        returns.append(daily_return)

    return returns


# ============================================================
# ALIGN DATA
# ============================================================

def align_returns(all_returns):

    """
    Make sure every asset has the same number of observations.

    We use the most recent common number of trading days.
    """

    minimum_length = min(
        len(returns)
        for returns in all_returns
    )

    aligned = []

    for returns in all_returns:
        aligned.append(
            returns[-minimum_length:]
        )

    return aligned


# ============================================================
# COVARIANCE MATRIX
# ============================================================

def calculate_covariance_matrix(returns):

    """
    Calculate the covariance matrix using numpy.

    Each row of `returns` represents one asset.
    Each column represents one trading day.
    """

    return np.cov(
        returns,
        ddof=1
    )


# ============================================================
# PORTFOLIO VOLATILITY
# ============================================================

def calculate_portfolio_volatility(
    covariance_matrix,
    weights
):

    weights = np.array(weights)

    # Portfolio variance:
    #
    #       w' C w
    #
    # where:
    # w = vector of portfolio weights
    # C = covariance matrix

    portfolio_variance = (
        weights
        @ covariance_matrix
        @ weights
    )

    # Daily volatility
    daily_volatility = math.sqrt(
        portfolio_variance
    )

    # Annualized volatility
    annual_volatility = (
        daily_volatility
        * math.sqrt(TRADING_DAYS_PER_YEAR)
    )

    return annual_volatility


# ============================================================
# INDIVIDUAL VOLATILITY
# ============================================================

def calculate_individual_volatility(
    returns
):

    return np.std(
        returns,
        ddof=1
    ) * math.sqrt(
        TRADING_DAYS_PER_YEAR
    )


# ============================================================
# CORRELATION MATRIX
# ============================================================

def calculate_correlation_matrix(
    returns
):

    return np.corrcoef(returns)


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Build portfolio
    # --------------------------------------------------------

    portfolio = build_portfolio()

    validate_weights(portfolio)

    tickers = [
        ticker
        for ticker, _ in portfolio
    ]

    weights = [
        weight
        for _, weight in portfolio
    ]

    # --------------------------------------------------------
    # Display portfolio
    # --------------------------------------------------------

    print()
    print("=" * 65)
    print("PORTFOLIO")
    print("=" * 65)

    for ticker, weight in portfolio:
        print(
            f"{ticker:<10} {weight:>8.1%}"
        )

    print(
        f"{'TOTAL':<10} "
        f"{sum(weights):>8.1%}"
    )

    # --------------------------------------------------------
    # Download historical prices
    # --------------------------------------------------------

    print()
    print("Downloading historical data...")

    all_returns = []

    for ticker in tickers:

        print(f"  {ticker}")

        prices = get_prices(ticker)

        returns = calculate_returns(prices)

        all_returns.append(returns)

    # --------------------------------------------------------
    # Align the datasets
    # --------------------------------------------------------

    all_returns = align_returns(
        all_returns
    )

    # Convert to numpy array.
    #
    # Shape:
    #
    #       assets × trading_days
    #

    returns_array = np.array(
        all_returns
    )

    # --------------------------------------------------------
    # Individual volatility
    # --------------------------------------------------------

    individual_volatilities = []

    for returns in all_returns:

        volatility = (
            calculate_individual_volatility(
                returns
            )
        )

        individual_volatilities.append(
            volatility
        )

    # --------------------------------------------------------
    # Covariance matrix
    # --------------------------------------------------------

    covariance_matrix = (
        calculate_covariance_matrix(
            returns_array
        )
    )

    # --------------------------------------------------------
    # Portfolio volatility
    # --------------------------------------------------------

    portfolio_volatility = (
        calculate_portfolio_volatility(
            covariance_matrix,
            weights
        )
    )

    # --------------------------------------------------------
    # Correlation matrix
    # --------------------------------------------------------

    correlation_matrix = (
        calculate_correlation_matrix(
            returns_array
        )
    )

    # ========================================================
    # RESULTS
    # ========================================================

    print()
    print("=" * 65)
    print("INDIVIDUAL VOLATILITY")
    print("=" * 65)

    print(
        f"{'Asset':<12}"
        f"{'Weight':>12}"
        f"{'Volatility':>18}"
    )

    print("-" * 65)

    for i, ticker in enumerate(tickers):

        print(
            f"{ticker:<12}"
            f"{weights[i]:>11.1%}"
            f"{individual_volatilities[i]:>17.2%}"
        )

    print()
    print("=" * 65)
    print("PORTFOLIO VOLATILITY")
    print("=" * 65)

    print(
        f"\nAnnualized portfolio volatility: "
        f"{portfolio_volatility:.2%}"
    )

    print()
    print("=" * 65)
    print("CORRELATION MATRIX")
    print("=" * 65)

    # Print correlation matrix
    print()

    print(
        f"{'':<10}",
        end=""
    )

    for ticker in tickers:
        print(
            f"{ticker:>10}",
            end=""
        )

    print()

    for i, ticker in enumerate(tickers):

        print(
            f"{ticker:<10}",
            end=""
        )

        for j in range(len(tickers)):

            print(
                f"{correlation_matrix[i][j]:>10.2f}",
                end=""
            )

        print()


if __name__ == "__main__":
    main()
