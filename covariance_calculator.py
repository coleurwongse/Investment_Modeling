import usstock
from collections import defaultdict
import statistics

ticker = "TSLA"

rows = usstock.chart(ticker, period="max", interval="1d").get("rows", [])

rows.sort(key=lambda row: row["date"])

by_year = defaultdict(list)

for row in rows:
    year = int(str(row["date"])[:4])
    by_year[year].append(row)

# Get the last closing price of each year
stock_year_end_prices = {}

for year in sorted(by_year):
    year_rows = by_year[year]

    stock_year_end_prices[year] = float(year_rows[-1]["close"])

sp500_data = usstock.ohlcv_history("SPX")

sp500_rows = sp500_data["rows"]

sp500_rows.sort(key=lambda row: row["date"])

sp500_by_year = defaultdict(list)

for row in sp500_rows:
    year = int(str(row["date"])[:4])
    sp500_by_year[year].append(row)

# Get the last closing price of each year
sp500_year_end_prices = {}

for year in sorted(sp500_by_year):
    year_rows = sp500_by_year[year]

    sp500_year_end_prices[year] = float(year_rows[-1]["close"])

stock_returns = []
sp500_returns = []

for year in sorted(stock_year_end_prices):

    if year - 1 not in stock_year_end_prices:
        continue

    if year not in sp500_year_end_prices:
        continue

    if year - 1 not in sp500_year_end_prices:
        continue

    stock_returns.append(stock_year_end_prices[year] / stock_year_end_prices[year - 1] - 1)
    sp500_returns.append(sp500_year_end_prices[year] / sp500_year_end_prices[year - 1] - 1)

stock_average = statistics.mean(stock_returns)
sp500_average = statistics.mean(sp500_returns)

covariance = 0

for i in range(len(stock_returns)):
    stock_difference = (stock_returns[i] - stock_average)
    sp500_difference = (sp500_returns[i] - sp500_average)
    covariance += (stock_difference * sp500_difference)

covariance /= len(stock_returns) - 1

stock_std = statistics.stdev(stock_returns)
sp500_std = statistics.stdev(sp500_returns)

correlation = covariance / (stock_std * sp500_std)

print(f"Correlation: {correlation:.6f}")