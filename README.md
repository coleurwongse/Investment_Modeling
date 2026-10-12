Programs for use in the 2026 Wharton Investment Competition.
Uses:
  https://github.com/pinion05/usstock 
to collect stock data.

Monte Carlo Sim
Uses portfolio concentrations (stocks, bonds, etc.) to simulate a portfolio with set requirements and records performance.

Phase 1:
No withdrawals, so primarily focus on stocks to generate high returns. Allocations are set.

Phase 2:
One variable withdrawal at the beginning of 2033. At the beginning of 2031, analyze the portfolio and calculate an approximate withdrawal. In 2033, review the range and select an appropriate value for the withdrawal.
In addition to the variable withdrawal, there will be a 50k annual withdrawal from 2033-2042.
Shift the balance towards bonds, but use a dynamic allocation instead of setting allocations at the beginning (e.g. if returns won't give the cash for the withdrawals, increase concentration of stocks)

Individual Sim
Same as the Monte Carlo sim, except it only runs once. Good for finding what a "typical" run looks like.

Correlation Calculator
Calculates the correlation between an individual stock and the S&P 500. Use for diversification. (Also calculates covariance.)

Volatility Calculator
Given portfolio allocations, calculates the raw volatility of the portfolio. (Not beta, because beta uses the S&P 500 as a baseline, and assets that have little to no correlation to the S&P 500 will have a beta of 0 regardless of volatility.)
