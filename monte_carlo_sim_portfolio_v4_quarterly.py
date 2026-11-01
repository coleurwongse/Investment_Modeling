"""
Quarterly version of monte_carlo_sim_portfolio_v4.py: the same simulation, with every year
split into four quarters (see quarterly_sim.py). Year-end results are
identical to monte_carlo_sim_portfolio_v4.py; change the portfolio or settings there.
"""

import monte_carlo_sim_portfolio_v4 as model
import quarterly_sim


if __name__ == "__main__":
    quarterly_sim.main(model)
