"""
Quarterly version of monte_carlo_sim_portfolio.py: the same simulation, with every year
split into four quarters (see quarterly_sim.py). Year-end results are
identical to monte_carlo_sim_portfolio.py; change the portfolio or settings there.
"""

import monte_carlo_sim_portfolio as model
import quarterly_sim


if __name__ == "__main__":
    quarterly_sim.main(model)
