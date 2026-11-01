"""
Run ONE simulated 2027-2042 path and print every year and quarter.

Quarterly version of individual_simulation.py: it uses the same model
(monte_carlo_sim_portfolio.py) and splits each year into quarters (see
quarterly_sim.py). With the same SEED, the yearly numbers match
individual_simulation.py exactly.
"""

import monte_carlo_sim_portfolio as model
import quarterly_sim


# ============================================================
# SETTINGS
# ============================================================

# Set to a number (e.g. 42) to replay the same path every run.
# None = a different random path each time.
SEED = None


if __name__ == "__main__":
    quarterly_sim.run_single(model, SEED)
