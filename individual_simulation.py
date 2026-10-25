"""
Run ONE simulated 2027-2042 path and print every year.

This uses the exact same portfolio, market models and withdrawal /
allocation policy as monte_carlo_sim_portfolio.py (it calls that
file's simulate_path function), so a single run here is one of the
paths the Monte Carlo averages over. Change the portfolio or policy
in monte_carlo_sim_portfolio.py and this file follows automatically.
"""

import random

import numpy

import monte_carlo_sim_portfolio as model


# ============================================================
# SETTINGS
# ============================================================

# Set to a number (e.g. 42) to replay the same path every run.
# None = a different random path each time.
SEED = None


# ============================================================
# RUN ONE PATH
# ============================================================

if SEED is not None:
    random.seed(SEED)
    numpy.random.seed(SEED)  # scipy's Student-t draws use numpy

print()
print("=" * 70)
print("SINGLE SIMULATED PATH")
print("=" * 70)

path = model.simulate_path(verbose=True)

print()
print("=" * 70)

if path["success"]:
    print("SIMULATION SUCCESSFUL")
else:
    print(f"SIMULATION FAILED in {path['failure_year']}")

total_2033 = path["variable_withdrawal"] + model.ANNUAL_WITHDRAWAL

print(
    f"2033 withdrawal:       ${total_2033:,.2f} "
    f"(${path['variable_withdrawal']:,.2f} variable "
    f"+ ${model.ANNUAL_WITHDRAWAL:,.0f} regular)"
)
print(
    f"2033-2042 allocation:  "
    + (
        "all T-bills (above the retained value)"
        if path["all_bonds_after_2033"]
        else "stock/bond mix adjusted each year"
    )
)
print(f"Final Portfolio Value: ${path['final_value']:,.2f}")
print("=" * 70)
