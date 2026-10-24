"""
Find the withdrawal settings that give the highest average 2033
variable withdrawal while keeping the success rate at or above
TARGET_SUCCESS_RATE, for the portfolio in monte_carlo_sim_new_portfolio.py.

Settings searched:
    WITHDRAWAL_RETAINED_VALUE   amount kept in T-bills from 2033 on;
                                every $1 lower adds $1 to the variable
                                withdrawal of paths above it
    WITHDRAWAL_STANDARD_AMOUNT  variable withdrawal for paths that end
                                up below the retained value (where
                                nearly all failures happen)

Every candidate runs on the same random paths (same seed), so the
differences between candidates come from the settings, not luck. The
winner is then re-checked on a fresh, larger set of paths and under
the stress-test scenarios.

This doesn't edit the simulation file; copy the recommended values
into monte_carlo_sim_new_portfolio.py.
"""

import random
import statistics

import numpy

import monte_carlo_sim_new_portfolio as model


# ============================================================
# SETTINGS
# ============================================================

TARGET_SUCCESS_RATE = 0.95

# The search requires this much extra so the result still clears the
# target on new random paths (sampling error is ~0.2-0.3%).
SAFETY_MARGIN = 0.005

# Scenario the target must hold in ("normal" = no stress test).
OPTIMIZE_SCENARIO = ("normal", model.SCENARIO_START_YEAR)

COARSE_RETAINED_VALUES = range(300_000, 500_001, 20_000)
STANDARD_AMOUNTS = [0, 25_000, 50_000, 75_000, 100_000]
COARSE_PATHS = 5_000

REFINE_STEP = 5_000
REFINE_PATHS = 20_000

VERIFY_PATHS = 50_000

SEARCH_SEED = 2027
VERIFY_SEED = 9999

CURRENT_STANDARD = model.WITHDRAWAL_STANDARD_AMOUNT

# (scenario, start year) combinations reported for the winner.
STRESS_TESTS = [
    ("recession", 2031),
    ("recession", 2033),
    ("market_crash", 2032),
    ("sector_crash", 2032),
]


# ============================================================
# EVALUATION
# ============================================================

def evaluate(retained, standard, paths, seed, scenario=OPTIMIZE_SCENARIO):
    """(success rate, mean variable withdrawal, median) for one setting."""

    model.WITHDRAWAL_RETAINED_VALUE = retained
    model.WITHDRAWAL_RETAINED_FLOOR_VALUE = retained - model.ANNUAL_WITHDRAWAL
    model.WITHDRAWAL_STANDARD_AMOUNT = standard
    model.MARKET_SCENARIO, model.SCENARIO_START_YEAR = scenario

    random.seed(seed)
    numpy.random.seed(seed)

    results = [model.simulate_path() for _ in range(paths)]
    withdrawals = [r["variable_withdrawal"] for r in results]

    return (
        sum(r["success"] for r in results) / paths,
        statistics.mean(withdrawals),
        statistics.median(withdrawals),
    )


def best_of(candidates):
    """
    Highest mean withdrawal among candidates meeting the target. On a
    tie (e.g. when every path below the retained value is in the
    danger zone, so the standard amount never applies), keep the
    current standard amount.
    """

    required = TARGET_SUCCESS_RATE + SAFETY_MARGIN
    passing = [c for c in candidates if c["success"] >= required]
    if not passing:
        return None
    return max(
        passing,
        key=lambda c: (round(c["mean"]), c["standard"] == CURRENT_STANDARD),
    )


def run_grid(retained_values, standard_amounts, paths):
    candidates = []
    for standard in standard_amounts:
        for retained in retained_values:
            success, mean, median = evaluate(retained, standard, paths, SEARCH_SEED)
            candidates.append({
                "retained": retained, "standard": standard,
                "success": success, "mean": mean, "median": median,
            })
    return candidates


def print_rows(rows):
    print(f"{'retained':>10s} {'standard':>9s} {'success':>8s} "
          f"{'mean var. w/d':>14s} {'median':>10s}")
    for c in rows:
        print(f"${c['retained']:>9,} ${c['standard']:>8,} {c['success']:>8.2%} "
              f"${c['mean']:>13,.0f} ${c['median']:>9,.0f}")


# ============================================================
# SEARCH
# ============================================================

def main():
    original = (
        model.WITHDRAWAL_RETAINED_VALUE,
        model.WITHDRAWAL_STANDARD_AMOUNT,
    )

    print()
    print(f"Target: {TARGET_SUCCESS_RATE:.0%} success "
          f"(searching for {TARGET_SUCCESS_RATE + SAFETY_MARGIN:.1%} "
          f"to leave a margin), scenario {OPTIMIZE_SCENARIO[0]}")

    current = evaluate(*original, VERIFY_PATHS, VERIFY_SEED)
    print(f"Current settings (retained ${original[0]:,}, standard "
          f"${original[1]:,}): success {current[0]:.2%}, "
          f"mean variable withdrawal ${current[1]:,.0f}")

    print(f"\n1. Coarse search ({COARSE_PATHS:,} paths per setting)...")
    coarse = run_grid(COARSE_RETAINED_VALUES, STANDARD_AMOUNTS, COARSE_PATHS)
    coarse_best = best_of(coarse)
    if coarse_best is None:
        raise SystemExit("No setting reaches the target success rate.")

    print("\n   Best retained value for each standard amount:")
    print_rows([
        best_of([c for c in coarse if c["standard"] == s])
        for s in STANDARD_AMOUNTS
        if best_of([c for c in coarse if c["standard"] == s])
    ])

    print(f"\n2. Refining around ${coarse_best['retained']:,} "
          f"({REFINE_PATHS:,} paths per setting)...")
    # One coarse step either side, and the neighboring standard amounts.
    coarse_step = COARSE_RETAINED_VALUES.step
    refined = run_grid(
        range(
            coarse_best["retained"] - coarse_step,
            coarse_best["retained"] + coarse_step + 1,
            REFINE_STEP,
        ),
        [s for s in STANDARD_AMOUNTS if abs(s - coarse_best["standard"]) <= 25_000],
        REFINE_PATHS,
    )
    best = best_of(refined)
    if best is None:
        best = coarse_best

    print_rows(sorted(
        (c for c in refined if c["standard"] == best["standard"]),
        key=lambda c: c["retained"],
    ))

    print(f"\n3. Verifying on {VERIFY_PATHS:,} new paths...")
    success, mean, median = evaluate(
        best["retained"], best["standard"], VERIFY_PATHS, VERIFY_SEED
    )
    standard_error = (success * (1 - success) / VERIFY_PATHS) ** 0.5

    print()
    print("=" * 70)
    print("RECOMMENDED SETTINGS")
    print("=" * 70)
    print(f"WITHDRAWAL_RETAINED_VALUE  = {best['retained']}")
    print(f"WITHDRAWAL_STANDARD_AMOUNT = {best['standard']}")
    print(f"Success rate:              {success:.2%} (+/- {2 * standard_error:.2%})")
    print(f"Mean variable withdrawal:  ${mean:,.0f}  (was ${current[1]:,.0f})")
    print(f"Median variable withdrawal: ${median:,.0f}")

    print("\nStress tests with these settings (success / mean withdrawal):")
    for scenario in STRESS_TESTS:
        s, m_, _ = evaluate(best["retained"], best["standard"], VERIFY_PATHS // 5,
                            VERIFY_SEED, scenario)
        so, mo, _ = evaluate(*original, VERIFY_PATHS // 5, VERIFY_SEED, scenario)
        print(f"  {scenario[0]:13s} {scenario[1]}:  {s:6.2%} / ${m_:>9,.0f}"
              f"    (current settings: {so:6.2%} / ${mo:>9,.0f})")
    print("=" * 70)


if __name__ == "__main__":
    main()
