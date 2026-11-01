"""
Quarterly versions of the Monte Carlo sims.

The quarterly files (monte_carlo_sim_quarterly.py, ...) run the
matching annual sim unchanged and split each simulated year into four
quarters, so every year-end value, failure, withdrawal and allocation
is exactly what the annual sim produced; the quarters only add detail
inside each year.

How a year is split: each holding's simulated annual return is spread
over the four quarters along a random path that always ends exactly at
that annual return (a "Brownian bridge" in log space). The path's
wiggle matches the holding's volatility, and holdings move together
within the year by their correlation with the S&P 500. Holdings are
bought at the start of the year and held, so the stock percentage
drifts from quarter to quarter as stocks and bonds move apart.

The quarter paths use their own random generator, so running a
quarterly sim with the same seed as the annual sim gives identical
annual results.

Timing, as in the annual sims: withdrawals come out at the start of a
year (before Q1), and the 2027 contribution arrives at the end of Q4.
"""

import math
import random
import statistics

import numpy


QUARTERS = 4

# Correlations are capped like the annual model's, so no holding moves
# in perfect lockstep with the S&P 500 (except the S&P 500 itself).
MAX_MARKET_CORRELATION = 0.95


# ============================================================
# SPLITTING A YEAR INTO QUARTERS
# ============================================================

def log_volatility(mean, std):
    """Annual std of log returns for an arithmetic mean / std."""
    growth = max(1.0 + mean, 0.05)
    return math.sqrt(math.log(1.0 + (std / growth) ** 2))


def brownian_path(rng):
    """Standard Brownian motion at the end of each quarter (t = 1/4 .. 1)."""
    position = 0.0
    path = []
    for _ in range(QUARTERS):
        position += rng.gauss(0.0, math.sqrt(1.0 / QUARTERS))
        path.append(position)
    return path


def quarter_growths(holding, market_path, rng):
    """
    Growth factor (1 + cumulative return) of one holding at the end of
    each quarter. The last quarter is exactly 1 + its annual return.
    """

    annual_growth = 1.0 + holding["return"]
    log_total = math.log(max(annual_growth, 1e-12))
    volatility = log_volatility(holding["mean"], holding["std"])

    correlation = holding["market_correlation"]
    if correlation != 1.0:
        correlation = max(
            -MAX_MARKET_CORRELATION,
            min(correlation, MAX_MARKET_CORRELATION)
        )
    own_path = brownian_path(rng)
    path = [
        correlation * market + math.sqrt(1.0 - correlation ** 2) * own
        for market, own in zip(market_path, own_path)
    ]

    growths = []
    for quarter in range(1, QUARTERS):
        t = quarter / QUARTERS
        # Bridge: the random path pinned to 0 at the start and end of
        # the year, so the year still totals log_total.
        bridge = path[quarter - 1] - t * path[-1]
        growths.append(math.exp(t * log_total + volatility * bridge))

    growths.append(annual_growth)
    return growths


def split_path(model, path, return_log, rng):
    """
    Quarterly values and stock percentages for one annual path.

    Returns:
        values:     {(year, quarter): portfolio value at quarter end}
        stock_pct:  {(year, quarter): % in stocks at quarter start,
                     None once the path has run out of money}
    """

    entries = {entry["year"]: entry for entry in return_log}
    values = {}
    stock_pct = {}

    for year in model.tracked_years:
        entry = entries.get(year)

        # The path ran out of money at the start of this year.
        if entry is None:
            for quarter in range(1, QUARTERS + 1):
                values[(year, quarter)] = 0.0
                stock_pct[(year, quarter)] = None
            continue

        start_value = path["start_values"][year]
        holdings = [h for h in entry["holdings"] if h["weight"]]

        # Any part of the allocation not assigned to a holding (only if
        # the weights don't add to 100%) earns nothing, as in the annual
        # sim's weighted return.
        cash = 1.0 - sum(h["weight"] for h in holdings)

        market_path = brownian_path(rng)
        growths = [quarter_growths(h, market_path, rng) for h in holdings]

        stock_share = sum(h["weight"] for h in holdings if h["is_stock"])
        total_share = 1.0

        for quarter in range(1, QUARTERS + 1):
            stock_pct[(year, quarter)] = (
                100 * stock_share / total_share if total_share > 0 else None
            )

            stock_share = sum(
                h["weight"] * g[quarter - 1]
                for h, g in zip(holdings, growths)
                if h["is_stock"]
            )
            total_share = cash + sum(
                h["weight"] * g[quarter - 1]
                for h, g in zip(holdings, growths)
            )
            values[(year, quarter)] = start_value * total_share

        # Exactly the annual sim's year-end value (this also adds the
        # 2027 contribution at the end of Q4).
        values[(year, QUARTERS)] = path["values"][year]

    return values, stock_pct


def simulate_quarterly_path(model, rng, verbose=False):
    """
    Run one path of the annual sim and split it into quarters. Returns
    the annual sim's path dict plus "quarter_values" and
    "quarter_stock_pct".
    """

    return_log = []
    model.RETURN_LOG = return_log

    try:
        path = model.simulate_path(verbose=verbose)
    finally:
        model.RETURN_LOG = None

    values, stock_pct = split_path(model, path, return_log, rng)
    path["quarter_values"] = values
    path["quarter_stock_pct"] = stock_pct
    return path


def quarters(model):
    """[(year, quarter), ...] in order."""
    return [
        (year, quarter)
        for year in model.tracked_years
        for quarter in range(1, QUARTERS + 1)
    ]


def label(key):
    year, quarter = key
    return f"{year} Q{quarter}"


# ============================================================
# MONTE CARLO
# ============================================================

def main(model, seed=None):
    """Run model.NUM_SIMULATIONS quarterly paths and print / graph them."""

    rng = random.Random(seed)
    keys = quarters(model)

    values_by_quarter = {key: [] for key in keys}
    stock_pct_by_quarter = {key: [] for key in keys}
    variable_withdrawals = []
    final_values = []
    failures = 0

    for simulation in range(1, model.NUM_SIMULATIONS + 1):
        path = simulate_quarterly_path(model, rng)

        for key in keys:
            values_by_quarter[key].append(path["quarter_values"][key])
            if path["quarter_stock_pct"][key] is not None:
                stock_pct_by_quarter[key].append(path["quarter_stock_pct"][key])

        variable_withdrawals.append(path["variable_withdrawal"])
        final_values.append(path["final_value"])
        failures += not path["success"]

        if simulation % 100 == 0 or simulation == model.NUM_SIMULATIONS:
            print(
                f"\rSimulation {simulation:,}/{model.NUM_SIMULATIONS:,} | "
                f"success rate so far: {1 - failures / simulation:.2%}",
                end="",
                flush=True
            )

    count = model.NUM_SIMULATIONS

    print()
    print()
    print("=" * 70)
    print("FINAL MONTE CARLO RESULTS (quarterly)")
    print("=" * 70)
    print(
        f"Average 2033 variable withdrawal: "
        f"${statistics.mean(variable_withdrawals):,.0f}"
    )
    print(
        f"Median 2033 variable withdrawal:  "
        f"${statistics.median(variable_withdrawals):,.0f}"
    )
    print(f"Simulations:       {count:,}")
    print(f"Successful:        {count - failures:,}")
    print(f"Failed:            {failures:,}")
    print(f"Success Rate:      {(count - failures) / count:.2%}")
    print(f"Median 2042 value: ${statistics.median(final_values):,.2f}")

    print()
    print("=" * 70)
    print("PORTFOLIO VALUE BY QUARTER (end of quarter)")
    print("% in stocks is at the start of the quarter")
    print("=" * 70)
    print(
        f"{'quarter':9s} {'10th pct':>12s} {'median':>12s} "
        f"{'90th pct':>12s} {'average':>12s} {'% stocks':>9s}"
    )

    for key in keys:
        values = values_by_quarter[key]
        stock = stock_pct_by_quarter[key]
        stock_text = f"{statistics.mean(stock):8.1f}%" if stock else "      n/a"
        print(
            f"{label(key):9s} "
            f"${model.percentile_value(values, 0.10):>11,.0f} "
            f"${model.percentile_value(values, 0.50):>11,.0f} "
            f"${model.percentile_value(values, 0.90):>11,.0f} "
            f"${statistics.mean(values):>11,.0f} "
            f"{stock_text}"
        )

    print("=" * 70)

    plot_results(model, values_by_quarter, stock_pct_by_quarter)


# ============================================================
# SINGLE PATH
# ============================================================

def run_single(model, seed=None):
    """One path, printed year by year and then quarter by quarter."""

    if seed is not None:
        random.seed(seed)
        numpy.random.seed(seed)  # the sims' Student-t draws use numpy

    print()
    print("=" * 70)
    print("SINGLE SIMULATED PATH (quarterly)")
    print("=" * 70)

    path = simulate_quarterly_path(model, random.Random(seed), verbose=True)

    print()
    print("=" * 70)
    print(f"{'quarter':9s} {'value at end':>16s} {'% stocks at start':>18s}")
    for key in quarters(model):
        stock = path["quarter_stock_pct"][key]
        stock_text = f"{stock:17.1f}%" if stock is not None else f"{'n/a':>18s}"
        print(f"{label(key):9s} ${path['quarter_values'][key]:>15,.2f} {stock_text}")

    print("=" * 70)
    if path["success"]:
        print("SIMULATION SUCCESSFUL")
    else:
        print(f"SIMULATION FAILED in {path['failure_year']}")
    print(
        f"2033 withdrawal:       "
        f"${path['variable_withdrawal'] + model.ANNUAL_WITHDRAWAL:,.2f} "
        f"(${path['variable_withdrawal']:,.2f} variable "
        f"+ ${model.ANNUAL_WITHDRAWAL:,.0f} regular)"
    )
    print(f"Final Portfolio Value: ${path['final_value']:,.2f}")
    print("=" * 70)

    return path


# ============================================================
# GRAPHS
# ============================================================

def plot_results(model, values_by_quarter, stock_pct_by_quarter):
    """
    Quarterly versions of the annual sims' two graphs: 10th/50th/90th
    percentile portfolio value at each quarter end, and the average
    percent in stocks at the start of each quarter.
    """

    # Only needed here, so the sims can be imported without it.
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    keys = quarters(model)
    positions = list(range(len(keys)))
    year_ticks = [i for i, (_, quarter) in enumerate(keys) if quarter == 1]
    year_labels = [str(keys[i][0]) for i in year_ticks]

    percentile_lines = [
        (0.90, "90th percentile", "#104281"),
        (0.50, "Median", "#2a78d6"),
        (0.10, "10th percentile", "#86b6ef"),
    ]

    figure, (value_axis, stock_axis) = plt.subplots(
        1, 2, figsize=(15, 5.5)
    )

    for percentile, line_label, color in percentile_lines:
        line = [
            model.percentile_value(values_by_quarter[key], percentile)
            for key in keys
        ]

        value_axis.plot(
            positions, line, color=color, linewidth=2, label=line_label
        )

        value_axis.annotate(
            f"${line[-1] / 1000:,.0f}k", (positions[-1], line[-1]),
            xytext=(6, 0), textcoords="offset points",
            va="center", fontsize=9, color="#52514e"
        )

    value_axis.set_title("Portfolio value by quarter", loc="left")
    value_axis.set_ylabel("Portfolio value (end of quarter)")
    value_axis.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"${value / 1000:,.0f}k")
    )
    value_axis.set_ylim(bottom=0)
    value_axis.legend(frameon=False)

    stock_line = [
        statistics.mean(stock_pct_by_quarter[key])
        if stock_pct_by_quarter[key] else float("nan")
        for key in keys
    ]

    stock_axis.plot(positions, stock_line, color="#2a78d6", linewidth=2)

    stock_axis.set_title(
        "Average % of portfolio in stocks (rest in bonds)", loc="left"
    )
    stock_axis.set_ylabel("% in stocks (start of quarter)")
    stock_axis.set_ylim(0, 100)

    for axis in (value_axis, stock_axis):
        axis.set_xlabel("Quarter (labels mark Q1)")
        axis.set_xticks(year_ticks)
        axis.set_xticklabels(year_labels, rotation=45)
        axis.set_xticks(positions, minor=True)
        axis.grid(axis="y", color="#e5e4df", linewidth=0.8)
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)

    figure.tight_layout()
    plt.show()
