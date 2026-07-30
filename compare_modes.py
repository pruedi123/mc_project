#!/usr/bin/env python3
"""Run a single plan through all three return modes and print a comparison.

Usage:
    python3 compare_modes.py <plan.json> [--runs 1000]

Loads a Streamlit-format plan JSON (same shape as temp_roth_only.json), runs
it through Simulated (lognormal), Historical (master_global_factors), and
Bootstrap (circular block) using the same withdrawal/tax/guardrail config in
each case, and prints a side-by-side summary table. Useful for spotting how
much of a plan's success comes from the return engine vs. its inherent
robustness.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from plan_to_sim_params import build_sim_params_from_plan
from sim_engine import (
    get_all_historical_windows,
    run_historical_parallel,
    run_monte_carlo,
    BOOTSTRAP_MODE_NAME,
)


MODES = [
    ("Lognormal", "Simulated (lognormal)"),
    ("Historical", "Historical (master_global_factors)"),
    ("Bootstrap", BOOTSTRAP_MODE_NAME),
]


def _run_one(plan, mode_value, n_runs):
    """Build sim_params *per mode* via the faithful translator and dispatch.

    Per-mode build is necessary because Streamlit recomputes blended_mu /
    blended_sigma differently depending on return_mode (lognormal uses UI
    drift/vol; historical/bootstrap use LBM history). Other sim_params are
    mode-independent — we rebuild the whole dict for simplicity / safety.
    """
    built = build_sim_params_from_plan(plan, return_mode=mode_value)
    sim_params = built["sim_params"]
    sim_years = built["sim_years"]
    inheritor_rate = built["inheritor_marginal_rate"]

    if mode_value == "Historical (master_global_factors)":
        # Use the plan's actual allocation column (e.g. LBM 60E), matching the
        # Streamlit UI — not the 100E/100F reconstruction.
        windows, _ = get_all_historical_windows(
            sim_years, built["target_stock_pct"], built["use_actual_allocation_columns"])
        results, ayd = run_historical_parallel(windows, sim_years, inheritor_rate, sim_params)
        return results, ayd, sim_years, built["base_spending"]
    results, ayd = run_monte_carlo(
        num_runs=n_runs,
        years=sim_years,
        inheritor_rate=inheritor_rate,
        taxable_log_drift=plan.get("taxable_log_drift", 0.09038261),
        taxable_log_volatility=plan.get("taxable_log_volatility", 0.20485277),
        bond_log_drift=plan.get("bond_log_drift", 0.0172918),
        bond_log_volatility=plan.get("bond_log_volatility", 0.04796435),
        return_mode=mode_value,
        **sim_params,
    )
    return results, ayd, sim_years, built["base_spending"]


def _metrics(results, all_yearly_df, sim_years, target_spending):
    if not results:
        return None
    after_tax_end = np.array([r.get("after_tax_end", 0.0) for r in results])
    portfolio_cagr = np.array([r.get("portfolio_cagr", 0.0) for r in results])
    # Same depletion definition as compute_scenario_summary ($1 floor)
    success = float((after_tax_end >= 1.0).mean())

    run_avg_spend = all_yearly_df.groupby("run")["after_tax_spending"].mean()
    # $1 tolerance for the withdrawal solver's +/- $0.50 target
    spending_success = (
        float((run_avg_spend >= target_spending - 1.0).mean()) if target_spending > 0 else float("nan")
    )

    pct = lambda a, q: float(np.percentile(a, q)) if len(a) else float("nan")
    # Median-ending path (same path Streamlit displays as "Single run detail — 50th percentile"):
    # the path whose after_tax_end sits closest to the median across all paths.
    median_end_val = float(np.median(after_tax_end))
    median_run_idx = int(np.argmin(np.abs(after_tax_end - median_end_val)))
    median_path = all_yearly_df[all_yearly_df["run"] == median_run_idx]
    median_path_avg_spending = float(median_path["after_tax_spending"].mean()) if not median_path.empty else float("nan")

    return {
        "n_paths": len(results),
        "portfolio_survival_%": success * 100,
        "spending_success_%": spending_success * 100 if not np.isnan(spending_success) else float("nan"),
        "mean_CAGR_%": portfolio_cagr.mean() * 100,
        "median_CAGR_%": float(np.median(portfolio_cagr)) * 100,
        "P5_after_tax_end": pct(after_tax_end, 5),
        "P10_after_tax_end": pct(after_tax_end, 10),
        "P50_after_tax_end": pct(after_tax_end, 50),
        "P90_after_tax_end": pct(after_tax_end, 90),
        "mean_spending": float(run_avg_spend.mean()),
        "P10_spending": pct(run_avg_spend.values, 10),
        "P50_spending": pct(run_avg_spend.values, 50),
        "median_path_avg_spending": median_path_avg_spending,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("plan", help="Path to plan JSON (Streamlit/session-state schema)")
    ap.add_argument("--runs", type=int, default=1000, help="Monte Carlo paths per stochastic mode (default 1000)")
    ap.add_argument("--show-spending-target", type=float, default=None,
                    help="Override target spending used for spending_success_%% (default: first period amount)")
    args = ap.parse_args()

    plan = json.loads(Path(args.plan).read_text())
    # Build once just to surface horizon/spending/stock% for the header.
    preview = build_sim_params_from_plan(plan, return_mode="Simulated (lognormal)")
    sim_years = preview["sim_years"]
    base_spending = preview["base_spending"]
    target_stock_pct = preview["target_stock_pct"]
    inheritor_rate = preview["inheritor_marginal_rate"]
    target_spending = (
        args.show_spending_target if args.show_spending_target is not None else float(base_spending)
    )

    print(f"Plan: {args.plan}")
    print(f"  Horizon: {sim_years} years   Stock %: {target_stock_pct*100:.0f}   "
          f"Inheritor MTR: {inheritor_rate*100:.0f}%")
    print(f"  Base spending: ${base_spending:,.0f}   Target for success-%: ${target_spending:,.0f}")
    print(f"  Stochastic-mode paths: {args.runs}")
    print()

    rows = []
    for label, mode_value in MODES:
        print(f"Running {label} ...", flush=True)
        results, ayd, _sy, _bs = _run_one(plan, mode_value, args.runs)
        m = _metrics(results, ayd, sim_years, target_spending)
        if m is None:
            print(f"  {label}: no results")
            continue
        m["mode"] = label
        rows.append(m)

    if not rows:
        print("No results to display.")
        return 1

    df = pd.DataFrame(rows).set_index("mode")
    cols = [
        ("n_paths", "Paths", "int"),
        ("portfolio_survival_%", "Survival %", "pct"),
        ("spending_success_%", f"Spending ≥ ${target_spending:,.0f}", "pct"),
        ("mean_CAGR_%", "Mean CAGR", "pct"),
        ("median_CAGR_%", "Median CAGR", "pct"),
        ("P5_after_tax_end", "P5 after-tax ending", "dollar"),
        ("P10_after_tax_end", "P10 after-tax ending", "dollar"),
        ("P50_after_tax_end", "P50 after-tax ending", "dollar"),
        ("P90_after_tax_end", "P90 after-tax ending", "dollar"),
        ("mean_spending", "Mean avg-annual spending", "dollar"),
        ("P10_spending", "P10 avg-annual spending", "dollar"),
        ("P50_spending", "P50 avg-annual spending", "dollar"),
        ("median_path_avg_spending", "Median-ending-path avg spending", "dollar"),
    ]

    def fmt(v, kind):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "—"
        if kind == "int":
            return f"{int(v):,}"
        if kind == "pct":
            return f"{v:.1f}%"
        if kind == "dollar":
            return f"${v:,.0f}"
        return str(v)

    label_width = max(len(label) for _, label, _ in cols) + 2
    mode_width = max(len(m) for m in df.index) + 2
    headers = ["Metric"] + list(df.index)
    widths = [label_width] + [max(mode_width, 18)] * len(df.index)
    sep = "  "
    print(sep.join(h.ljust(w) for h, w in zip(headers, widths)))
    print(sep.join("-" * w for w in widths))
    for key, label, kind in cols:
        row = [label.ljust(widths[0])]
        for i, mode in enumerate(df.index):
            row.append(fmt(df.loc[mode, key], kind).rjust(widths[i + 1]))
        print(sep.join(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
