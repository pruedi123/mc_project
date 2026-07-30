#!/usr/bin/env python3
"""Worst-path drill-down across all three return modes.

For each of {Lognormal, Historical, Bootstrap}, identify the worst paths by
deepest first-10-year drawdown, then report:
  - Worst single path: year and depth of drawdown, spending in worst year,
    avg spending across worst 5-year stretch, ending portfolio
  - Worst 10% of paths: same metrics averaged across that cohort

Maps the "1929 window" narrative onto the synthetic modes — gives you a
"bad twin" story per mode for client conversations.

Usage:
    python3 worst_paths.py <plan.json> [--runs 1000] [--worst-pct 10]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_full_process import build_sim_params
from sim_engine import (
    BOOTSTRAP_MODE_NAME,
    get_all_historical_windows,
    run_historical_parallel,
    run_monte_carlo,
)


MODES = [
    ("Lognormal", "Simulated (lognormal)"),
    ("Historical", "Historical (master_global_factors)"),
    ("Bootstrap", BOOTSTRAP_MODE_NAME),
]


def _run_one(plan, sim_params, sim_years, inheritor_rate, mode_value, n_runs):
    if mode_value == "Historical (master_global_factors)":
        windows, dates = get_all_historical_windows(sim_years)
        results, ayd = run_historical_parallel(windows, sim_years, inheritor_rate, sim_params)
        return results, ayd, dates
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
    return results, ayd, None


def _path_metrics(ayd: pd.DataFrame, starting_portfolio: float, dd_horizon_years: int = 10):
    """For each run, return per-path metrics.

    Drawdown is portfolio's lowest value vs. peak observed up to that year,
    measured over the first dd_horizon_years. The "trough" is the minimum
    portfolio value over that window; depth is (trough/peak) - 1.
    """
    if "total_portfolio" not in ayd.columns:
        raise ValueError("all_yearly_df missing 'total_portfolio' column")

    rows = []
    grouped = ayd.groupby("run")
    for run_id, df_run in grouped:
        df_run = df_run.sort_values("year")
        years = df_run["year"].values
        port = df_run["total_portfolio"].values
        spend = df_run["after_tax_spending"].values

        # Drawdown over first dd_horizon years, peak resets to starting portfolio
        window = port[:dd_horizon_years]
        if len(window) == 0:
            continue
        baseline = np.concatenate([[starting_portfolio], window])
        running_peak = np.maximum.accumulate(baseline)
        dd_series = baseline / running_peak - 1.0  # length dd_horizon+1
        dd_idx = int(np.argmin(dd_series))
        depth = float(dd_series[dd_idx])
        trough_year = int(dd_idx)  # 0 means baseline (year 0), 1..dd_horizon means yr1..yr10
        trough_value = float(baseline[dd_idx])

        # Worst single year for spending
        ws_idx = int(np.argmin(spend))
        worst_year_spending = float(spend[ws_idx])
        worst_year_index = int(years[ws_idx])

        # Worst 5-yr-avg spending stretch
        if len(spend) >= 5:
            rolling = np.array([spend[i:i + 5].mean() for i in range(len(spend) - 4)])
            wsa_idx = int(np.argmin(rolling))
            worst_5yr_avg = float(rolling[wsa_idx])
            worst_5yr_start = int(years[wsa_idx])
        else:
            worst_5yr_avg = float(spend.mean())
            worst_5yr_start = int(years[0])

        ending = float(port[-1])
        mean_spend = float(spend.mean())

        rows.append({
            "run": int(run_id),
            "dd_depth": depth,
            "dd_year": trough_year,
            "dd_trough": trough_value,
            "worst_year_spending": worst_year_spending,
            "worst_year_index": worst_year_index,
            "worst_5yr_avg_spending": worst_5yr_avg,
            "worst_5yr_start": worst_5yr_start,
            "ending_portfolio": ending,
            "mean_spending": mean_spend,
        })
    return pd.DataFrame(rows)


def _format_single(label, mode_label, row, dates, ending_baseline):
    out = [f"=== {mode_label} — Worst single path ==="]
    run_id = int(row["run"])
    if dates is not None and len(dates) > run_id:
        out.append(f"  Historical window start: {dates[run_id].strftime('%b %Y')}")
    out.append(f"  Sim id: {run_id}")
    out.append(f"  First-10yr drawdown trough: ${row['dd_trough']:,.0f} in year {int(row['dd_year'])} ({row['dd_depth']*100:.1f}%)")
    out.append(f"  Worst single year of spending: ${row['worst_year_spending']:,.0f} (year {int(row['worst_year_index'])})")
    out.append(f"  Worst 5-year avg spending: ${row['worst_5yr_avg_spending']:,.0f} (years {int(row['worst_5yr_start'])}–{int(row['worst_5yr_start']) + 4})")
    out.append(f"  Mean spending across all 30 yrs: ${row['mean_spending']:,.0f}")
    out.append(f"  Ending portfolio: ${row['ending_portfolio']:,.0f}  (vs baseline ${ending_baseline:,.0f})")
    return "\n".join(out)


def _format_cohort(label, mode_label, cohort, worst_pct, dates=None):
    out = [f"=== {mode_label} — Worst {worst_pct}% of paths (n={len(cohort)}) ==="]
    if dates is not None and len(cohort) > 0:
        sample_dates = [dates[int(r)].strftime("%b %Y") for r in cohort["run"].iloc[:6] if int(r) < len(dates)]
        if sample_dates:
            out.append(f"  Window starts (first 6): {', '.join(sample_dates)}")
    out.append(f"  Avg first-10yr drawdown depth: {cohort['dd_depth'].mean()*100:.1f}%")
    out.append(f"  Avg trough portfolio value: ${cohort['dd_trough'].mean():,.0f}  (worst-of-worst: ${cohort['dd_trough'].min():,.0f})")
    out.append(f"  Avg worst-year spending: ${cohort['worst_year_spending'].mean():,.0f}  (worst-of-worst: ${cohort['worst_year_spending'].min():,.0f})")
    out.append(f"  Avg worst-5yr avg spending: ${cohort['worst_5yr_avg_spending'].mean():,.0f}")
    out.append(f"  Avg ending portfolio: ${cohort['ending_portfolio'].mean():,.0f}")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("plan", help="Path to plan JSON (Streamlit/session-state schema)")
    ap.add_argument("--runs", type=int, default=1000, help="Monte Carlo paths per stochastic mode (default 1000)")
    ap.add_argument("--worst-pct", type=int, default=10, help="Cohort percentile cutoff (default 10 = worst 10%)")
    args = ap.parse_args()

    plan = json.loads(Path(args.plan).read_text())
    sim_params, sim_years, target_stock_pct, base_spending, _mg = build_sim_params(plan)
    inheritor_rate = float(plan.get("inheritor_marginal_rate", 0.32))
    starting_portfolio = (
        float(plan.get("taxable_start", 0))
        + float(plan.get("tda_start", 0))
        + float(plan.get("tda_spouse_start", 0))
        + float(plan.get("roth_start", 0))
    )

    print(f"Plan: {args.plan}")
    print(f"  Horizon: {sim_years} years   Stock %: {target_stock_pct*100:.0f}   "
          f"Starting portfolio: ${starting_portfolio:,.0f}")
    print(f"  Base spending: ${base_spending:,.0f}   Stochastic paths: {args.runs}   "
          f"Cohort: worst {args.worst_pct}%")
    print()

    for label, mode_value in MODES:
        print(f"--- {label} ({mode_value}) ---", flush=True)
        results, ayd, dates = _run_one(plan, sim_params, sim_years, inheritor_rate, mode_value, args.runs)
        m = _path_metrics(ayd, starting_portfolio)
        if m.empty:
            print(f"  no paths\n")
            continue

        # Sort by deepest first-10yr drawdown (most negative = worst)
        m_sorted = m.sort_values("dd_depth", ascending=True)
        worst_single = m_sorted.iloc[0]
        cohort_n = max(1, int(round(len(m_sorted) * args.worst_pct / 100.0)))
        cohort = m_sorted.head(cohort_n)

        ending_median = float(m["ending_portfolio"].median())

        print(_format_single(label, label, worst_single, dates, ending_median))
        print()
        print(_format_cohort(label, label, cohort, args.worst_pct, dates))
        print()


if __name__ == "__main__":
    raise SystemExit(main())
