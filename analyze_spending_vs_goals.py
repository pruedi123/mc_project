#!/usr/bin/env python3
"""Spending vs goal/essential across all 820 historical cohorts.

Reports two distinct lenses on the same run:
  (1) AVERAGE lens  -- % of cohorts whose 30-year AVERAGE annual after-tax spend
      meets goal ($80k) and essential ($72k). This is what a household
      experiences "on average" over retirement.
  (2) SINGLE-YEAR lens -- even when the average clears essential, individual
      years can dip below it. We count those lean years so the gap between
      "averaged fine" and "every single year fine" is explicit.

Usage: python3 analyze_spending_vs_goals.py truth_layer_sample_plan.json
"""
import os
os.environ.setdefault("MC_SEED", "424242")  # match the export's reproducible seed

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from run_full_process import build_sim_params
from sim_engine import get_all_historical_windows, run_historical_parallel


def main():
    plan = json.loads(Path(sys.argv[1]).read_text())
    sim_params, sim_years, target_stock_pct, base_spending, _ = build_sim_params(plan)
    inheritor_rate = float(plan.get("inheritor_marginal_rate", 0.32))
    goal = float(base_spending)                       # 80,000
    essential = float(plan.get("essential_spending", 0.0))  # 72,000

    windows, dates = get_all_historical_windows(sim_years, target_stock_pct)
    _results, ayd = run_historical_parallel(windows, sim_years, inheritor_rate, sim_params)

    n = ayd["run"].nunique()
    run_avg = ayd.groupby("run")["after_tax_spending"].mean()
    spend = ayd["after_tax_spending"].values
    total_years = len(spend)

    print(f"Cohorts: {n}   Horizon: {sim_years} yrs   Goal: ${goal:,.0f}   Essential: ${essential:,.0f}\n")

    # (1) AVERAGE lens
    print("=== AVERAGE annual spend over the full retirement ===")
    for label, thr in [("goal", goal), ("essential", essential)]:
        pct = (run_avg >= thr).mean() * 100
        print(f"  cohorts whose 30-yr AVERAGE >= {label} (${thr:,.0f}): {(run_avg>=thr).sum():>3}/{n}  ({pct:.1f}%)")
    print(f"  median cohort average: ${run_avg.median():,.0f}")

    # (2) SINGLE-YEAR lens
    print("\n=== SINGLE-YEAR detail (the gap people miss) ===")
    every_year_ess = ayd.groupby("run")["after_tax_spending"].min() >= essential
    every_year_goal = ayd.groupby("run")["after_tax_spending"].min() >= goal
    print(f"  cohorts where EVERY year >= essential (${essential:,.0f}): {every_year_ess.sum():>3}/{n}  ({every_year_ess.mean()*100:.1f}%)")
    print(f"  cohorts where EVERY year >= goal      (${goal:,.0f}): {every_year_goal.sum():>3}/{n}  ({every_year_goal.mean()*100:.1f}%)")

    below_ess = spend < essential
    print(f"\n  total cohort-years below essential: {below_ess.sum():,} of {total_years:,} ({below_ess.mean()*100:.2f}% of all years)")

    # The key group: averaged >= essential, yet had >=1 lean year
    avg_ok = run_avg >= essential
    dipped = ~every_year_ess
    both = avg_ok & dipped
    print(f"  cohorts that AVERAGED >= essential but still had >=1 year below it: {both.sum()}/{n}")
    if dipped.sum() > 0:
        years_below = ayd[ayd["after_tax_spending"] < essential].groupby("run").size()
        print(f"  among cohorts that ever dip: median {int(years_below.median())} lean years, "
              f"max {int(years_below.max())} lean years")
        print(f"  deepest single lean year anywhere: ${spend.min():,.0f}")


if __name__ == "__main__":
    raise SystemExit(main())
