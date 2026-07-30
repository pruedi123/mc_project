#!/usr/bin/env python3
"""Export the website truth-layer CSV: one row per real historical outcome.

The website says "we ran N outcomes; here are 3." This file IS those N outcomes.
N is deterministic and citable: it equals the number of overlapping monthly-start
historical return windows for the plan's horizon (30 years -> 820 windows,
Jun-1927 ... Sep-1995). Monte Carlo is NOT used here -- 1000 is arbitrary; only
the historical windows give a fixed, auditable count.

3 rows are flagged in `shown_on_site` as the representatives the website displays:
  typical = median outcome by average annual spending
  worst   = deepest first-10yr drawdown (the 1929-window story)
  best    = highest ending portfolio (most left to heirs)

Reproducibility: sets MC_SEED so the guardrail inner-sims are deterministic, so
re-running produces a byte-identical CSV. This MUST be set before importing
sim_engine (the seed base is read at import time).

Usage:
    python3 export_truth_layer_csv.py truth_layer_sample_plan.json [--out outcomes_820.csv]
"""
import os
os.environ.setdefault("MC_SEED", "424242")  # fixed -> reproducible; set before sim_engine import

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from run_full_process import build_sim_params, find_spending
from sim_engine import get_all_historical_windows, get_allocation_column, run_historical_parallel
from worst_paths import _path_metrics  # reuse existing per-path metrics


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("plan", help="Path to a representative (non-client) plan JSON")
    ap.add_argument("--out", default="truth_layer_outcomes.csv", help="Output CSV path")
    ap.add_argument("--solve-spending", type=float, default=None, metavar="PCT",
                    help="Solve max sustainable base spending so PCT (e.g. 0.85) of "
                         "the 820 cohorts average at least the base. Overrides plan's periods amount.")
    args = ap.parse_args()

    plan_path = Path(args.plan)
    plan = json.loads(plan_path.read_text())

    sim_params, sim_years, target_stock_pct, base_spending, _mg = build_sim_params(plan)
    inheritor_rate = float(plan.get("inheritor_marginal_rate", 0.32))

    # Match the Streamlit app: essential_spending is DISPLAY-ONLY (a reference
    # success threshold shown in results), NOT a hard floor. The engine flexes the
    # full base via guardrails; spending CAN dip below essential in bad cohorts.
    # We report how often each cohort kept every year at/above essential, but we do
    # NOT inject it as goal_schedule (that would diverge from what the app produces).
    essential = float(plan.get("essential_spending", 0.0))
    starting_portfolio = (
        float(plan.get("taxable_start", 0))
        + float(plan.get("tda_start", 0))
        + float(plan.get("tda_spouse_start", 0))
        + float(plan.get("roth_start", 0))
    )

    windows, dates = get_all_historical_windows(sim_years, target_stock_pct)
    ret_col = get_allocation_column(target_stock_pct)
    n_outcomes = len(windows)
    print(f"Horizon: {sim_years} yrs   Historical windows (outcomes): {n_outcomes}")
    print(f"  Returns column: {ret_col!r}  (actual balanced portfolio for {target_stock_pct*100:.0f}% equity)")
    print(f"  Window starts: {dates[0]:%b %Y} ... {dates[-1]:%b %Y}")
    print(f"  Starting portfolio: ${starting_portfolio:,.0f}   Stock %: {target_stock_pct*100:.0f}   "
          f"Annual fee: {plan.get('investment_fee_bps', 0):.0f} bps   Taxes: {plan.get('taxes_enabled', True)}", flush=True)

    solved_pct = None
    if args.solve_spending is not None:
        solved_pct = float(args.solve_spending)
        seed = base_spending if base_spending > 0 else 50000.0
        print(f"\nSolving base spending so {solved_pct*100:.0f}% of {n_outcomes} cohorts "
              f"average >= base (seed ${seed:,.0f})...", flush=True)
        found, rate, min_avg = find_spending(
            sim_params, seed, solved_pct, seed, True, windows, sim_years, inheritor_rate, plan)
        base_spending = float(found)
        sim_params = dict(sim_params)
        sim_params["withdrawal_schedule"] = [base_spending] * sim_years
        print(f"  -> Base spending ${base_spending:,.0f}/yr  ({rate*100:.0f}% of cohorts met it; "
              f"worst-cohort avg ${min_avg:,.0f})", flush=True)
    print(f"  Base spending used: ${base_spending:,.0f}", flush=True)

    results, ayd = run_historical_parallel(windows, sim_years, inheritor_rate, sim_params)
    m = _path_metrics(ayd, starting_portfolio)
    if len(m) != n_outcomes:
        print(f"WARNING: metrics rows ({len(m)}) != windows ({n_outcomes})", file=sys.stderr)

    m = m.sort_values("run").reset_index(drop=True)

    # Build the public, client-free outcome table
    df = pd.DataFrame({
        "outcome_id": m["run"].astype(int) + 1,
        "window_start": [f"{dates[int(r)]:%Y-%m}" for r in m["run"]],
        "avg_annual_spending": m["mean_spending"].round(0).astype(int),
        "worst_year_spending": m["worst_year_spending"].round(0).astype(int),
        "worst_5yr_avg_spending": m["worst_5yr_avg_spending"].round(0).astype(int),
        "max_drawdown_pct": (m["dd_depth"] * 100).round(1),
        "drawdown_year": m["dd_year"].astype(int),
        "ending_portfolio": m["ending_portfolio"].round(0).astype(int),
    })

    # Single-year lens: per-cohort count of years below goal / essential.
    yb_goal = ayd[ayd["after_tax_spending"] < base_spending].groupby("run").size()
    df["years_below_goal"] = [int(yb_goal.get(int(r), 0)) for r in m["run"]]
    if essential > 0:
        yb_ess = ayd[ayd["after_tax_spending"] < essential].groupby("run").size()
        df["years_below_essential"] = [int(yb_ess.get(int(r), 0)) for r in m["run"]]
    else:
        df["years_below_essential"] = 0

    df["shown_on_site"] = ""

    # --- pick the 3 representatives ---
    # "best" is the 90th-percentile cohort by lifetime spending, NOT the single
    # max -- the max (a 1932 bottom-buyer) is an outlier and reads as cherry-picking
    # the upside. The 90th percentile is a strong-but-defensible "good" outcome.
    typical_id = int(df.iloc[(df["avg_annual_spending"] - df["avg_annual_spending"].median()).abs().argmin()]["outcome_id"])
    worst_id = int(df.iloc[m["dd_depth"].values.argmin()]["outcome_id"])  # most negative drawdown
    p90 = df["avg_annual_spending"].quantile(0.90)
    best_id = int(df.iloc[(df["avg_annual_spending"] - p90).abs().argmin()]["outcome_id"])

    df.loc[df["outcome_id"] == typical_id, "shown_on_site"] = "typical"
    df.loc[df["outcome_id"] == worst_id, "shown_on_site"] = "worst"
    df.loc[df["outcome_id"] == best_id, "shown_on_site"] = "best"

    out_path = Path(args.out)
    df.to_csv(out_path, index=False)

    # --- provenance sidecar: makes the claim auditable ---
    meta = {
        "n_outcomes": int(n_outcomes),
        "website_claim": f"{n_outcomes} outcomes simulated; 3 shown",
        "horizon_years": int(sim_years),
        "window_start_first": f"{dates[0]:%Y-%m}",
        "window_start_last": f"{dates[-1]:%Y-%m}",
        "return_mode": "Historical (master_global_factors)",
        "data_source": "master_global_factors.xlsx",
        "returns_column": ret_col,
        "starting_portfolio": int(round(starting_portfolio)),
        "target_stock_pct": int(round(target_stock_pct * 100)),
        "investment_fee_bps": int(round(plan.get("investment_fee_bps", 0))),
        "ss_person1": {"amount": plan.get("ss_income", 0), "start_age": plan.get("ss_start_age_p1")},
        "ss_person2": {"amount": plan.get("ss_income_spouse", 0), "start_age": plan.get("ss_start_age_p2")},
        "spending_goal": int(round(base_spending)),
        "essential_threshold": int(round(essential)),
        "essential_semantics": "display-only (matches app: not enforced as a hard floor)",
        "base_spending_solved": None if solved_pct is None else f"solved so {int(solved_pct*100)}% of cohorts meet base",
        # Two-lens summary: AVERAGE-over-retirement vs EVERY-single-year.
        "average_lens": {
            "pct_cohorts_avg_ge_goal": round(float((df["avg_annual_spending"] >= base_spending).mean() * 100), 1),
            "pct_cohorts_avg_ge_essential": None if essential <= 0 else round(float((df["avg_annual_spending"] >= essential).mean() * 100), 1),
            "median_cohort_average": int(round(float(df["avg_annual_spending"].median()))),
        },
        "single_year_lens": {
            "pct_cohorts_every_year_ge_essential": None if essential <= 0 else round(float((df["years_below_essential"] == 0).mean() * 100), 1),
            "pct_cohorts_every_year_ge_goal": round(float((df["years_below_goal"] == 0).mean() * 100), 1),
            "cohorts_every_year_ge_essential": None if essential <= 0 else int((df["years_below_essential"] == 0).sum()),
            "cohorts_with_a_year_below_essential": None if essential <= 0 else int((df["years_below_essential"] > 0).sum()),
            "total_cohort_years_below_essential": None if essential <= 0 else int(df["years_below_essential"].sum()),
            "total_cohort_years": int(n_outcomes * sim_years),
            "deepest_single_year": int(df["worst_year_spending"].min()),
            "lean_years_distribution": {
                "0": int((df["years_below_essential"] == 0).sum()),
                "1-2": int(df["years_below_essential"].between(1, 2).sum()),
                "3-5": int(df["years_below_essential"].between(3, 5).sum()),
                "6-10": int(df["years_below_essential"].between(6, 10).sum()),
                "11+": int((df["years_below_essential"] >= 11).sum()),
            },
        },
        "spending_cap": "none" if plan.get("guardrail_max_spending_pct", 25.0) < 0 else f"{plan.get('guardrail_max_spending_pct')}% above base",
        "taxes_enabled": bool(plan.get("taxes_enabled", True)),
        "guardrails": {
            "lower": plan.get("guardrail_lower"),
            "target": plan.get("guardrail_target"),
            "upper": plan.get("guardrail_upper"),
        },
        "representatives": {
            "typical": {"id": typical_id, "rule": "median by average lifetime spending"},
            "worst": {"id": worst_id, "rule": "deepest first-10yr drawdown"},
            "best": {"id": best_id, "rule": "90th percentile by average lifetime spending (not the max)"},
        },
        "mc_seed": os.environ["MC_SEED"],
        "engine_commit": _git_commit(),
        "plan_file": plan_path.name,
    }
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))

    # --- acceptance checks ---
    assert len(df) == n_outcomes, "row count != window count"
    assert (df["shown_on_site"] != "").sum() == 3, "must flag exactly 3 representatives"

    print(f"\nWrote {len(df)} rows -> {out_path}")
    print(f"Provenance  -> {meta_path}")
    print("\nThe 3 shown:")
    print(df[df["shown_on_site"] != ""].to_string(index=False))


if __name__ == "__main__":
    raise SystemExit(main())
