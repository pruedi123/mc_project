#!/usr/bin/env python3
"""EXPERIMENT: bear-triggered Social Security acceleration vs. the website baseline.

Same canonical plan as the firm stress test (truth_layer_sample_plan.json:
$80k goal / $72k essential, 60/40 LBM 60E, taxes off, 120 bps). One behavioral
rule added:

  If a cohort's FIRST YEAR equity return (LBM 100E) is worse than -20%,
  everyone not yet claiming Social Security claims early -- the year AFTER the
  bad year, floored at age 62 -- instead of waiting to their planned age.
  The FRA benefit (PIA) is held fixed; the engine's actuarial reduction
  (ss_adjustment_factor) makes the early check permanently smaller.

Compares triggered cohorts head-to-head (baseline claiming vs. accelerated) on
worst-year spending, whether the $72k floor held every year, 30-yr average
spending, and ending portfolio. Non-triggered cohorts are identical to the
website, so the headline floor-held count only moves via triggered cohorts.

Usage: python3 experiment_bear_ss.py
"""
import os
os.environ.setdefault("MC_SEED", "424242")
import json
from pathlib import Path
import numpy as np
import pandas as pd

from run_full_process import build_sim_params
from sim_engine import (
    get_all_historical_windows, run_historical_parallel,
    load_master_global, ss_adjustment_factor,
)

ROOT = Path(__file__).resolve().parent
TRIGGER = -0.20          # year-1 equity return threshold
FLOOR_AGE = 62           # earliest SS eligibility


def cohort_metrics(ayd, dates, essential):
    """Return dict start_date -> metrics from an all-yearly df."""
    out = {}
    for run_id, g in ayd.groupby("run"):
        g = g.sort_values("year")
        spend = g["after_tax_spending"].values
        port = g["total_portfolio"].values
        start = f"{dates[int(run_id)]:%Y-%m}"
        out[start] = dict(
            avg=spend.mean(), worst=spend.min(), first10=spend[:10].mean(),
            ending=port[-1], floor_held=bool((spend >= essential).all()),
        )
    return out


def main():
    plan = json.loads((ROOT / "truth_layer_sample_plan.json").read_text())
    sp, yrs, tsp, base, _ = build_sim_params(plan)
    goal = float(base)
    essential = float(plan.get("essential_spending", 0.0))
    inh = float(plan.get("inheritor_marginal_rate", 0.32))

    # blended windows used for the portfolio sim (LBM 60E)
    windows, dates = get_all_historical_windows(yrs, tsp)

    # pure equity (LBM 100E) year-1 return per start month, for the trigger
    df = load_master_global()
    eq = df["LBM 100E"].values  # rolling 12-mo growth factors
    year1_equity = np.array([eq[i] - 1.0 for i in range(len(windows))])
    triggered = [i for i in range(len(windows)) if year1_equity[i] < TRIGGER]

    # ---- baseline: all 820, planned claiming (reproduces the website) ----
    _r, ayd_base = run_historical_parallel(windows, yrs, inh, sp)
    base_m = cohort_metrics(ayd_base, dates, essential)

    # ---- accelerated SS for triggered cohorts ----
    fra1, fra2 = plan["ss_fra_age_p1"], plan["ss_fra_age_p2"]
    age1, age2 = plan["start_age"], plan["start_age_spouse"]
    pia1 = plan["ss_income"] / ss_adjustment_factor(plan["ss_start_age_p1"], fra1)
    pia2 = plan["ss_income_spouse"] / ss_adjustment_factor(plan["ss_start_age_p2"], fra2)
    # claim the year after the bad year (age in year 2 = start_age+1), floored at 62,
    # never later than the originally planned age
    new_age1 = min(plan["ss_start_age_p1"], max(FLOOR_AGE, age1 + 1))
    new_age2 = min(plan["ss_start_age_p2"], max(FLOOR_AGE, age2 + 1))
    new_inc1 = pia1 * ss_adjustment_factor(new_age1, fra1)
    new_inc2 = pia2 * ss_adjustment_factor(new_age2, fra2)

    plan_mod = dict(plan)
    plan_mod.update(ss_income=new_inc1, ss_start_age_p1=new_age1,
                    ss_income_spouse=new_inc2, ss_start_age_p2=new_age2)
    sp_mod, *_ = build_sim_params(plan_mod)

    trig_windows = [windows[i] for i in triggered]
    trig_dates = [dates[i] for i in triggered]
    _r2, ayd_mod = run_historical_parallel(trig_windows, yrs, inh, sp_mod)
    mod_m = cohort_metrics(ayd_mod, trig_dates, essential)

    # ---- report ----
    print("=" * 78)
    print("BEAR-TRIGGERED SS ACCELERATION  vs  WEBSITE BASELINE")
    print("=" * 78)
    print(f"Plan: goal ${goal:,.0f} | essential ${essential:,.0f} | {len(windows)} retirements")
    print(f"Trigger: year-1 equity (LBM 100E) < {TRIGGER:.0%}")
    print()
    print("SS claiming change applied ONLY to triggered cohorts:")
    print(f"  Person 1: ${plan['ss_income']:,.0f} @ age {plan['ss_start_age_p1']}"
          f"   ->  ${new_inc1:,.0f} @ age {new_age1}   (PIA ${pia1:,.0f})")
    print(f"  Person 2: ${plan['ss_income_spouse']:,.0f} @ age {plan['ss_start_age_p2']}"
          f"   ->  ${new_inc2:,.0f} @ age {new_age2}   (PIA ${pia2:,.0f})")
    print()
    print(f"Triggered cohorts: {len(triggered)} of {len(windows)}")
    for a in ["1929-09", "1966-01", "1973-01"]:
        i = next((k for k, d in enumerate(dates) if f"{d:%Y-%m}" == a), None)
        hit = (i in triggered) if i is not None else False
        y1 = year1_equity[i] if i is not None else float("nan")
        print(f"  {a}: year-1 equity {y1:+.1%}  -> {'TRIGGERS' if hit else 'no trigger'}")
    print()

    # head-to-head on triggered cohorts
    rows = []
    for i in triggered:
        s = f"{dates[i]:%Y-%m}"
        b, m = base_m[s], mod_m[s]
        rows.append(dict(start=s, y1eq=year1_equity[i],
                         worst_b=b["worst"], worst_m=m["worst"],
                         avg_b=b["avg"], avg_m=m["avg"],
                         end_b=b["ending"], end_m=m["ending"],
                         held_b=b["floor_held"], held_m=m["floor_held"]))
    R = pd.DataFrame(rows).sort_values("start")

    worse_year_help = (R.worst_m - R.worst_b)
    flips_up = int(((~R.held_b) & (R.held_m)).sum())   # floor failed -> now holds
    flips_dn = int(((R.held_b) & (~R.held_m)).sum())   # floor held -> now fails
    print("--- Effect on triggered cohorts ---")
    print(f"  Worst-year spending: improved in {(worse_year_help > 1).sum()}, "
          f"worse in {(worse_year_help < -1).sum()}, ~equal in {(worse_year_help.abs() <= 1).sum()}")
    print(f"  Median worst-year change: ${worse_year_help.median():+,.0f}  "
          f"(mean ${worse_year_help.mean():+,.0f})")
    print(f"  Floor-held flips: {flips_up} cohorts fail->hold, {flips_dn} hold->fail")
    print(f"  30-yr avg spending change: median ${ (R.avg_m-R.avg_b).median():+,.0f}  "
          f"(mean ${(R.avg_m-R.avg_b).mean():+,.0f})")
    print(f"  Ending portfolio change:   median ${(R.end_m-R.end_b).median():+,.0f}  "
          f"(mean ${(R.end_m-R.end_b).mean():+,.0f})")
    print()

    # headline floor-held count (only triggered cohorts can change)
    base_floor = sum(1 for v in base_m.values() if v["floor_held"])
    new_floor = base_floor + flips_up - flips_dn
    print("--- Headline (all 820) ---")
    print(f"  $72k floor held every year:  baseline {base_floor}/{len(windows)}"
          f"  ->  with rule {new_floor}/{len(windows)}")
    print()

    # show the worst handful by baseline worst-year
    print("--- Worst 12 triggered cohorts (by baseline worst single year) ---")
    show = R.sort_values("worst_b").head(12)
    print(f"  {'start':>8} {'y1eq':>7} {'worst(base)':>11} {'worst(rule)':>11} "
          f"{'Δworst':>8} {'held b/r':>9}")
    for _, x in show.iterrows():
        print(f"  {x.start:>8} {x.y1eq:>+6.0%} {x.worst_b:>11,.0f} {x.worst_m:>11,.0f} "
              f"{x.worst_m-x.worst_b:>+8,.0f}  {('Y' if x.held_b else 'N')}->{('Y' if x.held_m else 'N')}")

    R.to_csv(ROOT / "experiment_bear_ss_triggered.csv", index=False)
    print(f"\nFull triggered-cohort detail -> experiment_bear_ss_triggered.csv")


if __name__ == "__main__":
    raise SystemExit(main())
