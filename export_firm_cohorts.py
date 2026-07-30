#!/usr/bin/env python3
"""Export the 820 cohorts in the RUEDIWEALTH SITE schema:
   start,regime,avgSpend,pctGoal,first10Avg,worstYear,finalPortfolio,survived

Feeds ~/ruediwealth-site/public/methodology/cohorts-820.csv (the site's single
source of truth -> build-cohorts.mjs -> cohorts-data.json). Uses the canonical
plan (truth_layer_sample_plan.json): $80k goal / $72k essential, taxes off,
120 bps, LBM 60E. With taxes off, spending == gross withdrawal (before taxes).

Usage: python3 export_firm_cohorts.py truth_layer_sample_plan.json [--out cohorts-820.csv]
"""
import os
os.environ.setdefault("MC_SEED", "424242")

import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from run_full_process import build_sim_params
from sim_engine import get_all_historical_windows, run_historical_parallel

ANCHORS = {"1929-09", "1966-01", "1973-01"}

def regime_for(start):
    y, m = int(start[:4]), int(start[5:7])
    if start == "1929-09": return "1929 crash and Depression"
    if start == "1966-01": return "Peak before 1970s inflation"
    if start == "1973-01": return "1973 oil shock and inflation"
    if (y, m) < (1929, 9):           return "Late-1920s boom"
    if y <= 1932:                    return "Crash and Depression"
    if y <= 1938:                    return "1930s recovery, 1937 slump"
    if y <= 1945:                    return "World War II"
    if y <= 1953:                    return "Postwar boom"
    if y <= 1965:                    return "1950s–60s rising market"
    if y <= 1972:                    return "Late-1960s market peak"
    if y <= 1981:                    return "1970s oil shock and inflation"
    if y <= 1989:                    return "Early-1980s recovery"
    return "Early-1990s growth"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plan")
    ap.add_argument("--out", default="cohorts-820.csv")
    args = ap.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    sp, yrs, tsp, base, _ = build_sim_params(plan)
    goal = float(base); essential = float(plan.get("essential_spending", 0.0))
    inh = float(plan.get("inheritor_marginal_rate", 0.32))
    windows, dates = get_all_historical_windows(yrs, tsp)
    _r, ayd = run_historical_parallel(windows, yrs, inh, sp)

    rows = []
    anchor_detail = {}
    for run_id, g in ayd.groupby("run"):
        g = g.sort_values("year")
        spend = g["after_tax_spending"].values
        port = g["total_portfolio"].values
        start = f"{dates[int(run_id)]:%Y-%m}"
        avg = int(round(spend.mean()))
        rows.append({
            "start": start,
            "regime": regime_for(start),
            "avgSpend": avg,
            "pctGoal": round(avg / goal, 2),
            "first10Avg": int(round(spend[:10].mean())),
            "worstYear": int(round(spend.min())),
            "finalPortfolio": int(round(port[-1])),
            "survived": "true" if (essential <= 0 or (spend >= essential).all()) else "false",
        })
        if start in ANCHORS:
            anchor_detail[start] = {
                "avg": avg, "first10": int(round(spend[:10].mean())),
                "worst": int(round(spend.min())), "final": int(round(port[-1])),
                "pctGoal": round(avg / goal, 3),
                "years": [int(round(s)) for s in spend],
                "ports": [int(round(p)) for p in port],
            }

    df = pd.DataFrame(rows).sort_values("start")
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} cohorts -> {args.out}  (goal ${goal:,.0f}, essential ${essential:,.0f})")
    print(f"survived (every year >= essential): {(df.survived=='true').sum()}/{len(df)}")
    print()
    print("=== ANCHOR before/after (old hardcoded -> new canonical run) ===")
    old = {"1929-09": (77412, 1284), "1966-01": (87820, 337596), "1973-01": (91950, 522190)}
    for a in ["1929-09", "1966-01", "1973-01"]:
        d = anchor_detail.get(a)
        if not d: print(f"  {a}: MISSING"); continue
        oa, of = old[a]
        print(f"  {a} [{regime_for(a)}]")
        print(f"     avgSpend   ${oa:>8,}  ->  ${d['avg']:>8,}   ({d['pctGoal']*100:.0f}% of $80k goal)")
        print(f"     finalPort  ${of:>8,}  ->  ${d['final']:>8,}")
        print(f"     worst yr ${d['worst']:,} | first-decade avg ${d['first10']:,}")
    # save anchor detail for prose work
    Path("anchor_detail.json").write_text(json.dumps(anchor_detail, indent=2))
    print("\nFull anchor year-by-year saved -> anchor_detail.json")

if __name__ == "__main__":
    raise SystemExit(main())
