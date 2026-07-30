#!/usr/bin/env python3
"""Per-year after-tax spending distribution for the MC back-end (years 11-30),
all-Roth $40k target, at a given starting corpus. Prints P10/P50/P90 by year."""
import copy, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from compare_modes import _run_one
from solve_capital_roth import BASE   # reuse the all-Roth $40k plan

MODE = "Historical (master_global_factors)"

def per_year(corpus):
    plan = copy.deepcopy(BASE)
    plan["roth_start"] = corpus
    plan["tda_start"] = 0.0
    results, ayd, sim_years, _ = _run_one(plan, MODE, 1000)
    g = ayd.groupby("year")["after_tax_spending"]
    p10 = g.quantile(0.10); p50 = g.quantile(0.50); p90 = g.quantile(0.90)
    return p10, p50, p90, sim_years

def main():
    corpora = {"hybrid $655k": 655_000.0, "recapture ~$956k": 956_000.0}
    out = {}
    for name, c in corpora.items():
        p10, p50, p90, n = per_year(c)
        out[name] = (p10, p50, p90)
        print(f"\n=== {name}  (years map to plan-years 11..{10+n}) ===")
        print(f"{'simYr':>5} {'planYr':>6} {'P10':>9} {'P50(median)':>12} {'P90':>9}")
        for y in range(1, n+1):
            print(f"{y:>5} {10+y:>6} ${p10[y]:>8,.0f} ${p50[y]:>11,.0f} ${p90[y]:>8,.0f}")

if __name__ == "__main__":
    main()
