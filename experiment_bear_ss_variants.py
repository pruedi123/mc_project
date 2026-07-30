#!/usr/bin/env python3
"""Variants of the bear-triggered SS rule: both spouses vs. higher earner only."""
import os
os.environ.setdefault("MC_SEED", "424242")
import json
from pathlib import Path
import numpy as np
from run_full_process import build_sim_params
from sim_engine import (get_all_historical_windows, run_historical_parallel,
                        load_master_global, ss_adjustment_factor)

ROOT = Path(__file__).resolve().parent


def metrics(ayd, ds, ess):
    o = {}
    for rid, g in ayd.groupby("run"):
        g = g.sort_values("year")
        s = g["after_tax_spending"].values
        p = g["total_portfolio"].values
        o[f"{ds[int(rid)]:%Y-%m}"] = dict(avg=s.mean(), worst=s.min(),
                                          ending=p[-1], held=bool((s >= ess).all()))
    return o


def main():
    plan = json.loads((ROOT / "truth_layer_sample_plan.json").read_text())
    sp, yrs, tsp, base, _ = build_sim_params(plan)
    ess = float(plan["essential_spending"]); inh = float(plan["inheritor_marginal_rate"])
    windows, dates = get_all_historical_windows(yrs, tsp)
    eq = load_master_global()["LBM 100E"].values
    trig = [i for i in range(len(windows)) if eq[i] - 1.0 < -0.20]

    _, ab = run_historical_parallel(windows, yrs, inh, sp)
    B = metrics(ab, dates, ess)
    bf = sum(1 for v in B.values() if v["held"])

    fra1, fra2 = plan["ss_fra_age_p1"], plan["ss_fra_age_p2"]
    a1, a2 = plan["start_age"], plan["start_age_spouse"]
    pia1 = plan["ss_income"] / ss_adjustment_factor(plan["ss_start_age_p1"], fra1)
    pia2 = plan["ss_income_spouse"] / ss_adjustment_factor(plan["ss_start_age_p2"], fra2)

    def run_variant(name, accel_p1, accel_p2):
        pm = dict(plan)
        if accel_p1:
            na1 = min(plan["ss_start_age_p1"], max(62, a1 + 1))
            pm["ss_start_age_p1"] = na1; pm["ss_income"] = pia1 * ss_adjustment_factor(na1, fra1)
        if accel_p2:
            na2 = min(plan["ss_start_age_p2"], max(62, a2 + 1))
            pm["ss_start_age_p2"] = na2; pm["ss_income_spouse"] = pia2 * ss_adjustment_factor(na2, fra2)
        spm, *_ = build_sim_params(pm)
        tw = [windows[i] for i in trig]; td = [dates[i] for i in trig]
        _, am = run_historical_parallel(tw, yrs, inh, spm)
        M = metrics(am, td, ess)
        k = lambda i: f"{dates[i]:%Y-%m}"
        dW = np.array([M[k(i)]["worst"] - B[k(i)]["worst"] for i in trig])
        dA = np.array([M[k(i)]["avg"] - B[k(i)]["avg"] for i in trig])
        dE = np.array([M[k(i)]["ending"] - B[k(i)]["ending"] for i in trig])
        fu = sum(1 for i in trig if (not B[k(i)]["held"]) and M[k(i)]["held"])
        fd = sum(1 for i in trig if B[k(i)]["held"] and not M[k(i)]["held"])
        print(f"\n=== {name} ===")
        print(f"  worst-year:  median ${np.median(dW):>+8,.0f}  mean ${dW.mean():>+8,.0f}  | {(dW>1).sum():>2} better, {(dW<-1).sum():>2} worse")
        print(f"  30yr avg:    median ${np.median(dA):>+8,.0f}  mean ${dA.mean():>+8,.0f}")
        print(f"  ending:      median ${np.median(dE):>+8,.0f}  mean ${dE.mean():>+8,.0f}")
        print(f"  floor flips: {fu} fail->hold, {fd} hold->fail  |  headline {bf} -> {bf+fu-fd} / 820")

    print(f"Triggered: {len(trig)}/820  (baseline floor held {bf}/820)")
    run_variant("BOTH claim early (P1 67->66, P2 65->62)", True, True)
    run_variant("ONLY higher earner P1 claims early (67->66)", True, False)


if __name__ == "__main__":
    raise SystemExit(main())
