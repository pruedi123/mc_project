#!/usr/bin/env python3
"""Year-by-year walk down the Sept-1929 historical path for two plans:
  (A) Full 30-yr worst-case lockbox  -> spends $40k every year, guaranteed.
  (B) Capture method = front-end lockbox (yrs 1-10, sweep excess>$40k into the
      MC corpus) + MC guardrails back-end (yrs 11-30) along the actual returns.
Front-end uses the iris real per-allocation factors (same series that built the
cost factors); back-end uses mc_project's guardrails engine on the same data.
All real dollars, all-Roth (tax-free)."""
import copy, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd

from plan_to_sim_params import build_sim_params_from_plan
from sim_engine import simulate_withdrawals, compute_run_pp_factors, load_master_global
from solve_capital_roth import BASE

IRIS = Path("/Users/paulruedi/Desktop/Updated Web Calcs/updated_iris/master_global_factors.xlsx")
INCOME = 40_000.0
EQPCT = {1:10,2:10,3:20,4:30,5:30,6:30,7:30,8:50,9:50,10:60}
CF = {1:1.231406562,2:1.292126847,3:1.282723002,4:1.187658741,5:1.13755828,
      6:1.187268433,7:1.132553958,8:1.035426612,9:1.063703604,10:0.972778868}
CF_FULL = {11:0.874630846,12:0.84017019,13:0.817495223,14:0.772040398,15:0.673003405,
           16:0.588978652,17:0.534879174,18:0.589253821,19:0.569080068,20:0.578858285,
           21:0.465174493,22:0.354242196,23:0.330070407,24:0.34065398,25:0.282058842,
           26:0.239057965,27:0.213614258,28:0.177219885,29:0.177605917,30:0.144337551}
EQ_FULL = {11:70,12:70,13:70,14:70,15:80,16:90,17:90,18:90,19:90,20:90,
           21:90,22:90,23:90,24:90,25:90,26:90,27:90,28:90,29:90,30:100}
SEED_BACKEND = 572_883.0   # grows to ~$655k at yr11 (LBM 70E worst-case)

def col(eq): return "LBM 100F" if eq == 0 else f"LBM {int(eq)}E"

def main():
    dfi = pd.read_excel(IRIS, engine="openpyxl"); dfi.columns = dfi.columns.str.strip()
    bm = pd.to_datetime(dfi["begin month"])
    s = int((bm - pd.Timestamp("1929-09-30")).abs().values.argmin())
    print(f"Sept-1929 start row: {bm.iloc[s].date()} (iris)")
    def fac(eq, k):  # annual factor for allocation eq, k years after start (k=0..)
        return float(dfi[col(eq)].iloc[s + 12*k])

    # ---------- FRONT-END (capture) yrs 1-10 + recapture ----------
    print("\nCAPTURE METHOD — front-end lockbox, years 1-10 (real $)")
    print(f"{'Yr':>3} {'Eq%':>3} {'bucket opens':>13} {'spend':>8} {'recaptured':>11}")
    recap_rolled = 0.0; recap_each = []
    for y in range(1, 11):
        cum = 1.0
        for k in range(y): cum *= fac(EQPCT[y], k)
        bucket = INCOME * CF[y] * cum
        excess = max(0.0, bucket - INCOME)
        roll = 1.0
        for k in range(y, 11): roll *= fac(60, k)   # roll to yr11 in LBM 60E
        recap_rolled += excess * roll
        recap_each.append(excess)
        print(f"{y:>3} {EQPCT[y]:>3} ${bucket:>11,.0f} ${INCOME:>7,.0f} ${excess:>10,.0f}")
    seed_grown = SEED_BACKEND
    for k in range(11): seed_grown *= fac(70, k)     # seed in LBM 70E for 11 yrs
    corpus = seed_grown + recap_rolled
    print(f"  total recaptured (real, at sweep): ${sum(recap_each):,.0f}")
    print(f"  recaptured rolled to yr11        : ${recap_rolled:,.0f}")
    print(f"  back-end seed grown to yr11       : ${seed_grown:,.0f}")
    print(f"  ==> MC corpus entering year 11    : ${corpus:,.0f}")

    # ---------- BACK-END (capture) yrs 11-30 via mc_project guardrails ----------
    plan = copy.deepcopy(BASE); plan["roth_start"] = corpus; plan["tda_start"] = 0.0
    built = build_sim_params_from_plan(plan, return_mode="Historical (master_global_factors)")
    sp = built["sim_params"]; ny = built["sim_years"]
    dfm = load_master_global(); bmm = pd.to_datetime(dfm["begin month"])
    s11 = int((bmm - pd.Timestamp("1940-09-30")).abs().values.argmin())  # yr11 = 11y after 1929
    idx = [s11 + 12*k for k in range(ny)]
    stock = dfm["LBM 100E"].iloc[idx].values - 1.0
    bond  = dfm["LBM 100 F"].iloc[idx].values - 1.0
    dfr = simulate_withdrawals(years=ny, stock_return_series=stock, bond_return_series=bond,
                               pp_factors_run=compute_run_pp_factors(0, ny), **sp)
    spend_be = dfr["after_tax_spending"].tolist()

    # ---------- LOCKBOX (full 30-yr) along same path ----------
    lb_spend = [INCOME]*30
    lb_legacy = 0.0
    for y in range(1, 31):
        eq = EQPCT[y] if y <= 10 else EQ_FULL[y]
        cf = CF[y] if y <= 10 else CF_FULL[y]
        cum = 1.0
        for k in range(y): cum *= fac(eq, k)
        lb_legacy += max(0.0, INCOME*cf*cum - INCOME)

    # ---------- SIDE BY SIDE ----------
    print("\n\nYEAR-BY-YEAR SPENDING — Sept 1929 path (real $)")
    print(f"{'PlanYr':>6} {'Calendar':>9} {'Lockbox':>10} {'Capture':>10}   note")
    for y in range(1, 31):
        cal = 1929 + y
        if y <= 10:
            cap = INCOME; note = "front-end floor (+sweep)"
        else:
            cap = spend_be[y-11]; note = "MC guardrails back-end"
        print(f"{y:>6} {cal:>9} ${lb_spend[y-1]:>9,.0f} ${cap:>9,.0f}   {note}")
    print(f"\nLockbox unspent surplus over 30 yrs (legacy): ${lb_legacy:,.0f}")
    end_bal = float(dfr['end_taxable_total'].iloc[-1] + dfr['end_tda_total'].iloc[-1] + dfr['end_roth'].iloc[-1])
    print(f"Capture back-end ending balance (yr30): ${end_bal:,.0f}")
    print(f"Capture total spend 30y: ${INCOME*10 + sum(spend_be):,.0f}   "
          f"Lockbox total spend 30y: ${INCOME*30:,.0f}")

if __name__ == "__main__":
    main()
