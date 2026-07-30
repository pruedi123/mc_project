import os, json
os.environ.setdefault("MC_SEED","424242")
from pathlib import Path
from run_full_process import build_sim_params
from sim_engine import get_all_historical_windows, simulate_withdrawals, compute_run_pp_factors

plan=json.loads(Path("truth_layer_sample_plan.json").read_text())
sp,yrs,tsp,base,_=build_sim_params(plan)
windows,dates=get_all_historical_windows(yrs,tsp)
i=next(k for k,d in enumerate(dates) if f"{d:%Y-%m}"=="1929-09")
sr,br=windows[i]; pp=compute_run_pp_factors(i,yrs)
df=simulate_withdrawals(years=yrs, stock_return_series=sr, bond_return_series=br, pp_factors_run=pp, **sp)
df["total_portfolio"]=df["end_taxable_total"]+df["end_tda_total"]+df["end_roth"]
spend=df["after_tax_spending"].values; port=df["total_portfolio"].values
print("WEBSITE (mc_project) 1929-09  [total spend = gross withdrawals, incl real variable SS]:")
print(f"  30-yr avg spending: ${spend.mean():,.0f}")
print(f"  WORST single year:  ${spend.min():,.0f}  (year {int(spend.argmin())+1})")
print(f"  first-decade avg:   ${spend[:10].mean():,.0f}")
print(f"  ending portfolio:   ${port[-1]:,.0f}")
sscol = [c for c in df.columns if c=="ss_income"]
print(f"\n{'yr':>3} {'spend(total)':>13} {'portfolio':>13}")
for y in range(yrs):
    print(f"{y+1:>3} {spend[y]:>13,.0f} {port[y]:>13,.0f}")
