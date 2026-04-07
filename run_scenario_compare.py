#!/usr/bin/env python3
"""Compare two find_spending scenarios at different target success rates."""
import json, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, '/Users/paulruedi/Desktop/Updated Web Calcs/mc_project')
from sim_engine import (
    get_all_historical_windows, run_historical_parallel,
    store_distribution_results, load_master_global, load_bond_factors,
    PP_FACTORS, compute_run_pp_factors, simulate_withdrawals,
)
from run_full_process import build_sim_params, run_sim, find_spending


def run_scenario(label, sim_params, original_spending, target_pct, is_historical,
                 windows, window_start_dates, sim_years, inheritor_rate, plan,
                 ending_balance_goal):
    """Run find_spending + re-run + stress tests for one target rate. Returns summary dict."""
    print(f"\n{'='*60}")
    print(f"  SCENARIO: {label} (target {target_pct*100:.0f}%)")
    print(f"{'='*60}")

    t1 = time.time()
    found_spending, found_rate, found_min = find_spending(
        sim_params, original_spending, target_pct=target_pct, guess=original_spending,
        is_historical=is_historical, windows=windows, sim_years=sim_years,
        inheritor_rate=inheritor_rate, plan=plan)
    print(f"  >> ${found_spending:,.0f}/yr ({found_rate*100:.0f}% ideal) | Essential floor: ${found_min:,.0f} ({time.time()-t1:.1f}s)")

    # Re-run at found spending
    rerun_params = dict(sim_params)
    scale = found_spending / original_spending if original_spending > 0 else 1.0
    rerun_params['withdrawal_schedule'] = [v * scale for v in sim_params['withdrawal_schedule']]
    results, all_yearly = run_sim(rerun_params, is_historical, windows, sim_years, inheritor_rate, plan)
    dist = store_distribution_results(results, all_yearly, 'historical_dist' if is_historical else 'simulated',
        ending_balance_goal, spending_target=found_spending, essential_spending=found_min)

    # Compute overall and first-10-year averages across all runs
    run_avg = all_yearly.groupby('run')['after_tax_spending'].mean()
    first10 = all_yearly[all_yearly['year'] <= 10]
    run_first10 = first10.groupby('run')['after_tax_spending'].mean() if not first10.empty else run_avg

    # Percentile stats
    pcts = [0, 10, 25, 50, 75, 90]
    avg_pctiles = {p: np.percentile(run_avg, p) for p in pcts}
    first10_pctiles = {p: np.percentile(run_first10, p) for p in pcts}

    print(f"\n  {'Pctl':>6}  {'Full Avg':>12}  {'First 10yr':>12}")
    print(f"  {'─'*6}  {'─'*12}  {'─'*12}")
    for p in pcts:
        print(f"  {p:>5}%  ${avg_pctiles[p]:>11,.0f}  ${first10_pctiles[p]:>11,.0f}")

    # Stress test: 1929 and 1966
    stress_results = {}
    if is_historical:
        stress_codes = [('1929-09', 'Sept 1929'), ('1966-01', 'Jan 1966')]
        for code, name in stress_codes:
            idx = None
            for i, d in enumerate(window_start_dates):
                ds = str(d)
                yr_mo = code.split('-')
                if yr_mo[0] in ds and f'-{yr_mo[1]}-' in ds:
                    idx = i
                    break
            if idx is not None:
                stock_rets, bond_rets = windows[idx]
                pp_run = compute_run_pp_factors(idx, sim_years)
                df_stress = simulate_withdrawals(
                    years=sim_years, stock_return_series=stock_rets,
                    bond_return_series=bond_rets, pp_factors_run=pp_run,
                    **rerun_params)
                df_stress['total_portfolio'] = df_stress['end_taxable_total'] + df_stress['end_tda_total'] + df_stress['end_roth']
                s_first10 = df_stress[df_stress['year'] <= 10]
                avg_all = df_stress['after_tax_spending'].mean()
                avg_10 = s_first10['after_tax_spending'].mean() if not s_first10.empty else avg_all
                min_spend = df_stress['after_tax_spending'].min()
                shortfall_years = int((df_stress['after_tax_spending'] < found_spending).sum())
                shortfall_10 = int((s_first10['after_tax_spending'] < found_spending).sum()) if not s_first10.empty else 0
                stress_results[code] = {
                    'name': name, 'avg_all': avg_all, 'avg_10': avg_10,
                    'min': min_spend, 'shortfall_years': shortfall_years,
                    'shortfall_10': shortfall_10, 'total_years': len(df_stress),
                    'df': df_stress,
                }
                print(f"\n  {name} Start:")
                print(f"    Full avg: ${avg_all:,.0f}  |  First 10yr avg: ${avg_10:,.0f}  |  Min: ${min_spend:,.0f}")
                print(f"    Shortfall years: {shortfall_years}/{len(df_stress)} total, {shortfall_10}/{min(10, len(df_stress))} in first 10")

    return {
        'label': label, 'target_pct': target_pct,
        'found_spending': found_spending, 'found_rate': found_rate, 'found_min': found_min,
        'avg_pctiles': avg_pctiles, 'first10_pctiles': first10_pctiles,
        'stress': stress_results, 'dist': dist,
    }


if __name__ == '__main__':
    plan_path = sys.argv[1] if len(sys.argv) > 1 else \
        '/Users/paulruedi/RWM/Current Client Plans/Black, Larry & Lisa/black_larry_lisa.json'
    with open(plan_path) as f:
        plan = json.load(f)

    print(f"Plan: {plan_path}")
    print(f"Portfolio: taxable=${plan['taxable_start']:,.0f}  TDA=${plan['tda_start']:,.0f}  "
          f"TDA-spouse=${plan['tda_spouse_start']:,.0f}  Roth=${plan['roth_start']:,.0f}")

    sim_params, sim_years, target_stock_pct, base_spending, mg_df = build_sim_params(plan)
    inheritor_rate = plan['inheritor_marginal_rate']
    ending_balance_goal = plan.get('ending_balance_goal', 0.0)
    is_historical = 'Historical' in plan.get('return_mode', 'Historical')
    windows = None
    window_start_dates = None

    t0 = time.time()
    if is_historical:
        print(f"\nLoading historical windows for {sim_years} years...")
        windows, window_start_dates = get_all_historical_windows(sim_years)
        print(f"  {len(windows)} windows loaded in {time.time()-t0:.1f}s")

    # Auto-calculate spending if needed
    portfolio_total = plan['taxable_start'] + plan['tda_start'] + plan['tda_spouse_start'] + plan['roth_start']
    four_pct = portfolio_total * 0.04
    income_sum = (plan.get('ss_income', 0) + plan.get('ss_income_spouse', 0) +
        plan.get('pension_income', 0) + plan.get('pension_income_spouse', 0) + plan.get('other_income', 0))
    if base_spending <= 0:
        original_spending = round((four_pct + income_sum) / 1000) * 1000
        sim_params['withdrawal_schedule'] = [original_spending] * sim_years
        print(f"\nAuto spending: ${original_spending:,.0f}/yr")
    else:
        original_spending = base_spending
        print(f"\nUsing entered spending: ${original_spending:,.0f}/yr")

    # ── Run both scenarios ──
    sc1 = run_scenario('Scenario 1 (95%)', sim_params, original_spending, 0.95,
        is_historical, windows, window_start_dates, sim_years, inheritor_rate, plan, ending_balance_goal)

    sc2 = run_scenario('Scenario 2 (85%)', sim_params, original_spending, 0.85,
        is_historical, windows, window_start_dates, sim_years, inheritor_rate, plan, ending_balance_goal)

    # ── Comparison Summary ──
    print(f"\n{'='*70}")
    print(f"  SCENARIO COMPARISON")
    print(f"{'='*70}")
    print(f"  {'':>30}  {'95% Target':>14}  {'85% Target':>14}  {'Delta':>14}")
    print(f"  {'─'*30}  {'─'*14}  {'─'*14}  {'─'*14}")
    print(f"  {'Found Spending':>30}  ${sc1['found_spending']:>13,.0f}  ${sc2['found_spending']:>13,.0f}  ${sc2['found_spending']-sc1['found_spending']:>+13,.0f}")
    print(f"  {'Essential Floor':>30}  ${sc1['found_min']:>13,.0f}  ${sc2['found_min']:>13,.0f}  ${sc2['found_min']-sc1['found_min']:>+13,.0f}")

    # Percentile comparison
    print(f"\n  Full-Horizon Avg Spending:")
    for p in [0, 10, 25, 50, 75, 90]:
        v1 = sc1['avg_pctiles'][p]
        v2 = sc2['avg_pctiles'][p]
        print(f"    {p:>3}th pctl:{'':>18}  ${v1:>13,.0f}  ${v2:>13,.0f}  ${v2-v1:>+13,.0f}")

    print(f"\n  First 10-Year Avg Spending:")
    for p in [0, 10, 25, 50, 75, 90]:
        v1 = sc1['first10_pctiles'][p]
        v2 = sc2['first10_pctiles'][p]
        print(f"    {p:>3}th pctl:{'':>18}  ${v1:>13,.0f}  ${v2:>13,.0f}  ${v2-v1:>+13,.0f}")

    # Stress test comparison
    for code in ['1929-09', '1966-01']:
        if code in sc1['stress'] and code in sc2['stress']:
            s1 = sc1['stress'][code]
            s2 = sc2['stress'][code]
            print(f"\n  {s1['name']} Start:")
            print(f"    {'Full avg:':>30}  ${s1['avg_all']:>13,.0f}  ${s2['avg_all']:>13,.0f}  ${s2['avg_all']-s1['avg_all']:>+13,.0f}")
            print(f"    {'First 10yr avg:':>30}  ${s1['avg_10']:>13,.0f}  ${s2['avg_10']:>13,.0f}  ${s2['avg_10']-s1['avg_10']:>+13,.0f}")
            print(f"    {'Min spending:':>30}  ${s1['min']:>13,.0f}  ${s2['min']:>13,.0f}  ${s2['min']-s1['min']:>+13,.0f}")
            print(f"    {'Shortfall yrs (total):':>30}  {s1['shortfall_years']:>13}  {s2['shortfall_years']:>13}  {s2['shortfall_years']-s1['shortfall_years']:>+13}")
            print(f"    {'Shortfall yrs (first 10):':>30}  {s1['shortfall_10']:>13}  {s2['shortfall_10']:>13}  {s2['shortfall_10']-s1['shortfall_10']:>+13}")

            # Full year-by-year side-by-side
            df1 = s1['df'][['year', 'after_tax_spending', 'total_portfolio']].set_index('year')
            df2 = s2['df'][['year', 'after_tax_spending', 'total_portfolio']].set_index('year')
            t1 = sc1['found_spending']
            t2 = sc2['found_spending']
            print(f"\n    Year-by-Year ({s1['name']})  [target: 95%=${t1:,.0f}  85%=${t2:,.0f}]")
            print(f"    {'Yr':>4}  {'95% Spend':>12} {'':>1} {'95% Portf':>12}  {'85% Spend':>12} {'':>1} {'85% Portf':>12}  {'Spend Δ':>12}")
            print(f"    {'─'*4}  {'─'*12} {'─':>1} {'─'*12}  {'─'*12} {'─':>1} {'─'*12}  {'─'*12}")
            max_yr = max(df1.index.max(), df2.index.max())
            for yr in range(1, max_yr + 1):
                if yr in df1.index and yr in df2.index:
                    v1 = df1.loc[yr, 'after_tax_spending']
                    p1 = df1.loc[yr, 'total_portfolio']
                    v2 = df2.loc[yr, 'after_tax_spending']
                    p2 = df2.loc[yr, 'total_portfolio']
                    flag1 = '*' if v1 < t1 else ' '
                    flag2 = '*' if v2 < t2 else ' '
                    sep = '  ───' if yr == 10 else ''
                    print(f"    {yr:>4}  ${v1:>11,.0f} {flag1} ${p1:>11,.0f}  ${v2:>11,.0f} {flag2} ${p2:>11,.0f}  ${v2-v1:>+11,.0f}{sep}")
            # Summary line
            avg1_10 = df1.loc[1:10, 'after_tax_spending'].mean()
            avg2_10 = df2.loc[1:10, 'after_tax_spending'].mean()
            avg1_all = df1['after_tax_spending'].mean()
            avg2_all = df2['after_tax_spending'].mean()
            print(f"    {'─'*4}  {'─'*12}   {'─'*12}  {'─'*12}   {'─'*12}  {'─'*12}")
            print(f"    {'1-10':>4}  ${avg1_10:>11,.0f}   {'avg':>12}  ${avg2_10:>11,.0f}   {'avg':>12}  ${avg2_10-avg1_10:>+11,.0f}")
            print(f"    {'All':>4}  ${avg1_all:>11,.0f}   {'avg':>12}  ${avg2_all:>11,.0f}   {'avg':>12}  ${avg2_all-avg1_all:>+11,.0f}")
            print(f"    * = below target")

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"{'='*70}")
