#!/usr/bin/env python3
"""Run Full Process for a client plan without Streamlit."""
import json, sys, time, math
import numpy as np
import pandas as pd

# Add project to path
sys.path.insert(0, '/Users/paulruedi/Desktop/Updated Web Calcs/mc_project')
from sim_engine import (
    get_all_historical_windows, run_historical_parallel,
    store_distribution_results, load_master_global, load_bond_factors,
    PP_FACTORS, compute_run_pp_factors, simulate_withdrawals,
)


def build_sim_params(plan):
    """Build sim_params dict from a plan JSON."""
    sim_years = max(1, max(plan['life_expectancy_primary'] - plan['start_age'],
                          plan['life_expectancy_spouse'] - plan['start_age_spouse']) + 1)
    target_stock_pct = plan['target_stock_pct'] / 100.0 if plan['target_stock_pct'] > 1 else plan['target_stock_pct']

    periods = plan.get('periods', [{'amount': 0.0}])
    base_spending = periods[0].get('amount', 0.0)
    withdrawal_schedule = [base_spending] * sim_years

    mg_df = load_master_global()
    stock_log_rets = np.log(mg_df['LBM 100E'].dropna().values)
    bond_log_returns = np.log(1.0 + load_bond_factors())
    stock_mu = float(np.mean(stock_log_rets))
    bond_mu = float(np.mean(bond_log_returns))
    n = min(len(stock_log_rets), len(bond_log_returns))
    blended_log_rets = target_stock_pct * stock_log_rets[:n] + (1 - target_stock_pct) * bond_log_returns[:n]
    blended_mu = target_stock_pct * stock_mu + (1 - target_stock_pct) * bond_mu
    blended_sigma = float(np.std(blended_log_rets))

    sim_params = dict(
        start_age_primary=plan['start_age'],
        start_age_spouse=plan['start_age_spouse'],
        taxable_start=plan['taxable_start'],
        stock_total_return=0.0,
        stock_dividend_yield=plan.get('stock_dividend_yield', 0.02),
        stock_turnover=plan.get('stock_turnover', 0.10),
        investment_fee_bps=plan.get('investment_fee_bps', 0.0),
        bond_return=0.0,
        roth_start=plan['roth_start'],
        tda_start=plan['tda_start'],
        tda_spouse_start=plan['tda_spouse_start'],
        target_stock_pct=target_stock_pct,
        taxable_stock_basis_pct=plan.get('taxable_stock_basis_pct', 50.0),
        taxable_bond_basis_pct=plan.get('taxable_bond_basis_pct', 100.0),
        withdrawal_schedule=withdrawal_schedule,
        rmd_start_age=plan.get('rmd_start_age', 73),
        rmd_start_age_spouse=plan.get('rmd_start_age_spouse', 73),
        ss_income_annual=plan.get('ss_income', 0.0),
        ss_income_spouse_annual=plan.get('ss_income_spouse', 0.0),
        ss_cola=plan.get('ss_cola', 0.0),
        pension_income_annual=plan.get('pension_income', 0.0),
        pension_income_spouse_annual=plan.get('pension_income_spouse', 0.0),
        pension_cola_p1=plan.get('pension_cola_p1', 0.0),
        pension_cola_p2=plan.get('pension_cola_p2', 0.0),
        pension_survivor_pct_p1=plan.get('pension_survivor_pct_p1', 0.0),
        pension_survivor_pct_p2=plan.get('pension_survivor_pct_p2', 0.0),
        pp_factors=PP_FACTORS,
        other_income_annual=plan.get('other_income', 0.0),
        filing_status='mfj' if plan.get('filing_status', '') == 'Married Filing Jointly' else 'single',
        use_itemized_deductions=plan.get('use_itemized', False),
        itemized_deduction_amount=plan.get('itemized_deduction', 0.0),
        roth_conversion_amount=plan.get('roth_conversion_amount', 0.0),
        roth_conversion_years=plan.get('roth_conversion_years', 0),
        roth_conversion_tax_source='tda',
        roth_conversion_source='person1',
        roth_conversion_mode='none',
        roth_bracket_fill_rate=plan.get('roth_bracket_fill_rate', 0.24),
        ss_start_age_p1=plan.get('ss_start_age_p1', 67),
        ss_start_age_p2=plan.get('ss_start_age_p2', 67),
        ss_fra_age_p1=plan.get('ss_fra_age_p1', 67),
        ss_fra_age_p2=plan.get('ss_fra_age_p2', 67),
        state_tax_rate=plan.get('state_tax_rate', 0.05),
        state_exempt_retirement=plan.get('state_exempt_retirement', False),
        life_expectancy_primary=plan['life_expectancy_primary'],
        life_expectancy_spouse=plan['life_expectancy_spouse'],
        guardrails_enabled=plan.get('guardrails_enabled', True),
        guardrail_lower=plan.get('guardrail_lower', 0.75),
        guardrail_upper=plan.get('guardrail_upper', 0.90),
        guardrail_target=plan.get('guardrail_target', 0.85),
        guardrail_inner_sims=plan.get('guardrail_inner_sims', 200),
        guardrail_max_spending_pct=plan.get('guardrail_max_spending_pct', 25.0),
        taxes_enabled=plan.get('taxes_enabled', True),
        goal_schedule=None,
        flex_goal_schedule=None,
        flex_goal_min_pct=plan.get('flex_goal_min_pct', 50.0),
        base_is_essential=False,
        flex_capped_base_schedule=None,
        flex_cap_max_schedule=None,
        inheritance_enabled=plan.get('inheritance_enabled', False),
        inheritance_year=plan.get('inheritance_year', 1),
        inheritance_taxable_amount=plan.get('inheritance_taxable_amount', 0.0),
        inheritance_ira_amount=plan.get('inheritance_ira_amount', 0.0),
        tcja_sunset=plan.get('tcja_sunset', False),
        tcja_sunset_year=plan.get('tcja_sunset_year', 2),
        qcd_annual=plan.get('qcd_annual', 0.0),
        earned_income_annual=plan.get('earned_income', 0.0),
        earned_income_years=plan.get('earned_income_years', 0),
        irmaa_enabled=plan.get('irmaa_enabled', False),
        prefer_tda_before_taxable=plan.get('prefer_tda_before_taxable', False),
        goal_taxable_start=0.0,
        goal_tda_start=0.0,
        goal_tda_p1_fraction=0.0,
        goal_liquidation_schedule=None,
        goal_stock_pct=target_stock_pct,
        blended_mu=blended_mu,
        blended_sigma=blended_sigma,
    )
    return sim_params, sim_years, target_stock_pct, base_spending, mg_df


def run_sim(params, is_historical, windows, sim_years, inheritor_rate, plan):
    if is_historical:
        return run_historical_parallel(windows, sim_years, inheritor_rate, params)
    else:
        from sim_engine import run_monte_carlo
        return run_monte_carlo(
            num_runs=1000, years=sim_years, inheritor_rate=inheritor_rate,
            taxable_log_drift=plan.get('taxable_log_drift', 0.09),
            taxable_log_volatility=plan.get('taxable_log_volatility', 0.205),
            bond_log_drift=plan.get('bond_log_drift', 0.017),
            bond_log_volatility=plan.get('bond_log_volatility', 0.048),
            **params)


def find_spending(params, original_spending, target_pct, guess, is_historical, windows, sim_years, inheritor_rate, plan, tol=1000.0, max_iter=15):
    def _run(spend_amt):
        test_params = dict(params)
        scale = spend_amt / original_spending if original_spending > 0 else 1.0
        test_params['withdrawal_schedule'] = [v * scale for v in test_params['withdrawal_schedule']]
        _, ayd = run_sim(test_params, is_historical, windows, sim_years, inheritor_rate, plan)
        run_avg = ayd.groupby('run')['after_tax_spending'].mean()
        return float((run_avg >= spend_amt).mean()), float(run_avg.min())

    rate, min_avg = _run(guess)
    if abs(rate - target_pct) <= 0.01:
        return round(guess / 1000) * 1000, rate, min_avg

    if rate >= target_pct:
        lo, hi = guess, guess * 1.5
        for _ in range(5):
            r, _ = _run(hi)
            if r < target_pct:
                break
            hi = lo + (hi - lo) * 1.5
    else:
        lo, hi = guess * 0.5, guess
        for _ in range(5):
            r, _ = _run(lo)
            if r >= target_pct:
                break
            lo = max(hi - (hi - lo) * 1.5, 0.0)

    for i in range(max_iter):
        mid = (lo + hi) / 2.0
        print(f"  iter {i+1}: trying ${mid:,.0f} ...", end=' ', flush=True)
        r, _ = _run(mid)
        print(f"-> {r*100:.0f}%")
        if r >= target_pct:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    result = round((lo + hi) / 2.0 / 1000) * 1000
    final_rate, final_min = _run(result)
    return result, final_rate, final_min


def find_decline(params, spending_target, target_rate, is_historical, windows, sim_years, inheritor_rate, plan, guess_decline=20.0, tol=1.0, max_iter=15):
    balance_keys = ['taxable_start', 'tda_start', 'tda_spouse_start', 'roth_start',
        'goal_taxable_start', 'goal_tda_start']

    def _run(pct_decline):
        factor = 1.0 - pct_decline / 100.0
        test_params = dict(params)
        for k in balance_keys:
            if k in test_params:
                test_params[k] = params[k] * factor
        _, ayd = run_sim(test_params, is_historical, windows, sim_years, inheritor_rate, plan)
        run_avg = ayd.groupby('run')['after_tax_spending'].mean()
        return float((run_avg >= spending_target).mean())

    rate = _run(guess_decline)
    if abs(rate - target_rate) <= 0.01:
        return guess_decline

    if rate > target_rate:
        lo, hi = guess_decline, min(guess_decline * 1.5, 90.0)
        for _ in range(5):
            r = _run(hi)
            if r <= target_rate:
                break
            hi = min(lo + (hi - lo) * 1.5, 95.0)
    else:
        hi, lo = guess_decline, max(guess_decline * 0.5, 0.0)
        for _ in range(5):
            r = _run(lo)
            if r > target_rate:
                break
            lo = max(hi - (hi - lo) * 1.5, 0.0)

    for i in range(max_iter):
        mid = (lo + hi) / 2.0
        print(f"  iter {i+1}: trying {mid:.1f}% decline ...", end=' ', flush=True)
        r = _run(mid)
        print(f"-> {r*100:.0f}%")
        if r > target_rate:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return round((lo + hi) / 2.0, 1)


if __name__ == '__main__':
    # ── Load plan JSON ──
    # Usage: python run_full_process.py [plan.json] [--target-pct 0.90] [--shortfall-pct 80]
    target_pct_override = 0.90
    shortfall_pct = 80.0
    skip_next = False
    positional_args = []
    for i, a in enumerate(sys.argv[1:], 1):
        if skip_next:
            skip_next = False
            continue
        if a == '--target-pct':
            target_pct_override = float(sys.argv[i + 1])
            skip_next = True
        elif a == '--shortfall-pct':
            shortfall_pct = float(sys.argv[i + 1])
            skip_next = True
        elif not a.startswith('--'):
            positional_args.append(a)
    plan_path = positional_args[0] if positional_args else \
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

    # ── Pre-load historical windows ──
    t0 = time.time()
    if is_historical:
        print(f"\nLoading historical windows for {sim_years} years...")
        windows, window_start_dates = get_all_historical_windows(sim_years)
        print(f"  {len(windows)} windows loaded in {time.time()-t0:.1f}s")

    # ── Phase 1: Auto-calculate spending ──
    print("\n=== PHASE 1: Auto-calculate spending ===")
    portfolio_total = plan['taxable_start'] + plan['tda_start'] + plan['tda_spouse_start'] + plan['roth_start']
    four_pct = portfolio_total * 0.04
    income_sum = (plan.get('ss_income', 0) + plan.get('ss_income_spouse', 0) +
        plan.get('pension_income', 0) + plan.get('pension_income_spouse', 0) + plan.get('other_income', 0))
    if base_spending <= 0:
        auto_spending = round((four_pct + income_sum) / 1000) * 1000
        withdrawal_schedule = [auto_spending] * sim_years
        sim_params['withdrawal_schedule'] = list(withdrawal_schedule)
        print(f"  4% of ${portfolio_total:,.0f} = ${four_pct:,.0f} + ${income_sum:,.0f} income = ${auto_spending:,.0f}/yr")
    else:
        auto_spending = base_spending
        print(f"  Using entered spending: ${auto_spending:,.0f}/yr")
    original_spending = auto_spending

    # ── Phase 2: Initial simulation ──
    print("\n=== PHASE 2: Initial simulation ===")
    t1 = time.time()
    results, all_yearly = run_sim(sim_params, is_historical, windows, sim_years, inheritor_rate, plan)
    dist = store_distribution_results(results, all_yearly, 'historical_dist' if is_historical else 'simulated',
        ending_balance_goal, spending_target=original_spending, essential_spending=0.0)
    initial_success = dist.get('mc_spending_success_rate', 0)
    print(f"  Spending ${original_spending:,.0f}/yr -> {initial_success*100:.0f}% ideal success ({time.time()-t1:.1f}s)")

    # ── Phase 3: Spending Finder ──
    print(f"\n=== PHASE 3: Find spending at {target_pct_override*100:.0f}% ideal success ===")
    t1 = time.time()
    found_spending, found_rate, found_min = find_spending(
        sim_params, original_spending, target_pct=target_pct_override, guess=original_spending,
        is_historical=is_historical, windows=windows, sim_years=sim_years,
        inheritor_rate=inheritor_rate, plan=plan)
    print(f"  >> ${found_spending:,.0f}/yr ({found_rate*100:.0f}% ideal) | Essential floor: ${found_min:,.0f} ({time.time()-t1:.1f}s)")

    # ── Phase 4: Re-run with found spending ──
    print("\n=== PHASE 4: Re-run simulation at found spending ===")
    t1 = time.time()
    rerun_params = dict(sim_params)
    scale = found_spending / original_spending if original_spending > 0 else 1.0
    rerun_params['withdrawal_schedule'] = [v * scale for v in sim_params['withdrawal_schedule']]
    results2, all_yearly2 = run_sim(rerun_params, is_historical, windows, sim_years, inheritor_rate, plan)
    dist2 = store_distribution_results(results2, all_yearly2, 'historical_dist' if is_historical else 'simulated',
        ending_balance_goal, spending_target=found_spending, essential_spending=found_min)
    print(f"  ${found_spending:,.0f}/yr -> {dist2.get('mc_spending_success_rate',0)*100:.0f}% ideal, "
          f"{dist2.get('mc_essential_success_rate',0)*100:.0f}% essential ({time.time()-t1:.1f}s)")

    # After-tax ending balance distribution
    after_tax_ends = np.array([r['after_tax_end'] for r in results2])
    pcts = [0, 5, 10, 25, 50, 75, 90, 95]
    vals = np.percentile(after_tax_ends, pcts)
    print("\n  After-Tax Ending Balance Distribution:")
    print(f"    {'Percentile':>12}  {'Value':>14}")
    print(f"    {'─'*12}  {'─'*14}")
    for p, v in zip(pcts, vals):
        print(f"    {p:>11}%  ${v:>13,.0f}")
    print(f"    {'─'*12}  {'─'*14}")
    print(f"    {'Mean':>12}  ${np.mean(after_tax_ends):>13,.0f}")

    # ── Worst-case drill-down (Sept 1929 window) ──
    if is_historical:
        # Find the Sept 1929 window
        sept_1929_idx = None
        for i, d in enumerate(window_start_dates):
            ds = str(d)
            if '1929' in ds and ('-09-' in ds or 'Sep' in ds):
                sept_1929_idx = i
                break
        if sept_1929_idx is not None:
            stock_rets, bond_rets = windows[sept_1929_idx]
            pp_run = compute_run_pp_factors(sept_1929_idx, sim_years)
            df_1929 = simulate_withdrawals(
                years=sim_years, stock_return_series=stock_rets,
                bond_return_series=bond_rets, pp_factors_run=pp_run,
                **rerun_params)
            df_1929['total_portfolio'] = df_1929['end_taxable_total'] + df_1929['end_tda_total'] + df_1929['end_roth']
            print(f"\n  Sept 1929 Start — Year-by-Year Detail (target=${found_spending:,.0f}/yr)")
            print(f"    {'Yr':>3}  {'AgeP1':>5}  {'AgeP2':>5}  {'Input Target':>12}  {'Guardrail Adj':>13}  {'Actual Spend':>12}  {'Portfolio':>12}")
            print(f"    {'─'*3}  {'─'*5}  {'─'*5}  {'─'*12}  {'─'*13}  {'─'*12}  {'─'*12}")
            for _, r in df_1929.iterrows():
                guardrail_target = r['net_spending_target']
                actual = r['after_tax_spending']
                print(f"    {int(r['year']):>3}  {int(r['age_p1']):>5}  {int(r['age_p2']):>5}  "
                      f"${found_spending:>11,.0f}  ${guardrail_target:>12,.0f}  ${actual:>11,.0f}  ${r['total_portfolio']:>11,.0f}")
            print(f"    {'─'*3}  {'─'*5}  {'─'*5}  {'─'*12}  {'─'*13}  {'─'*12}  {'─'*12}")
            first10 = df_1929[df_1929['year'] <= 10]
            avg_first10 = first10['after_tax_spending'].mean() if not first10.empty else df_1929['after_tax_spending'].mean()
            shortfall_threshold = found_min * (shortfall_pct / 100.0)
            n_below = int((df_1929['after_tax_spending'] < shortfall_threshold).sum())
            pct_below = n_below / len(df_1929) * 100
            n_below_10 = int((first10['after_tax_spending'] < shortfall_threshold).sum()) if not first10.empty else 0
            pct_below_10 = n_below_10 / min(10, len(df_1929)) * 100
            print(f"    Avg actual: ${df_1929['after_tax_spending'].mean():,.0f}  |  "
                  f"First 10yr avg: ${avg_first10:,.0f}  |  "
                  f"Min actual: ${df_1929['after_tax_spending'].min():,.0f}  |  "
                  f"Final portfolio: ${df_1929['total_portfolio'].iloc[-1]:,.0f}")
            print(f"    Below {shortfall_pct:.0f}% of essential (${shortfall_threshold:,.0f}): "
                  f"{n_below}/{len(df_1929)} yrs ({pct_below:.1f}%) overall  |  "
                  f"{n_below_10}/{min(10, len(df_1929))} yrs ({pct_below_10:.1f}%) in first 10")

    # ── Phase 5: Balance Decline Finder (75% target) ──
    print("\n=== PHASE 5: Find balance decline to reach 75% success ===")
    t1 = time.time()
    decline_pct = find_decline(rerun_params, found_spending, target_rate=0.75,
        is_historical=is_historical, windows=windows, sim_years=sim_years,
        inheritor_rate=inheritor_rate, plan=plan, guess_decline=20.0)
    balance_keys = ['taxable_start', 'tda_start', 'tda_spouse_start', 'roth_start', 'goal_taxable_start', 'goal_tda_start']
    orig_total = sum(rerun_params.get(k, 0.0) for k in balance_keys)
    decline_factor = 1.0 - decline_pct / 100.0
    reduced_total = orig_total * decline_factor
    dollar_drop = orig_total - reduced_total

    # Historical probability of this decline (3-month horizon)
    stock_f = mg_df['LBM 100E'].dropna().values
    bond_f = mg_df['LBM 100 F'].dropna().values
    nf = min(len(stock_f), len(bond_f))
    stock_f = stock_f[:nf]; bond_f = bond_f[:nf]
    horizon_months = 3

    def _extract_monthly(factors_12mo):
        nn = len(factors_12mo)
        monthly = np.zeros(nn + 11)
        seed = factors_12mo[0] ** (1.0 / 12.0)
        for k in range(12):
            monthly[k] = seed
        for t in range(nn - 1):
            monthly[t + 12] = monthly[t] * factors_12mo[t + 1] / factors_12mo[t]
        return monthly

    stock_monthly = _extract_monthly(stock_f)
    bond_monthly = _extract_monthly(bond_f)
    nm = min(len(stock_monthly), len(bond_monthly))
    blended_monthly = target_stock_pct * stock_monthly[:nm] + (1 - target_stock_pct) * bond_monthly[:nm]
    wealth = np.concatenate([[1.0], np.cumprod(blended_monthly)])
    w = wealth[12:]
    rolling_rets = w[horizon_months:] / w[:len(w) - horizon_months] - 1.0
    n_declines = int(np.sum(rolling_rets <= -(decline_pct / 100.0)))
    empirical_prob = n_declines / len(rolling_rets) if len(rolling_rets) > 0 else 0.0

    def _norm_cdf(x):
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
    blended_annual = target_stock_pct * stock_f + (1 - target_stock_pct) * bond_f
    annual_log_rets = np.log(blended_annual)
    mu = float(np.mean(annual_log_rets))
    sigma = float(np.std(annual_log_rets))
    frac = horizon_months / 12.0
    mu_h = mu * frac; sigma_h = sigma * np.sqrt(frac)
    log_thr = np.log(1.0 - decline_pct / 100.0)
    simulated_prob = _norm_cdf((log_thr - mu_h) / sigma_h)

    print(f"  >> {decline_pct:.1f}% decline (${orig_total:,.0f} -> ${reduced_total:,.0f}, -${dollar_drop:,.0f})")
    print(f"     Historical prob: {empirical_prob*100:.1f}% | Simulated prob: {simulated_prob*100:.1f}% (3-month horizon)")
    print(f"     ({time.time()-t1:.1f}s)")

    # ── Phase 6: Stressed Spending (85% target at declined balances) ──
    print("\n=== PHASE 6: Find spending at declined balances (85% target) ===")
    t1 = time.time()
    stressed_params = dict(rerun_params)
    for k in balance_keys:
        if k in stressed_params:
            stressed_params[k] = rerun_params[k] * decline_factor
    stressed_spending, stressed_rate, stressed_min = find_spending(
        stressed_params, found_spending, target_pct=0.85, guess=found_spending,
        is_historical=is_historical, windows=windows, sim_years=sim_years,
        inheritor_rate=inheritor_rate, plan=plan)
    spending_delta = stressed_spending - found_spending
    print(f"  >> ${stressed_spending:,.0f}/yr ({stressed_rate*100:.0f}% ideal)")
    print(f"     Delta from plan: {'+'if spending_delta>=0 else '-'}${abs(spending_delta):,.0f}/yr "
          f"({spending_delta/found_spending*100:+.1f}%)")
    print(f"     ({time.time()-t1:.1f}s)")

    # ── Final Summary ──
    elapsed = time.time() - t0
    print(f"""
{'='*60}
           FULL PROCESS SUMMARY
{'='*60}
  Starting portfolio          ${orig_total:,.0f}
  Optimal spending (90%)      ${found_spending:,.0f}/yr
  Essential floor (100%)      ${found_min:,.0f}/yr
  Decline to reach 75%        {decline_pct:.1f}% (-${dollar_drop:,.0f})
  Decline prob (3-mo)         Hist: {empirical_prob*100:.1f}% | Sim: {simulated_prob*100:.1f}%
  Stressed portfolio           ${reduced_total:,.0f}
  Stressed spending (85%)     ${stressed_spending:,.0f}/yr
  Spending change if decline  {'+'if spending_delta>=0 else '-'}${abs(spending_delta):,.0f}/yr ({spending_delta/found_spending*100:+.1f}%)
{'='*60}
  Total time: {elapsed:.1f}s
""")
