"""Headless Monte Carlo retirement simulation runner.

No Streamlit dependency. Takes a clean plan dict, runs simulation,
generates PDF report, returns results.
"""

import numpy as np
import pandas as pd

from sim_engine import (
    PP_FACTORS, compute_run_pp_factors, load_master_global, load_bond_factors,
    get_all_historical_windows, compute_summary_metrics, build_scenario_params,
    compute_scenario_summary, run_monte_carlo, simulate_withdrawals,
)
from pdf_report import generate_report
from plan_schema import apply_defaults, validate_plan, plan_to_session_state


# ── Withdrawal schedule builder ───────────────────────────────

def _build_withdrawal_schedule(plan: dict, years: int) -> list:
    """Build year-by-year withdrawal schedule from plan spending config."""
    spending = plan['spending']
    if spending.get('periods'):
        schedule = []
        for p in spending['periods']:
            n = p.get('years', years - len(schedule))  # last period fills remaining
            schedule.extend([float(p['amount'])] * n)
    else:
        schedule = [float(spending['annual'])] * years

    # Pad or trim to match horizon
    if len(schedule) < years:
        last = schedule[-1] if schedule else 0.0
        schedule.extend([last] * (years - len(schedule)))
    return schedule[:years]


# ── Goal schedule builder ─────────────────────────────────────

def _build_goal_schedules(plan: dict, years: int, withdrawal_schedule: list):
    """Process goals and update withdrawal_schedule in place.

    Returns (goal_schedule, flex_goal_schedule) or (None, None) if no goals.
    """
    goals = plan.get('goals')
    if not goals:
        return None, None

    goal_schedule = [0.0] * years       # Essential goals
    flex_goal_schedule = [0.0] * years   # Flexible goals

    for g in goals:
        amount = float(g.get('amount', 0))
        begin = int(g.get('begin', 1))     # 1-based year
        end = int(g.get('end', years))
        priority = g.get('priority', 'Essential')
        # Normalize priority names
        if priority in ('Need', 'need', 'essential'):
            priority = 'Essential'
        elif priority in ('Want', 'want', 'flexible'):
            priority = 'Flexible'

        for y_idx in range(begin - 1, min(end, years)):
            withdrawal_schedule[y_idx] += amount
            if priority == 'Essential':
                goal_schedule[y_idx] += amount
            else:
                flex_goal_schedule[y_idx] += amount

    has_essential = any(g > 0 for g in goal_schedule)
    has_flexible = any(g > 0 for g in flex_goal_schedule)
    return (goal_schedule if has_essential else None,
            flex_goal_schedule if has_flexible else None)


# ── sim_params builder ────────────────────────────────────────

def _build_sim_params(plan: dict, years: int, withdrawal_schedule: list) -> dict:
    """Translate clean plan dict into the flat sim_params dict
    that simulate_withdrawals expects."""
    accts = plan['accounts']
    alloc = plan['allocation']
    ss = plan['social_security']
    pensions = plan['pensions']
    annuities = plan.get('annuities', {})
    tax = plan['tax']
    gr = plan['guardrails']
    ret = plan['returns']
    earned = plan.get('earned_income', {})

    target_stock_pct = alloc['stock_pct'] / 100.0

    params = dict(
        start_age_primary=int(plan['person1']['age']),
        start_age_spouse=int(plan['person2']['age']),
        life_expectancy_primary=int(plan['person1']['life_expectancy']),
        life_expectancy_spouse=int(plan['person2']['life_expectancy']),
        taxable_start=float(accts['taxable']),
        roth_start=float(accts.get('roth', 0)),
        tda_start=float(accts['tda_p1']),
        tda_spouse_start=float(accts.get('tda_p2', 0)),
        taxable_stock_basis_pct=accts.get('taxable_stock_basis_pct', 50) / 100.0,
        taxable_bond_basis_pct=accts.get('taxable_bond_basis_pct', 100) / 100.0,
        target_stock_pct=target_stock_pct,
        withdrawal_schedule=withdrawal_schedule,
        stock_total_return=0.0,
        bond_return=0.0,
        stock_dividend_yield=float(ret.get('stock_dividend_yield', 0.02)),
        stock_turnover=float(ret.get('stock_turnover', 0.10)),
        investment_fee_bps=float(ret.get('investment_fee_bps', 0)),
        # Social Security
        ss_income_annual=float(ss['person1']['benefit']),
        ss_income_spouse_annual=float(ss['person2']['benefit']),
        ss_cola=float(ss.get('cola', 0)),
        ss_start_age_p1=int(ss['person1']['start_age']),
        ss_start_age_p2=int(ss['person2']['start_age']),
        ss_fra_age_p1=int(ss['person1'].get('fra', 67)),
        ss_fra_age_p2=int(ss['person2'].get('fra', 67)),
        # Pensions
        pension_income_annual=float(pensions['person1'].get('income', 0)),
        pension_income_spouse_annual=float(pensions['person2'].get('income', 0)),
        pension_cola_p1=float(pensions['person1'].get('cola', 0)),
        pension_cola_p2=float(pensions['person2'].get('cola', 0)),
        pension_survivor_pct_p1=float(pensions['person1'].get('survivor_pct', 0)),
        pension_survivor_pct_p2=float(pensions['person2'].get('survivor_pct', 0)),
        pp_factors=PP_FACTORS,
        # Annuities
        annuity_income_p1=float(annuities.get('person1', {}).get('income', 0)),
        annuity_income_p2=float(annuities.get('person2', {}).get('income', 0)),
        annuity_cola_p1=float(annuities.get('person1', {}).get('cola', 0)),
        annuity_cola_p2=float(annuities.get('person2', {}).get('cola', 0)),
        annuity_survivor_pct_p1=float(annuities.get('person1', {}).get('survivor_pct', 0)),
        annuity_survivor_pct_p2=float(annuities.get('person2', {}).get('survivor_pct', 0)),
        annuity_start_year=int(annuities.get('start_year', 1)),
        # Other income
        other_income_annual=float(plan.get('other_income', 0)),
        earned_income_annual=float(earned.get('annual', 0)),
        earned_income_years=int(earned.get('years', 0)),
        qcd_annual=float(plan.get('qcd_annual', 0)),
        # Tax
        filing_status=tax['filing_status'],
        use_itemized_deductions=bool(tax.get('use_itemized', False)),
        itemized_deduction_amount=float(tax.get('itemized_deduction', 0)),
        taxes_enabled=bool(tax.get('enabled', True)),
        state_tax_rate=float(tax.get('state_tax_rate', 0.05)),
        state_exempt_retirement=bool(tax.get('state_exempt_retirement', False)),
        tcja_sunset=bool(tax.get('tcja_sunset', False)),
        tcja_sunset_year=int(tax.get('tcja_sunset_year', 2)),
        irmaa_enabled=bool(tax.get('irmaa_enabled', False)),
        # RMDs
        rmd_start_age=int(plan.get('rmd_start_age_p1', 73)),
        rmd_start_age_spouse=int(plan.get('rmd_start_age_p2', 73)),
        # Roth conversions (Phase 1: disabled)
        roth_conversion_amount=0.0,
        roth_conversion_years=0,
        roth_conversion_tax_source='taxable',
        roth_conversion_source='person1',
        roth_conversion_mode='none',
        roth_bracket_fill_rate=0.22,
        # Guardrails
        guardrails_enabled=bool(gr.get('enabled', True)),
        guardrail_lower=float(gr.get('lower', 0.75)),
        guardrail_upper=float(gr.get('upper', 0.90)),
        guardrail_target=float(gr.get('target', 0.85)),
        guardrail_inner_sims=int(gr.get('inner_sims', 200)),
        guardrail_max_spending_pct=float(gr.get('max_spending_pct', -1)),
        # Goals (Phase 1: disabled)
        goal_schedule=None,
        flex_goal_schedule=None,
        flex_goal_min_pct=0.5,
        base_is_essential=False,
        flex_capped_base_schedule=None,
        flex_cap_max_schedule=None,
        goal_taxable_start=0.0,
        goal_tda_start=0.0,
        goal_tda_p1_fraction=0.0,
        goal_liquidation_schedule=None,
        goal_stock_pct=0.5,
        # Inheritance
        inheritance_enabled=False,
        inheritance_year=10,
        inheritance_taxable_amount=0.0,
        inheritance_ira_amount=0.0,
        # Withdrawal preference
        prefer_tda_before_taxable=bool(alloc.get('prefer_tda_before_taxable', False)),
    )

    # Handle inheritance if specified
    inh = plan.get('inheritance')
    if inh:
        params['inheritance_enabled'] = True
        params['inheritance_year'] = int(inh.get('year', 10))
        params['inheritance_taxable_amount'] = float(inh.get('taxable', 0))
        params['inheritance_ira_amount'] = float(inh.get('ira', 0))

    return params


# ── Blended return parameters ─────────────────────────────────

def _compute_blended_params(plan: dict, sim_params: dict):
    """Compute and set blended_mu/sigma on sim_params for guardrails."""
    ret = plan['returns']
    target_stock_pct = sim_params['target_stock_pct']

    if ret['mode'] == 'lognormal':
        stock_mu = float(ret['stock_log_drift'])
        stock_sigma = float(ret['stock_log_volatility'])
        bond_mu = float(ret['bond_log_drift'])
        bond_sigma = float(ret['bond_log_volatility'])
    else:
        mg_df = load_master_global()
        stock_factors = mg_df['LBM 100E'].dropna().values
        stock_log_rets = np.log(stock_factors)
        stock_mu = float(np.mean(stock_log_rets))
        stock_sigma = float(np.std(stock_log_rets))
        bond_log_returns = np.log(1.0 + load_bond_factors())
        bond_mu = float(np.mean(bond_log_returns))
        bond_sigma = float(np.std(bond_log_returns))

    if sim_params['guardrails_enabled']:
        blended_mu = target_stock_pct * stock_mu + (1 - target_stock_pct) * bond_mu
        if ret['mode'] != 'lognormal':
            n = min(len(stock_log_rets), len(bond_log_returns))
            blended_log_rets = target_stock_pct * stock_log_rets[:n] + (1 - target_stock_pct) * bond_log_returns[:n]
            blended_sigma = float(np.std(blended_log_rets))
        else:
            rho = 0.10
            w_s, w_b = target_stock_pct, 1 - target_stock_pct
            blended_sigma = float(np.sqrt(
                (w_s * stock_sigma) ** 2 + (w_b * bond_sigma) ** 2
                + 2 * w_s * w_b * stock_sigma * bond_sigma * rho
            ))
        sim_params['blended_mu'] = blended_mu
        sim_params['blended_sigma'] = blended_sigma
    else:
        sim_params['blended_mu'] = 0.0
        sim_params['blended_sigma'] = 0.0

    return stock_mu, stock_sigma, bond_mu, bond_sigma


# ── Scenario runner ───────────────────────────────────────────

def _parse_scenario_overrides(scenario_def: dict) -> dict:
    """Convert a clean scenario definition into the override dict
    that build_scenario_params expects."""
    ovr = {}
    if 'spend_scale' in scenario_def:
        ovr['spend_scale'] = scenario_def['spend_scale']
    if 'spend_flat' in scenario_def:
        ovr['spend_flat'] = float(scenario_def['spend_flat'])
    if 'spending' in scenario_def and 'annual' in scenario_def['spending']:
        ovr['spend_flat'] = float(scenario_def['spending']['annual'])
    if 'stock_pct' in scenario_def:
        ovr['target_stock_pct'] = scenario_def['stock_pct'] / 100.0
    if 'allocation' in scenario_def and 'stock_pct' in scenario_def['allocation']:
        ovr['target_stock_pct'] = scenario_def['allocation']['stock_pct'] / 100.0
    if 'tda_p1' in scenario_def:
        ovr['tda_delta_p1'] = float(scenario_def['tda_p1']) - float(scenario_def.get('_base_tda_p1', 0))
    if 'tda_delta_p1' in scenario_def:
        ovr['tda_delta_p1'] = float(scenario_def['tda_delta_p1'])
    if 'tda_delta_p2' in scenario_def:
        ovr['tda_delta_p2'] = float(scenario_def['tda_delta_p2'])
    # Annuity overrides
    for key in ['annuity_income_p1', 'annuity_income_p2', 'annuity_cola_p1', 'annuity_cola_p2',
                'annuity_survivor_pct_p1', 'annuity_survivor_pct_p2', 'annuity_start_year']:
        if key in scenario_def:
            ovr[key] = scenario_def[key]
    # Annuity purchase (from taxable)
    if 'annuity_purchase' in scenario_def:
        ovr['annuity_purchase'] = float(scenario_def['annuity_purchase'])
        ovr['annuity_annual_income'] = float(scenario_def.get('annuity_annual_income', 0))
        ovr['annuity_cola'] = float(scenario_def.get('annuity_cola', 0))
        ovr['annuity_person'] = scenario_def.get('annuity_person', 'Person 1')
        ovr['annuity_survivor_pct'] = float(scenario_def.get('annuity_survivor_pct', 0))
        ovr['annuity_start_year'] = int(scenario_def.get('annuity_start_year', 1))
    # Life expectancy overrides
    if 'life_expectancy_primary' in scenario_def:
        ovr['life_expectancy_primary'] = int(scenario_def['life_expectancy_primary'])
    if 'life_expectancy_spouse' in scenario_def:
        ovr['life_expectancy_spouse'] = int(scenario_def['life_expectancy_spouse'])
    return ovr


def _run_single(sim_params: dict, years: int, ret: dict, inheritor_rate: float,
                ending_balance_goal: float, num_runs: int, name: str) -> dict:
    """Run a single simulation and return a scenario summary dict."""
    if ret['mode'] == 'lognormal':
        results, all_yearly_df = run_monte_carlo(
            num_runs=num_runs, years=years,
            inheritor_rate=inheritor_rate,
            taxable_log_drift=float(ret.get('stock_log_drift', 0.09038261)),
            taxable_log_volatility=float(ret.get('stock_log_volatility', 0.20485277)),
            bond_log_drift=float(ret.get('bond_log_drift', 0.0172918)),
            bond_log_volatility=float(ret.get('bond_log_volatility', 0.04796435)),
            **sim_params,
        )
    else:
        windows, _ = get_all_historical_windows(years)
        results = []
        all_yearly = []
        for run_idx, (stock_rets, bond_rets) in enumerate(windows):
            run_pp = compute_run_pp_factors(run_idx, years)
            df_run = simulate_withdrawals(
                years=years, stock_return_series=stock_rets,
                bond_return_series=bond_rets, pp_factors_run=run_pp, **sim_params)
            df_run['total_portfolio'] = df_run['end_taxable_total'] + df_run['end_tda_total'] + df_run['end_roth']
            metrics = compute_summary_metrics(df_run, inheritor_rate)
            results.append(metrics)
            df_run['run'] = run_idx
            all_yearly.append(df_run)
        all_yearly_df = pd.concat(all_yearly, ignore_index=True)

    return compute_scenario_summary(name, results, all_yearly_df,
        inheritor_rate, ending_balance_goal)


def _select_median_run(all_yearly_df: pd.DataFrame) -> pd.DataFrame:
    """Select the simulation run closest to the median ending portfolio."""
    run_ends = all_yearly_df.groupby('run')['total_portfolio'].last()
    median_val = run_ends.median()
    median_run_idx = int((run_ends - median_val).abs().idxmin())
    median_df = all_yearly_df[all_yearly_df['run'] == median_run_idx].copy()
    median_df = median_df.drop(columns=['run', 'total_portfolio'], errors='ignore').reset_index(drop=True)
    return median_df


# ── Main entry point ──────────────────────────────────────────

def run_plan(plan: dict, verbose: bool = False) -> dict:
    """Execute a retirement plan simulation.

    Args:
        plan: Clean plan dict (sparse -- missing keys filled from defaults).
        verbose: Print progress messages.

    Returns:
        dict with keys:
            'success_rate': float (0-1)
            'percentile_rows': list of dicts
            'spending_percentiles': list of dicts
            'pdf_bytes': bytes
            'plan': dict (complete plan with defaults)
            'all_yearly': DataFrame
            'sim_df': DataFrame (median run)
            'multi_scenario_results': list or None
            'num_sims': int
            'pct_non_positive': float
    """
    # Step 1: Apply defaults and validate
    plan = apply_defaults(plan)
    errors = validate_plan(plan)
    if errors:
        raise ValueError(f"Invalid plan: {'; '.join(errors)}")

    # Step 2: Compute horizon
    years = max(1,
        max(plan['person1']['life_expectancy'] - plan['person1']['age'],
            plan['person2']['life_expectancy'] - plan['person2']['age']) + 1)

    # Step 3: Build withdrawal schedule
    withdrawal_schedule = _build_withdrawal_schedule(plan, years)

    # Step 3b: Process goals (adds to withdrawal_schedule, builds goal schedules)
    goal_schedule, flex_goal_schedule = _build_goal_schedules(plan, years, withdrawal_schedule)

    # Step 4: Build sim_params
    sim_params = _build_sim_params(plan, years, withdrawal_schedule)

    # Step 4b: Apply goal schedules
    if goal_schedule:
        sim_params['goal_schedule'] = goal_schedule
    if flex_goal_schedule:
        sim_params['flex_goal_schedule'] = flex_goal_schedule

    # Step 5: Compute blended return params
    stock_mu, stock_sigma, bond_mu, bond_sigma = _compute_blended_params(plan, sim_params)

    ret = plan['returns']
    inheritor_rate = float(plan['tax']['inheritor_marginal_rate'])
    ending_balance_goal = float(plan.get('ending_balance_goal', 0))
    num_runs = int(plan['num_sims'])

    # Step 6: Run simulation(s)
    scenarios_def = plan.get('scenarios')
    multi_scenario_results = None

    if scenarios_def and len(scenarios_def) > 0:
        # Multi-scenario: run each as a full independent plan
        from plan_schema import _deep_merge, describe_plan_diffs
        multi_scenario_results = []
        scenario_input_diffs = {}

        # Run baseline first
        if verbose:
            print(f"  Running scenario 1/{len(scenarios_def) + 1}: Baseline...")
        baseline_summary = _run_single(sim_params, years, ret, inheritor_rate, ending_balance_goal, num_runs, 'Baseline')
        multi_scenario_results.append(baseline_summary)

        # Run each scenario as a full plan with overrides applied at plan level
        for s_num, sc_def in enumerate(scenarios_def):
            s_name = sc_def.get('name', f'Scenario {s_num + 2}')
            if verbose:
                print(f"  Running scenario {s_num + 2}/{len(scenarios_def) + 1}: {s_name}...")

            # Deep-merge scenario overrides into the base plan and rebuild everything
            sc_plan = _deep_merge(plan, sc_def)
            sc_plan.pop('scenarios', None)  # don't recurse
            sc_plan.pop('name', None)

            # Compute input diffs vs baseline
            scenario_input_diffs[s_name] = describe_plan_diffs(plan, sc_plan)

            sc_years = max(1,
                max(sc_plan['person1']['life_expectancy'] - sc_plan['person1']['age'],
                    sc_plan['person2']['life_expectancy'] - sc_plan['person2']['age']) + 1)

            sc_withdrawal = _build_withdrawal_schedule(sc_plan, sc_years)
            sc_goal, sc_flex_goal = _build_goal_schedules(sc_plan, sc_years, sc_withdrawal)
            sc_sim_params = _build_sim_params(sc_plan, sc_years, sc_withdrawal)
            if sc_goal:
                sc_sim_params['goal_schedule'] = sc_goal
            if sc_flex_goal:
                sc_sim_params['flex_goal_schedule'] = sc_flex_goal
            _compute_blended_params(sc_plan, sc_sim_params)

            sc_ret = sc_plan['returns']
            sc_inheritor = float(sc_plan['tax']['inheritor_marginal_rate'])
            sc_end_goal = float(sc_plan.get('ending_balance_goal', 0))
            sc_num_runs = int(sc_plan.get('num_sims', num_runs))

            sc_summary = _run_single(sc_sim_params, sc_years, sc_ret, sc_inheritor, sc_end_goal, sc_num_runs, s_name)
            multi_scenario_results.append(sc_summary)

        # Use baseline for primary results
        baseline = multi_scenario_results[0]
        percentile_rows = baseline['percentile_rows']
        spending_percentiles = baseline['spending_percentiles']
        pct_non_positive = baseline['pct_non_positive']
        primary_all_yearly = baseline['all_yearly_df']
        sim_df = _select_median_run(primary_all_yearly)

    else:
        # Single scenario
        if verbose:
            print("  Running simulation...")

        summary = _run_single(sim_params, years, ret, inheritor_rate, ending_balance_goal, num_runs, 'Baseline')
        percentile_rows = summary['percentile_rows']
        spending_percentiles = summary['spending_percentiles']
        pct_non_positive = summary['pct_non_positive']
        primary_all_yearly = summary['all_yearly_df']
        sim_df = _select_median_run(primary_all_yearly)

    if verbose:
        print("  Generating PDF report...")

    # Step 7: Build session_state for PDF report
    result_data = {
        'percentile_rows': percentile_rows,
        'spending_percentiles': spending_percentiles,
        'pct_non_positive': pct_non_positive,
        'all_yearly': primary_all_yearly,
        'sim_df': sim_df,
        'num_sims': num_runs,
        'multi_scenario_results': multi_scenario_results,
        'scenario_input_diffs': scenario_input_diffs if scenarios_def else None,
    }
    session_state = plan_to_session_state(plan, result_data)

    # Step 8: Generate PDF
    pdf_bytes = generate_report(session_state)

    # Step 9: Return
    return {
        'success_rate': 1.0 - pct_non_positive,
        'percentile_rows': percentile_rows,
        'spending_percentiles': spending_percentiles,
        'pdf_bytes': pdf_bytes,
        'plan': plan,
        'all_yearly': primary_all_yearly,
        'sim_df': sim_df,
        'multi_scenario_results': multi_scenario_results,
        'num_sims': num_runs,
        'pct_non_positive': pct_non_positive,
    }


# ── CLI entry point ───────────────────────────────────────────

if __name__ == '__main__':
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python headless_runner.py <plan.json> [--save] [--verbose]")
        sys.exit(1)

    plan_path = sys.argv[1]
    do_save = '--save' in sys.argv
    verbose = '--verbose' in sys.argv

    with open(plan_path) as f:
        plan = json.load(f)

    if verbose:
        print(f"Running plan: {plan.get('client', 'Unknown')}")

    result = run_plan(plan, verbose=verbose)

    print(f"\nSuccess rate: {result['success_rate']:.1%}")
    m = next((r for r in result['percentile_rows'] if r['percentile'] == 50), None)
    if m:
        print(f"Median ending portfolio: ${m['after_tax_end']:,.0f}")
        print(f"Median total taxes: ${m['total_taxes']:,.0f}")
    s = next((r for r in result['spending_percentiles'] if r['percentile'] == 50), None)
    if s:
        print(f"Median avg annual spending: ${s['avg_annual_after_tax_spending']:,.0f}")

    if do_save:
        from headless_store import save_plan
        paths = save_plan(result['plan'], result['pdf_bytes'])
        print(f"\nSaved JSON: {paths['json_path']}")
        print(f"Saved PDF:  {paths['pdf_path']}")
    else:
        # Save PDF to same directory as input
        import os
        pdf_path = os.path.splitext(plan_path)[0] + '_report.pdf'
        with open(pdf_path, 'wb') as f:
            f.write(result['pdf_bytes'])
        print(f"\nPDF saved: {pdf_path}")
