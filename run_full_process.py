#!/usr/bin/env python3
"""Run Full Process for a client plan without Streamlit."""
import json, sys, time, math
import numpy as np
import pandas as pd

# Add project to path
sys.path.insert(0, '/Users/paulruedi/Desktop/Updated Web Calcs/mc_project')
from sim_engine import (
    get_all_historical_windows, run_historical_parallel,
    store_distribution_results, load_portfolio_factors, should_use_actual_allocation_columns,
    PP_FACTORS, compute_run_pp_factors, simulate_withdrawals, set_mc_seed,
)
from spending_safety import essential_reserve_analysis, solve_essential_floor_for_reserve
from plan_to_sim_params import build_sim_params_from_plan

# Pin the guardrail inner-MC so repeated runs on identical inputs give
# identical results (same canonical seed used by main.py and the fitting
# room app) — without this, every run_full_process.py invocation draws from
# an unseeded RNG and can return a different answer for the same plan.
set_mc_seed(20260705)


def build_sim_params(plan):
    """Translate a saved plan JSON into sim_params via the faithful
    Streamlit-equivalent translator (plan_to_sim_params) so headless numbers
    match on-screen numbers: multi-period spending, add_goals, Roth
    conversions, filing status, and UI-matching defaults all flow through.
    Raises NotImplementedError (loudly) for pension-buyout plans and
    separately-funded goals rather than silently dropping them."""
    tr = build_sim_params_from_plan(plan)
    return (tr['sim_params'], tr['sim_years'], tr['target_stock_pct'],
            tr['base_spending'], tr['use_actual_allocation_columns'], tr)


def run_sim(params, is_historical, windows, sim_years, inheritor_rate, plan):
    if is_historical:
        return run_historical_parallel(windows, sim_years, inheritor_rate, params)
    else:
        from sim_engine import run_monte_carlo
        # Pass return_mode through so Bootstrap plans actually use the
        # precomputed bootstrap grids instead of silently running lognormal.
        return run_monte_carlo(
            num_runs=int(plan.get('monte_carlo_runs', 1000)),
            years=sim_years, inheritor_rate=inheritor_rate,
            taxable_log_drift=plan.get('taxable_log_drift', 0.09038261),
            taxable_log_volatility=plan.get('taxable_log_volatility', 0.20485277),
            bond_log_drift=plan.get('bond_log_drift', 0.0172918),
            bond_log_volatility=plan.get('bond_log_volatility', 0.04796435),
            return_mode=plan.get('return_mode', 'Simulated (lognormal)'),
            **params)


def find_spending(params, original_spending, target_pct, guess, is_historical, windows, sim_years, inheritor_rate, plan, tol=1000.0, max_iter=15):
    goal_schedule = params.get('goal_schedule')
    flex_goal_schedule = params.get('flex_goal_schedule')

    def _run(spend_amt):
        test_params = dict(params)
        base_schedule = test_params['withdrawal_schedule']
        if goal_schedule is not None or flex_goal_schedule is not None:
            # Essential dollars are a fixed floor (never scaled, matching how
            # compute_prioritized_target treats them at simulate time) — only
            # the flexible + base portion moves toward spend_amt. Scaling the
            # combined schedule uniformly here would silently drive the
            # implied base below zero once spend_amt dips under the fixed
            # essential+flex total, breaking the guardrail's cut-priority math.
            n = len(base_schedule)
            essential_arr = goal_schedule if goal_schedule is not None else [0.0] * n
            flex_arr = flex_goal_schedule if flex_goal_schedule is not None else [0.0] * n
            essential_avg = float(np.mean(essential_arr))
            scalable_original = original_spending - essential_avg
            scalable_scale = max(0.0, (spend_amt - essential_avg) / scalable_original) if scalable_original > 0 else 1.0
            new_flex = [f * scalable_scale for f in flex_arr]
            test_schedule = []
            for i in range(n):
                base_i = base_schedule[i] - essential_arr[i] - flex_arr[i]
                test_schedule.append(essential_arr[i] + new_flex[i] + base_i * scalable_scale)
            test_params['withdrawal_schedule'] = test_schedule
            if flex_goal_schedule is not None:
                test_params['flex_goal_schedule'] = new_flex
        else:
            scale = spend_amt / original_spending if original_spending > 0 else 1.0
            test_schedule = [v * scale for v in base_schedule]
            test_params['withdrawal_schedule'] = test_schedule
        # Grade against the scaled schedule's planned AVERAGE (== spend_amt for
        # flat plans) so go-go/slow-go schedules aren't judged on period 1 alone.
        planned_avg = float(np.mean(test_schedule)) if test_schedule else spend_amt
        _, ayd = run_sim(test_params, is_historical, windows, sim_years, inheritor_rate, plan)
        run_avg = ayd.groupby('run')['after_tax_spending'].mean()
        # $1 tolerance: the withdrawal solver targets net spending +/- $0.50
        return float((run_avg >= planned_avg - 1.0).mean()), float(run_avg.min())

    rate, _ = _run(guess)
    # No early-return here even when the first guess already lands within
    # tolerance of target_pct — a guess this close to a HIGH target (e.g.
    # 100%) may still be far below the true achievable ceiling (a modest
    # guess like the UI default can trivially clear a loose success bar),
    # so we must always bracket-and-bisect to find the actual boundary.
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
        return float((run_avg >= spending_target - 1.0).mean())

    rate = _run(guess_decline)
    # No early-return on the first guess — see find_spending for why this
    # shortcut is unsafe (a guess already near target_rate can still be far
    # from the true boundary the bracket-and-bisect search would find).
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


def find_increase(params, spending_target, target_rate, is_historical, windows, sim_years, inheritor_rate, plan, guess_increase=20.0, tol=1.0, max_iter=15):
    balance_keys = ['taxable_start', 'tda_start', 'tda_spouse_start', 'roth_start',
        'goal_taxable_start', 'goal_tda_start']

    def _run(pct_increase):
        factor = 1.0 + pct_increase / 100.0
        test_params = dict(params)
        for k in balance_keys:
            if k in test_params:
                test_params[k] = params[k] * factor
        _, ayd = run_sim(test_params, is_historical, windows, sim_years, inheritor_rate, plan)
        run_avg = ayd.groupby('run')['after_tax_spending'].mean()
        return float((run_avg >= spending_target - 1.0).mean())

    rate = _run(guess_increase)
    # No early-return on the first guess — see find_spending for why this
    # shortcut is unsafe (a guess already near target_rate can still be far
    # from the true boundary the bracket-and-bisect search would find).
    if rate < target_rate:
        lo, hi = guess_increase, min(guess_increase * 1.5, 200.0)
        for _ in range(5):
            r = _run(hi)
            if r >= target_rate:
                break
            hi = min(lo + (hi - lo) * 1.5, 200.0)
            if hi >= 200.0:
                break
    else:
        hi, lo = guess_increase, max(guess_increase * 0.5, 0.0)
        for _ in range(5):
            r = _run(lo)
            if r < target_rate:
                break
            lo = max(hi - (hi - lo) * 1.5, 0.0)

    for i in range(max_iter):
        mid = (lo + hi) / 2.0
        print(f"  iter {i+1}: trying {mid:.1f}% increase ...", end=' ', flush=True)
        r = _run(mid)
        print(f"-> {r*100:.0f}%")
        if r < target_rate:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return round((lo + hi) / 2.0, 1)


if __name__ == '__main__':
    # ── Load plan JSON ──
    # Usage: python run_full_process.py [plan.json] [--target-pct 0.90] [--shortfall-pct 80]
    #        [--reserve-amount 50000] [--withdrawal-rate 0.04]
    target_pct_override = 0.90
    shortfall_pct = 80.0
    redline_override = None
    reserve_amount_override = None
    withdrawal_rate_override = None
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
        elif a == '--redline':
            redline_override = float(sys.argv[i + 1])
            skip_next = True
        elif a == '--reserve-amount':
            reserve_amount_override = float(sys.argv[i + 1])
            skip_next = True
        elif a == '--withdrawal-rate':
            withdrawal_rate_override = float(sys.argv[i + 1])
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

    # Resolve the reserve override before translation so the default legacy
    # goal (starting portfolio + stated cash reserve) can see it.
    if reserve_amount_override is not None:
        plan['reserve_amount'] = reserve_amount_override

    sim_params, sim_years, target_stock_pct, base_spending, use_actual_allocation_columns, translated = build_sim_params(plan)
    _legacy = float(sim_params.get('legacy_target', 0.0))
    if 'legacy_target' in plan:
        print(f"Legacy goal: ${_legacy:,.0f} (from plan)" if _legacy > 0
              else "Legacy goal: none (plan sets 0)")
    else:
        print(f"Legacy goal: ${_legacy:,.0f} (default: starting portfolio + reserve)")
    inheritor_rate = translated['inheritor_marginal_rate']
    ending_balance_goal = translated['ending_balance_goal']
    is_historical = 'Historical' in plan.get('return_mode', 'Historical')
    windows = None

    # ── Pre-load historical windows ──
    t0 = time.time()
    if is_historical:
        print(f"\nLoading historical windows for {sim_years} years...")
        windows, window_start_dates = get_all_historical_windows(
            sim_years, target_stock_pct, use_actual_allocation_columns)
        print(f"  {len(windows)} windows loaded in {time.time()-t0:.1f}s")

    # ── Phase 1: Auto-calculate spending ──
    print("\n=== PHASE 1: Auto-calculate spending ===")
    portfolio_total = plan['taxable_start'] + plan['tda_start'] + plan['tda_spouse_start'] + plan['roth_start']
    four_pct = portfolio_total * 0.04

    def _worst_draw_avg_pp(horizon, cola=0.0):
        """Average real value multiplier for a $1/yr nominal income stream over
        the horizon, under the single worst rolling historical CPI window.
        A fixed pension counted at face value overstates what it funds — under
        the worst inflation draw a no-COLA dollar averages ~42 cents of
        purchasing power across 30 years. SS is excluded from this (ss_cola=0
        already means inflation-tracking in this engine's real-dollar terms)."""
        from sim_engine import CPI_MO_FACTORS, compute_run_pp_factors
        n_starts = len(CPI_MO_FACTORS) - horizon * 12 + 1
        worst = None
        for s in range(n_starts):
            pp = compute_run_pp_factors(s, horizon)
            avg = sum(((1 + cola) ** y) * pp[y] for y in range(horizon)) / horizon
            if worst is None or avg < worst:
                worst = avg
        return worst if worst is not None else 1.0

    ss_sum = plan.get('ss_income', 0) + plan.get('ss_income_spouse', 0)
    fixed_sum = (plan.get('pension_income', 0) + plan.get('pension_income_spouse', 0) +
        plan.get('other_income', 0) +
        sim_params.get('annuity_income_p1', 0.0) + sim_params.get('annuity_income_p2', 0.0))
    if withdrawal_rate_override is not None and fixed_sum > 0:
        pp_mult = _worst_draw_avg_pp(sim_years)
        income_sum = ss_sum + fixed_sum * pp_mult
        print(f"  Fixed income streams discounted to worst-CPI-draw average value: "
              f"${fixed_sum:,.0f} x {pp_mult:.2f} = ${fixed_sum * pp_mult:,.0f}")
    else:
        income_sum = ss_sum + fixed_sum
    if withdrawal_rate_override is not None:
        # Classic withdrawal-rate goal (4%, 4.5%, etc.) instead of a solved
        # number — replaces any entered/multi-period schedule with one flat
        # rate-based figure, since the point is to stress-test the plain
        # textbook heuristic against history, not layer it onto other goals.
        rate_dollars = portfolio_total * withdrawal_rate_override
        auto_spending = round((rate_dollars + income_sum) / 1000) * 1000
        sim_params['withdrawal_schedule'] = [auto_spending] * len(sim_params['withdrawal_schedule'])
        print(f"  {withdrawal_rate_override*100:.1f}% of ${portfolio_total:,.0f} = ${rate_dollars:,.0f} + "
              f"${income_sum:,.0f} income = ${auto_spending:,.0f}/yr")
    elif base_spending <= 0:
        auto_spending = round((four_pct + income_sum) / 1000) * 1000
        # Layer auto-spending on top of the translated schedule (which may
        # already carry add_goal amounts) instead of replacing it.
        sim_params['withdrawal_schedule'] = [auto_spending + v for v in sim_params['withdrawal_schedule']]
        print(f"  4% of ${portfolio_total:,.0f} = ${four_pct:,.0f} + ${income_sum:,.0f} income = ${auto_spending:,.0f}/yr")
    else:
        auto_spending = base_spending
        print(f"  Using entered spending: ${auto_spending:,.0f}/yr")
    original_spending = auto_spending
    ideal_spending = float(plan.get('ideal_spending') or plan.get('spending', {}).get('ideal') or original_spending)
    acceptable_spending = float(plan.get('acceptable_spending') or plan.get('spending', {}).get('acceptable') or ideal_spending * 0.90)
    redline_spending = float(redline_override or plan.get('redline_spending') or plan.get('essential_spending') or plan.get('spending', {}).get('redline') or ideal_spending * 0.80)
    # 0/absent = skip the reserve-protected floor solve entirely (opt-in,
    # since most plans don't carry a stated cash-reserve bucket).
    reserve_amount = float(reserve_amount_override if reserve_amount_override is not None else (plan.get('reserve_amount') or 0.0))

    # ── Phase 2: Initial simulation ──
    print("\n=== PHASE 2: Initial simulation ===")
    t1 = time.time()
    planned_avg_initial = float(np.mean(sim_params['withdrawal_schedule']))
    results, all_yearly = run_sim(sim_params, is_historical, windows, sim_years, inheritor_rate, plan)
    dist = store_distribution_results(results, all_yearly, 'historical_dist' if is_historical else 'simulated',
        ending_balance_goal, spending_target=planned_avg_initial, essential_spending=redline_spending,
        acceptable_spending=acceptable_spending, redline_spending=redline_spending)
    initial_success = dist.get('mc_spending_success_rate', 0)
    print(f"  Spending ${original_spending:,.0f}/yr -> {initial_success*100:.0f}% ideal success ({time.time()-t1:.1f}s)")

    # ── Phase 3: Spending Finder (skipped in withdrawal-rate mode) ──
    if withdrawal_rate_override is not None:
        print(f"\n=== PHASE 3: Skipped (using the {withdrawal_rate_override*100:.1f}% withdrawal-rate goal as-is) ===")
        run_avg_initial = all_yearly.groupby('run')['after_tax_spending'].mean()
        found_spending = auto_spending
        found_rate = initial_success
        found_min = float(run_avg_initial.min())
        print(f"  >> ${found_spending:,.0f}/yr entered | Historical worst 30-yr avg: ${found_min:,.0f} "
              f"| Median 30-yr avg: ${run_avg_initial.median():,.0f}")
    else:
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
    found_planned_avg = float(np.mean(rerun_params['withdrawal_schedule']))
    results2, all_yearly2 = run_sim(rerun_params, is_historical, windows, sim_years, inheritor_rate, plan)
    dist2 = store_distribution_results(results2, all_yearly2, 'historical_dist' if is_historical else 'simulated',
        ending_balance_goal, spending_target=found_planned_avg, essential_spending=redline_spending,
        acceptable_spending=acceptable_spending, redline_spending=redline_spending)
    print(f"  ${found_spending:,.0f}/yr -> {dist2.get('mc_spending_success_rate',0)*100:.0f}% ideal, "
          f"{dist2.get('mc_essential_success_rate',0)*100:.0f}% essential ({time.time()-t1:.1f}s)")
    reserve_analysis = essential_reserve_analysis(all_yearly2, found_spending, redline_spending)
    before_ess = reserve_analysis['before_reserve']['essential']
    after_ess = reserve_analysis['after_reserve']['essential']
    print(f"  Essential reserve needed: ${reserve_analysis['reserve_needed']:,.0f} "
          f"({before_ess['runs_below']} runs below before, {after_ess['runs_below']} after)")

    withdrawal_rate_stats = None
    if withdrawal_rate_override is not None:
        run_avg2 = all_yearly2.groupby('run')['after_tax_spending'].mean()
        withdrawal_rate_stats = {
            'lowest_single_year': float(all_yearly2['after_tax_spending'].min()),
            'lowest_30yr_avg': float(run_avg2.min()),
            'median_30yr_avg': float(run_avg2.median()),
        }
        print(f"\n  Lowest single year (any window):   ${withdrawal_rate_stats['lowest_single_year']:,.0f}")
        print(f"  Lowest 30-yr average (worst window): ${withdrawal_rate_stats['lowest_30yr_avg']:,.0f}")
        print(f"  Median 30-yr average (50th pctile):   ${withdrawal_rate_stats['median_30yr_avg']:,.0f}")

    # ── Reserve Protected Floor Solver (opt-in via --reserve-amount or plan['reserve_amount']) ──
    # Same solve_essential_floor_for_reserve used by main.py's interactive
    # "Reserve Protected Floor Solver" and the fitting-room site's export —
    # given a stated cash reserve held OUTSIDE the portfolio, finds the
    # highest essential floor that reserve can fully protect across every
    # historical run, rather than essential_reserve_analysis's reverse
    # direction (fixed floor -> reserve needed).
    reserve_floor_result = None
    if reserve_amount > 0:
        print(f"\n=== Reserve Protected Floor Solver (${reserve_amount:,.0f} reserve) ===")
        t1 = time.time()
        reserve_floor_result = solve_essential_floor_for_reserve(
            all_yearly2, reserve_amount=reserve_amount, max_floor=found_spending)
        rf_analysis = reserve_floor_result['analysis']
        print(f"  >> Protected floor: ${reserve_floor_result['essential_floor']:,.0f}/yr "
              f"(reserve needed: ${reserve_floor_result['reserve_needed']:,.0f}, "
              f"unused: ${reserve_floor_result['unused_reserve']:,.0f}) ({time.time()-t1:.1f}s)")
        print(f"     Minimum year after reserve: ${rf_analysis['after_reserve']['minimum_year']:,.0f}, "
              f"max breach-years in one run: {rf_analysis['max_breach_years_in_one_run']}")

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
            rerun_schedule = rerun_params['withdrawal_schedule']
            for _, r in df_1929.iterrows():
                guardrail_target = r['net_spending_target']
                actual = r['after_tax_spending']
                input_target = rerun_schedule[int(r['year']) - 1]
                print(f"    {int(r['year']):>3}  {int(r['age_p1']):>5}  {int(r['age_p2']):>5}  "
                      f"${input_target:>11,.0f}  ${guardrail_target:>12,.0f}  ${actual:>11,.0f}  ${r['total_portfolio']:>11,.0f}")
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
    decline_pct = find_decline(rerun_params, found_planned_avg, target_rate=0.75,
        is_historical=is_historical, windows=windows, sim_years=sim_years,
        inheritor_rate=inheritor_rate, plan=plan, guess_decline=20.0)
    balance_keys = ['taxable_start', 'tda_start', 'tda_spouse_start', 'roth_start', 'goal_taxable_start', 'goal_tda_start']
    orig_total = sum(rerun_params.get(k, 0.0) for k in balance_keys)
    decline_factor = 1.0 - decline_pct / 100.0
    reduced_total = orig_total * decline_factor
    dollar_drop = orig_total - reduced_total

    # Historical probability of this decline (3- and 6-month horizons) from the
    # monthly wealth path implied by the 12-month rolling allocation factors.
    allocation_f = load_portfolio_factors(target_stock_pct, use_actual_allocation_columns)
    GUARDRAIL_HORIZONS = (3, 6)

    def _extract_monthly(factors_12mo):
        nn = len(factors_12mo)
        monthly = np.zeros(nn + 11)
        seed = factors_12mo[0] ** (1.0 / 12.0)
        for k in range(12):
            monthly[k] = seed
        for t in range(nn - 1):
            monthly[t + 12] = monthly[t] * factors_12mo[t + 1] / factors_12mo[t]
        return monthly

    # Prefer the true monthly return panel (bootstrap-returns) — the 12-month
    # rolling workbook factors cannot recover short-horizon moves, and the
    # flat-seed reconstruction badly overstates 3/6-month decline frequency.
    _PANEL_CSV = '/Users/paulruedi/Desktop/Updated Web Calcs/bootstrap-returns/data/global_panel.csv'
    w = None
    monthly_source = 'reconstructed from 12-mo factors'
    try:
        _panel = pd.read_csv(_PANEL_CSV)
        _pct = target_stock_pct * 100.0 if target_stock_pct <= 1.0 else float(target_stock_pct)
        _col = f"{int(round(_pct / 10.0) * 10)}E"
        if _col in _panel.columns:
            _r = _panel[_col].to_numpy(dtype=float)
            w = np.concatenate([[1.0], np.cumprod(1.0 + _r)])
            monthly_source = f'true monthly panel, {_col}'
    except Exception:
        pass
    if w is None:
        allocation_monthly = _extract_monthly(allocation_f)
        wealth = np.concatenate([[1.0], np.cumprod(allocation_monthly)])
        w = wealth[12:]

    def _norm_cdf(x):
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
    annual_log_rets = np.log(allocation_f)
    mu = float(np.mean(annual_log_rets))
    sigma = float(np.std(annual_log_rets))

    def _move_probs(move_frac):
        """{horizon_months: (historical, lognormal)} probability of a rolling
        return <= move_frac (declines, move_frac < 0) or >= it (gains).
        Historical = share of all overlapping monthly-start windows."""
        out = {}
        for h in GUARDRAIL_HORIZONS:
            rolling = w[h:] / w[:len(w) - h] - 1.0
            if len(rolling) == 0:
                out[h] = (0.0, 0.0)
                continue
            emp = float(np.mean(rolling <= move_frac if move_frac < 0 else rolling >= move_frac))
            frac = h / 12.0
            mu_h = mu * frac; sigma_h = sigma * np.sqrt(frac)
            z = (np.log(1.0 + move_frac) - mu_h) / sigma_h
            out[h] = (emp, _norm_cdf(z) if move_frac < 0 else 1.0 - _norm_cdf(z))
        return out

    def _prob_line(probs):
        hist = " / ".join(f"{probs[h][0]*100:.1f}% ({h}-mo)" for h in GUARDRAIL_HORIZONS)
        sim = " / ".join(f"{probs[h][1]*100:.1f}% ({h}-mo)" for h in GUARDRAIL_HORIZONS)
        return f"Historical prob ({monthly_source}): {hist} | Simulated: {sim}"

    decline_probs = _move_probs(-(decline_pct / 100.0))
    empirical_prob, simulated_prob = decline_probs[3]

    print(f"  >> {decline_pct:.1f}% decline (${orig_total:,.0f} -> ${reduced_total:,.0f}, -${dollar_drop:,.0f})")
    print(f"     {_prob_line(decline_probs)}")
    print(f"     ({time.time()-t1:.1f}s)")

    # ── Phase 5b: Balance Increase Finder (95% target) ──
    print("\n=== PHASE 5b: Find balance increase to reach 95% success ===")
    t1 = time.time()
    increase_pct = find_increase(rerun_params, found_planned_avg, target_rate=0.95,
        is_historical=is_historical, windows=windows, sim_years=sim_years,
        inheritor_rate=inheritor_rate, plan=plan, guess_increase=20.0)
    increase_factor = 1.0 + increase_pct / 100.0
    increased_total = orig_total * increase_factor
    dollar_gain = increased_total - orig_total

    # Historical probability of this gain (same monthly wealth path as decline)
    increase_probs = _move_probs(increase_pct / 100.0)
    empirical_prob_up, simulated_prob_up = increase_probs[3]

    print(f"  >> {increase_pct:.1f}% increase (${orig_total:,.0f} -> ${increased_total:,.0f}, +${dollar_gain:,.0f})")
    print(f"     {_prob_line(increase_probs)}")
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
    reserve_floor_line = ""
    if reserve_floor_result is not None:
        reserve_floor_line = (
            f"  Reserve-protected floor      ${reserve_floor_result['essential_floor']:,.0f}/yr "
            f"(${reserve_amount:,.0f} reserve)\n"
        )
    if withdrawal_rate_override is not None:
        spending_label = f"Spending ({withdrawal_rate_override*100:.1f}% rule)"
        floor_label = "Lowest 30-yr average"
        floor_value = withdrawal_rate_stats['lowest_30yr_avg']
        withdrawal_rate_line = (
            f"  Lowest single year           ${withdrawal_rate_stats['lowest_single_year']:,.0f}\n"
            f"  Median 30-yr average         ${withdrawal_rate_stats['median_30yr_avg']:,.0f}\n"
        )
    else:
        spending_label = "Optimal spending (90%)"
        floor_label = "Essential floor (100%)"
        floor_value = found_min
        withdrawal_rate_line = ""
    print(f"""
{'='*60}
           FULL PROCESS SUMMARY
{'='*60}
  Starting portfolio          ${orig_total:,.0f}
  {spending_label + ' ' * max(0, 28 - len(spending_label))}${found_spending:,.0f}/yr
  Essential reserve needed    ${reserve_analysis['reserve_needed']:,.0f}
  {floor_label + ' ' * max(0, 28 - len(floor_label))}${floor_value:,.0f}/yr
{withdrawal_rate_line}{reserve_floor_line}  Decline to reach 75%        {decline_pct:.1f}% (-${dollar_drop:,.0f})
  Decline prob (hist)         {decline_probs[3][0]*100:.1f}% (3-mo) | {decline_probs[6][0]*100:.1f}% (6-mo)
  Increase to reach 95%       {increase_pct:.1f}% (+${dollar_gain:,.0f})
  Increase prob (hist)        {increase_probs[3][0]*100:.1f}% (3-mo) | {increase_probs[6][0]*100:.1f}% (6-mo)
  Stressed portfolio           ${reduced_total:,.0f}
  Stressed spending (85%)     ${stressed_spending:,.0f}/yr
  Spending change if decline  {'+'if spending_delta>=0 else '-'}${abs(spending_delta):,.0f}/yr ({spending_delta/found_spending*100:+.1f}%)
{'='*60}
  Total time: {elapsed:.1f}s
""")
