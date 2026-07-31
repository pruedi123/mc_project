#!/usr/bin/env python3
"""Monthly guardrail check for a client plan.

Ages the plan to the check date, re-draws the guardrail band from actual
account balances, renders a verdict (OK / cut / raise available), appends
to a running history log, and optionally regenerates the client-facing
one-page check-in HTML.

Usage:
  monthly_check.py plan.json --as-of 2026-10 \\
      --balances taxable=180000,tda=380000,tda_spouse=290000,roth=0 \\
      [--plan-start 2026-07] [--report out.html] [--history file.jsonl]

--plan-start (or a "plan_start": "YYYY-MM" key in the plan JSON) anchors
plan aging; whole elapsed years bump both ages, which shrinks the horizon
and moves income start dates naturally. The family (legacy) goal is pinned
at its day-1 value — it never re-defaults from current balances.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, '/Users/paulruedi/Desktop/Updated Web Calcs/mc_project')
from run_full_process import (build_sim_params, run_sim, find_spending,
                              find_decline, find_increase)
from sim_engine import get_all_historical_windows

PANEL_CSV = '/Users/paulruedi/Desktop/Updated Web Calcs/bootstrap-returns/data/global_panel.csv'
BALANCE_KEYS = ('taxable', 'tda', 'tda_spouse', 'roth')


def parse_month(s):
    y, m = s.split('-')
    return int(y), int(m)


def months_between(start, end):
    (y1, m1), (y2, m2) = parse_month(start), parse_month(end)
    return (y2 - y1) * 12 + (m2 - m1)


def age_plan(plan, elapsed_years, balances, pinned_legacy):
    """Return a copy of the plan advanced by whole elapsed years with
    current balances and the day-1 legacy goal pinned explicitly."""
    aged = dict(plan)
    aged['start_age'] = int(plan['start_age']) + elapsed_years
    aged['start_age_spouse'] = int(plan['start_age_spouse']) + elapsed_years
    for key, bal in balances.items():
        aged[f'{key}_start'] = bal
    aged['legacy_target'] = pinned_legacy
    if elapsed_years > 0 and plan.get('add_goals'):
        shifted = []
        for g in plan['add_goals']:
            g2 = dict(g)
            g2['begin'] = max(1, int(g.get('begin', 1)) - elapsed_years)
            g2['end'] = int(g.get('end', 9999)) - elapsed_years
            if g2['end'] >= 1:
                shifted.append(g2)
        aged['add_goals'] = shifted
    if elapsed_years > 0 and len(plan.get('periods', [])) > 1:
        print('  WARNING: multi-period withdrawal schedules are not aged; '
              'review the schedule manually.')
    return aged


def rescale_schedule(params, original_spending, spend_amt):
    """Reproduce find_spending's schedule scaling: essential dollars are a
    fixed floor; only the flexible + base portion moves toward spend_amt."""
    test_params = dict(params)
    goal_schedule = params.get('goal_schedule')
    flex_goal_schedule = params.get('flex_goal_schedule')
    base_schedule = params['withdrawal_schedule']
    n = len(base_schedule)
    if goal_schedule is not None or flex_goal_schedule is not None:
        essential_arr = goal_schedule if goal_schedule is not None else [0.0] * n
        flex_arr = flex_goal_schedule if flex_goal_schedule is not None else [0.0] * n
        essential_avg = float(np.mean(essential_arr))
        scalable_original = original_spending - essential_avg
        s = max(0.0, (spend_amt - essential_avg) / scalable_original) if scalable_original > 0 else 1.0
        new_flex = [f * s for f in flex_arr]
        sched = []
        for i in range(n):
            base_i = base_schedule[i] - essential_arr[i] - flex_arr[i]
            sched.append(essential_arr[i] + new_flex[i] + base_i * s)
        test_params['withdrawal_schedule'] = sched
        if flex_goal_schedule is not None:
            test_params['flex_goal_schedule'] = new_flex
    else:
        scale = spend_amt / original_spending if original_spending > 0 else 1.0
        test_params['withdrawal_schedule'] = [v * scale for v in base_schedule]
    return test_params


def hit_probabilities(move_frac, target_stock_pct, horizons=(3, 6)):
    """Historical probability of an h-month portfolio move at least as far as
    move_frac (negative = decline), from the true monthly return panel."""
    try:
        panel = pd.read_csv(PANEL_CSV)
        pct = target_stock_pct * 100.0 if target_stock_pct <= 1.0 else float(target_stock_pct)
        col = f'{int(round(pct / 10.0) * 10)}E'
        r = panel[col].to_numpy(dtype=float)
    except Exception:
        return None
    w = np.concatenate([[1.0], np.cumprod(1.0 + r)])
    out = {}
    for h in horizons:
        roll = w[h:] / w[:len(w) - h] - 1.0
        out[h] = float(np.mean(roll <= move_frac if move_frac < 0 else roll >= move_frac))
    return out


def usd(v):
    return f'${v:,.0f}'


MONTH_NAMES = ('January', 'February', 'March', 'April', 'May', 'June', 'July',
               'August', 'September', 'October', 'November', 'December')


def next_anniversary_label(plan_start, elapsed_years):
    """Month-name label for the start of the next plan year — the point where
    a year lops off the horizon and the band re-draws structurally."""
    y, m = parse_month(plan_start)
    total = (y * 12 + (m - 1)) + (elapsed_years + 1) * 12
    return f'{MONTH_NAMES[total % 12]} {total // 12}'


def render_report(ctx, out_path):
    """Lean one-page client check-in: verdict, band, what-would-change-it.
    No success percentages — dollar band + hit likelihood only."""
    frac = (ctx['total'] - ctx['lower']) / max(1.0, ctx['upper'] - ctx['lower'])
    marker = 18.0 + 64.0 * min(1.0, max(0.0, frac))
    if ctx['verdict'] == 'OK':
        verdict_html = (
            f"<h1><span class=\"check\">&#10003;</span>You're OK.</h1>"
            f"<p class=\"walkaway\">Your plan is on track. You can spend "
            f"<strong>{usd(ctx['spending'] / 12)} this month</strong>, just as planned &mdash; "
            f"{usd(ctx['spending'])} for the year. Nothing needs to change.</p>")
    elif ctx['verdict'] == 'CUT':
        verdict_html = (
            f"<h1>A change this month.</h1>"
            f"<p class=\"walkaway\">Markets pulled your portfolio below its lower guardrail, so spending "
            f"adjusts to <strong>{usd(ctx['new_spending'] / 12)} a month</strong> "
            f"({usd(ctx['new_spending'])} for the year) "
            f"until the plan recovers. This is the guardrail doing its job &mdash; not the plan failing.</p>")
    else:
        verdict_html = (
            f"<h1>Good news: a raise may be available.</h1>"
            f"<p class=\"walkaway\">Your portfolio is above its upper guardrail. The remaining plan supports "
            f"about <strong>{usd(ctx['new_spending'] / 12)} a month</strong> "
            f"({usd(ctx['new_spending'])} a year) &mdash; we'll confirm the raise after "
            f"checking your family goal is still safe.</p>")
    floor_note = ''
    if ctx['essential'] > 0:
        floor_note = (f"<p class=\"floor-note\"><strong>{usd(ctx['essential'])}</strong> of your spending is "
                      f"protected as essential &mdash; it is never reduced by the guardrails.</p>")
    dp = ctx['decline_probs']
    h2 = ctx['months_to_anniv']
    anniv = ctx['anniv_label']
    h2_desc = f"the {h2} month{'s' if h2 != 1 else ''} before your plan re-sets for its next year in {anniv}"
    if dp:
        if max(dp.values()) > 0:
            drop_freq = (f"A drop that size within three months has happened in {dp[3] * 100:.1f}% of "
                         f"all three-month stretches since 1927. The odds of it happening in "
                         f"{h2_desc}: {dp.get(h2, dp[3]) * 100:.1f}%.")
        else:
            drop_freq = (f"A drop that size has never happened within three months &mdash; or within "
                         f"{h2_desc} &mdash; in 99 years of market history.")
    else:
        drop_freq = ''
    up = ctx['increase_probs']
    up_freq = (f"Historically, {up[3] * 100:.1f}% of three-month stretches gained this much; the odds "
               f"of it happening in {h2_desc}: {up.get(h2, up[3]) * 100:.1f}%." if up else '')
    history_line = ''
    if ctx['n_checks'] > 0:
        changes = ctx['n_changes']
        history_line = (f"<p class=\"history\">Checked {ctx['n_checks']} time{'s' if ctx['n_checks'] != 1 else ''} "
                        f"since the plan started &mdash; spending changed "
                        f"{changes} time{'s' if changes != 1 else ''}.</p>")
    legacy_note = ''
    if ctx['legacy'] > 0:
        legacy_note = (f" Raises above the plan are only granted while the {usd(ctx['legacy'])} "
                       f"family goal still looks safe.")
    html = f"""<title>Monthly Check-In &middot; {ctx['as_of']}</title>
<style>
  :root {{
    --navy: #1F2A33; --ink: #26313A; --ink-2: #55616B; --ink-3: #8A949C;
    --good: #1E7A46; --warn: #B8860B; --bad: #B0281E;
    --sheet: #FFFFFF; --bg: #F2F0EB; --rule: #E3E0D8; --band: #E8F2EC;
    color-scheme: light;
  }}
  * {{ box-sizing: border-box; }}
  body {{ background: var(--bg); color: var(--ink); font: 16px/1.55 Georgia, 'Times New Roman', serif; margin: 0; padding: 24px 16px; }}
  .sheet {{ max-width: 720px; margin: 0 auto; background: var(--sheet); border: 1px solid var(--rule); padding: 40px 44px 28px; }}
  .masthead {{ display: flex; justify-content: space-between; align-items: baseline; border-bottom: 3px solid var(--navy); padding-bottom: 10px; }}
  .firm {{ font-weight: 700; font-size: 15px; letter-spacing: .04em; }}
  .doc-meta {{ font-size: 13px; color: var(--ink-2); }}
  .prepared {{ display: flex; justify-content: space-between; font-size: 13px; color: var(--ink-2); padding: 10px 0 4px; border-bottom: 1px solid var(--rule); }}
  .verdict {{ padding: 26px 0 8px; }}
  .verdict h1 {{ font-size: 30px; margin: 0 0 10px; letter-spacing: -.01em; }}
  .check {{ color: var(--good); margin-right: 10px; }}
  .walkaway {{ font-size: 18px; margin: 0 0 8px; }}
  .floor-note, .history {{ font-size: 14px; color: var(--ink-2); margin: 6px 0 0; }}
  .eyebrow {{ font-size: 12px; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; color: var(--ink-3); margin: 26px 0 8px; }}
  .plain {{ font-size: 15px; margin: 0 0 14px; }}
  .band-track {{ position: relative; height: 14px; background: linear-gradient(90deg, #F6E3E1 0 18%, var(--band) 18% 82%, #FDF3DC 82% 100%); border-radius: 7px; margin: 46px 0 10px; }}
  .gr-tick {{ position: absolute; top: -5px; width: 3px; height: 24px; background: var(--navy); }}
  .you {{ position: absolute; left: {marker:.1f}%; top: 50%; width: 16px; height: 16px; transform: translate(-50%,-50%); background: var(--good); border: 3px solid var(--sheet); border-radius: 50%; box-shadow: 0 0 0 1px rgba(31,42,51,.18); }}
  .you-label {{ position: absolute; left: {marker:.1f}%; top: -38px; transform: translateX(-{'92' if marker > 55 else '8'}%); font-size: 12px; font-weight: 700; color: var(--good); white-space: nowrap; }}
  .you-label .amt {{ display: block; font-size: 14px; }}
  .band-ends {{ display: flex; justify-content: space-between; font-size: 12.5px; color: var(--ink-2); }}
  .band-end .amt {{ display: block; font-weight: 700; font-size: 15px; color: var(--ink); }}
  .band-end.right {{ text-align: right; }}
  .panels {{ display: flex; gap: 14px; margin-top: 6px; flex-wrap: wrap; }}
  .panel {{ flex: 1 1 280px; border: 1px solid var(--rule); padding: 14px 16px; }}
  .panel h3 {{ font-size: 14.5px; margin: 0 0 6px; }}
  .panel p {{ font-size: 13.5px; margin: 0 0 6px; color: var(--ink-2); }}
  .panel .consequence {{ color: var(--ink); }}
  footer {{ font-size: 11.5px; color: var(--ink-3); border-top: 1px solid var(--rule); margin-top: 30px; padding-top: 12px; line-height: 1.6; }}
</style>
<div class="sheet">
  <div class="masthead">
    <div class="firm">Ruedi Wealth Management</div>
    <div class="doc-meta">Monthly Check-In &middot; {ctx['as_of']}</div>
  </div>
  <div class="prepared">
    <span>Prepared for <strong>{ctx['client']}</strong></span>
    <span>Portfolio: {usd(ctx['total'])}</span>
  </div>
  <div class="verdict">
    {verdict_html}
    {floor_note}
    {history_line}
  </div>
  <div class="eyebrow">Your guardrails this month</div>
  <p class="plain">As long as your portfolio stays between the two guardrails, your spending stays the same. These lines are re-drawn at every check.</p>
  <div class="band-track">
    <div class="gr-tick" style="left:18%"></div>
    <div class="gr-tick" style="right:18%"></div>
    <div class="you-label">You are here<span class="amt">{usd(ctx['total'])}</span></div>
    <div class="you"></div>
  </div>
  <div class="band-ends">
    <div class="band-end"><span class="amt">{usd(ctx['lower'])}</span>Lower guardrail</div>
    <div class="band-end right"><span class="amt">{usd(ctx['upper'])}</span>Upper guardrail</div>
  </div>
  <div class="eyebrow">What would change it</div>
  <div class="panels">
    <div class="panel">
      <h3>If your portfolio falls below {usd(ctx['lower'])}</h3>
      <p class="consequence">Spending adjusts to <strong>{usd(ctx['stressed_spending'])}</strong> a year.
        {'Your ' + usd(ctx['essential']) + ' essential amount is untouched.' if ctx['essential'] > 0 else ''}</p>
      <p>{drop_freq}</p>
    </div>
    <div class="panel">
      <h3>If your portfolio rises above {usd(ctx['upper'])}</h3>
      <p class="consequence">A <strong>raise</strong> is considered.{legacy_note}</p>
      <p>{up_freq}</p>
    </div>
  </div>
  <footer>
    All figures are in today's dollars. Guardrails are re-computed each check from actual balances and the
    years remaining in the plan; crossing odds come from 99 years of monthly return history for this
    portfolio mix. This report describes the plan's rules, not a guarantee of results.
    Generated automatically from mc_project &middot; Ruedi Wealth Management &middot; Champaign, Illinois
  </footer>
</div>
"""
    Path(out_path).write_text(html)


def main():
    args = sys.argv[1:]
    plan_path = None
    as_of = None
    plan_start = None
    balances_arg = None
    report_path = None
    history_path = None
    client_label = None
    i = 0
    while i < len(args):
        a = args[i]
        if a == '--as-of':
            as_of = args[i + 1]; i += 2
        elif a == '--plan-start':
            plan_start = args[i + 1]; i += 2
        elif a == '--balances':
            balances_arg = args[i + 1]; i += 2
        elif a == '--report':
            report_path = args[i + 1]; i += 2
        elif a == '--history':
            history_path = args[i + 1]; i += 2
        elif a == '--client':
            client_label = args[i + 1]; i += 2
        else:
            plan_path = a; i += 1
    if not plan_path or not as_of or not balances_arg:
        print(__doc__)
        sys.exit(1)

    with open(plan_path) as f:
        plan = json.load(f)
    plan_start = plan_start or plan.get('plan_start')
    if not plan_start:
        print('ERROR: need --plan-start YYYY-MM (or a "plan_start" key in the plan).')
        sys.exit(1)

    balances = {}
    for part in balances_arg.split(','):
        k, v = part.split('=')
        if k not in BALANCE_KEYS:
            print(f'ERROR: unknown balance key "{k}" (use {", ".join(BALANCE_KEYS)})')
            sys.exit(1)
        balances[k] = float(v)
    for k in BALANCE_KEYS:
        balances.setdefault(k, 0.0)
    total = sum(balances.values())

    elapsed = months_between(plan_start, as_of)
    if elapsed < 0:
        print('ERROR: --as-of is before --plan-start.')
        sys.exit(1)
    elapsed_years, month_of_year = divmod(elapsed, 12)

    # Pin the day-1 family goal before balances change: translating the
    # original plan resolves the default (starting portfolio + reserve) or the
    # explicit key, and the aged plan carries it verbatim so the goal never
    # drifts with the market.
    day1 = build_sim_params(plan)
    pinned_legacy = float(day1[0].get('legacy_target', 0.0))

    aged = age_plan(plan, elapsed_years, balances, pinned_legacy)
    sim_params, sim_years, target_stock_pct, base_spending, use_cols, translated = build_sim_params(aged)
    inheritor_rate = translated['inheritor_marginal_rate']
    spending = float(np.mean(sim_params['withdrawal_schedule']))
    essential = float(np.mean(sim_params['goal_schedule'])) if sim_params.get('goal_schedule') else 0.0
    is_historical = 'Historical' in aged.get('return_mode', 'Historical')

    print(f'Monthly check: {as_of} (plan year {elapsed_years + 1}, month {month_of_year + 1})')
    print(f'Portfolio: {usd(total)}  (' + '  '.join(f'{k}={usd(v)}' for k, v in balances.items() if v > 0) + ')')
    print(f'Plan spending: {usd(spending)}/yr | Family goal (pinned day-1): {usd(pinned_legacy)}'
          if pinned_legacy > 0 else f'Plan spending: {usd(spending)}/yr | No family goal')
    print(f'Horizon remaining: {sim_years} years')

    windows = None
    if is_historical:
        windows, _ = get_all_historical_windows(sim_years, target_stock_pct, use_cols)
        print(f'{len(windows)} historical windows loaded')

    print('\n-- Current position --')
    t0 = time.time()
    _, ayd = run_sim(sim_params, is_historical, windows, sim_years, inheritor_rate, aged)
    run_avg = ayd.groupby('run')['after_tax_spending'].mean()
    sr_now = float((run_avg >= spending - 1.0).mean())
    print(f'Remaining-plan success at current balances: {sr_now * 100:.0f}% ({time.time() - t0:.1f}s)')

    raise_deferred = False
    if sr_now < 0.75:
        verdict = 'CUT'
        print('\n-- Below the lower guardrail: solving the adjusted spending --')
        new_spending, _, _ = find_spending(sim_params, spending, 0.85, spending * 0.9,
                                           is_historical, windows, sim_years, inheritor_rate, aged)
    elif sr_now > 0.95:
        # Year 1 is literal when the plan says so: the household chose to start
        # at the entered amount, so raises wait for the year-2 check. Cuts are
        # never deferred — a breach is acted on whenever it happens.
        if elapsed_years == 0 and aged.get('guardrail_year1_literal', False):
            verdict = 'OK'
            raise_deferred = True
            new_spending = spending
        else:
            verdict = 'RAISE'
            print('\n-- Above the upper guardrail: sizing the available raise --')
            new_spending, _, _ = find_spending(sim_params, spending, 0.85, spending * 1.15,
                                               is_historical, windows, sim_years, inheritor_rate, aged)
    else:
        verdict = 'OK'
        new_spending = spending

    # The published band always describes the GO-FORWARD plan: on a change
    # month the rails are re-drawn at the adjusted spending, so the client
    # never sees rails their new spending has already moved away from.
    if verdict == 'OK':
        band_params, band_target = sim_params, spending
        print('\n-- Re-drawing the band --')
    else:
        band_params = rescale_schedule(sim_params, spending, new_spending)
        band_target = float(np.mean(band_params['withdrawal_schedule']))
        print(f'\n-- Re-drawing the band at the adjusted spending ({usd(band_target)}) --')
    decline_pct = find_decline(band_params, band_target, 0.75, is_historical, windows,
                               sim_years, inheritor_rate, aged, guess_decline=20.0)
    increase_pct = find_increase(band_params, band_target, 0.95, is_historical, windows,
                                 sim_years, inheritor_rate, aged, guess_increase=10.0)
    lower = total * (1.0 - decline_pct / 100.0)
    upper = total * (1.0 + increase_pct / 100.0)

    print('\n-- Stressed spending at the lower rail --')
    stressed_params = dict(band_params)
    for k in ('taxable_start', 'tda_start', 'tda_spouse_start', 'roth_start'):
        stressed_params[k] = band_params.get(k, 0.0) * (1.0 - decline_pct / 100.0)
    stressed_spending, _, _ = find_spending(stressed_params, band_target, 0.85, band_target * 0.85,
                                            is_historical, windows, sim_years, inheritor_rate, aged)

    # Horizons: 3 months as the steady near-term gauge, plus the months left
    # until the plan's next anniversary — the odds a change lands before this
    # plan year rolls, which naturally shrink as the year progresses.
    months_to_anniv = 12 - month_of_year
    anniv_label = next_anniversary_label(plan_start, elapsed_years)
    horizons = (3, months_to_anniv) if months_to_anniv != 3 else (3,)
    decline_probs = hit_probabilities(-(decline_pct / 100.0), target_stock_pct, horizons)
    increase_probs = hit_probabilities(increase_pct / 100.0, target_stock_pct, horizons)

    history_path = history_path or str(Path(plan_path).with_suffix('')) + '_checks.jsonl'
    prior = []
    if Path(history_path).exists():
        with open(history_path) as f:
            prior = [json.loads(line) for line in f if line.strip()]
    n_checks = len(prior) + 1
    n_changes = sum(1 for r in prior if r.get('verdict') != 'OK') + (0 if verdict == 'OK' else 1)
    with open(history_path, 'a') as f:
        f.write(json.dumps({
            'as_of': as_of, 'balances': balances, 'total': total,
            'lower': round(lower), 'upper': round(upper),
            'verdict': verdict, 'spending': round(spending),
            'new_spending': round(new_spending),
        }) + '\n')

    print(f"""
{'=' * 56}
  MONTHLY CHECK  ·  {as_of}
{'=' * 56}
  Portfolio                {usd(total)}
  Lower guardrail          {usd(lower)}  (-{decline_pct:.1f}%)
  Upper guardrail          {usd(upper)}  (+{increase_pct:.1f}%)
  Verdict                  {verdict}{' (raise deferred: year 1 starts at plan)' if raise_deferred else ''}
  Spending                 {usd(new_spending)}/yr  ({usd(new_spending / 12)}/month)
  If it falls to the rail  {usd(stressed_spending)}/yr""")
    if decline_probs:
        def _fmt_probs(probs):
            parts = [f'{probs[3] * 100:.1f}% (3-mo)']
            if months_to_anniv != 3:
                parts.append(f'{probs[months_to_anniv] * 100:.1f}% (by {anniv_label}, {months_to_anniv} mo)')
            return ' | '.join(parts)
        print(f'  Odds of hitting lower   {_fmt_probs(decline_probs)}')
        print(f'  Odds of hitting upper   {_fmt_probs(increase_probs)}')
    print(f'  History                  {n_checks} checks, {n_changes} spending changes')
    if verdict != 'OK':
        print('  Band shown is drawn at the adjusted (go-forward) spending.')
    print(f'{"=" * 56}')
    print(f'History appended: {history_path}')

    if report_path:
        render_report({
            'as_of': as_of, 'client': client_label or Path(plan_path).stem,
            'total': total, 'lower': lower, 'upper': upper,
            'verdict': verdict, 'spending': spending, 'new_spending': new_spending,
            'stressed_spending': stressed_spending, 'essential': essential,
            'legacy': pinned_legacy, 'decline_probs': decline_probs,
            'increase_probs': increase_probs,
            'months_to_anniv': months_to_anniv, 'anniv_label': anniv_label,
            'n_checks': n_checks, 'n_changes': n_changes,
        }, report_path)
        print(f'Report written: {report_path}')


if __name__ == '__main__':
    main()
