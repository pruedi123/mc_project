"""Save/load plan JSON and PDF reports to ~/RWM/Headless Plans/."""

import os
import json
from datetime import datetime

HEADLESS_DIR = os.path.expanduser('~/RWM/Current Client Plans')


def _client_dir(client_name: str) -> str:
    return os.path.join(HEADLESS_DIR, client_name)


def _plan_stem(client_name: str) -> str:
    """Convert 'Jones, Bob & Mary' -> 'jones_bob_mary'."""
    stem = client_name.lower()
    for ch in [',', '&', '  ']:
        stem = stem.replace(ch, ' ')
    return '_'.join(stem.split())


def _plan_to_streamlit_format(plan: dict) -> dict:
    """Convert clean plan dict to the flat Streamlit session_state format
    so plans are loadable by the Streamlit app."""
    ss = plan.get('social_security', {})
    pensions = plan.get('pensions', {})
    tax = plan.get('tax', {})
    ret = plan.get('returns', {})
    gr = plan.get('guardrails', {})
    alloc = plan.get('allocation', {})
    accts = plan.get('accounts', {})
    earned = plan.get('earned_income', {})
    spending = plan.get('spending', {})

    _RETURN_MODE_MAP = {
        'historical': 'Historical (master_global_factors)',
        'lognormal': 'Simulated (lognormal)',
    }
    _FILING_MAP = {
        'mfj': 'Married Filing Jointly',
        'single': 'Single',
    }

    data = {
        'person1_label': plan['person1'].get('label', ''),
        'person2_label': plan['person2'].get('label', ''),
        'start_age': plan['person1']['age'],
        'start_age_spouse': plan['person2']['age'],
        'life_expectancy_primary': plan['person1']['life_expectancy'],
        'life_expectancy_spouse': plan['person2']['life_expectancy'],
        'taxable_start': float(accts.get('taxable', 300000)),
        'taxable_stock_basis_pct': float(accts.get('taxable_stock_basis_pct', 50)),
        'taxable_bond_basis_pct': float(accts.get('taxable_bond_basis_pct', 100)),
        'roth_start': float(accts.get('roth', 0)),
        'tda_start': float(accts.get('tda_p1', 400000)),
        'tda_spouse_start': float(accts.get('tda_p2', 0)),
        'target_stock_pct': int(alloc.get('stock_pct', 60)),
        'prefer_tda_before_taxable': bool(alloc.get('prefer_tda_before_taxable', False)),
        'roth_conversion_mode': 'None',
        'num_withdrawal_periods': 1,
        'rmd_start_age': plan.get('rmd_start_age_p1', 73),
        'rmd_start_age_spouse': plan.get('rmd_start_age_p2', 73),
        'ending_balance_goal': float(plan.get('ending_balance_goal', 0)),
        'num_add_goals': 0,
        'ss_income': float(ss.get('person1', {}).get('benefit', 25000)),
        'ss_start_age_p1': int(ss.get('person1', {}).get('start_age', 67)),
        'ss_fra_age_p1': int(ss.get('person1', {}).get('fra', 67)),
        'ss_income_spouse': float(ss.get('person2', {}).get('benefit', 20000)),
        'ss_start_age_p2': int(ss.get('person2', {}).get('start_age', 65)),
        'ss_fra_age_p2': int(ss.get('person2', {}).get('fra', 67)),
        'ss_cola': float(ss.get('cola', 0)),
        'pension_income': float(pensions.get('person1', {}).get('income', 0)),
        'pension_cola_p1': float(pensions.get('person1', {}).get('cola', 0)),
        'pension_survivor_pct_p1': float(pensions.get('person1', {}).get('survivor_pct', 0)),
        'pension_income_spouse': float(pensions.get('person2', {}).get('income', 0)),
        'pension_cola_p2': float(pensions.get('person2', {}).get('cola', 0)),
        'pension_survivor_pct_p2': float(pensions.get('person2', {}).get('survivor_pct', 0)),
        'other_income': float(plan.get('other_income', 0)),
        'earned_income': float(earned.get('annual', 0) if isinstance(earned, dict) else earned),
        'earned_income_years': int(earned.get('years', 0) if isinstance(earned, dict) else 0),
        'qcd_annual': float(plan.get('qcd_annual', 0)),
        'pension_buyout_enabled': False,
        'taxes_enabled': bool(tax.get('enabled', True)),
        'filing_status': _FILING_MAP.get(tax.get('filing_status', 'mfj'), 'Married Filing Jointly'),
        'use_itemized': bool(tax.get('use_itemized', False)),
        'itemized_deduction': float(tax.get('itemized_deduction', 0)),
        'inheritor_marginal_rate': float(tax.get('inheritor_marginal_rate', 0.35)),
        'state_tax_rate': float(tax.get('state_tax_rate', 0.05)),
        'state_exempt_retirement': bool(tax.get('state_exempt_retirement', True)),
        'return_mode': _RETURN_MODE_MAP.get(ret.get('mode', 'historical'), 'Historical (master_global_factors)'),
        'taxable_log_drift': float(ret.get('stock_log_drift', 0.09038261)),
        'taxable_log_volatility': float(ret.get('stock_log_volatility', 0.20485277)),
        'bond_log_drift': float(ret.get('bond_log_drift', 0.0172918)),
        'bond_log_volatility': float(ret.get('bond_log_volatility', 0.04796435)),
        'stock_dividend_yield': float(ret.get('stock_dividend_yield', 0.02)),
        'stock_turnover': float(ret.get('stock_turnover', 0.10)),
        'investment_fee_bps': float(ret.get('investment_fee_bps', 0)),
        'guardrails_enabled': bool(gr.get('enabled', True)),
        'guardrail_lower': float(gr.get('lower', 0.75)),
        'guardrail_upper': float(gr.get('upper', 0.90)),
        'guardrail_target': float(gr.get('target', 0.85)),
        'guardrail_inner_sims': int(gr.get('inner_sims', 200)),
        'guardrail_max_spending_pct': float(gr.get('max_spending_pct', 50)),
        'flex_goal_min_pct': 50.0,
        'tcja_sunset': bool(tax.get('tcja_sunset', False)),
        'tcja_sunset_year': int(tax.get('tcja_sunset_year', 2)),
        'irmaa_enabled': bool(tax.get('irmaa_enabled', False)),
        'display_decimals': 0,
        'monte_carlo_runs': int(plan.get('num_sims', 1000)),
        'num_scenarios': 1,
        'random_seed': 0,
        'seed_mode': 'Random each run',
    }

    # Withdrawal periods
    if spending.get('periods'):
        periods = [{'amount': float(p['amount'])} for p in spending['periods']]
        for i, p in enumerate(spending['periods']):
            if 'years' in p and i < len(spending['periods']) - 1:
                periods[i]['end_year'] = p['years']
        data['num_withdrawal_periods'] = len(periods)
    else:
        periods = [{'amount': float(spending.get('annual', 60000))}]
    data['periods'] = periods

    # Annuities
    ann = plan.get('annuities', {})
    if ann and (ann.get('person1', {}).get('income', 0) > 0 or ann.get('person2', {}).get('income', 0) > 0):
        data['pension_buyout_enabled'] = True
        if ann.get('person1', {}).get('income', 0) > 0:
            data['pension_buyout_person'] = 'Person 1'
            data['pension_buyout_income'] = float(ann['person1']['income'])
            data['pension_buyout_cola'] = float(ann['person1'].get('cola', 0))
            data['pension_buyout_survivor'] = float(ann['person1'].get('survivor_pct', 0))
        else:
            data['pension_buyout_person'] = 'Person 2'
            data['pension_buyout_income'] = float(ann['person2']['income'])
            data['pension_buyout_cola'] = float(ann['person2'].get('cola', 0))
            data['pension_buyout_survivor'] = float(ann['person2'].get('survivor_pct', 0))

    # Goals
    goals = plan.get('goals')
    if goals:
        data['num_add_goals'] = len(goals)
        data['add_goals'] = []
        for g in goals:
            data['add_goals'].append({
                'label': g.get('label', ''),
                'amount': float(g.get('amount', 0)),
                'begin': int(g.get('begin', 1)),
                'end': int(g.get('end', 1)),
                'priority': g.get('priority', 'Essential'),
                'cap': float(g.get('cap', -1)),
            })
    else:
        data['add_goals'] = []
    return data


def save_plan(plan: dict, pdf_bytes: bytes = None) -> dict:
    """Save plan in Streamlit-compatible format + PDF to ~/RWM/Current Client Plans/{client}/.

    Returns dict with 'json_path', 'pdf_path', 'client_dir'.
    """
    client = plan.get('client', 'Client')
    cdir = _client_dir(client)
    os.makedirs(cdir, exist_ok=True)

    stem = _plan_stem(client)
    name = stem  # always overwrite the same file

    # Convert to Streamlit format
    streamlit_data = _plan_to_streamlit_format(plan)

    # Also save the clean plan format as a companion file for the headless runner
    json_path = os.path.join(cdir, f'{name}.json')
    with open(json_path, 'w') as f:
        json.dump(streamlit_data, f, indent=2, default=str)

    # Save clean plan as _plan.json for headless runner to reload
    clean_path = os.path.join(cdir, f'{name}_plan.json')
    plan_to_save = dict(plan)
    plan_to_save['_meta'] = {
        'saved': datetime.now().isoformat(timespec='seconds'),
    }
    with open(clean_path, 'w') as f:
        json.dump(plan_to_save, f, indent=2, default=str)

    pdf_path = None
    if pdf_bytes:
        pdf_path = os.path.join(cdir, f'{name}.pdf')
        with open(pdf_path, 'wb') as f:
            f.write(pdf_bytes)

    return {'json_path': json_path, 'pdf_path': pdf_path, 'client_dir': cdir}


def update_plan(plan: dict, json_path: str, pdf_bytes: bytes = None) -> dict:
    """Overwrite an existing plan JSON (and optionally its PDF).

    Returns dict with 'json_path', 'pdf_path'.
    """
    plan_to_save = dict(plan)
    plan_to_save['_meta'] = plan.get('_meta', {})
    plan_to_save['_meta']['modified'] = datetime.now().isoformat(timespec='seconds')

    with open(json_path, 'w') as f:
        json.dump(plan_to_save, f, indent=2, default=str)

    pdf_path = None
    if pdf_bytes:
        pdf_path = json_path.replace('.json', '.pdf')
        with open(pdf_path, 'wb') as f:
            f.write(pdf_bytes)

    return {'json_path': json_path, 'pdf_path': pdf_path}


def load_plan(client_name: str, plan_name: str = None) -> dict:
    """Load a saved plan in clean format. If plan_name is None, load the most recent.

    Loads the Streamlit-format .json (authoritative) and converts to clean format.
    """
    cdir = _client_dir(client_name)
    if not os.path.isdir(cdir):
        raise FileNotFoundError(f'No client folder: {cdir}')

    # Load Streamlit format files (authoritative)
    jsons = sorted([f for f in os.listdir(cdir)
                    if f.endswith('.json') and not f.endswith('_results.json')
                    and not f.endswith('_plan.json')],
                   key=lambda f: os.path.getmtime(os.path.join(cdir, f)))

    if not jsons:
        raise FileNotFoundError(f'No plans found for {client_name}')

    fname = jsons[-1]
    with open(os.path.join(cdir, fname)) as f:
        data = json.load(f)

    # If it's already clean format (has 'person1' dict), return as-is
    if isinstance(data.get('person1'), dict):
        return data

    # Convert Streamlit format to clean format
    return _streamlit_to_plan_format(data, client_name)


def _streamlit_to_plan_format(data: dict, client_name: str) -> dict:
    """Convert Streamlit flat format back to clean plan format."""
    plan = {
        'client': client_name,
        'person1': {
            'label': data.get('person1_label', ''),
            'age': data.get('start_age', 65),
            'life_expectancy': data.get('life_expectancy_primary', 84),
        },
        'person2': {
            'label': data.get('person2_label', ''),
            'age': data.get('start_age_spouse', 60),
            'life_expectancy': data.get('life_expectancy_spouse', 89),
        },
        'accounts': {
            'taxable': data.get('taxable_start', 300000),
            'taxable_stock_basis_pct': data.get('taxable_stock_basis_pct', 50),
            'taxable_bond_basis_pct': data.get('taxable_bond_basis_pct', 100),
            'tda_p1': data.get('tda_start', 400000),
            'tda_p2': data.get('tda_spouse_start', 0),
            'roth': data.get('roth_start', 0),
        },
        'spending': {},
        'allocation': {
            'stock_pct': data.get('target_stock_pct', 60),
            'prefer_tda_before_taxable': data.get('prefer_tda_before_taxable', False),
        },
        'social_security': {
            'person1': {
                'benefit': data.get('ss_income', 25000),
                'start_age': data.get('ss_start_age_p1', 67),
                'fra': data.get('ss_fra_age_p1', 67),
            },
            'person2': {
                'benefit': data.get('ss_income_spouse', 20000),
                'start_age': data.get('ss_start_age_p2', 65),
                'fra': data.get('ss_fra_age_p2', 67),
            },
            'cola': data.get('ss_cola', 0),
        },
        'pensions': {
            'person1': {
                'income': data.get('pension_income', 0),
                'cola': data.get('pension_cola_p1', 0),
                'survivor_pct': data.get('pension_survivor_pct_p1', 0),
            },
            'person2': {
                'income': data.get('pension_income_spouse', 0),
                'cola': data.get('pension_cola_p2', 0),
                'survivor_pct': data.get('pension_survivor_pct_p2', 0),
            },
        },
        'other_income': data.get('other_income', 0),
        'earned_income': {
            'annual': data.get('earned_income', 0),
            'years': data.get('earned_income_years', 0),
        },
        'qcd_annual': data.get('qcd_annual', 0),
        'tax': {
            'enabled': data.get('taxes_enabled', True),
            'filing_status': 'mfj' if 'Joint' in str(data.get('filing_status', '')) else 'single',
            'use_itemized': data.get('use_itemized', False),
            'itemized_deduction': data.get('itemized_deduction', 0),
            'inheritor_marginal_rate': data.get('inheritor_marginal_rate', 0.35),
            'state_tax_rate': data.get('state_tax_rate', 0.05),
            'state_exempt_retirement': data.get('state_exempt_retirement', True),
            'tcja_sunset': data.get('tcja_sunset', False),
            'tcja_sunset_year': data.get('tcja_sunset_year', 2),
            'irmaa_enabled': data.get('irmaa_enabled', False),
        },
        'rmd_start_age_p1': data.get('rmd_start_age', 73),
        'rmd_start_age_p2': data.get('rmd_start_age_spouse', 73),
        'ending_balance_goal': data.get('ending_balance_goal', 0),
        'guardrails': {
            'enabled': data.get('guardrails_enabled', True),
            'lower': data.get('guardrail_lower', 0.75),
            'upper': data.get('guardrail_upper', 0.90),
            'target': data.get('guardrail_target', 0.85),
            'inner_sims': data.get('guardrail_inner_sims', 200),
            'max_spending_pct': data.get('guardrail_max_spending_pct', 50),
        },
        'returns': {
            'mode': 'historical' if 'Historical' in str(data.get('return_mode', '')) else 'lognormal',
            'stock_log_drift': data.get('taxable_log_drift', 0.09038261),
            'stock_log_volatility': data.get('taxable_log_volatility', 0.20485277),
            'bond_log_drift': data.get('bond_log_drift', 0.0172918),
            'bond_log_volatility': data.get('bond_log_volatility', 0.04796435),
            'stock_dividend_yield': data.get('stock_dividend_yield', 0.02),
            'stock_turnover': data.get('stock_turnover', 0.10),
            'investment_fee_bps': data.get('investment_fee_bps', 0),
        },
        'num_sims': data.get('monte_carlo_runs', 1000),
    }

    # Spending
    periods = data.get('periods', [])
    if periods:
        plan['spending']['annual'] = periods[0].get('amount', 60000)
        if len(periods) > 1:
            plan['spending']['periods'] = [{'amount': p['amount']} for p in periods]
    else:
        plan['spending']['annual'] = 60000

    return plan


def list_clients() -> list:
    """Return sorted list of client folder names."""
    os.makedirs(HEADLESS_DIR, exist_ok=True)
    return sorted([d for d in os.listdir(HEADLESS_DIR)
                   if os.path.isdir(os.path.join(HEADLESS_DIR, d)) and not d.startswith('.')])


def list_plans(client_name: str) -> list:
    """Return list of {name, path, modified} for a client's plans."""
    cdir = _client_dir(client_name)
    if not os.path.isdir(cdir):
        return []
    plans = []
    for f in sorted(os.listdir(cdir)):
        if f.endswith('.json') and not f.endswith('_results.json'):
            path = os.path.join(cdir, f)
            plans.append({
                'name': f[:-5],
                'path': path,
                'modified': datetime.fromtimestamp(os.path.getmtime(path)).isoformat(timespec='seconds'),
            })
    return plans


def find_client(query: str) -> list:
    """Fuzzy-match a client name query against saved clients."""
    query_lower = query.lower()
    clients = list_clients()
    # Exact substring match first
    exact = [c for c in clients if query_lower in c.lower()]
    if exact:
        return exact
    # Word-level match
    query_words = set(query_lower.split())
    scored = []
    for c in clients:
        c_words = set(c.lower().replace(',', '').replace('&', '').split())
        overlap = len(query_words & c_words)
        if overlap > 0:
            scored.append((overlap, c))
    scored.sort(reverse=True)
    return [c for _, c in scored]
