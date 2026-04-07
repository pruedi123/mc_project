"""Clean plan JSON schema, defaults, validation, and session_state adapter.

The clean plan format is the public API for the headless runner.
Users (or Claude) specify only what differs from defaults.
"""

import copy
from datetime import date

PLAN_DEFAULTS = {
    # Identity
    "client": "Client",

    # People
    "person1": {"label": "", "age": 65, "life_expectancy": 84},
    "person2": {"label": "", "age": 60, "life_expectancy": 89},

    # Accounts
    "accounts": {
        "taxable": 300000,
        "taxable_stock_basis_pct": 50,
        "taxable_bond_basis_pct": 100,
        "tda_p1": 700000,
        "tda_p2": 0,
        "roth": 0,
    },

    # Spending
    "spending": {
        "annual": 60000,
        "periods": None,  # [{"amount": 60000, "years": 10}, {"amount": 40000}]
    },

    # Allocation
    "allocation": {
        "stock_pct": 60,
        "prefer_tda_before_taxable": False,
    },

    # Social Security
    "social_security": {
        "person1": {"benefit": 0, "start_age": 67, "fra": 67},
        "person2": {"benefit": 0, "start_age": 67, "fra": 67},
        "cola": 0.0,
    },

    # Pensions
    "pensions": {
        "person1": {"income": 0, "cola": 0.0, "survivor_pct": 0.0},
        "person2": {"income": 0, "cola": 0.0, "survivor_pct": 0.0},
    },

    # Annuities
    "annuities": {
        "person1": {"income": 0, "cola": 0.0, "survivor_pct": 0.0},
        "person2": {"income": 0, "cola": 0.0, "survivor_pct": 0.0},
        "start_year": 1,
    },

    # Other income
    "other_income": 0,
    "earned_income": {"annual": 0, "years": 0},
    "qcd_annual": 0,

    # Tax
    "tax": {
        "enabled": True,
        "filing_status": "mfj",
        "use_itemized": False,
        "itemized_deduction": 0,
        "inheritor_marginal_rate": 0.35,
        "state_tax_rate": 0.05,
        "state_exempt_retirement": True,
        "tcja_sunset": False,
        "tcja_sunset_year": 2,
        "irmaa_enabled": False,
    },

    # RMDs
    "rmd_start_age_p1": 73,
    "rmd_start_age_p2": 73,
    "ending_balance_goal": 0,

    # Guardrails
    "guardrails": {
        "enabled": True,
        "lower": 0.75,
        "upper": 0.90,
        "target": 0.85,
        "inner_sims": 200,
        "max_spending_pct": 50.0,
    },

    # Returns
    "returns": {
        "mode": "historical",
        "stock_log_drift": 0.09038261,
        "stock_log_volatility": 0.20485277,
        "bond_log_drift": 0.0172918,
        "bond_log_volatility": 0.04796435,
        "stock_dividend_yield": 0.02,
        "stock_turnover": 0.10,
        "investment_fee_bps": 0,
    },

    # Simulation
    "num_sims": 1000,

    # Roth conversions (Phase 2)
    "roth_conversions": None,

    # Goals: [{"label": "LTC", "amount": 100000, "begin": 28, "end": 30, "priority": "Flexible"}]
    # priority: "Essential" (never reduced by guardrails) or "Flexible" (can be reduced)
    # begin/end: simulation year numbers (1-based)
    "goals": None,

    # Inheritance
    "inheritance": None,

    # Scenarios
    "scenarios": None,
}

# ── Deep merge ────────────────────────────────────────────────

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base. Returns new dict."""
    result = copy.deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result


def apply_defaults(plan: dict) -> dict:
    """Deep-merge user plan over PLAN_DEFAULTS. Returns a complete plan dict."""
    return _deep_merge(PLAN_DEFAULTS, plan)


# ── Validation ────────────────────────────────────────────────

def validate_plan(plan: dict) -> list:
    """Return list of error strings. Empty = valid."""
    errors = []
    p1 = plan.get('person1', {})
    p2 = plan.get('person2', {})
    if p1.get('age', 65) >= p1.get('life_expectancy', 84):
        errors.append('person1 age must be less than life_expectancy')
    if p2.get('age', 60) >= p2.get('life_expectancy', 89):
        errors.append('person2 age must be less than life_expectancy')
    accts = plan.get('accounts', {})
    if accts.get('taxable', 0) < 0:
        errors.append('taxable account cannot be negative')
    spending = plan.get('spending', {})
    if spending.get('annual', 0) < 0 and spending.get('periods') is None:
        errors.append('spending annual cannot be negative')
    return errors


# ── Session state adapter ─────────────────────────────────────

_FILING_STATUS_DISPLAY = {
    'mfj': 'Married Filing Jointly',
    'single': 'Single',
}

_RETURN_MODE_DISPLAY = {
    'lognormal': 'Simulated (lognormal)',
    'historical': 'Historical (master_global_factors)',
}


def plan_to_session_state(plan: dict, results: dict) -> dict:
    """Convert clean plan + simulation results into the flat dict
    that pdf_report.generate_report() expects."""
    ss = {}

    # Client
    ss['client_select'] = plan['client']
    ss['save_file_name'] = plan['client'].lower().replace(' ', '_').replace(',', '')

    # Ages
    ss['start_age'] = plan['person1']['age']
    ss['start_age_spouse'] = plan['person2']['age']
    ss['life_expectancy_primary'] = plan['person1']['life_expectancy']
    ss['life_expectancy_spouse'] = plan['person2']['life_expectancy']

    # Accounts
    ss['taxable_start'] = plan['accounts']['taxable']
    ss['tda_start'] = plan['accounts']['tda_p1']
    ss['tda_spouse_start'] = plan['accounts']['tda_p2']
    ss['roth_start'] = plan['accounts']['roth']
    ss['target_stock_pct'] = plan['allocation']['stock_pct']  # 0-100 for PDF display

    # Social Security
    ss['ss_income'] = plan['social_security']['person1']['benefit']
    ss['ss_start_age_p1'] = plan['social_security']['person1']['start_age']
    ss['ss_income_spouse'] = plan['social_security']['person2']['benefit']
    ss['ss_start_age_p2'] = plan['social_security']['person2']['start_age']

    # Pensions
    ss['pension_income'] = plan['pensions']['person1']['income']
    ss['pension_income_spouse'] = plan['pensions']['person2']['income']

    # Other income
    ss['other_income'] = plan.get('other_income', 0)

    # Withdrawal periods
    spending = plan['spending']
    if spending.get('periods'):
        periods = spending['periods']
        ss['num_withdrawal_periods'] = len(periods)
        year_cursor = 0
        for i, p in enumerate(periods):
            ss[f'wd_amount_{i}'] = p['amount']
            if 'years' in p:
                year_cursor += p['years']
                ss[f'wd_end_{i}'] = year_cursor
    else:
        ss['num_withdrawal_periods'] = 1
        ss['wd_amount_0'] = spending['annual']

    # Simulation settings
    ss['num_sims'] = results.get('num_sims', plan['num_sims'])
    ss['sim_mode'] = 'simulated' if plan['returns']['mode'] == 'lognormal' else 'historical_dist'
    ss['return_mode'] = _RETURN_MODE_DISPLAY.get(plan['returns']['mode'], 'Simulated (lognormal)')
    ss['taxes_enabled'] = plan['tax']['enabled']
    ss['guardrails_enabled'] = plan['guardrails']['enabled']
    ss['investment_fee_bps'] = plan['returns']['investment_fee_bps']
    ss['guardrail_max_spending_pct'] = plan['guardrails']['max_spending_pct']
    ss['filing_status'] = _FILING_STATUS_DISPLAY.get(plan['tax']['filing_status'], 'Married Filing Jointly')

    # Simulation results
    ss['mc_percentile_rows'] = results.get('percentile_rows', [])
    ss['mc_pct_non_positive'] = results.get('pct_non_positive', 0.0)
    ss['mc_spending_pct_rows'] = results.get('spending_percentiles', [])
    ss['mc_all_yearly'] = results.get('all_yearly')
    ss['sim_df'] = results.get('sim_df')
    if results.get('multi_scenario_results'):
        ss['multi_scenario_results'] = results['multi_scenario_results']
    if results.get('scenario_input_diffs'):
        ss['scenario_input_diffs'] = results['scenario_input_diffs']

    return ss


def describe_plan_diffs(baseline: dict, scenario: dict) -> list:
    """Compare two clean plan dicts and return a list of human-readable difference strings."""
    diffs = []

    # Accounts
    b_accts = baseline.get('accounts', {})
    s_accts = scenario.get('accounts', {})
    for key, label in [('taxable', 'Taxable'), ('tda_p1', 'TDA P1'), ('tda_p2', 'TDA P2'), ('roth', 'Roth')]:
        bv = b_accts.get(key, 0)
        sv = s_accts.get(key, 0)
        if bv != sv:
            diffs.append(f"{label}: ${bv:,.0f} -> ${sv:,.0f}")

    # Allocation
    b_alloc = baseline.get('allocation', {}).get('stock_pct', 60)
    s_alloc = scenario.get('allocation', {}).get('stock_pct', 60)
    if b_alloc != s_alloc:
        diffs.append(f"Allocation: {b_alloc}% -> {s_alloc}% stocks")

    # Spending
    b_spend = baseline.get('spending', {}).get('annual', 60000)
    s_spend = scenario.get('spending', {}).get('annual', 60000)
    if b_spend != s_spend:
        diffs.append(f"Spending: ${b_spend:,.0f} -> ${s_spend:,.0f}/yr")

    # Social Security
    for person, label in [('person1', 'SS P1'), ('person2', 'SS P2')]:
        b_ss = baseline.get('social_security', {}).get(person, {})
        s_ss = scenario.get('social_security', {}).get(person, {})
        if b_ss.get('benefit', 0) != s_ss.get('benefit', 0):
            diffs.append(f"{label}: ${b_ss.get('benefit', 0):,.0f} -> ${s_ss.get('benefit', 0):,.0f}")
        if b_ss.get('start_age', 67) != s_ss.get('start_age', 67):
            diffs.append(f"{label} claim age: {b_ss.get('start_age', 67)} -> {s_ss.get('start_age', 67)}")

    # Pensions
    for person, label in [('person1', 'Pension P1'), ('person2', 'Pension P2')]:
        b_pen = baseline.get('pensions', {}).get(person, {}).get('income', 0)
        s_pen = scenario.get('pensions', {}).get(person, {}).get('income', 0)
        if b_pen != s_pen:
            diffs.append(f"{label}: ${b_pen:,.0f} -> ${s_pen:,.0f}/yr")

    # Annuities
    for person, label in [('person1', 'Annuity P1'), ('person2', 'Annuity P2')]:
        b_ann = baseline.get('annuities', {}).get(person, {}).get('income', 0)
        s_ann = scenario.get('annuities', {}).get(person, {}).get('income', 0)
        if b_ann != s_ann:
            diffs.append(f"{label}: ${b_ann:,.0f} -> ${s_ann:,.0f}/yr")

    # Goals
    b_goals = baseline.get('goals') or []
    s_goals = scenario.get('goals') or []
    if b_goals != s_goals:
        if not b_goals and s_goals:
            for g in s_goals:
                diffs.append(f"Added goal: {g['label']} ${g['amount']:,.0f}/yr yrs {g['begin']}-{g['end']}")
        elif b_goals and not s_goals:
            diffs.append("Removed all goals")
        else:
            for g in s_goals:
                if g not in b_goals:
                    diffs.append(f"Added goal: {g['label']} ${g['amount']:,.0f}/yr yrs {g['begin']}-{g['end']}")
            for g in b_goals:
                if g not in s_goals:
                    diffs.append(f"Removed goal: {g['label']}")

    # Life expectancy
    for person, label in [('person1', 'Life exp P1'), ('person2', 'Life exp P2')]:
        ble = baseline.get(person, {}).get('life_expectancy', 84 if person == 'person1' else 89)
        sle = scenario.get(person, {}).get('life_expectancy', 84 if person == 'person1' else 89)
        if ble != sle:
            diffs.append(f"{label}: {ble} -> {sle}")

    if not diffs:
        diffs.append("No input differences")

    return diffs
