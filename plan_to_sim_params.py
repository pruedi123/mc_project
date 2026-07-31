"""Faithful Streamlit-equivalent translator from a saved plan JSON to sim_params.

Mirrors main.py:624-863 1:1 so headless tools (compare_modes, etc.) produce
the same numbers Streamlit produces. Use this in place of run_full_process's
`build_sim_params` when fidelity to the Streamlit UI matters.

Limitations (matches v1 scope — extend as plans need it):
- Pension-buyout scenarios: not handled (assumes `pension_buyout_enabled=False`)
- Separately-funded goals: not handled (assumes no `fund_separately` goals)
- Multi-period withdrawal schedules: handled via `periods[*].end_year` field
"""
import math
from datetime import date as _date
from typing import Dict, Tuple

import numpy as np

from sim_engine import (
    PP_FACTORS,
    load_portfolio_factors,
    resolve_mortgage_params,
    should_use_actual_allocation_columns,
)


_FILING_STATUS_MAP = {
    "Single": "single",
    "single": "single",
    "Married Filing Jointly": "mfj",
    "mfj": "mfj",
}

def _parse_fill_rate(value, default=0.24):
    """Coerce roth_bracket_fill_rate to a fraction.

    Legacy saved plans stored the selectbox LABEL ('22%'); newer plans store
    a float. Accept '22%', '22', 22, 0.22 — all mean 0.22."""
    if value is None:
        return default
    if isinstance(value, str):
        value = value.strip().rstrip('%')
        try:
            value = float(value)
        except ValueError:
            return default
    value = float(value)
    return value / 100.0 if value > 1.0 else value


_ROTH_MODE_MAP = {
    "None": "none",
    "none": "none",
    "Fixed amount": "fixed",
    "fixed": "fixed",
    "Fill to bracket": "bracket_fill",
    "bracket_fill": "bracket_fill",
}


def _sim_years_from_plan(plan: dict) -> int:
    return max(
        1,
        max(
            int(plan["life_expectancy_primary"]) - int(plan["start_age"]),
            int(plan["life_expectancy_spouse"]) - int(plan["start_age_spouse"]),
        )
        + 1,
    )


def _build_withdrawal_schedule(plan: dict, sim_years: int) -> Tuple[list, list]:
    """Replicate ui_inputs._render_withdrawal_section: convert saved periods
    [{amount, end_year?}, ...] into (num_years, amount) tuples, then expand
    into a year-by-year schedule of length sim_years."""
    periods = plan.get("periods") or [{"amount": 0.0}]
    inputs = []
    period_start = 1
    n = len(periods)
    for i, p in enumerate(periods):
        amount = float(p.get("amount", 0.0))
        if i == n - 1:
            period_end = sim_years
        else:
            period_end = int(p.get("end_year", min(period_start + 4, sim_years)))
        num_years = max(0, int(period_end) - period_start + 1)
        inputs.append((num_years, amount))
        period_start = int(period_end) + 1

    schedule = []
    for num_years, amount in inputs:
        schedule.extend([amount] * num_years)
    if len(schedule) < sim_years:
        last = schedule[-1] if schedule else 0.0
        schedule.extend([last] * (sim_years - len(schedule)))
    schedule = schedule[:sim_years]
    return schedule, inputs


def _build_goal_layers(plan: dict, sim_years: int, withdrawal_schedule: list):
    """Layer add_goals onto withdrawal_schedule and build essential/flexible
    goal_schedule + flex_goal_schedule. Returns dict of all goal-related
    schedules used by main.py's sim_params block. Does NOT handle the
    fund_separately path (set-aside accounting); raises if encountered."""
    goal_schedule = [0.0] * sim_years
    flex_goal_schedule = [0.0] * sim_years
    flex_capped_base = [0.0] * sim_years
    flex_cap_max = [0.0] * sim_years
    has_flex_caps = False
    goal_liquidation = [0.0] * sim_years

    for g in plan.get("add_goals", []) or []:
        amount = float(g.get("amount", 0.0))
        begin = int(g.get("begin", 1))
        end = int(g.get("end", sim_years))
        priority = g.get("priority", "Essential")
        if priority in ("Need", "need"):
            priority = "Essential"
        if priority in ("Want", "want"):
            priority = "Flexible"
        cap = float(g.get("cap", -1.0))
        fund_separately = bool(g.get("fund_separately", False))
        if fund_separately:
            raise NotImplementedError(
                "Separately-funded goals are not yet supported by plan_to_sim_params; "
                "use the Streamlit app directly for plans with set-aside funding."
            )
        if amount <= 0:
            continue
        for y in range(begin - 1, min(end, sim_years)):
            withdrawal_schedule[y] += amount
            if priority == "Essential":
                goal_schedule[y] += amount
            elif priority == "Flexible":
                flex_goal_schedule[y] += amount
                if cap >= 0:
                    has_flex_caps = True
                    flex_capped_base[y] += amount
                    flex_cap_max[y] += amount * (1.0 + cap / 100.0)

    return {
        "goal_schedule": goal_schedule if any(g > 0 for g in goal_schedule) else None,
        "flex_goal_schedule": flex_goal_schedule if any(g > 0 for g in flex_goal_schedule) else None,
        "flex_capped_base_schedule": flex_capped_base if has_flex_caps else None,
        "flex_cap_max_schedule": flex_cap_max if has_flex_caps else None,
        "goal_liquidation_schedule": goal_liquidation if any(g > 0 for g in goal_liquidation) else None,
    }


def _blended_mu_sigma(plan: dict, target_stock_pct: float, return_mode: str, guardrails_enabled: bool, historical_allocation_source: str):
    """Mirror main.py:829-863."""
    if not guardrails_enabled:
        return 0.0, 0.0
    if return_mode == "Simulated (lognormal)":
        stock_mu = float(plan.get("taxable_log_drift", 0.09038261))
        stock_sigma = float(plan.get("taxable_log_volatility", 0.20485277))
        bond_mu = float(plan.get("bond_log_drift", 0.0172918))
        bond_sigma = float(plan.get("bond_log_volatility", 0.04796435))
        blended_mu = target_stock_pct * stock_mu + (1 - target_stock_pct) * bond_mu
        rho = 0.10
        w_s, w_b = target_stock_pct, 1 - target_stock_pct
        blended_sigma = float(
            math.sqrt(
                (w_s * stock_sigma) ** 2
                + (w_b * bond_sigma) ** 2
                + 2 * w_s * w_b * stock_sigma * bond_sigma * rho
            )
        )
        return blended_mu, blended_sigma
    use_actual = should_use_actual_allocation_columns(return_mode, historical_allocation_source)
    allocation_log_rets = np.log(load_portfolio_factors(target_stock_pct, use_actual))
    blended_mu = float(np.mean(allocation_log_rets))
    blended_sigma = float(np.std(allocation_log_rets))
    return blended_mu, blended_sigma


def build_sim_params_from_plan(plan: dict, return_mode: str | None = None) -> Dict:
    """Translate a saved plan JSON into the sim_params dict + supporting
    values, replicating main.py:624-863 logic. Returns:

        {
            'sim_params': dict (ready to pass to run_monte_carlo / run_historical_parallel),
            'sim_years': int,
            'target_stock_pct': float (0-1),
            'base_spending': float (the first-period amount; pre-goal layering),
            'return_mode': str (mode used for blended_mu/sigma — either from plan or override),
            'inheritor_marginal_rate': float,
            'monte_carlo_runs': int,
            'ending_balance_goal': float,
        }
    """
    if plan.get("pension_buyout_enabled"):
        raise NotImplementedError(
            "Pension-buyout plans are not yet supported by plan_to_sim_params; "
            "use the Streamlit app directly for plans with pension buyouts."
        )

    sim_years = _sim_years_from_plan(plan)
    target_stock_pct_raw = float(plan.get("target_stock_pct", 60))
    target_stock_pct = target_stock_pct_raw / 100.0 if target_stock_pct_raw > 1 else target_stock_pct_raw

    base_withdrawal_schedule, _periods = _build_withdrawal_schedule(plan, sim_years)
    withdrawal_schedule = list(base_withdrawal_schedule)
    base_spending = withdrawal_schedule[0] if withdrawal_schedule else 0.0
    goals = _build_goal_layers(plan, sim_years, withdrawal_schedule)

    # Streamlit's filing_status_key resolution: 'single' or 'mfj'.
    filing_choice = plan.get("filing_status", "Married Filing Jointly")
    filing_status_key = _FILING_STATUS_MAP.get(filing_choice, "mfj")

    roth_mode_ui = plan.get("roth_conversion_mode", "None")
    roth_mode_engine = _ROTH_MODE_MAP.get(roth_mode_ui, "none")
    roth_tax_source = "taxable" if plan.get("roth_conversion_tax_source") == "Taxable" else "tda"
    roth_source_ui = plan.get("roth_conversion_source_tda", "Person 1 TDA")
    roth_source = "person1" if roth_source_ui == "Person 1 TDA" else "person2"

    selected_return_mode = return_mode or plan.get("return_mode", "Simulated (lognormal)")
    historical_allocation_source = plan.get("historical_allocation_source", "Actual allocation columns")
    guardrails_enabled = bool(plan.get("guardrails_enabled", True))
    blended_mu, blended_sigma = _blended_mu_sigma(
        plan, target_stock_pct, selected_return_mode, guardrails_enabled, historical_allocation_source
    )

    taxable_start = float(plan.get("taxable_start", 0.0))
    tda_start = float(plan.get("tda_start", 0.0))
    tda_spouse_start = float(plan.get("tda_spouse_start", 0.0))
    roth_start = float(plan.get("roth_start", 0.0))

    # Mortgage: fixed nominal payment for the loan term, or payoff from taxable at start
    taxable_start, mortgage_payment_annual, mortgage_years_remaining = resolve_mortgage_params(
        float(plan.get("mortgage_balance", 0.0)),
        float(plan.get("mortgage_rate", 0.0)),
        int(plan.get("mortgage_years_remaining", 0)),
        bool(plan.get("mortgage_payoff_at_start", False)),
        taxable_start,
    )

    # Default estate goal: maintain the starting balance — investable portfolio
    # (after any mortgage payoff) plus the stated cash reserve. An explicit
    # "legacy_target" in the plan overrides; 0 means no estate goal.
    if "legacy_target" in plan:
        legacy_target = float(plan.get("legacy_target") or 0.0)
    else:
        legacy_target = (taxable_start + tda_start + tda_spouse_start + roth_start
                         + float(plan.get("reserve_amount") or 0.0))

    sim_params = dict(
        start_age_primary=int(plan["start_age"]),
        start_age_spouse=int(plan["start_age_spouse"]),
        taxable_start=taxable_start,
        stock_total_return=0.0,
        stock_dividend_yield=float(plan.get("stock_dividend_yield", 0.02)),
        stock_turnover=float(plan.get("stock_turnover", 0.10)),
        investment_fee_bps=float(plan.get("investment_fee_bps", 0.0)),
        bond_return=0.0,
        roth_start=roth_start,
        tda_start=tda_start,
        tda_spouse_start=tda_spouse_start,
        target_stock_pct=target_stock_pct,
        # Streamlit's UI divides these by 100 before passing to engine
        # (ui_inputs.py:589 — widget shows percent but engine expects fraction).
        taxable_stock_basis_pct=float(plan.get("taxable_stock_basis_pct", 50.0)) / 100.0,
        taxable_bond_basis_pct=float(plan.get("taxable_bond_basis_pct", 100.0)) / 100.0,
        withdrawal_schedule=withdrawal_schedule,
        rmd_start_age=int(plan.get("rmd_start_age", 73)),
        rmd_start_age_spouse=int(plan.get("rmd_start_age_spouse", 73)),
        ss_income_annual=float(plan.get("ss_income", 0.0)),
        ss_income_spouse_annual=float(plan.get("ss_income_spouse", 0.0)),
        ss_cola=float(plan.get("ss_cola", 0.0)),
        pension_income_annual=float(plan.get("pension_income", 0.0)),
        pension_income_spouse_annual=float(plan.get("pension_income_spouse", 0.0)),
        pension_cola_p1=float(plan.get("pension_cola_p1", 0.0)),
        pension_cola_p2=float(plan.get("pension_cola_p2", 0.0)),
        pension_survivor_pct_p1=float(plan.get("pension_survivor_pct_p1", 0.0)),
        pension_survivor_pct_p2=float(plan.get("pension_survivor_pct_p2", 0.0)),
        pp_factors=PP_FACTORS,
        mortgage_payment_annual=mortgage_payment_annual,
        mortgage_years_remaining=mortgage_years_remaining,
        other_income_annual=float(plan.get("other_income", 0.0)),
        filing_status=filing_status_key,
        use_itemized_deductions=bool(plan.get("use_itemized", False)),
        itemized_deduction_amount=float(plan.get("itemized_deduction", 0.0)),
        roth_conversion_amount=float(plan.get("roth_conversion_amount", 0.0)),
        roth_conversion_years=int(plan.get("roth_conversion_years", 0)),
        roth_conversion_tax_source=roth_tax_source,
        roth_conversion_source=roth_source,
        roth_conversion_mode=roth_mode_engine,
        roth_bracket_fill_rate=_parse_fill_rate(plan.get("roth_bracket_fill_rate")),
        ss_start_age_p1=int(plan.get("ss_start_age_p1", 67)),
        ss_start_age_p2=int(plan.get("ss_start_age_p2", 67)),
        ss_fra_age_p1=int(plan.get("ss_fra_age_p1", 67)),
        ss_fra_age_p2=int(plan.get("ss_fra_age_p2", 67)),
        state_tax_rate=float(plan.get("state_tax_rate", 0.05)),
        state_exempt_retirement=bool(plan.get("state_exempt_retirement", False)),
        life_expectancy_primary=int(plan["life_expectancy_primary"]),
        life_expectancy_spouse=int(plan["life_expectancy_spouse"]),
        guardrails_enabled=guardrails_enabled,
        guardrail_lower=float(plan.get("guardrail_lower", 0.75)),
        guardrail_upper=float(plan.get("guardrail_upper", 0.90)),
        guardrail_target=float(plan.get("guardrail_target", 0.85)),
        guardrail_inner_sims=int(plan.get("guardrail_inner_sims", 200)),
        guardrail_max_spending_pct=float(plan.get("guardrail_max_spending_pct", 25.0)),
        guardrail_year1_literal=bool(plan.get("guardrail_year1_literal", False)),
        guardrail_cap_release_underwater=bool(plan.get("guardrail_cap_release_underwater", False)),
        legacy_target=legacy_target,
        taxes_enabled=bool(plan.get("taxes_enabled", True)),
        goal_schedule=goals["goal_schedule"],
        flex_goal_schedule=goals["flex_goal_schedule"],
        flex_goal_min_pct=float(plan.get("flex_goal_min_pct", 50.0)) / 100.0,
        base_is_essential=False,
        flex_capped_base_schedule=goals["flex_capped_base_schedule"],
        flex_cap_max_schedule=goals["flex_cap_max_schedule"],
        inheritance_enabled=bool(plan.get("inheritance_enabled", False)),
        inheritance_year=int(plan.get("inheritance_year", 1)),
        inheritance_taxable_amount=float(plan.get("inheritance_taxable_amount", 0.0)),
        inheritance_ira_amount=float(plan.get("inheritance_ira_amount", 0.0)),
        tcja_sunset=bool(plan.get("tcja_sunset", False)),
        tcja_sunset_year=int(plan.get("tcja_sunset_year", 2)),
        qcd_annual=float(plan.get("qcd_annual", 0.0)),
        earned_income_annual=float(plan.get("earned_income", 0.0)),
        earned_income_years=int(plan.get("earned_income_years", 0)),
        irmaa_enabled=bool(plan.get("irmaa_enabled", False)),
        prefer_tda_before_taxable=bool(plan.get("prefer_tda_before_taxable", False)),
        goal_taxable_start=0.0,
        goal_tda_start=0.0,
        goal_tda_p1_fraction=0.0,
        goal_liquidation_schedule=goals["goal_liquidation_schedule"],
        goal_stock_pct=target_stock_pct,
        blended_mu=blended_mu,
        blended_sigma=blended_sigma,
        # Year 1 = the year the plan is run (drives the OBBBA senior-bonus
        # window); plans may pin it explicitly for reproducibility
        sim_start_calendar_year=int(plan.get("sim_start_calendar_year", _date.today().year)),
    )

    return {
        "sim_params": sim_params,
        "sim_years": sim_years,
        "target_stock_pct": target_stock_pct,
        "base_spending": float(base_spending),
        "return_mode": selected_return_mode,
        "historical_allocation_source": historical_allocation_source,
        "use_actual_allocation_columns": should_use_actual_allocation_columns(selected_return_mode, historical_allocation_source),
        "inheritor_marginal_rate": float(plan.get("inheritor_marginal_rate", 0.32)),
        "monte_carlo_runs": int(plan.get("monte_carlo_runs", 1000)),
        "ending_balance_goal": float(plan.get("ending_balance_goal", 0.0)),
    }
