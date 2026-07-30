#!/usr/bin/env python3
"""Binary-search the starting portfolio needed to sustain a target after-tax
essential income at a given success rate. Reuses the headless engine glue."""
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_modes import _run_one, _metrics

MODE = "Historical (master_global_factors)"
TARGET_AFTER_TAX = 40_000.0
SUCCESS_BAR = 90.0  # percent of paths with avg after-tax spending >= target

BASE = {
    "start_age": 65, "start_age_spouse": 65,
    "life_expectancy_primary": 85, "life_expectancy_spouse": 85,  # 20-year horizon
    "taxable_start": 0.0, "taxable_stock_basis_pct": 50.0, "taxable_bond_basis_pct": 100.0,
    "roth_start": 0.0, "tda_start": 0.0, "tda_spouse_start": 0.0,
    "target_stock_pct": 60, "prefer_tda_before_taxable": False,
    "roth_conversion_mode": "None", "num_withdrawal_periods": 1,
    "rmd_start_age": 73, "rmd_start_age_spouse": 73, "ending_balance_goal": 0.0,
    "num_add_goals": 0,
    "ss_income": 0.0, "ss_start_age_p1": 67, "ss_fra_age_p1": 67,
    "ss_income_spouse": 0.0, "ss_start_age_p2": 67, "ss_fra_age_p2": 67, "ss_cola": 0.0,
    "pension_income": 0.0, "pension_cola_p1": 0.0, "pension_survivor_pct_p1": 0.0,
    "pension_income_spouse": 0.0, "pension_cola_p2": 0.0, "pension_survivor_pct_p2": 0.0,
    "other_income": 0.0, "earned_income": 0.0, "earned_income_years": 0, "qcd_annual": 0.0,
    "pension_buyout_enabled": False,
    "taxes_enabled": True, "filing_status": "Married Filing Jointly",
    "use_itemized": False, "itemized_deduction": 0.0, "inheritor_marginal_rate": 0.32,
    "state_tax_rate": 0.0495, "state_exempt_retirement": True,
    "return_mode": MODE,
    "taxable_log_drift": 0.09038261, "taxable_log_volatility": 0.20485277,
    "bond_log_drift": 0.0172918, "bond_log_volatility": 0.04796435,
    "stock_dividend_yield": 0.02, "stock_turnover": 0.1,
    "investment_fee_bps": 100.0,
    "guardrails_enabled": True, "guardrail_lower": 0.75, "guardrail_upper": 0.90,
    "guardrail_target": 0.90, "guardrail_inner_sims": 200,
    "guardrail_max_spending_pct": 25.0, "flex_goal_min_pct": 50.0,
    "tcja_sunset": False, "tcja_sunset_year": 2, "irmaa_enabled": False,
    "essential_spending": TARGET_AFTER_TAX, "display_decimals": 0,
    "monte_carlo_runs": 1000, "num_scenarios": 1, "random_seed": 0,
    "seed_mode": "Random each run",
    "periods": [{"amount": TARGET_AFTER_TAX}],
    "add_goals": [],
}


def success_at(portfolio):
    plan = copy.deepcopy(BASE)
    pass
    plan["roth_start"] = portfolio
    plan["tda_start"] = 0.0
    results, ayd, sim_years, _ = _run_one(plan, MODE, 1000)
    m = _metrics(results, ayd, sim_years, TARGET_AFTER_TAX)
    return m["spending_success_%"], m


def main():
    lo, hi = 300_000.0, 1_200_000.0
    print(f"Target: ${TARGET_AFTER_TAX:,.0f}/yr after-tax met in {SUCCESS_BAR:.0f}% of historical paths\n")
    for label, p in (("lo", lo), ("hi", hi)):
        s, _ = success_at(p)
        print(f"  bracket {label}=${p:,.0f} -> {s:.1f}%")

    for i in range(11):
        mid = (lo + hi) / 2.0
        s, m = success_at(mid)
        print(f"  iter {i+1}: ${mid:,.0f} -> {s:.1f}%")
        if s >= SUCCESS_BAR:
            hi = mid
        else:
            lo = mid

    ans = round((lo + hi) / 2.0 / 5000) * 5000
    s, m = success_at(ans)
    print(f"\n==> Required portfolio ~= ${ans:,.0f}  (spending success {s:.1f}%)")
    print(f"    P10/P50 avg after-tax spending: ${m['P10_spending']:,.0f} / ${m['P50_spending']:,.0f}")
    print(f"    Portfolio survival: {m['portfolio_survival_%']:.1f}%   Median ending: ${m['P50_after_tax_end']:,.0f}")


if __name__ == "__main__":
    main()
