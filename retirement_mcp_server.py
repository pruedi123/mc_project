"""MCP Server for the retirement simulation headless runner.

Exposes run_plan, load_plan, list_clients as tools for Claude Desktop.
"""

import os
import sys
import json

# Ensure mc_project is on the path
sys.path.insert(0, os.path.dirname(__file__))

from mcp.server.fastmcp import FastMCP

import argparse as _argparse
_parser = _argparse.ArgumentParser()
_parser.add_argument('--http', action='store_true')
_parser.add_argument('--port', type=int, default=8787)
_args, _ = _parser.parse_known_args()

mcp = FastMCP(
    "Retirement Simulator",
    host="0.0.0.0",
    port=_args.port,
    instructions="""You are a retirement planning assistant with access to a Monte Carlo simulation engine.

WORKFLOW FOR NEW CLIENTS:
When the user describes a new client in natural language, follow these steps:

1. EXTRACT the inputs they provided (ages, accounts, SS, spending, etc.)

2. ASK about any MUST-HAVE inputs they didn't mention:
   - Client name (both persons if couple)
   - Ages for both persons
   - Account balances (taxable, TDA/IRA for each person, Roth)
   - Social Security benefit amounts and claiming ages
   - Desired annual after-tax spending
   Only ask about these if they were not provided. Do NOT ask about optional details yet.

3. SHOW ASSUMPTIONS before running. Display all inputs including defaults:
   "Here's what I'll use for [Client Name]:
   - Ages: Person 1 [age], Person 2 [age]
   - Life expectancy: Person 1 [84], Person 2 [89]
   - Taxable: $X (50% stock basis, 100% bond basis)
   - TDA Person 1: $X, TDA Person 2: $X
   - Roth: $X
   - SS Person 1: $X/yr at age X, Person 2: $X/yr at age X
   - Spending: $X/yr
   - Allocation: 60% stocks / 40% bonds
   - Tax: MFJ, 5% state, retirement income exempt
   - Pensions/Annuities: none
   - Guardrails: on, spending cap 50%
   - Investment fees: 0 bps
   - Returns: historical
   Anything you want to change before I run it?"

4. RUN the simulation only after the user confirms or says to proceed.

5. PRESENT RESULTS clearly:
   - Success rate
   - Median ending portfolio
   - Median avg annual spending
   - Median total taxes and effective rate
   - Where the PDF report was saved

WORKFLOW FOR EXISTING CLIENTS:
When the user says something like "load the Jones plan" or "change Thompson allocation to 80%":
1. Use load_client_plan or update_client_plan
2. Show what changed
3. Present results

DEFAULTS (used when not specified):
- Life expectancy: person1=84, person2=89
- Accounts: taxable=300000 (50% stock basis, 100% bond basis), tda_p1=400000, tda_p2=300000, roth=0
- SS: person1=25000 at 67 (FRA 67), person2=20000 at 65 (FRA 67), no COLA
- Spending: 60000/yr
- Allocation: 60% stocks
- Tax: MFJ, 5% state rate, retirement income exempt, inheritor rate 35%
- Guardrails: enabled, lower=75%, upper=90%, target=85%, spending cap=50%
- Returns: historical mode
- Pensions/Annuities: none
- RMD start: age 73 both
- Roth conversions: none
- Investment fees: 0 bps
- Simulations: 1000

SCENARIO SUPPORT:
Users can request comparisons like "also run it with 80% stocks" or "compare with a $500k annuity."
Add scenarios to the plan JSON:
  "scenarios": [{"name": "80% Stocks", "stock_pct": 80}]

Scenario override options:
  - stock_pct: change allocation (0-100)
  - spend_flat: replace spending with fixed amount
  - spend_scale: multiply spending by factor (e.g., 0.9 for 90%)
  - tda_delta_p1/p2: add/subtract from TDA balance
  - annuity_income_p1/p2: set annuity income
  - annuity_cola_p1/p2: annuity COLA rate
  - annuity_survivor_pct_p1/p2: survivor benefit fraction
  - life_expectancy_primary/spouse: stress test ages
""",
)


@mcp.tool()
def run_retirement_plan(plan_json: str) -> str:
    """Run a Monte Carlo retirement simulation from a JSON plan and save the results.

    Takes a JSON string with client details. Only include fields that differ from defaults.

    REQUIRED FIELDS:
    - client: "Last, First & Spouse" format
    - person1.age and person2.age
    - At least one account balance
    - spending.annual

    EXAMPLE (minimal):
    {"client": "Smith, John & Jane", "person1": {"age": 62}, "person2": {"age": 58},
     "accounts": {"taxable": 500000, "tda_p1": 300000, "tda_p2": 200000},
     "social_security": {"person1": {"benefit": 28000, "start_age": 67},
                         "person2": {"benefit": 18000, "start_age": 65}},
     "spending": {"annual": 55000}}

    EXAMPLE (with scenarios):
    {"client": "Smith, John & Jane", "person1": {"age": 62}, "person2": {"age": 58},
     "accounts": {"taxable": 500000, "tda_p1": 300000, "tda_p2": 200000},
     "spending": {"annual": 55000},
     "scenarios": [{"name": "80% Stocks", "stock_pct": 80}]}

    EXAMPLE (with annuity comparison):
    {"client": "Smith, John & Jane", "person1": {"age": 62}, "person2": {"age": 58},
     "accounts": {"taxable": 500000, "tda_p1": 300000, "tda_p2": 200000},
     "spending": {"annual": 55000},
     "scenarios": [{"name": "Buy Annuity", "tda_delta_p1": -500000,
                    "annuity_income_p1": 36000, "annuity_cola_p1": 0,
                    "annuity_survivor_pct_p1": 1.0}]}

    ALL DEFAULTS (used when field is omitted):
    person1: age=65, life_expectancy=84
    person2: age=60, life_expectancy=89
    accounts: taxable=300000 (basis 50%/100%), tda_p1=400000, tda_p2=300000, roth=0
    social_security: p1=25000@67(fra67), p2=20000@65(fra67), cola=0
    spending: annual=60000
    allocation: stock_pct=60, prefer_tda_before_taxable=false
    tax: enabled, mfj, state_tax_rate=0.05, state_exempt_retirement=true, inheritor_rate=0.35
    guardrails: enabled, lower=0.75, upper=0.90, target=0.85, max_spending_pct=50
    returns: mode=historical, fee=0bps
    pensions: none
    annuities: none
    rmd_start: 73 both
    num_sims: 1000

    Returns JSON summary with success rate, median outcomes, and file paths.
    Plan JSON and PDF report are automatically saved to ~/RWM/Headless Plans/{client}/.
    """
    from headless_runner import run_plan
    from headless_store import save_plan

    plan = json.loads(plan_json)
    result = run_plan(plan, verbose=False)

    # Save plan + PDF
    paths = save_plan(result['plan'], result['pdf_bytes'])

    # Build summary
    m = next((r for r in result['percentile_rows'] if r['percentile'] == 50), {})
    s = next((r for r in result['spending_percentiles'] if r['percentile'] == 50), {})

    summary = {
        'client': result['plan']['client'],
        'success_rate': f"{result['success_rate']:.1%}",
        'median_ending_portfolio': f"${m.get('after_tax_end', 0):,.0f}",
        'median_total_taxes': f"${m.get('total_taxes', 0):,.0f}",
        'median_effective_tax_rate': f"{m.get('effective_tax_rate', 0):.1%}",
        'median_avg_annual_spending': f"${s.get('avg_annual_after_tax_spending', 0):,.0f}",
        'median_total_lifetime_spending': f"${s.get('total_lifetime_after_tax_spending', 0):,.0f}",
        'json_saved': paths['json_path'],
        'pdf_saved': paths['pdf_path'],
    }

    # Add scenario comparison if multi-scenario
    if result.get('multi_scenario_results'):
        scenarios = []
        for sc in result['multi_scenario_results']:
            sc_m = next((r for r in sc['percentile_rows'] if r['percentile'] == 50), {})
            sc_s = next((r for r in sc['spending_percentiles'] if r['percentile'] == 50), {})
            scenarios.append({
                'name': sc['name'],
                'success_rate': f"{(1 - sc['pct_non_positive']):.1%}",
                'median_ending_portfolio': f"${sc_m.get('after_tax_end', 0):,.0f}",
                'median_avg_annual_spending': f"${sc_s.get('avg_annual_after_tax_spending', 0):,.0f}",
            })
        summary['scenarios'] = scenarios

    return json.dumps(summary, indent=2)


@mcp.tool()
def load_client_plan(client_name: str) -> str:
    """Load an existing client plan by name. Use fuzzy matching.

    Returns the full plan JSON so you can review inputs, show them to the user,
    modify them, and pass back to run_retirement_plan.

    Examples:
      load_client_plan("Jones")
      load_client_plan("Thompson, Mike")
      load_client_plan("Jones, Bob & Mary")
    """
    from headless_store import load_plan, find_client

    matches = find_client(client_name)
    if not matches:
        return json.dumps({'error': f'No client found matching "{client_name}"'})

    plan = load_plan(matches[0])
    plan.pop('_meta', None)
    return json.dumps(plan, indent=2)


@mcp.tool()
def list_retirement_clients() -> str:
    """List all saved retirement client plans.

    Returns client names and their saved plan files with dates.
    Use this when the user asks to see their clients or find a plan.
    """
    from headless_store import list_clients, list_plans

    clients = list_clients()
    if not clients:
        return json.dumps({'clients': [], 'message': 'No saved clients yet.'})

    result = []
    for c in clients:
        plans = list_plans(c)
        result.append({
            'client': c,
            'plans': [{'name': p['name'], 'modified': p['modified']} for p in plans],
        })
    return json.dumps(result, indent=2)


@mcp.tool()
def update_client_plan(client_name: str, changes_json: str) -> str:
    """Load an existing client plan, apply changes, re-run simulation, and save.

    client_name: fuzzy matched (e.g., "Jones" or "Thompson")
    changes_json: JSON string with ONLY the fields to change.

    Examples:
      update_client_plan("Jones", '{"allocation": {"stock_pct": 80}}')
      update_client_plan("Thompson", '{"spending": {"annual": 75000}}')
      update_client_plan("Jones", '{"scenarios": [{"name": "More stocks", "stock_pct": 80}]}')

    The changes are deep-merged into the existing plan, so you only need
    to specify what's different. Everything else stays the same.

    Returns JSON summary with results and file paths.
    """
    from headless_runner import run_plan
    from headless_store import load_plan, save_plan, find_client
    from plan_schema import _deep_merge

    matches = find_client(client_name)
    if not matches:
        return json.dumps({'error': f'No client found matching "{client_name}"'})

    plan = load_plan(matches[0])
    plan.pop('_meta', None)

    changes = json.loads(changes_json)
    updated_plan = _deep_merge(plan, changes)

    result = run_plan(updated_plan, verbose=False)
    paths = save_plan(result['plan'], result['pdf_bytes'])

    m = next((r for r in result['percentile_rows'] if r['percentile'] == 50), {})
    s = next((r for r in result['spending_percentiles'] if r['percentile'] == 50), {})

    summary = {
        'client': result['plan']['client'],
        'changes_applied': list(changes.keys()),
        'success_rate': f"{result['success_rate']:.1%}",
        'median_ending_portfolio': f"${m.get('after_tax_end', 0):,.0f}",
        'median_avg_annual_spending': f"${s.get('avg_annual_after_tax_spending', 0):,.0f}",
        'json_saved': paths['json_path'],
        'pdf_saved': paths['pdf_path'],
    }

    if result.get('multi_scenario_results'):
        scenarios = []
        for sc in result['multi_scenario_results']:
            sc_m = next((r for r in sc['percentile_rows'] if r['percentile'] == 50), {})
            sc_s = next((r for r in sc['spending_percentiles'] if r['percentile'] == 50), {})
            scenarios.append({
                'name': sc['name'],
                'success_rate': f"{(1 - sc['pct_non_positive']):.1%}",
                'median_ending_portfolio': f"${sc_m.get('after_tax_end', 0):,.0f}",
                'median_avg_annual_spending': f"${sc_s.get('avg_annual_after_tax_spending', 0):,.0f}",
            })
        summary['scenarios'] = scenarios

    return json.dumps(summary, indent=2)


if __name__ == "__main__":
    if _args.http:
        mcp.run(transport="sse")
    else:
        mcp.run()
