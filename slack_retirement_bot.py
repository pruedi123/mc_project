"""Slack bot for running retirement simulations.

Listens to a Slack channel. When a message describes a client plan
(new or update), uses Claude API to parse it into plan JSON, runs
the simulation, emails the PDF, and posts results back to Slack.

Usage:
    export SLACK_BOT_TOKEN=xoxb-...
    export SLACK_APP_TOKEN=xapp-...
    export ANTHROPIC_API_KEY=sk-ant-...
    export RETIREMENT_EMAIL=your@email.com
    export SMTP_PASSWORD=your-gmail-app-password

    python slack_retirement_bot.py
"""

import os
import sys
import json
import smtplib
import logging
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders

sys.path.insert(0, os.path.dirname(__file__))

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
import anthropic

from headless_runner import run_plan
from headless_store import save_plan, load_plan, find_client, list_clients
from plan_schema import apply_defaults, _deep_merge, PLAN_DEFAULTS

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────

SLACK_BOT_TOKEN = os.environ.get('SLACK_BOT_TOKEN', '')
SLACK_APP_TOKEN = os.environ.get('SLACK_APP_TOKEN', '')
ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY', '')
RETIREMENT_EMAIL = os.environ.get('RETIREMENT_EMAIL', '')
SMTP_PASSWORD = os.environ.get('SMTP_PASSWORD', '')
SMTP_FROM = os.environ.get('SMTP_FROM', RETIREMENT_EMAIL)

CHANNEL_NAME = os.environ.get('RETIREMENT_CHANNEL', 'retirement-plans')

app = App(token=SLACK_BOT_TOKEN)
claude = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


# ── Claude NLP parser ─────────────────────────────────────────

PARSE_SYSTEM_PROMPT = """You are a retirement plan input parser. Convert natural language client descriptions into JSON plan format.

OUTPUT FORMAT: Return ONLY valid JSON, no explanation, no markdown code fences.

RULES:
- Client name format: "Last, First & Spouse" (e.g., "Thompson, Mike & Linda")
- Ages go in person1/person2 objects
- Account balances in accounts object (taxable, tda_p1, tda_p2, roth)
- Social Security in social_security object with person1/person2 sub-objects (benefit, start_age, fra)
- Spending in spending.annual
- Allocation in allocation.stock_pct (0-100)
- Only include fields that differ from defaults

DEFAULTS (omit these if they match):
- life_expectancy: person1=84, person2=89
- accounts: taxable=300000, tda_p1=400000, tda_p2=300000, roth=0
- SS: person1: benefit=25000, start_age=67, fra=67; person2: benefit=20000, start_age=65, fra=67
- spending: annual=60000
- allocation: stock_pct=60
- guardrails: enabled=true, max_spending_pct=50
- tax: filing_status=mfj, state_tax_rate=0.05, state_exempt_retirement=true
- pensions/annuities: none

FOR UPDATES to existing plans:
If the user says something like "change Thompson allocation to 80%" or "update Jones spending to 70k",
return JSON with:
  {"_action": "update", "_client": "Thompson", "allocation": {"stock_pct": 80}}

FOR NEW PLANS:
Return the full plan JSON with client name and all specified fields.
  {"client": "Thompson, Mike & Linda", "person1": {"age": 63}, ...}

FOR SCENARIO REQUESTS:
If the user wants to compare options, add a scenarios list:
  {"_action": "update", "_client": "Jones", "scenarios": [{"name": "80% Stocks", "stock_pct": 80}]}

FOR LOADING/VIEWING existing plans:
If the user says "show me the Thompson plan", "load Kranks", "get Thompson results", "what does the Jones plan look like", etc.:
  {"_action": "load", "_client": "Thompson"}

FOR RE-RUNNING an existing plan:
If the user says "run the Kranks plan again", "rerun Thompson":
  {"_action": "rerun", "_client": "Kranks"}

FOR GOALS (additional spending on top of base withdrawal):
Goals have: label, amount (annual), begin (year number), end (year number), priority ("Essential" or "Flexible").
Essential goals are never reduced by guardrails. Flexible goals can be reduced in bad markets.
Example - adding goals to a new or existing plan:
  "goals": [
    {"label": "Long-term care", "amount": 100000, "begin": 28, "end": 30, "priority": "Flexible"},
    {"label": "New roof", "amount": 30000, "begin": 5, "end": 5, "priority": "Essential"}
  ]
When the user says "add a long-term care goal of $100k from years 28-30 to the Kranks plan":
  {"_action": "update", "_client": "Kranks", "goals": [{"label": "Long-term care", "amount": 100000, "begin": 28, "end": 30, "priority": "Flexible"}]}

If the message is not about a retirement plan (greeting, question, etc.), return:
  {"_action": "ignore"}
"""


def parse_with_claude(text: str) -> dict:
    """Use Claude to parse natural language into plan JSON."""
    # Include existing clients for context
    clients = list_clients()
    client_list = ", ".join(clients) if clients else "none"

    response = claude.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        system=PARSE_SYSTEM_PROMPT + f"\n\nExisting clients: {client_list}",
        messages=[{"role": "user", "content": text}],
    )
    raw = response.content[0].text.strip()
    # Strip markdown code fences if present
    if raw.startswith('```'):
        raw = raw.split('\n', 1)[1]
        if raw.endswith('```'):
            raw = raw[:-3]
        raw = raw.strip()
    return json.loads(raw)


# ── Email sender ──────────────────────────────────────────────

def send_email(pdf_bytes: bytes, client_name: str, summary: dict, recipient: str):
    """Send PDF report as email attachment via Gmail SMTP."""
    if not recipient or not SMTP_PASSWORD:
        logger.warning("Email not configured — skipping")
        return False

    msg = MIMEMultipart()
    msg['From'] = SMTP_FROM
    msg['To'] = recipient
    msg['Subject'] = f'Retirement Plan Report: {client_name}'

    body = f"""Retirement Plan Report: {client_name}

Success Rate: {summary.get('success_rate', 'N/A')}
Median Ending Portfolio: {summary.get('median_ending_portfolio', 'N/A')}
Median Avg Annual Spending: {summary.get('median_avg_annual_spending', 'N/A')}
Median Total Taxes: {summary.get('median_total_taxes', 'N/A')}
Effective Tax Rate: {summary.get('median_effective_tax_rate', 'N/A')}

PDF report attached.
"""
    msg.attach(MIMEText(body, 'plain'))

    # Attach PDF
    part = MIMEBase('application', 'pdf')
    part.set_payload(pdf_bytes)
    encoders.encode_base64(part)
    safe_name = client_name.replace(',', '').replace(' ', '_').lower()
    part.add_header('Content-Disposition', f'attachment; filename="{safe_name}_report.pdf"')
    msg.attach(part)

    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(SMTP_FROM, SMTP_PASSWORD)
            server.send_message(msg)
        logger.info(f"Email sent to {recipient}")
        return True
    except Exception as e:
        logger.error(f"Email failed: {e}")
        return False


# ── Result formatting ─────────────────────────────────────────

def format_slack_results(summary: dict) -> str:
    """Format results for Slack message."""
    lines = [
        f"*Retirement Plan: {summary['client']}*",
        "",
        f"Success Rate: *{summary['success_rate']}*",
        f"Median Ending Portfolio: *{summary['median_ending_portfolio']}*",
        f"Median Annual Spending: *{summary['median_avg_annual_spending']}*/yr",
        f"Median Total Taxes: {summary['median_total_taxes']}",
        f"Effective Tax Rate: {summary['median_effective_tax_rate']}",
        f"Total Lifetime Spending: {summary['median_total_lifetime_spending']}",
    ]

    if summary.get('scenarios'):
        # Find baseline (first scenario or the main result)
        baseline = summary['scenarios'][0] if summary['scenarios'] else None
        baseline_spending = baseline.get('median_base_spending_raw', 0) if baseline else 0
        baseline_ending = baseline.get('median_ending_raw', 0) if baseline else 0

        lines.append("")
        lines.append("*Scenario Comparison:*")
        for sc in summary['scenarios']:
            lines.append(f"")
            lines.append(f"  *{sc['name']}*")
            lines.append(f"  Success Rate: {sc['success_rate']}")
            lines.append(f"  Annual Spending: {sc['median_avg_annual_spending']}/yr")
            lines.append(f"  Ending Portfolio: {sc['median_ending_portfolio']}")

            # Show cost of goals compared to baseline
            if sc != baseline and baseline:
                sc_spending_raw = sc.get('median_base_spending_raw', sc.get('median_spending_raw', 0))
                sc_ending_raw = sc.get('median_ending_raw', 0)

                if baseline_spending > 0 and sc_spending_raw > 0:
                    spending_diff = sc_spending_raw - baseline_spending
                    ending_diff = sc_ending_raw - baseline_ending

                    if sc.get('goal_description'):
                        lines.append(f"")
                        lines.append(f"  *Cost of {sc['goal_description']}:*")
                    else:
                        lines.append(f"")
                        lines.append(f"  *Impact vs Baseline:*")

                    if spending_diff < 0:
                        lines.append(f"  Annual spending reduced by ${abs(spending_diff):,.0f}/yr")
                    elif spending_diff > 0:
                        lines.append(f"  Annual spending increased by ${spending_diff:,.0f}/yr")

                    if ending_diff < 0:
                        lines.append(f"  Ending portfolio reduced by ${abs(ending_diff):,.0f}")
                    elif ending_diff > 0:
                        lines.append(f"  Ending portfolio increased by ${ending_diff:,.0f}")

    lines.append("")
    lines.append(f"_PDF saved: {summary.get('pdf_saved', 'N/A')}_")
    return "\n".join(lines)


# ── Slack event handlers ──────────────────────────────────────

def _process_message(text, channel, thread_ts, say_fn, slack_client):
    """Process a retirement plan message in a background thread."""
    try:
        # Parse with Claude
        logger.info(f"Parsing message: {text[:100]}...")
        parsed = parse_with_claude(text)
        logger.info(f"Parsed result: {json.dumps(parsed, indent=2)}")

        if parsed.get('_action') == 'ignore':
            return

        if parsed.get('_action') == 'load':
            # Load and display existing plan
            client_name = parsed.get('_client', '')
            matches = find_client(client_name)
            if not matches:
                slack_client.chat_postMessage(channel=channel, text=f"No client found matching '{client_name}'", thread_ts=thread_ts)
                return

            plan = load_plan(matches[0])
            plan.pop('_meta', None)

            # Format plan summary for Slack
            p1 = plan.get('person1', {})
            p2 = plan.get('person2', {})
            accts = plan.get('accounts', {})
            ss = plan.get('social_security', {})
            spending = plan.get('spending', {})
            alloc = plan.get('allocation', {})

            pensions = plan.get('pensions', {})
            annuities = plan.get('annuities', {})
            tax = plan.get('tax', {})
            gr = plan.get('guardrails', {})
            ret = plan.get('returns', {})
            goals = plan.get('goals') or []
            earned = plan.get('earned_income', {})

            lines = [
                f"*{matches[0]}*",
                "",
                "*Demographics:*",
                f"  Person 1: age {p1.get('age', '?')}, life expectancy {p1.get('life_expectancy', '?')}",
                f"  Person 2: age {p2.get('age', '?')}, life expectancy {p2.get('life_expectancy', '?')}",
                "",
                "*Accounts:*",
                f"  Taxable: ${accts.get('taxable', 0):,.0f} (basis: {accts.get('taxable_stock_basis_pct', 50)}% stock, {accts.get('taxable_bond_basis_pct', 100)}% bond)",
                f"  TDA P1: ${accts.get('tda_p1', 0):,.0f}",
                f"  TDA P2: ${accts.get('tda_p2', 0):,.0f}",
                f"  Roth: ${accts.get('roth', 0):,.0f}",
                "",
                "*Income:*",
                f"  SS P1: ${ss.get('person1', {}).get('benefit', 0):,.0f}/yr at age {ss.get('person1', {}).get('start_age', '?')} (FRA {ss.get('person1', {}).get('fra', 67)})",
                f"  SS P2: ${ss.get('person2', {}).get('benefit', 0):,.0f}/yr at age {ss.get('person2', {}).get('start_age', '?')} (FRA {ss.get('person2', {}).get('fra', 67)})",
                f"  SS COLA: {ss.get('cola', 0):.1%}" if ss.get('cola', 0) > 0 else None,
            ]
            pen1 = pensions.get('person1', {}).get('income', 0)
            pen2 = pensions.get('person2', {}).get('income', 0)
            if pen1 > 0:
                lines.append(f"  Pension P1: ${pen1:,.0f}/yr (COLA {pensions['person1'].get('cola', 0):.1%}, survivor {pensions['person1'].get('survivor_pct', 0):.0%})")
            if pen2 > 0:
                lines.append(f"  Pension P2: ${pen2:,.0f}/yr (COLA {pensions['person2'].get('cola', 0):.1%}, survivor {pensions['person2'].get('survivor_pct', 0):.0%})")
            ann1 = annuities.get('person1', {}).get('income', 0)
            ann2 = annuities.get('person2', {}).get('income', 0)
            if ann1 > 0:
                lines.append(f"  Annuity P1: ${ann1:,.0f}/yr (COLA {annuities['person1'].get('cola', 0):.1%}, survivor {annuities['person1'].get('survivor_pct', 0):.0%})")
            if ann2 > 0:
                lines.append(f"  Annuity P2: ${ann2:,.0f}/yr (COLA {annuities['person2'].get('cola', 0):.1%}, survivor {annuities['person2'].get('survivor_pct', 0):.0%})")
            other = plan.get('other_income', 0)
            if other > 0:
                lines.append(f"  Other income: ${other:,.0f}/yr")
            earned_amt = earned.get('annual', 0) if isinstance(earned, dict) else 0
            if earned_amt > 0:
                lines.append(f"  Earned income: ${earned_amt:,.0f}/yr for {earned.get('years', 0)} years")

            lines.extend([
                "",
                "*Spending & Allocation:*",
                f"  Spending: ${spending.get('annual', 0):,.0f}/yr",
                f"  Allocation: {alloc.get('stock_pct', 60)}% stocks / {100 - alloc.get('stock_pct', 60)}% bonds",
            ])

            if goals:
                lines.append("")
                lines.append("*Additional Goals:*")
                for g in goals:
                    lines.append(f"  {g.get('label', '?')}: ${g.get('amount', 0):,.0f}/yr, years {g.get('begin', '?')}-{g.get('end', '?')} ({g.get('priority', '?')})")

            filing = tax.get('filing_status', 'mfj')
            filing_display = 'Married Filing Jointly' if filing == 'mfj' else 'Single'
            lines.extend([
                "",
                "*Tax:*",
                f"  Filing: {filing_display}",
                f"  State tax: {tax.get('state_tax_rate', 0.05):.1%}",
                f"  Retirement income exempt: {'Yes' if tax.get('state_exempt_retirement', True) else 'No'}",
                f"  Inheritor marginal rate: {tax.get('inheritor_marginal_rate', 0.35):.0%}",
                "",
                "*Guardrails:*",
                f"  Enabled: {'Yes' if gr.get('enabled', True) else 'No'}",
                f"  Lower/Upper/Target: {gr.get('lower', 0.75):.0%} / {gr.get('upper', 0.90):.0%} / {gr.get('target', 0.85):.0%}",
                f"  Spending cap: {gr.get('max_spending_pct', 50):.0f}% above base",
                "",
                "*Returns:*",
                f"  Mode: {'Historical' if ret.get('mode', 'historical') == 'historical' else 'Lognormal (Monte Carlo)'}",
                f"  Investment fees: {ret.get('investment_fee_bps', 0):.0f} bps",
            ])
            lines = [l for l in lines if l is not None]
            slack_client.chat_postMessage(channel=channel, text="\n".join(lines), thread_ts=thread_ts)
            return

        if parsed.get('_action') == 'rerun':
            # Re-run existing plan
            client_name = parsed.get('_client', '')
            matches = find_client(client_name)
            if not matches:
                slack_client.chat_postMessage(channel=channel, text=f"No client found matching '{client_name}'", thread_ts=thread_ts)
                return

            slack_client.chat_postMessage(channel=channel, text=f"Re-running plan for *{matches[0]}*... :hourglass:", thread_ts=thread_ts)

            existing = load_plan(matches[0])
            existing.pop('_meta', None)
            result = run_plan(existing, verbose=False)
            paths = save_plan(result['plan'], result['pdf_bytes'])

        elif parsed.get('_action') == 'update':
            # Update existing client
            client_name = parsed.pop('_client', '')
            parsed.pop('_action', None)

            matches = find_client(client_name)
            if not matches:
                slack_client.chat_postMessage(channel=channel, text=f"No client found matching '{client_name}'", thread_ts=thread_ts)
                return

            slack_client.chat_postMessage(channel=channel, text=f"Updating plan for *{matches[0]}*... :hourglass:", thread_ts=thread_ts)

            existing = load_plan(matches[0])
            existing.pop('_meta', None)
            updated = _deep_merge(existing, parsed)

            result = run_plan(updated, verbose=False)
            paths = save_plan(result['plan'], result['pdf_bytes'])

        else:
            # New client plan
            client_name = parsed.get('client', 'Unknown Client')
            slack_client.chat_postMessage(channel=channel, text=f"Running simulation for *{client_name}*... :hourglass:", thread_ts=thread_ts)

            result = run_plan(parsed, verbose=False)
            paths = save_plan(result['plan'], result['pdf_bytes'])

        # Build summary
        m = next((r for r in result['percentile_rows'] if r['percentile'] == 50), {})
        s = next((r for r in result['spending_percentiles'] if r['percentile'] == 50), {})
        summary = {
            'client': result['plan']['client'],
            'success_rate': f"{result['success_rate']:.1%}",
            'median_ending_portfolio': f"${m.get('after_tax_end', 0):,.0f}",
            'median_total_taxes': f"${m.get('total_taxes', 0):,.0f}",
            'median_effective_tax_rate': f"${m.get('effective_tax_rate', 0):.1%}",
            'median_avg_annual_spending': f"${s.get('avg_annual_after_tax_spending', 0):,.0f}",
            'median_total_lifetime_spending': f"${s.get('total_lifetime_after_tax_spending', 0):,.0f}",
            'median_base_spending': f"${s.get('avg_base_spending', 0):,.0f}" if 'avg_base_spending' in s else None,
            'median_goal_spending': f"${s.get('avg_goal_spending', 0):,.0f}" if 'avg_goal_spending' in s else None,
            'pdf_saved': paths.get('pdf_path', ''),
        }

        if result.get('multi_scenario_results'):
            scenarios = []
            for sc in result['multi_scenario_results']:
                sc_m = next((r for r in sc['percentile_rows'] if r['percentile'] == 50), {})
                sc_s = next((r for r in sc['spending_percentiles'] if r['percentile'] == 50), {})
                sc_entry = {
                    'name': sc['name'],
                    'success_rate': f"{(1 - sc['pct_non_positive']):.1%}",
                    'median_ending_portfolio': f"${sc_m.get('after_tax_end', 0):,.0f}",
                    'median_ending_raw': sc_m.get('after_tax_end', 0),
                    'median_avg_annual_spending': f"${sc_s.get('avg_annual_after_tax_spending', 0):,.0f}",
                    'median_spending_raw': sc_s.get('avg_annual_after_tax_spending', 0),
                }
                if 'avg_base_spending' in sc_s:
                    sc_entry['median_base_spending_raw'] = sc_s['avg_base_spending']
                else:
                    sc_entry['median_base_spending_raw'] = sc_s.get('avg_annual_after_tax_spending', 0)

                # Build goal description from the scenario's plan
                sc_plan_goals = None
                if result.get('plan', {}).get('scenarios'):
                    sc_idx = len(scenarios)  # current index
                    if sc_idx > 0 and sc_idx - 1 < len(result['plan']['scenarios']):
                        sc_def = result['plan']['scenarios'][sc_idx - 1]
                        sc_plan_goals = sc_def.get('goals')
                if sc_plan_goals:
                    goal_descs = []
                    for g in sc_plan_goals:
                        goal_descs.append(f"{g['label']} ${g['amount']:,.0f}/yr yrs {g['begin']}-{g['end']}")
                    sc_entry['goal_description'] = ', '.join(goal_descs)

                scenarios.append(sc_entry)
            summary['scenarios'] = scenarios

        # Post results to Slack
        slack_client.chat_postMessage(channel=channel, text=format_slack_results(summary), thread_ts=thread_ts)

        # Upload PDF to Slack thread
        if paths.get('pdf_path') and os.path.exists(paths['pdf_path']):
            slack_client.files_upload_v2(
                channel=channel,
                file=paths['pdf_path'],
                title=f"{result['plan']['client']} - Retirement Plan Report",
                thread_ts=thread_ts,
            )

        # Email PDF
        if RETIREMENT_EMAIL:
            emailed = send_email(result['pdf_bytes'], result['plan']['client'],
                                summary, RETIREMENT_EMAIL)
            if emailed:
                slack_client.chat_postMessage(channel=channel, text=f"_Report emailed to {RETIREMENT_EMAIL}_", thread_ts=thread_ts)

    except json.JSONDecodeError as e:
        slack_client.chat_postMessage(channel=channel, text=f"Couldn't parse that as a plan input: {e}", thread_ts=thread_ts)
    except Exception as e:
        logger.error(f"Error processing message: {e}", exc_info=True)
        slack_client.chat_postMessage(channel=channel, text=f"Error running simulation: {str(e)}", thread_ts=thread_ts)


@app.event("message")
def handle_message(event, say, client):
    """Handle incoming Slack messages — dispatch to background thread."""
    import threading

    if event.get('bot_id') or event.get('subtype'):
        return

    text = event.get('text', '').strip()
    if not text:
        return

    channel = event.get('channel', '')
    thread_ts = event.get('ts', '')

    t = threading.Thread(target=_process_message, args=(text, channel, thread_ts, say, client))
    t.start()


# ── Main ──────────────────────────────────────────────────────

if __name__ == "__main__":
    missing = []
    if not SLACK_BOT_TOKEN:
        missing.append('SLACK_BOT_TOKEN')
    if not SLACK_APP_TOKEN:
        missing.append('SLACK_APP_TOKEN')
    if not ANTHROPIC_API_KEY:
        missing.append('ANTHROPIC_API_KEY')
    if missing:
        print(f"Missing required environment variables: {', '.join(missing)}")
        print("\nSet them and try again:")
        print("  export SLACK_BOT_TOKEN=xoxb-...")
        print("  export SLACK_APP_TOKEN=xapp-...")
        print("  export ANTHROPIC_API_KEY=sk-ant-...")
        print("  export RETIREMENT_EMAIL=your@email.com  (optional)")
        print("  export SMTP_PASSWORD=your-gmail-app-password  (optional)")
        sys.exit(1)

    if not RETIREMENT_EMAIL:
        logger.warning("RETIREMENT_EMAIL not set — email delivery disabled")
    if not SMTP_PASSWORD:
        logger.warning("SMTP_PASSWORD not set — email delivery disabled")

    logger.info("Starting Retirement Simulator Slack bot...")
    logger.info(f"Listening for messages in any channel the bot is added to")
    handler = SocketModeHandler(app, SLACK_APP_TOKEN)
    handler.start()
