# CLAUDE.md

## Project Overview

Monte Carlo retirement withdrawal simulator built with Streamlit + Python. Simulates multi-account (taxable, TDA, Roth) withdrawals with full tax modeling, RMDs, dynamic guardrails, and scenario comparison. The main app is run with `streamlit run main.py`. When asked to run the app, use `streamlit run main.py` immediately.

## Problem Solving

When fixing bugs or implementing features, prefer the simplest approach using existing data/functions before creating new logic. Do not overcomplicate solutions with new abstractions when existing values can be used directly.

## UI & Visual Work

- Always ask clarifying questions before starting implementation when the request involves UI layout, visual design, or chart styling. Confirm the specific look/behavior expected rather than guessing.
- Do not remove existing working UI elements unless explicitly asked to. When refactoring or adding features, preserve all current outputs (cards, tables, charts) by default.

## Workflow

When a session starts or after /clear, read the project structure first before asking the user which files to look at. Key files are in the project root directory.

## Key Design Decisions

- **All returns are real (net of inflation).** Stock returns, bond returns, and all growth factors are expressed in real terms. This means tax brackets, standard deductions, NIIT thresholds, and other nominal tax parameters do NOT need inflation indexing — they are correct as stated in current-year dollars because all cash flows are in constant real dollars.

## Run

```
streamlit run main.py
```

## Stack

- Python 3.13, Streamlit, Pandas, NumPy, Altair, openpyxl, fpdf2, matplotlib
- Virtual environment in `mcproj/`
- Data files: `master_global_factors.xlsx`, `median_cpi_purchasing_power.xlsx`, `cpi_mo_factors.xlsx`

## Client Privacy & Pseudonymization — MANDATORY

**Claude must NEVER see, read, print, display, or access real client names.** This is a non-negotiable requirement. All client interaction uses pseudonymized aliases.

### Absolute prohibitions
- NEVER read `~/RWM/.client_aliases.json` directly
- NEVER run `pseudonymize.py list --reveal` — that flag is for the user only
- NEVER print, display, or capture output from `resolve_alias()` — it may only be called inside Python scripts where the return value is used programmatically and never echoed to stdout
- NEVER read client folder names from `~/RWM/Current Client Plans/` — folder names contain real client names
- NEVER call `save_plan()` directly with an alias as the client name — always use `save_pseudonymized_plan(alias, plan)` which resolves the real name internally
- NEVER list, glob, or enumerate files inside `~/RWM/Current Client Plans/` — file names contain real client names
- If a real client name accidentally appears in any tool output, do NOT repeat it — immediately move on

### How pseudonymization works
- `pseudonymize.py` maintains a local alias map (`~/RWM/.client_aliases.json`) that maps codenames to real client names
- The alias map and real names stay 100% local on the user's machine
- Claude only ever sees the alias (e.g., "Teal-Helm") and financial data (ages, balances, etc.)
- The user manages aliases through the **Alias Manager** Streamlit app (`alias_manager.py`) or CLI commands run locally

### Safe commands Claude CAN run
- `python pseudonymize.py list` — shows aliases only, no real names
- `python pseudonymize.py export ALIAS` — loads plan with alias replacing real name
- `save_pseudonymized_plan(alias, plan)` — saves plan under real name without exposing it
- Running simulations via `run_full_process.py` or `headless_runner.py` using temp files (e.g., `/tmp/`) that contain only the alias

## How the user prompts for simulations

- **"Run full process for [alias]"** — runs `run_full_process.py` with all 6 phases: auto-spending, spending finder at 90% success, percentile distribution, Sept 1929 worst-case drill-down, balance decline finder, and stressed spending.
- **"Run a quick sim for [alias]"** — runs `headless_runner.py` for a basic simulation with success rate, median ending portfolio, and PDF report. No spending search or stress tests.
- **Comfort Level threshold** defaults to 80% of the essential floor. The user can override this by saying "use 90% comfort level" (or any percentage). Pass as `--shortfall-pct 90` to `run_full_process.py`. Also update the HTML report's comfort level calculation to match (multiply `found_min` by the user's chosen percentage instead of hardcoded 0.80).
- New client flow: user runs `pseudonymize.py add` locally to get alias, then dictates inputs using the alias, then says "run full process" or "run a quick sim."
- **Person labels:** When creating a new plan, always include first names for person1 and person2 (e.g., `"label": "Bob"`). First names are safe to use — they are not identifying on their own. If the user doesn't provide first names, ask for them. These labels appear in the Alias Manager so the user can tell the spouses apart when reviewing plans later.

## Client-Facing Scenario Comparison Write-Up

**Whenever comparing two or more scenarios (e.g., different equity allocations), ALWAYS include a plain-language write-up after the data tables.** This is not optional. The write-up should:

1. **Be written for someone with no financial background** — 8th grade reading level, no jargon. Never use terms like "standard deviation," "percentile," or "Monte Carlo" in the client write-up.

2. **Frame each option around the person's flexibility and temperament**, not abstract risk tolerance:
   - Lower equity = values certainty and predictability, willing to accept less in exchange for a steadier ride
   - Middle equity = has some flexibility in spending, can adjust when needed
   - Higher equity = prioritizes legacy or has genuine ability to adapt through prolonged difficult periods

3. **Use real dollar amounts from the simulation** to make the worst case tangible. Don't say "spending could decline." Say "your spending would drop from $100,000 to $55,000 for several years."

4. **For worst-case narratives, always distinguish between the single worst year and the average over the full period.** The worst single year is a temporary dip — one tough year. The average annual spending across the full retirement (even in the worst historical window) is typically much higher and is what the client actually experiences over time. Always present both numbers clearly: "In one bad year, spending dipped to $X — but averaged across the full 30 years, even in the worst historical stretch, you still averaged $Y per year." For higher-equity options, also pull the cumulative market decline from the Sept 1929 window and describe the multi-year nature of the downturn.

5. **Name the real danger clearly:** The risk isn't that the plan fails on paper. The risk is that the person panics during a prolonged downturn and sells at the bottom, locking in losses permanently. The question is always: "If this happened, would you stick with the plan?"

6. **Frame the final recommendation around the tradeoff curve:**
   - Where does each additional dollar of spending come from (more equity)?
   - What does it cost in terms of worst-case experience?
   - Who actually benefits — the retiree (spending) or the heirs (ending balance)?

7. **End with a clear recommendation and a one-sentence reason.** Don't hedge. Pick the option that fits most people and explain why.

8. **When any scenario shows $0 ending balance**, always include the "gas tank" explanation: $0 doesn't mean the plan failed. The guardrail system managed spending so that essential needs were always met — every single year. The portfolio was fully consumed, crossing the finish line on fumes, but it did its job. Explain that $0 ending means the money worked for *the retiree* but left nothing for the next generation. If legacy matters, a higher stock allocation gives the portfolio room to both fund spending and leave something behind. Use the gas tank / cruise control analogy: the system slows down spending when the tank is low so you never run out on the side of the road with years left to go.

9. **The "Dutch Uncle" close — MANDATORY for every multi-scenario comparison.** After describing each option, the conclusion MUST confront the client directly with the worst-case spending numbers from the higher-equity scenarios. Use this structure:
   - Pull the exact worst single-year spending from each scenario being compared
   - For higher-equity options, also note how many consecutive years spending stays depressed (from the 1929 drill-down)
   - Dismiss the "small probability" comfort blanket directly: a 5th percentile event happens to 1 out of every 20 people. Somebody sits in that chair. It could be you.
   - Ask the question personally and specifically with real dollar amounts: "Could you live on $42,000 for a year? Could you do it for three years in a row? Could you watch your portfolio drop nearly 50% and not pick up the phone to sell everything?"
   - Make clear that "I think so" or "probably" isn't good enough — the answer needs to be truly yes
   - Contrast the two options in human terms: "Five thousand dollars apart — but the experience of living through it is a world apart"
   - Close with: "A little flexibility goes a long way. A lot of flexibility helps less than you'd think and asks more than most people expect."

## Client Plan Storage

Client plans are stored outside the project repo at:

```
~/RWM/Current Client Plans/
```

Each client gets their own subfolder named `Last, First` (or `Last, First (identifier)` for disambiguation):

```
~/RWM/Current Client Plans/
  Smith, John/
    smith_john.json
    smith_john_2.json
  Smith, John (Portland)/
    smith_john_portland.json
  Black, Bill/
    black_bill.json
```

## Save / Load Inputs vs Save Scenario

These are two separate features:

- **Save / Load Inputs** (top of sidebar) — Saves all sidebar input settings to a JSON file in the client's folder under `~/RWM/Current Client Plans/`. Persists across browser sessions. No simulation needed. Use this to save and reload plan configurations.
- **Save Scenario** (bottom of main page) — Saves simulation results (percentiles, ending balances, tax stats) to in-memory session state. Lost when the browser tab closes. Used to compare results side-by-side within one session.

### Save/Load implementation details

- Every sidebar widget has an explicit `key=` parameter so `st.session_state` can address it by name.
- `_SAVEABLE_KEYS` in `main.py` lists all widget keys that get saved/loaded.
- `_save_inputs_to_json(client, name)` collects widget values from session state + dynamic withdrawal period keys (`wd_amount_N`, `wd_end_N`) + scenario override keys, writes to `~/RWM/Current Client Plans/{client}/{name}.json`.
- `_load_inputs_from_json(client, name)` reads JSON, sets values into `st.session_state`, then `st.rerun()` causes all widgets to pick up the restored values.
- JSON files are flat dicts with two nested sections: `periods` (withdrawal schedule) and `scenario_overrides` (multi-scenario UI keys).
- Plan names are auto-generated from the client name (e.g. `smith_john`), with auto-incrementing (`smith_john_2`) when duplicates exist.

## Client PDF Report

- **Generate Client Report** button appears at the bottom of the results page after running a simulation.
- Uses `pdf_report.py` module (fpdf2 + matplotlib). No external services needed.
- Produces a 3–4 page PDF: (1) Big Picture with success rate and assumptions, (2) Range of Outcomes with percentile table and portfolio bands chart, (3) Year-by-year median income sources table and stacked chart, (4) Scenario Comparison (only if multiple scenarios were run).
- All charts rendered via matplotlib to PNG, embedded in PDF. No Altair/browser dependency.
- Entry point: `pdf_report.generate_report(dict(st.session_state)) -> bytes`
