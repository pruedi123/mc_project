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
