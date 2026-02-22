# CLAUDE.md

## Project Overview

Monte Carlo retirement withdrawal simulator built with Streamlit + Python. Simulates multi-account (taxable, TDA, Roth) withdrawals with full tax modeling, RMDs, dynamic guardrails, and scenario comparison.

## Key Design Decisions

- **All returns are real (net of inflation).** Stock returns, bond returns, and all growth factors are expressed in real terms. This means tax brackets, standard deductions, NIIT thresholds, and other nominal tax parameters do NOT need inflation indexing — they are correct as stated in current-year dollars because all cash flows are in constant real dollars.

## Run

```
streamlit run main.py
```

## Stack

- Python 3.13, Streamlit, Pandas, NumPy, Altair, openpyxl
- Virtual environment in `mcproj/`
- Data files: `master_global_factors.xlsx`, `median_cpi_purchasing_power.xlsx`, `cpi_mo_factors.xlsx`
- Saved input configurations: `saves/*.json`

## Save / Load Inputs vs Save Scenario

These are two separate features:

- **Save / Load Inputs** (top of sidebar) — Saves all sidebar input settings to a JSON file in `saves/`. Persists across browser sessions. No simulation needed. Use this to save and reload plan configurations (e.g. `Smiths_baseline.json`).
- **Save Scenario** (bottom of main page) — Saves simulation results (percentiles, ending balances, tax stats) to in-memory session state. Lost when the browser tab closes. Used to compare results side-by-side within one session.

### Save/Load implementation details

- Every sidebar widget has an explicit `key=` parameter so `st.session_state` can address it by name.
- `_SAVEABLE_KEYS` in `main.py` lists all widget keys that get saved/loaded.
- `_save_inputs_to_json()` collects widget values from session state + dynamic withdrawal period keys (`wd_amount_N`, `wd_end_N`) + scenario override keys, writes to `saves/{name}.json`.
- `_load_inputs_from_json()` reads JSON, sets values into `st.session_state`, then `st.rerun()` causes all widgets to pick up the restored values.
- JSON files are flat dicts with two nested sections: `periods` (withdrawal schedule) and `scenario_overrides` (multi-scenario UI keys).
