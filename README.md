# Monte Carlo Retirement Withdrawal Simulator

A Streamlit-based Monte Carlo simulator for retirement withdrawal planning. Simulates multi-account (taxable, TDA, Roth) withdrawals with full tax modeling, RMDs, dynamic guardrails, and scenario comparison.

## How to Run

```
streamlit run main.py
```

App opens at http://localhost:8501

## Stack

- Python 3.13, Streamlit, Pandas, NumPy, Altair, openpyxl
- Virtual environment in `mcproj/`
- Data files: `master_global_factors.xlsx`, `median_cpi_purchasing_power.xlsx`, `cpi_mo_factors.xlsx`

## Client Plan Storage

Client plans are stored **outside** the project folder at:

```
~/RWM/Current Client Plans/
```

### Folder structure

Each client gets their own subfolder named **Last, First**. If two clients share the same name, add an identifier in parentheses to tell them apart:

```
~/RWM/Current Client Plans/
  Smith, John/
    smith_john.json
    smith_john_2.json
    audit.log
    backups/
      smith_john.2026-02-23_1045.bak.json
  Smith, John (Portland)/
    smith_john_portland.json
    audit.log
  Black, Bill/
    black_bill.json
    audit.log
```

### How to create or select a client

In the sidebar under **Save / Load Inputs**:

1. Choose **-- New Client --** from the dropdown (or select an existing client).
2. Enter the client's **Last name** and **First name**.
3. Optionally enter an **Identifier** (e.g. "Portland") if needed to distinguish from another client with the same name.
4. A **Plan name** is auto-generated (e.g. `smith_john`). You can edit it before saving. If the name already exists, it auto-increments (`smith_john_2`, `smith_john_3`, etc.).

### Backups

Every time you save over an existing plan file, the previous version is automatically copied to a `backups/` subfolder inside the client's folder before it gets overwritten. Backup files are timestamped:

```
backups/smith_john.2026-02-23_1045.bak.json
```

This means you can always recover a previous version of any plan.

### Audit Log

Every save creates an entry in `audit.log` inside the client's folder. Each line records:

- **Timestamp** of the save
- **Filename** that was saved
- **Action** — `save` (new file) or `overwrite` (existing file)
- **What changed** — lists every field that differs from the previous version with old and new values

Example log entries:

```
[2026-02-23 10:45:12] save smith_john.json | new file
[2026-02-23 14:30:05] overwrite smith_john.json | target_stock_pct: 60 -> 70; ss_income: 25000.0 -> 30000.0
[2026-02-23 16:00:00] overwrite smith_john.json | no changes
```

## Save / Load Inputs vs Save Scenario

These are two **separate** features that do different things:

| Feature | Location | What it saves | Persists? |
|---|---|---|---|
| **Save / Load Inputs** | Sidebar expander | All sidebar settings (ages, balances, tax config, etc.) | Yes — written to a JSON file on disk |
| **Save Scenario** | Bottom of main page | Simulation results (percentiles, ending balances, tax stats) | No — stored in browser memory, lost when tab closes |

**Save / Load Inputs** is for preserving a client's plan configuration so you can come back to it later. **Save Scenario** is for comparing multiple simulation runs side-by-side during a single working session.

## Key Design Decision

All returns are real (net of inflation). Tax brackets, standard deductions, NIIT thresholds, and other nominal tax parameters do not need inflation indexing — all cash flows are in constant real dollars.
