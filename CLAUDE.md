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
