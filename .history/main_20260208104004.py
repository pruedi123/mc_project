import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from typing import Dict, Optional, Sequence

st.set_page_config(page_title='Withdrawal + RMD Simulator', layout='wide')

def interactive_line_chart(data_df, y_title='Value', fmt='$,.0f', height=400):
	"""Convert a DataFrame (index=x, columns=series) to an interactive Altair line chart with tooltips."""
	long = data_df.reset_index().melt(data_df.index.name or 'index', var_name='Series', value_name='value')
	x_col = data_df.index.name or 'index'
	nearest = alt.selection_point(nearest=True, on='pointerover', fields=[x_col], empty=False)
	base = alt.Chart(long).encode(
		x=alt.X(f'{x_col}:Q', title=x_col.replace('_', ' ').title()),
		y=alt.Y('value:Q', title=y_title, axis=alt.Axis(format=fmt)),
		color=alt.Color('Series:N', legend=alt.Legend(title='')),
	)
	lines = base.mark_line()
	points = base.mark_point(size=60, filled=True, opacity=0).encode(
		opacity=alt.condition(nearest, alt.value(1), alt.value(0)),
		tooltip=[alt.Tooltip(f'{x_col}:Q', title=x_col.replace('_', ' ').title())] +
			[alt.Tooltip('Series:N', title='Series'),
			 alt.Tooltip('value:Q', title=y_title, format=fmt)]
	).add_params(nearest)
	rules = alt.Chart(long).mark_rule(color='gray', strokeDash=[4, 4]).encode(
		x=f'{x_col}:Q',
	).transform_filter(nearest)
	chart = (lines + points + rules).properties(height=height).interactive()
	st.altair_chart(chart, use_container_width=True)

@st.cache_data
def load_master_global():
	"""Load the full master_global_factors.xlsx and return the DataFrame."""
	import os
	path = os.path.join(os.path.dirname(__file__), 'master_global_factors.xlsx')
	df = pd.read_excel(path)
	df['begin month'] = pd.to_datetime(df['begin month'])
	return df

@st.cache_data
def load_bond_factors():
	"""Load historical bond growth factors from master_global_factors.xlsx (LBM 100 F column).
	Returns an array of annual returns (growth_factor - 1)."""
	df = load_master_global()
	factors = df['LBM 100 F'].dropna().values
	return factors - 1.0  # convert growth factors to returns

@st.cache_data
def get_historical_annual_returns(start_year: int, years_needed: int):
	"""Extract non-overlapping annual stock (LBM 100E) and bond (LBM 100 F) returns
	starting from January of start_year, stepping every 12 months."""
	df = load_master_global()
	# find the row closest to January of the start year
	target = pd.Timestamp(year=start_year, month=1, day=1)
	idx = (df['begin month'] - target).abs().argmin()
	# step every 12 rows for non-overlapping annual periods
	indices = list(range(idx, len(df), 12))[:years_needed]
	stock_factors = df['LBM 100E'].iloc[indices].values
	bond_factors = df['LBM 100 F'].iloc[indices].values
	return stock_factors - 1.0, bond_factors - 1.0, len(indices)

def sample_bond_returns(years: int, bond_factors: np.ndarray, rng: np.random.Generator) -> np.ndarray:
	"""Randomly sample `years` annual bond returns from historical data (with replacement)."""
	indices = rng.integers(0, len(bond_factors), size=years)
	return bond_factors[indices]

def forward_success_rate(portfolio, remaining_schedule, scale_factor, blended_mu, blended_sigma, n_sims=200):
	"""Fast vectorized MC to estimate probability portfolio survives the remaining schedule.
	remaining_schedule is a list of base withdrawal amounts for each remaining year.
	scale_factor is a multiplier applied to every element of the schedule.
	Uses a simplified single-asset lognormal model (no tax engine)."""
	years_remaining = len(remaining_schedule)
	if years_remaining <= 0:
		return 1.0
	if portfolio <= 0:
		return 0.0
	rng = np.random.default_rng()
	log_returns = rng.normal(loc=blended_mu, scale=blended_sigma, size=(n_sims, years_remaining))
	growth_factors = np.exp(log_returns)
	balances = np.full(n_sims, portfolio, dtype=np.float64)
	for y in range(years_remaining):
		balances *= growth_factors[:, y]
		balances -= remaining_schedule[y] * scale_factor
		balances = np.maximum(balances, 0.0)
	return float(np.mean(balances > 0))

def find_sustainable_scale_factor(portfolio, remaining_schedule, blended_mu, blended_sigma, target_success=0.85, n_sims=200, tol=0.005):
	"""Binary search for the scaling factor on the remaining withdrawal schedule
	that gives target_success survival rate.
	Returns a multiplier (e.g. 1.0 = base schedule, 0.85 = 85% of base, 1.2 = 120% of base).
	Pre-generates one random return matrix and reuses it across all iterations for stability."""
	years_remaining = len(remaining_schedule)
	if portfolio <= 0 or years_remaining <= 0:
		return 0.0
	rng = np.random.default_rng()
	log_returns = rng.normal(loc=blended_mu, scale=blended_sigma, size=(n_sims, years_remaining))
	growth_factors = np.exp(log_returns)

	def check_survival(scale):
		balances = np.full(n_sims, portfolio, dtype=np.float64)
		for y in range(years_remaining):
			balances *= growth_factors[:, y]
			balances -= remaining_schedule[y] * scale
			balances = np.maximum(balances, 0.0)
		return float(np.mean(balances > 0))

	lo, hi = 0.0, 3.0
	# expand hi if needed
	for _ in range(10):
		if check_survival(hi) < target_success:
			break
		hi *= 2
	for _ in range(30):
		mid = (lo + hi) / 2
		if check_survival(mid) < target_success:
			hi = mid
		else:
			lo = mid
		if hi - lo < tol:
			break
	return lo

def get_all_historical_windows(years_needed: int):
	"""Return all possible annual return sequences from historical data.
	Each window starts at a different monthly offset and extracts years_needed
	annual returns by stepping every 12 rows through the rolling factors.
	Returns (windows, start_dates) where start_dates[i] is the begin month for window i."""
	df = load_master_global()
	step = 12
	last_needed_offset = (years_needed - 1) * step
	max_start = len(df) - last_needed_offset
	windows = []
	start_dates = []
	for start_idx in range(max_start):
		indices = list(range(start_idx, start_idx + years_needed * step, step))
		stock_returns = df['LBM 100E'].iloc[indices].values - 1.0
		bond_returns = df['LBM 100 F'].iloc[indices].values - 1.0
		windows.append((stock_returns, bond_returns))
		start_dates.append(df['begin month'].iloc[start_idx])
	return windows, start_dates

def get_uniform_lifetime_table():
	"""Return a dict age -> distribution period (divisor) using the IRS Uniform Lifetime Table.

	These values are the standard distribution periods used to compute RMD = balance / divisor.
	We include ages 70-120; RMDs begin at age 73 per the user's requirement but table covers a wider range.
	"""
	# Values taken from commonly-published IRS Uniform Lifetime Table (publication tables).
	# For our simulation we only need ages 73-120, but include a range for safety.
	table = {
		70:27.4,71:26.5,72:25.6,73:24.7,74:23.8,75:22.9,76:22.0,77:21.2,78:20.3,79:19.5,
		80:18.7,81:17.9,82:17.1,83:16.3,84:15.5,85:14.8,86:14.1,87:13.4,88:12.7,89:12.0,
		90:11.4,91:10.8,92:10.2,93:9.6,94:9.1,95:8.6,96:8.1,97:7.6,98:7.1,99:6.7,100:6.3,
		101:5.9,102:5.5,103:5.2,104:4.9,105:4.6,106:4.3,107:4.1,108:3.9,109:3.7,110:3.5,
		111:3.3,112:3.1,113:2.9,114:2.7,115:2.5,116:2.3,117:2.1,118:1.9,119:1.7,120:1.5
	}
	return table

def store_distribution_results(results, all_yearly_df, sim_mode_label):
	"""Process MC or historical distribution results and store in session state."""
	mc_df = pd.DataFrame(results)
	percentiles_list = [0, 10, 25, 50, 75, 90]
	summary_cols = ['after_tax_end', 'total_taxes', 'effective_tax_rate', 'portfolio_cagr', 'roth_cagr']
	pct_rows = []
	for p in percentiles_list:
		row = {'percentile': p}
		for col in summary_cols:
			row[col] = np.percentile(mc_df[col], p)
		pct_rows.append(row)
	st.session_state['mc_percentile_rows'] = pct_rows
	st.session_state['mc_pct_non_positive'] = float((mc_df['after_tax_end'] <= 0).mean())
	st.session_state['mc_all_yearly'] = all_yearly_df
	st.session_state['num_sims'] = len(results)
	run_ends = all_yearly_df.groupby('run')['total_portfolio'].last()
	median_val = run_ends.median()
	median_run_idx = int((run_ends - median_val).abs().idxmin())
	median_df = all_yearly_df[all_yearly_df['run'] == median_run_idx].drop(columns=['run', 'total_portfolio']).reset_index(drop=True)
	st.session_state['sim_df'] = median_df
	st.session_state['sim_mode'] = sim_mode_label

def compute_summary_metrics(df: pd.DataFrame, inheritor_rate: float) -> Dict[str, float]:
	"""Return the key summary values used for saved scenarios and Monte Carlo stats."""
	final = df.iloc[-1]
	years = len(df)
	after_tax_end = (final['end_stocks_mv'] + final['end_bonds_mv'] + final['end_roth'] +
					 final['end_tda_total'] * max(0.0, 1.0 - inheritor_rate))
	total_taxes = df['total_taxes'].sum()
	total_income = (df['ordinary_taxable_income'] + df['capital_gains']).sum()
	effective_tax_rate = total_taxes / total_income if total_income > 0 else 0.0
	portfolio_growth = (df['portfolio_return'] + 1.0).prod()
	roth_growth = (df['roth_return_used'] + 1.0).prod()
	return {
		'after_tax_end': after_tax_end,
		'total_taxes': total_taxes,
		'effective_tax_rate': effective_tax_rate,
		'portfolio_cagr': (portfolio_growth ** (1.0 / years) - 1.0) if years > 0 else 0.0,
		'roth_cagr': (roth_growth ** (1.0 / years) - 1.0) if years > 0 else 0.0,
	}

def run_monte_carlo(num_runs: int,
					start_age: int,
					start_age_spouse: int,
					taxable_start: float,
					stock_total_return: float,
					stock_dividend_yield: float,
					stock_turnover: float,
					bond_return: float,
					roth_start: float,
					tda_start: float,
					tda_spouse_start: float,
					withdrawal_schedule: Sequence[float],
					target_stock_pct: float,
					taxable_stock_basis_pct: float,
					taxable_bond_basis_pct: float,
					rmd_start_age: int,
					rmd_start_age_spouse: int,
					ss_income_annual: float,
					ss_income_spouse_annual: float,
					ss_cola: float,
					pension_income_annual: float,
					pension_income_spouse_annual: float,
					pension_cola: float,
					other_income_annual: float,
					filing_status: str,
					use_itemized_deductions: bool,
					itemized_deduction_amount: float,
					roth_conversion_amount: float,
					roth_conversion_years: int,
					roth_conversion_tax_source: str,
					taxable_log_drift: float,
					taxable_log_volatility: float,
					bond_log_drift: float,
					bond_log_volatility: float,
					life_expectancy_primary: int,
					life_expectancy_spouse: int,
					inheritor_rate: float,
					years: int,
					guardrails_enabled: bool = False,
					guardrail_lower: float = 0.75,
					guardrail_upper: float = 0.90,
					guardrail_target: float = 0.85,
					guardrail_inner_sims: int = 200,
					blended_mu: float = 0.0,
					blended_sigma: float = 0.0,
					guardrail_max_spending_pct: float = 0.0):
	results = []
	all_yearly = []
	for run_idx in range(num_runs):
		rng = np.random.default_rng()
		taxable_series = sample_lognormal_returns(years, taxable_log_drift, taxable_log_volatility, rng)
		bond_series = sample_lognormal_returns(years, bond_log_drift, bond_log_volatility, rng)
		df_run = simulate_withdrawals(
			start_age_primary=start_age, start_age_spouse=start_age_spouse,
			years=years,
			taxable_start=taxable_start,
			stock_total_return=stock_total_return,
			stock_dividend_yield=stock_dividend_yield,
			stock_turnover=stock_turnover,
			bond_return=bond_return,
			roth_start=roth_start,
			tda_start=tda_start,
			tda_spouse_start=tda_spouse_start,
			withdrawal_schedule=withdrawal_schedule,
			target_stock_pct=target_stock_pct,
			taxable_stock_basis_pct=taxable_stock_basis_pct,
			taxable_bond_basis_pct=taxable_bond_basis_pct,
			rmd_start_age=rmd_start_age,
			rmd_start_age_spouse=rmd_start_age_spouse,
			ss_income_annual=ss_income_annual,
			ss_income_spouse_annual=ss_income_spouse_annual,
			ss_cola=ss_cola,
			pension_income_annual=pension_income_annual,
			pension_income_spouse_annual=pension_income_spouse_annual,
			pension_cola=pension_cola,
			other_income_annual=other_income_annual,
			filing_status=filing_status,
			use_itemized_deductions=use_itemized_deductions,
			itemized_deduction_amount=itemized_deduction_amount,
			roth_conversion_amount=roth_conversion_amount,
			roth_conversion_years=roth_conversion_years,
			roth_conversion_tax_source=roth_conversion_tax_source,
			stock_return_series=taxable_series,
			bond_return_series=bond_series,
			life_expectancy_primary=life_expectancy_primary,
			life_expectancy_spouse=life_expectancy_spouse,
			guardrails_enabled=guardrails_enabled,
			guardrail_lower=guardrail_lower,
			guardrail_upper=guardrail_upper,
			guardrail_target=guardrail_target,
			guardrail_inner_sims=guardrail_inner_sims,
			blended_mu=blended_mu,
			blended_sigma=blended_sigma,
			guardrail_max_spending_pct=guardrail_max_spending_pct,
		)
		df_run['total_portfolio'] = df_run['end_taxable_total'] + df_run['end_tda_total'] + df_run['end_roth']
		metrics = compute_summary_metrics(df_run, inheritor_rate)
		results.append(metrics)
		df_run['run'] = run_idx
		all_yearly.append(df_run)
	all_yearly_df = pd.concat(all_yearly, ignore_index=True)
	return results, all_yearly_df
def sample_lognormal_returns(years: int, drift: float, volatility: float, rng: np.random.Generator) -> Sequence[float]:
	"""Return `years` draws from a lognormal distribution built from the provided normal parameters."""
	log_returns = rng.normal(loc=drift, scale=volatility, size=years)
	return np.exp(log_returns) - 1

def get_standard_deduction(filing_status: str) -> float:
	# 2024 standard deductions
	return 14600 if filing_status == 'single' else 29200

def get_ordinary_brackets(filing_status: str):
	# 2024 ordinary income brackets (taxable income after deductions)
	if filing_status == 'single':
		return [
			(0, 0.10),
			(11600, 0.12),
			(47150, 0.22),
			(100525, 0.24),
			(191950, 0.32),
			(243725, 0.35),
			(609350, 0.37),
		]
	else:
		return [
			(0, 0.10),
			(23200, 0.12),
			(94300, 0.22),
			(201050, 0.24),
			(383900, 0.32),
			(487450, 0.35),
			(731200, 0.37),
		]

def get_capital_gains_brackets(filing_status: str):
	# 2024 LTCG/QD thresholds (stacked on top of ordinary taxable income)
	if filing_status == 'single':
		return [
			(0, 0.00),
			(47025, 0.15),
			(518900, 0.20),
		]
	else:  # married filing jointly
		return [
			(0, 0.00),
			(94050, 0.15),
			(583750, 0.20),
		]

def compute_taxable_social_security(ss_income: float, other_income: float, cap_gains: float, filing_status: str) -> float:
	# Simplified provisional income calculation; treats all other income (ordinary + gains) as part of provisional.
	base = 25000 if filing_status == 'single' else 32000
	max_base = 34000 if filing_status == 'single' else 44000
	provisional = other_income + cap_gains + 0.5 * ss_income

	if provisional <= base:
		return 0.0
	if provisional <= max_base:
		return 0.5 * (provisional - base)
	# Above upper threshold
	excess = provisional - max_base
	amount = 0.85 * excess + min(0.5 * (max_base - base), 0.5 * ss_income)
	return min(amount, 0.85 * ss_income)

def apply_brackets(taxable: float, brackets):
	tax = 0.0
	for i, (start, rate) in enumerate(brackets):
		end = brackets[i+1][0] if i + 1 < len(brackets) else None
		if taxable <= start:
			break
		upper = taxable if end is None else min(taxable, end)
		tax += max(0.0, upper - start) * rate
		if end is None or taxable <= end:
			break
	return tax

def compute_capital_gains_tax(ordinary_taxable: float, cap_gains: float, filing_status: str) -> float:
	if cap_gains <= 0:
		return 0.0
	brackets = get_capital_gains_brackets(filing_status)
	remaining = cap_gains
	tax = 0.0
	for i, (threshold, rate) in enumerate(brackets):
		next_threshold = brackets[i+1][0] if i+1 < len(brackets) else None
		stack_start = max(0.0, threshold - ordinary_taxable)
		if next_threshold is None:
			tax += max(0.0, remaining) * rate
			break
		stack_end = max(0.0, next_threshold - ordinary_taxable)
		band = max(0.0, stack_end - stack_start)
		taxed_here = min(remaining, band)
		tax += taxed_here * rate
		remaining -= taxed_here
		if remaining <= 1e-9:
			break
	return tax

def get_marginal_rates(taxable_ordinary: float, cap_gains: float, filing_status: str):
	# marginal ordinary = bracket rate for next ordinary dollar
	ordinary_brackets = get_ordinary_brackets(filing_status)
	marginal_ordinary = 0.0
	for start, rate in ordinary_brackets:
		if taxable_ordinary >= start:
			marginal_ordinary = rate
		else:
			break

	# marginal cap gains rate given stacking (use top rate hit in cap gains bands)
	cg_brackets = get_capital_gains_brackets(filing_status)
	if cap_gains <= 0:
		marginal_cg = 0.0
	else:
		remaining = cap_gains
		marginal_cg = cg_brackets[0][1]
		for i, (threshold, rate) in enumerate(cg_brackets):
			next_threshold = cg_brackets[i+1][0] if i+1 < len(cg_brackets) else None
			stack_start = max(0.0, threshold - taxable_ordinary)
			stack_end = float('inf') if next_threshold is None else max(0.0, next_threshold - taxable_ordinary)
			band = max(0.0, stack_end - stack_start)
			if remaining > band:
				remaining -= band
				marginal_cg = rate
				continue
			else:
				marginal_cg = rate
				break

	return marginal_ordinary, marginal_cg


def process_rmd(tda_stocks_mv: float, tda_bonds_mv: float, current_age: int, rmd_start_age: int, table):
	"""Calculate and withdraw RMD for one person; returns updated balances plus rmd amount and divisor used."""
	rmd = 0.0
	divisor = None
	if current_age >= rmd_start_age:
		divisor = table.get(current_age)
		if divisor is None:
			divisor = max(1.0, 25.0 - (current_age - rmd_start_age))
		total_tda_balance = tda_stocks_mv + tda_bonds_mv
		rmd = total_tda_balance / divisor if divisor > 0 else 0.0
		take_rmd = min(rmd, total_tda_balance)
		tda_stock_ratio = (tda_stocks_mv / total_tda_balance) if total_tda_balance > 0 else 0.5
		tda_stocks_mv -= take_rmd * tda_stock_ratio
		tda_bonds_mv -= take_rmd * (1 - tda_stock_ratio)
	return tda_stocks_mv, tda_bonds_mv, rmd, divisor

def rebalance_accounts(target_stock_pct: float,
					   taxable_stock_mv: float,
					   taxable_bond_mv: float,
					   taxable_stock_basis: float,
					   taxable_bond_basis: float,
					   tda1_mv: float,
					   tda2_mv: float,
					   roth_mv: float):
	"""Rebalance household portfolio to hit target stock %.

	Priority for stocks: Roth first (tax-free growth), then taxable, then TDAs.
	Priority for bonds: TDAs first (interest taxed as ordinary anyway), then taxable, then Roth.
	TDAs and Roth can hold mixed allocations when needed to meet the household target.
	"""
	total_household = taxable_stock_mv + taxable_bond_mv + tda1_mv + tda2_mv + roth_mv
	if total_household <= 0:
		return (taxable_stock_mv, taxable_bond_mv, taxable_stock_basis, taxable_bond_basis,
				0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

	desired_stock_total = total_household * target_stock_pct
	desired_bond_total = total_household - desired_stock_total
	taxable_total = taxable_stock_mv + taxable_bond_mv
	tda_total = tda1_mv + tda2_mv

	# Allocate stocks: Roth first, then taxable, then TDAs
	stock_remaining = desired_stock_total
	roth_stock = min(stock_remaining, roth_mv)
	stock_remaining -= roth_stock
	taxable_stock = min(stock_remaining, taxable_total)
	stock_remaining -= taxable_stock
	# spill into TDAs if Roth + taxable can't hold all stocks
	tda_stock = min(stock_remaining, tda_total)
	stock_remaining -= tda_stock

	# Allocate bonds: TDAs first, then taxable, then Roth
	bond_remaining = desired_bond_total
	tda_bond = min(bond_remaining, tda_total - tda_stock)
	bond_remaining -= tda_bond
	taxable_bond = min(bond_remaining, taxable_total - taxable_stock)
	bond_remaining -= taxable_bond
	roth_bond = min(bond_remaining, roth_mv - roth_stock)
	bond_remaining -= roth_bond

	# Split TDA stock/bond between the two TDA accounts proportionally
	if tda_total > 0:
		tda1_frac = tda1_mv / tda_total
	else:
		tda1_frac = 0.5
	tda1_stock = tda_stock * tda1_frac
	tda1_bond = tda1_mv - tda1_stock
	tda2_stock = tda_stock - tda1_stock
	tda2_bond = tda2_mv - tda2_stock

	# adjust taxable basis proportionally
	taxable_basis_total = taxable_stock_basis + taxable_bond_basis
	if taxable_stock + taxable_bond > 0 and taxable_basis_total > 0:
		taxable_stock_basis = taxable_basis_total * (taxable_stock / (taxable_stock + taxable_bond))
		taxable_bond_basis = taxable_basis_total - taxable_stock_basis

	return (taxable_stock, taxable_bond, taxable_stock_basis, taxable_bond_basis,
			tda1_stock, tda1_bond, tda2_stock, tda2_bond, roth_stock, roth_bond)

def simulate_withdrawals(start_age_primary: int,
						 start_age_spouse: int,
						 years: int,
						 taxable_start: float,
						 stock_total_return: float,
						 stock_dividend_yield: float,
						 stock_turnover: float,
						 bond_return: float,
						 roth_start: float,
						 tda_start: float,
						 tda_spouse_start: float,
						 withdrawal_schedule: Sequence[float],
						 target_stock_pct: float,
						 taxable_stock_basis_pct: float,
						 taxable_bond_basis_pct: float,
						 rmd_start_age: int = 73,
						 rmd_start_age_spouse: int = 73,
						 ss_income_annual: float = 0.0,
						 ss_income_spouse_annual: float = 0.0,
						 ss_cola: float = 0.0,
						 pension_income_annual: float = 0.0,
						 pension_income_spouse_annual: float = 0.0,
						 pension_cola: float = 0.0,
						 other_income_annual: float = 0.0,
						 filing_status: str = 'single',
						 use_itemized_deductions: bool = False,
						 itemized_deduction_amount: float = 0.0,
						 roth_conversion_amount: float = 0.0,
						 roth_conversion_tax_source: str = 'taxable',
						 roth_conversion_years: int = 0,
						 stock_return_series: Optional[Sequence[float]] = None,
						 bond_return_series: Optional[Sequence[float]] = None,
						 life_expectancy_primary: int = 120,
						 life_expectancy_spouse: int = 120,
						 guardrails_enabled: bool = False,
						 guardrail_lower: float = 0.75,
						 guardrail_upper: float = 0.90,
						 guardrail_target: float = 0.85,
						 guardrail_inner_sims: int = 200,
						 blended_mu: float = 0.0,
						 blended_sigma: float = 0.0,
						 guardrail_max_spending_pct: float = 0.0):
	table = get_uniform_lifetime_table()

	# assume taxable holds 50% stocks / 50% bonds
	stocks_share = 0.5

	# initialize taxable split into stocks and bonds
	taxable = float(taxable_start)
	stocks_mv = taxable * stocks_share
	bonds_mv = taxable * (1 - stocks_share)
	# basis assumption via inputs: percent of market value
	stocks_basis = stocks_mv * taxable_stock_basis_pct
	bonds_basis = bonds_mv * taxable_bond_basis_pct

	# initial TDA/Roth totals
	tda1_total = float(tda_start)
	tda2_total = float(tda_spouse_start)
	roth_total = float(roth_start)

	(stocks_mv, bonds_mv, stocks_basis, bonds_basis,
		tda1_stocks_mv, tda1_bonds_mv, tda2_stocks_mv, tda2_bonds_mv, roth_stocks_mv, roth_bonds_mv) = rebalance_accounts(
		target_stock_pct,
		stocks_mv, bonds_mv, stocks_basis, bonds_basis,
		tda1_total, tda2_total, roth_total
	)

	if stock_return_series is not None and len(stock_return_series) < years:
		raise ValueError('stock_return_series must have at least `years` entries')
	if bond_return_series is not None and len(bond_return_series) < years:
		raise ValueError('bond_return_series must have at least `years` entries')

	rows = []

	# Guardrail: compute initial scaling factor via binary search
	if guardrails_enabled:
		total_portfolio_init = float(taxable_start) + float(tda_start) + float(tda_spouse_start) + float(roth_start)
		current_scale_factor = find_sustainable_scale_factor(
			total_portfolio_init, list(withdrawal_schedule), blended_mu, blended_sigma,
			guardrail_target, guardrail_inner_sims)
		# Apply max spending cap to scale factor
		if guardrail_max_spending_pct > 0:
			max_scale = 1.0 + guardrail_max_spending_pct / 100.0
			current_scale_factor = min(current_scale_factor, max_scale)
	else:
		current_scale_factor = 1.0

	for y in range(1, years+1):
		age_p1 = start_age_primary + y - 1
		age_p2 = start_age_spouse + y - 1
		# rebalance at start of each year to target allocation with rounding
		(stocks_mv, bonds_mv, stocks_basis, bonds_basis,
			tda1_stocks_mv, tda1_bonds_mv, tda2_stocks_mv, tda2_bonds_mv, roth_stocks_mv, roth_bonds_mv) = rebalance_accounts(
			target_stock_pct,
			stocks_mv, bonds_mv, stocks_basis, bonds_basis,
			tda1_stocks_mv + tda1_bonds_mv,
			tda2_stocks_mv + tda2_bonds_mv,
			roth_stocks_mv + roth_bonds_mv
		)
		# Guardrail check: after rebalance, before growth (skip year 1 — already solved)
		if guardrails_enabled and y > 1:
			total_portfolio_now = (stocks_mv + bonds_mv +
				tda1_stocks_mv + tda1_bonds_mv +
				tda2_stocks_mv + tda2_bonds_mv +
				roth_stocks_mv + roth_bonds_mv)
			remaining_schedule = list(withdrawal_schedule[y-1:])
			if len(remaining_schedule) > 1 and total_portfolio_now > 0:
				sr = forward_success_rate(total_portfolio_now, remaining_schedule,
					current_scale_factor, blended_mu, blended_sigma, guardrail_inner_sims)
				if sr < guardrail_lower or sr > guardrail_upper:
					current_scale_factor = find_sustainable_scale_factor(
						total_portfolio_now, remaining_schedule, blended_mu, blended_sigma,
						guardrail_target, guardrail_inner_sims)
					# Apply max spending cap to scale factor
					if guardrail_max_spending_pct > 0:
						max_scale = 1.0 + guardrail_max_spending_pct / 100.0
						current_scale_factor = min(current_scale_factor, max_scale)

		start_stocks_mv = stocks_mv
		start_bonds_mv = bonds_mv
		start_stocks_basis = stocks_basis
		start_bonds_basis = bonds_basis
		start_tda = tda1_stocks_mv + tda1_bonds_mv
		start_tda_spouse = tda2_stocks_mv + tda2_bonds_mv
		start_roth = roth_stocks_mv + roth_bonds_mv

		primary_alive = age_p1 <= life_expectancy_primary
		spouse_alive = age_p2 <= life_expectancy_spouse

		# apply Roth conversion at start of year before growth
		conversion_gross = min(roth_conversion_amount, tda1_stocks_mv + tda1_bonds_mv) if (y <= roth_conversion_years and primary_alive) else 0.0
		if conversion_gross > 0:
			total_tda_balance = tda1_stocks_mv + tda1_bonds_mv
			tda_stock_ratio = (tda1_stocks_mv / total_tda_balance) if total_tda_balance > 0 else 0.5
			tda1_stocks_mv -= conversion_gross * tda_stock_ratio
			tda1_bonds_mv -= conversion_gross * (1 - tda_stock_ratio)
			# temporarily park converted amount; taxes handled after computing tax delta
			pending_roth_conversion = conversion_gross
		else:
			pending_roth_conversion = 0.0

		stock_return_year = stock_return_series[y-1] if stock_return_series is not None else stock_total_return
		bond_return_year = bond_return_series[y-1] if bond_return_series is not None else bond_return
		# grow TDA and Roth sub-accounts using the same stock/bond returns
		tda1_stocks_mv *= (1 + stock_return_year)
		tda1_bonds_mv *= (1 + bond_return_year)
		tda2_stocks_mv *= (1 + stock_return_year)
		tda2_bonds_mv *= (1 + bond_return_year)
		roth_stocks_mv *= (1 + stock_return_year)
		roth_bonds_mv *= (1 + bond_return_year)

		# Stocks: split total return into price appreciation and dividend yield
		# Stocks: split total return into price appreciation and dividend yield
		price_return = (1 + stock_return_year) / (1 + stock_dividend_yield) - 1
		# apply price appreciation
		stocks_mv *= (1 + price_return)

		# dividends (qualified) - reinvest gross; tax computed later
		div = stocks_mv * stock_dividend_yield
		stocks_mv += div
		stocks_basis += div

		# Bonds: accrue interest, tax handled later
		interest = bonds_mv * bond_return_year
		bonds_mv += interest
		bonds_basis += interest

		# Stock turnover: track realized gains for tax; ignore drag in balances for now (assume immediate reinvest)
		turnover_sale = stocks_mv * stock_turnover
		turnover_basis_sold = stocks_basis * stock_turnover
		turnover_realized_gain = max(0.0, turnover_sale - turnover_basis_sold)
		# rebalance basis to reflect sale and repurchase at current market value
		stocks_mv = stocks_mv  # unchanged net of round trip
		stocks_basis = stocks_basis - turnover_basis_sold + turnover_sale

		# capture portfolio growth before any withdrawals/RMDs/taxes
		start_total_balance = start_stocks_mv + start_bonds_mv + start_tda + start_tda_spouse + start_roth
		investable_start_total = start_total_balance - conversion_gross
		total_after_return = (stocks_mv + bonds_mv +
							  tda1_stocks_mv + tda1_bonds_mv +
							  tda2_stocks_mv + tda2_bonds_mv +
							  roth_stocks_mv + roth_bonds_mv)
		investment_return_dollars = total_after_return - investable_start_total

		# compute RMD for each spouse if applicable
		if primary_alive:
			tda1_stocks_mv, tda1_bonds_mv, rmd_p1, divisor_p1 = process_rmd(tda1_stocks_mv, tda1_bonds_mv, age_p1, rmd_start_age, table)
		else:
			rmd_p1 = 0.0
			divisor_p1 = None
		if spouse_alive:
			tda2_stocks_mv, tda2_bonds_mv, rmd_p2, divisor_p2 = process_rmd(tda2_stocks_mv, tda2_bonds_mv, age_p2, rmd_start_age_spouse, table)
		else:
			rmd_p2 = 0.0
			divisor_p2 = None

		# Determine filing status and income for this year (needed by tax solve)
		filing_status_this_year = filing_status
		if filing_status == 'mfj' and (not spouse_alive or not primary_alive):
			filing_status_this_year = 'single'
		ss_income_p1 = ss_income_annual * ((1 + ss_cola) ** (y - 1)) if primary_alive else 0.0
		ss_income_p2 = ss_income_spouse_annual * ((1 + ss_cola) ** (y - 1)) if spouse_alive else 0.0
		ss_income = ss_income_p1 + ss_income_p2
		pension_income_p1 = pension_income_annual * ((1 + pension_cola) ** (y - 1)) if primary_alive else 0.0
		pension_income_p2 = pension_income_spouse_annual * ((1 + pension_cola) ** (y - 1)) if spouse_alive else 0.0
		pension_income = pension_income_p1 + pension_income_p2
		other_income = other_income_annual
		deduction = itemized_deduction_amount if use_itemized_deductions else get_standard_deduction(filing_status_this_year)

		# Snapshot mutable balances (post-growth, post-RMD) for iterative solve
		total_rmd_cash = rmd_p1 + rmd_p2
		snap_stocks_mv = stocks_mv
		snap_bonds_mv = bonds_mv
		snap_stocks_basis = stocks_basis
		snap_bonds_basis = bonds_basis
		snap_tda1_stocks = tda1_stocks_mv
		snap_tda1_bonds = tda1_bonds_mv
		snap_tda2_stocks = tda2_stocks_mv
		snap_tda2_bonds = tda2_bonds_mv
		snap_roth_stocks = roth_stocks_mv
		snap_roth_bonds = roth_bonds_mv

		def try_gross_withdrawal(gross_target):
			"""Run withdrawal waterfall + full tax computation for a given gross
			withdrawal from portfolio. Returns (net_after_tax_spending, result_dict)."""
			s_mv, b_mv = snap_stocks_mv, snap_bonds_mv
			s_basis, b_basis = snap_stocks_basis, snap_bonds_basis
			t1s, t1b = snap_tda1_stocks, snap_tda1_bonds
			t2s, t2b = snap_tda2_stocks, snap_tda2_bonds
			rs, rb = snap_roth_stocks, snap_roth_bonds

			out_taxable_cash = 0.0
			w_tda = total_rmd_cash
			w_roth = 0.0
			sold_bonds = 0.0
			sold_stocks = 0.0
			realized_gains = 0.0
			rmd_excess = 0.0

			if total_rmd_cash > gross_target:
				rmd_excess = total_rmd_cash - gross_target
				total_mv = s_mv + b_mv
				if total_mv > 0:
					s_mv += rmd_excess * (s_mv / total_mv)
					b_mv += rmd_excess * (b_mv / total_mv)
					total_basis = s_basis + b_basis
					if total_basis > 0:
						s_basis += rmd_excess * (s_basis / total_basis)
						b_basis += rmd_excess * (b_basis / total_basis)
					else:
						s_basis += rmd_excess * 0.5
						b_basis += rmd_excess * 0.5
				else:
					s_mv += rmd_excess * 0.5
					b_mv += rmd_excess * 0.5
					s_basis += rmd_excess * 0.5
					b_basis += rmd_excess * 0.5
				remaining = 0.0
			else:
				remaining = gross_target - total_rmd_cash

			# Waterfall: taxable bonds -> stocks -> TDA -> Roth
			if remaining > 0 and b_mv > 0:
				take = min(remaining, b_mv)
				basis_sold = take * (b_basis / b_mv) if b_mv > 0 else 0.0
				realized_gains += max(0.0, take - basis_sold)
				sold_bonds += take
				b_mv -= take
				b_basis -= basis_sold
				out_taxable_cash += take
				remaining -= take

			if remaining > 1e-8 and s_mv > 0:
				take = min(remaining, s_mv)
				basis_sold = take * (s_basis / s_mv) if s_mv > 0 else 0.0
				realized_gains += max(0.0, take - basis_sold)
				sold_stocks += take
				s_mv -= take
				s_basis -= basis_sold
				out_taxable_cash += take
				remaining -= take

			if remaining > 1e-8:
				t1_total = t1s + t1b
				take = min(remaining, t1_total)
				ratio = (t1s / t1_total) if t1_total > 0 else 0.5
				t1s -= take * ratio
				t1b -= take * (1 - ratio)
				w_tda += take
				remaining -= take

			if remaining > 1e-8:
				t2_total = t2s + t2b
				take = min(remaining, t2_total)
				ratio = (t2s / t2_total) if t2_total > 0 else 0.5
				t2s -= take * ratio
				t2b -= take * (1 - ratio)
				w_tda += take
				remaining -= take

			if remaining > 1e-8:
				r_total = rs + rb
				take = min(remaining, r_total)
				ratio = (rs / r_total) if r_total > 0 else 1.0
				rs -= take * ratio
				rb -= take * (1 - ratio)
				w_roth += take
				remaining -= take

			# Full tax computation
			ordinary_pre_ss_base = interest + w_tda + pension_income + other_income
			ordinary_pre_ss_with_conv = ordinary_pre_ss_base + pending_roth_conversion
			cg_total = div + turnover_realized_gain + realized_gains

			# With conversion
			t_ss = compute_taxable_social_security(ss_income, ordinary_pre_ss_with_conv, cg_total, filing_status_this_year)
			t_ordinary = max(0.0, ordinary_pre_ss_with_conv + t_ss - deduction)
			ord_tax = apply_brackets(t_ordinary, get_ordinary_brackets(filing_status_this_year))
			cg_tax = compute_capital_gains_tax(t_ordinary, cg_total, filing_status_this_year)
			total_tax = ord_tax + cg_tax

			# Without conversion (for delta)
			t_ss_nc = compute_taxable_social_security(ss_income, ordinary_pre_ss_base, cg_total, filing_status_this_year)
			t_ordinary_nc = max(0.0, ordinary_pre_ss_base + t_ss_nc - deduction)
			ord_tax_nc = apply_brackets(t_ordinary_nc, get_ordinary_brackets(filing_status_this_year))
			cg_tax_nc = compute_capital_gains_tax(t_ordinary_nc, cg_total, filing_status_this_year)
			total_tax_nc = ord_tax_nc + cg_tax_nc
			conv_tax_delta = max(0.0, total_tax - total_tax_nc)

			# NIIT
			niit_threshold = 200000 if filing_status_this_year == 'single' else 250000
			agi = ordinary_pre_ss_with_conv + t_ss + cg_total
			niit_base_val = max(0.0, agi - niit_threshold)
			net_inv = max(0.0, cg_total + interest + div)
			niit = 0.038 * min(niit_base_val, net_inv)
			total_tax += niit

			marg_ord, marg_cg = get_marginal_rates(t_ordinary, cg_total, filing_status_this_year)
			if niit_base_val > 0 and cg_total > 0:
				marg_cg += 0.038

			# Roth conversion tax payment
			conv_net_to_roth = 0.0
			if pending_roth_conversion > 0:
				if roth_conversion_tax_source == 'taxable':
					tax_to_pay = conv_tax_delta
					pay_b = min(tax_to_pay, b_mv)
					if pay_b > 0 and b_mv > 0:
						b_basis -= pay_b * (b_basis / b_mv)
						b_mv -= pay_b
					rem_tax = tax_to_pay - pay_b
					pay_s = min(rem_tax, s_mv)
					if pay_s > 0 and s_mv > 0:
						s_basis -= pay_s * (s_basis / s_mv)
						s_mv -= pay_s
					rem_tax -= pay_s
					conv_net_to_roth = max(0.0, pending_roth_conversion - rem_tax)
				else:
					conv_net_to_roth = max(0.0, pending_roth_conversion - conv_tax_delta)
				rs += conv_net_to_roth

			# Non-spending tax (portfolio drag, not spending reduction)
			nst = (
				rmd_excess * marg_ord
				+ max(0.0, interest) * marg_ord
				+ (div + turnover_realized_gain) * marg_cg
			)
			nst = max(0.0, min(nst, total_tax - conv_tax_delta))

			net_spending = (out_taxable_cash + w_tda + w_roth
				- rmd_excess + ss_income + pension_income + other_income
				- (total_tax - conv_tax_delta - nst))

			return net_spending, {
				'stocks_mv': s_mv, 'bonds_mv': b_mv,
				'stocks_basis': s_basis, 'bonds_basis': b_basis,
				'tda1_stocks': t1s, 'tda1_bonds': t1b,
				'tda2_stocks': t2s, 'tda2_bonds': t2b,
				'roth_stocks': rs, 'roth_bonds': rb,
				'out_taxable_cash': out_taxable_cash,
				'withdraw_from_tda': w_tda, 'withdraw_from_roth': w_roth,
				'rmd_excess_to_taxable': rmd_excess,
				'gross_sold_taxable_bonds': sold_bonds,
				'gross_sold_taxable_stocks': sold_stocks,
				'taxable_ss': t_ss, 'taxable_ordinary': t_ordinary,
				'ordinary_tax_total': ord_tax,
				'cap_gains_total': cg_total, 'cap_gains_tax': cg_tax,
				'niit_tax': niit, 'total_taxes': total_tax,
				'roth_conversion_tax_delta': conv_tax_delta,
				'non_spending_tax': nst,
				'marginal_ordinary_rate': marg_ord, 'marginal_cg_rate': marg_cg,
				'net_spending': net_spending,
				'gross_target': gross_target,
			}

		# Binary search: find gross withdrawal that delivers the net spending target
		base_withdrawal_this_year = withdrawal_schedule[y-1]
		net_target = base_withdrawal_this_year * current_scale_factor

		base_net, base_result = try_gross_withdrawal(0.0)

		if net_target <= 0 or base_net >= net_target:
			chosen = base_result
		else:
			max_available = (snap_stocks_mv + snap_bonds_mv +
				snap_tda1_stocks + snap_tda1_bonds +
				snap_tda2_stocks + snap_tda2_bonds +
				snap_roth_stocks + snap_roth_bonds + total_rmd_cash)
			lo = 0.0
			hi = min(net_target * 2.5, max_available)
			chosen = base_result
			for _ in range(20):
				mid = (lo + hi) / 2.0
				mid_net, mid_result = try_gross_withdrawal(mid)
				if mid_net < net_target - 5.0:
					lo = mid
				elif mid_net > net_target + 5.0:
					hi = mid
				else:
					chosen = mid_result
					break
				chosen = mid_result
				if hi - lo < 10.0:
					break

		# Commit the chosen withdrawal result
		stocks_mv = chosen['stocks_mv']
		bonds_mv = chosen['bonds_mv']
		stocks_basis = chosen['stocks_basis']
		bonds_basis = chosen['bonds_basis']
		tda1_stocks_mv = chosen['tda1_stocks']
		tda1_bonds_mv = chosen['tda1_bonds']
		tda2_stocks_mv = chosen['tda2_stocks']
		tda2_bonds_mv = chosen['tda2_bonds']
		roth_stocks_mv = chosen['roth_stocks']
		roth_bonds_mv = chosen['roth_bonds']

		actual_portfolio_return = (investment_return_dollars / investable_start_total) if investable_start_total > 0 else 0.0
		rows.append({
			'year': y,
			'portfolio_return': actual_portfolio_return,
			'roth_return_used': stock_return_year,
			'age_p1': age_p1,
			'age_p2': age_p2,
			'start_stocks_mv': start_stocks_mv,
			'start_bonds_mv': start_bonds_mv,
			'start_stocks_basis': start_stocks_basis,
			'start_bonds_basis': start_bonds_basis,
			'start_tda_p1': start_tda,
			'start_tda_p2': start_tda_spouse,
			'start_roth': start_roth,
			'rmd_divisor_p1': divisor_p1,
			'rmd_divisor_p2': divisor_p2,
			'rmd_p1': rmd_p1,
			'rmd_p2': rmd_p2,
			'rmd_total': rmd_p1 + rmd_p2,
			'withdraw_from_taxable_net': chosen['out_taxable_cash'],
			'withdraw_from_tda': chosen['withdraw_from_tda'],
			'withdraw_from_roth': chosen['withdraw_from_roth'],
			'rmd_excess_to_taxable': chosen['rmd_excess_to_taxable'],
			'gross_sold_taxable_bonds': chosen['gross_sold_taxable_bonds'],
			'gross_sold_taxable_stocks': chosen['gross_sold_taxable_stocks'],
			'ss_income_total': ss_income,
			'taxable_social_security': chosen['taxable_ss'],
			'pension_income_total': pension_income,
			'other_income': other_income,
			'roth_conversion': pending_roth_conversion,
			'roth_conversion_tax': chosen['roth_conversion_tax_delta'],
			'roth_conversion_tax_source': roth_conversion_tax_source,
			'ordinary_taxable_income': chosen['taxable_ordinary'],
			'ordinary_tax_total': chosen['ordinary_tax_total'],
			'capital_gains': chosen['cap_gains_total'],
			'capital_gains_tax': chosen['cap_gains_tax'],
			'niit_tax': chosen['niit_tax'],
			'marginal_ordinary_rate': chosen['marginal_ordinary_rate'],
			'marginal_cap_gains_rate': chosen['marginal_cg_rate'],
			'deduction_applied': deduction,
			'total_taxes': chosen['total_taxes'],
			'end_stocks_mv': stocks_mv,
			'end_bonds_mv': bonds_mv,
			'end_taxable_total': stocks_mv + bonds_mv,
			'end_stocks_basis': stocks_basis,
			'end_bonds_basis': bonds_basis,
			'investment_return_dollars': investment_return_dollars,
			'end_tda_p1': tda1_stocks_mv + tda1_bonds_mv,
			'end_tda_p2': tda2_stocks_mv + tda2_bonds_mv,
			'end_tda_total': (tda1_stocks_mv + tda1_bonds_mv + tda2_stocks_mv + tda2_bonds_mv),
			'end_roth': roth_stocks_mv + roth_bonds_mv,
			'withdrawal_used': chosen['gross_target'],
			'net_spending_target': net_target,
			'after_tax_spending': chosen['net_spending'],
		})

	df = pd.DataFrame(rows)
	return df

def main():
	st.title('Withdrawal + RMD Simulator (30-year)')

	if st.button('Reset saved scenarios'):
		for key in ['scenario_summaries', 'last_summary', 'mc_percentile_rows', 'mc_all_yearly',
					'mc_pct_non_positive', 'sim_df', 'sim_mode', 'mc_summaries', 'mc_percentiles']:
			st.session_state.pop(key, None)
		st.success('Saved scenarios cleared.')

	with st.sidebar:
		st.header('Inputs')

		with st.expander('Ages & Timeline', expanded=True):
			start_age = st.number_input('Starting age (person 1)', min_value=18, max_value=120, value=65)
			start_age_spouse = st.number_input('Starting age (person 2)', min_value=18, max_value=120, value=60)
			life_expectancy_primary = st.number_input('Primary life expectancy (last age lived through)', min_value=int(start_age), max_value=120, value=84, step=1)
			life_expectancy_spouse = st.number_input('Spouse life expectancy (last age lived through)', min_value=int(start_age_spouse), max_value=120, value=89, step=1)

		with st.expander('Account Balances'):
			taxable_start = st.number_input('Taxable account starting balance', value=300000.0, step=1000.0)
			taxable_stock_basis_pct = st.number_input('Taxable stock basis % of market value', value=50.0, min_value=0.0, max_value=100.0, step=1.0) / 100.0
			taxable_bond_basis_pct = st.number_input('Taxable bond basis % of market value', value=100.0, min_value=0.0, max_value=100.0, step=1.0) / 100.0
			roth_start = st.number_input('Roth account starting balance', value=100000.0, step=1000.0)
			tda_start = st.number_input('Tax-deferred account starting balance (IRA/401k) - person 1', value=600000.0, step=1000.0)
			tda_spouse_start = st.number_input('Tax-deferred account starting balance (IRA/401k) - person 2', value=0.0, step=1000.0)

		with st.expander('Allocation & Roth Conversions'):
			target_stock_pct = st.slider('Household target % in stocks', min_value=0, max_value=100, value=60, step=10) / 100.0
			roth_conversion_amount = st.number_input('Annual Roth conversion amount (from TDA)', value=0.0, step=1000.0)
			roth_conversion_years = st.number_input('Years to perform conversions', value=0, min_value=0, max_value=100, step=1)
			roth_conversion_tax_source = st.radio('Pay conversion taxes from', ['Taxable', 'TDA (reduce net conversion)'], horizontal=False)

		with st.expander('Withdrawal Schedule'):
			_horizon = max(1, max(int(life_expectancy_primary) - int(start_age),
								  int(life_expectancy_spouse) - int(start_age_spouse)) + 1)
			num_withdrawal_periods = st.number_input('Number of withdrawal periods', value=1, min_value=1, max_value=10, step=1)
			withdrawal_schedule_inputs = []
			period_start = 1
			for i in range(int(num_withdrawal_periods)):
				is_last = (i == int(num_withdrawal_periods) - 1)
				if is_last:
					period_end = _horizon
					st.markdown(f'**Period {i+1}:** years {period_start}–{period_end}')
					period_amount = st.number_input(f'Period {i+1} annual After-Tax Spending Goal', value=40000.0, step=1000.0, key=f'wd_amount_{i}')
				else:
					max_end = _horizon - (int(num_withdrawal_periods) - 1 - i)
					default_end = min(period_start + 4, max_end)
					period_end = st.number_input(
						f'Period {i+1}: years {period_start} through',
						value=default_end, min_value=period_start, max_value=max_end, step=1, key=f'wd_end_{i}')
					period_amount = st.number_input(f'Period {i+1} Annual After-Tax Spend Goal', value=40000.0, step=1000.0, key=f'wd_amount_{i}')
				num_years = int(period_end) - period_start + 1
				withdrawal_schedule_inputs.append((num_years, float(period_amount)))
				period_start = int(period_end) + 1
			rmd_start_age = st.number_input('RMD start age (person 1)', min_value=65, max_value=89, value=73)
			rmd_start_age_spouse = st.number_input('RMD start age (person 2)', min_value=65, max_value=90, value=73)

		with st.expander('Other Income'):
			ss_income_input = st.number_input('Annual Social Security - person 1 (current year)', value=0.0, step=1000.0)
			ss_income_spouse_input = st.number_input('Annual Social Security - person 2 (current year)', value=0.0, step=1000.0)
			ss_cola = st.number_input('Social Security COLA', value=0.02, format="%.4f")
			pension_income_input = st.number_input('Annual pension income - person 1', value=0.0, step=1000.0)
			pension_income_spouse_input = st.number_input('Annual pension income - person 2', value=0.0, step=1000.0)
			pension_cola = st.number_input('Pension COLA', value=0.00, format="%.4f")
			other_income_input = st.number_input('Other ordinary income', value=0.0, step=1000.0)

		with st.expander('Tax Settings'):
			filing_status_choice = st.radio('Filing status', ['Single', 'Married Filing Jointly'], horizontal=True, index=1)
			filing_status_key = 'single' if filing_status_choice == 'Single' else 'mfj'
			standard_deduction_display = 14600 if filing_status_key == 'single' else 29200
			use_itemized = st.checkbox('Use itemized deductions instead of standard', value=False)
			itemized_deduction_input = st.number_input('Itemized deduction amount', value=0.0, step=500.0)
			st.caption(f'Standard deduction used if not itemizing: ${standard_deduction_display:,.0f}')
			inheritor_marginal_rate = st.number_input(
				'Inheritor marginal tax rate on TDAs',
				value=0.22, min_value=0.0, max_value=0.37, format="%.4f")

		with st.expander('Return Assumptions'):
			return_mode = st.radio('Return mode', ['Simulated (lognormal)', 'Historical (master_global_factors)'], horizontal=False, index=1)
			stock_total_return = 0.0
			bond_return = 0.0
			if return_mode == 'Simulated (lognormal)':
				st.caption('Both stocks and bonds use lognormal draws.')
				taxable_log_drift = st.number_input('Stock log drift (µ)', value=0.09038261, format="%.8f")
				taxable_log_volatility = st.number_input('Stock log volatility (σ)', value=0.20485277, format="%.8f")
				bond_log_drift = st.number_input('Bond log drift (µ)', value=0.0172918, format="%.8f")
				bond_log_volatility = st.number_input('Bond log volatility (σ)', value=0.04796435, format="%.8f")
				random_seed_input = st.number_input('Random seed for returns', value=42, step=1)
				seed_mode = st.radio('Seed mode', ['Random each run', 'Fixed seed'], horizontal=True, index=0)
			else:
				st.caption('Returns from LBM 100E (stocks) and LBM 100 F (bonds).')
				historical_mode_type = st.radio('Historical mode',
					['All periods (distribution)', 'Specific start year'],
					horizontal=True, index=0)
				if historical_mode_type == 'Specific start year':
					mg_df = load_master_global()
					min_year = mg_df['begin month'].dt.year.min() + 1
					max_year = mg_df['begin month'].dt.year.max()
					historical_start_year = st.number_input('Historical start year', min_value=int(min_year), max_value=int(max_year), value=1966, step=1)
				else:
					historical_start_year = None
				taxable_log_drift = 0.0
				taxable_log_volatility = 0.0
				bond_log_drift = 0.0
				bond_log_volatility = 0.0
				random_seed_input = 42
				seed_mode = 'Fixed seed'
			stock_dividend_yield = st.number_input('Stock dividend (qualified) yield', value=0.02, format="%.4f")
			stock_turnover = st.number_input('Stock turnover rate', value=0.10, format="%.4f")

		with st.expander('Withdrawal Guardrails'):
			guardrails_enabled = st.checkbox('Enable dynamic withdrawal guardrails', value=False)
			if guardrails_enabled:
				st.caption('At each year, checks forward MC survival rate. If outside the dead band, resets withdrawal to the target success rate.')
				guardrail_lower = st.number_input('Lower guardrail (reduce spending if below)', value=0.75, min_value=0.50, max_value=0.95, format="%.2f", step=0.05)
				guardrail_upper = st.number_input('Upper guardrail (increase spending if above)', value=0.90, min_value=0.50, max_value=0.99, format="%.2f", step=0.05)
				guardrail_target = st.number_input('Target success rate (reset to)', value=0.85, min_value=0.50, max_value=0.99, format="%.2f", step=0.05)
				guardrail_inner_sims = st.number_input('Inner MC simulations per check', value=200, min_value=50, max_value=1000, step=50)
				guardrail_max_spending_pct = st.number_input(
					'Max spending cap (% above base withdrawal, 0=no cap)',
					value=50.0, min_value=0.0, max_value=200.0, format="%.0f", step=10.0,
					help='E.g. 50 means spending can never exceed 150% of the base withdrawal amount')
			else:
				guardrail_lower = 0.75
				guardrail_upper = 0.90
				guardrail_target = 0.85
				guardrail_inner_sims = 200
				guardrail_max_spending_pct = 0.0

		with st.expander('Simulation Settings'):
			display_decimals = st.number_input('Decimal places for tables/charts', min_value=0, max_value=6, value=0, step=1)
			monte_carlo_runs = st.number_input('Monte Carlo runs', min_value=50, max_value=5000, value=1000, step=50)

	years = max(1, max(life_expectancy_primary - start_age, life_expectancy_spouse - start_age_spouse) + 1)
	st.markdown(f"**Beginning portfolio: ${float(taxable_start) + float(tda_start) + float(tda_spouse_start) + float(roth_start):,.0f}**")

	# Build year-by-year withdrawal schedule from period inputs
	withdrawal_schedule = []
	for period_years, period_amount in withdrawal_schedule_inputs:
		withdrawal_schedule.extend([period_amount] * period_years)
	# Pad or trim to match simulation years
	if len(withdrawal_schedule) < years:
		last_val = withdrawal_schedule[-1] if withdrawal_schedule else 0.0
		withdrawal_schedule.extend([last_val] * (years - len(withdrawal_schedule)))
	withdrawal_schedule = withdrawal_schedule[:years]
	withdraw_amount = withdrawal_schedule[0]  # for backward-compat references

	currency_fmt = f'${{:,.{int(display_decimals)}f}}'

	# Common simulation parameters (used by all modes)
	sim_params = dict(
		start_age_primary=int(start_age), start_age_spouse=int(start_age_spouse),
		taxable_start=float(taxable_start), stock_total_return=float(stock_total_return),
		stock_dividend_yield=float(stock_dividend_yield), stock_turnover=float(stock_turnover),
		bond_return=float(bond_return), roth_start=float(roth_start),
		tda_start=float(tda_start), tda_spouse_start=float(tda_spouse_start),
		target_stock_pct=float(target_stock_pct),
		taxable_stock_basis_pct=float(taxable_stock_basis_pct),
		taxable_bond_basis_pct=float(taxable_bond_basis_pct),
		withdrawal_schedule=withdrawal_schedule,
		rmd_start_age=int(rmd_start_age), rmd_start_age_spouse=int(rmd_start_age_spouse),
		ss_income_annual=float(ss_income_input), ss_income_spouse_annual=float(ss_income_spouse_input),
		ss_cola=float(ss_cola),
		pension_income_annual=float(pension_income_input),
		pension_income_spouse_annual=float(pension_income_spouse_input),
		pension_cola=float(pension_cola), other_income_annual=float(other_income_input),
		filing_status=filing_status_key, use_itemized_deductions=bool(use_itemized),
		itemized_deduction_amount=float(itemized_deduction_input),
		roth_conversion_amount=float(roth_conversion_amount),
		roth_conversion_years=int(roth_conversion_years),
		roth_conversion_tax_source='taxable' if roth_conversion_tax_source == 'Taxable' else 'tda',
		life_expectancy_primary=int(life_expectancy_primary),
		life_expectancy_spouse=int(life_expectancy_spouse),
		guardrails_enabled=bool(guardrails_enabled),
		guardrail_lower=float(guardrail_lower),
		guardrail_upper=float(guardrail_upper),
		guardrail_target=float(guardrail_target),
		guardrail_inner_sims=int(guardrail_inner_sims),
		guardrail_max_spending_pct=float(guardrail_max_spending_pct),
	)

	# Compute blended return parameters for guardrail inner MC
	if guardrails_enabled:
		if return_mode == 'Simulated (lognormal)':
			stock_mu = float(taxable_log_drift)
			stock_sigma = float(taxable_log_volatility)
			bond_mu = float(bond_log_drift)
			bond_sigma = float(bond_log_volatility)
		else:
			# Estimate params from historical data
			mg_df = load_master_global()
			stock_factors = mg_df['LBM 100E'].dropna().values
			stock_log_rets = np.log(stock_factors)
			stock_mu = float(np.mean(stock_log_rets))
			stock_sigma = float(np.std(stock_log_rets))
			bond_log_returns = np.log(1.0 + load_bond_factors())
			bond_mu = float(np.mean(bond_log_returns))
			bond_sigma = float(np.std(bond_log_returns))
		blended_mu = target_stock_pct * stock_mu + (1 - target_stock_pct) * bond_mu
		blended_sigma = target_stock_pct * stock_sigma + (1 - target_stock_pct) * bond_sigma
		sim_params['blended_mu'] = blended_mu
		sim_params['blended_sigma'] = blended_sigma
	else:
		sim_params['blended_mu'] = 0.0
		sim_params['blended_sigma'] = 0.0

	if st.button('Run simulation'):
		sim_years = int(years)
		if return_mode == 'Historical (master_global_factors)' and historical_mode_type == 'All periods (distribution)':
			windows, window_start_dates = get_all_historical_windows(sim_years)
			n_windows = len(windows)
			hist_spinner = f'Running {n_windows} historical periods...'
			if guardrails_enabled:
				hist_spinner += ' (with guardrails — this may take longer)'
			with st.spinner(hist_spinner):
				results = []
				all_yearly = []
				for run_idx, (stock_rets, bond_rets) in enumerate(windows):
					df_run = simulate_withdrawals(
						years=sim_years, stock_return_series=stock_rets,
						bond_return_series=bond_rets, **sim_params)
					df_run['total_portfolio'] = df_run['end_taxable_total'] + df_run['end_tda_total'] + df_run['end_roth']
					metrics = compute_summary_metrics(df_run, float(inheritor_marginal_rate))
					results.append(metrics)
					df_run['run'] = run_idx
					all_yearly.append(df_run)
				all_yearly_df = pd.concat(all_yearly, ignore_index=True)
			store_distribution_results(results, all_yearly_df, 'historical_dist')
			st.session_state['window_start_dates'] = {i: d.strftime('%Y-%m') for i, d in enumerate(window_start_dates)}
		elif return_mode == 'Historical (master_global_factors)':
			hist_stock, hist_bond, hist_count = get_historical_annual_returns(int(historical_start_year), sim_years)
			if hist_count < sim_years:
				st.warning(f'Only {hist_count} years of historical data available from {historical_start_year}. Simulation truncated.')
				sim_years = hist_count
			df = simulate_withdrawals(
				years=sim_years, stock_return_series=hist_stock,
				bond_return_series=hist_bond, **sim_params)
			st.session_state['sim_mode'] = 'historical'
			st.session_state['sim_df'] = df
			st.session_state.pop('mc_all_yearly', None)
			st.session_state.pop('mc_percentile_rows', None)
		else:
			spinner_msg = f'Running {int(monte_carlo_runs)} simulations...'
			if guardrails_enabled:
				spinner_msg += ' (with guardrails — this may take longer)'
			with st.spinner(spinner_msg):
				mc_results, all_yearly_df = run_monte_carlo(
					num_runs=int(monte_carlo_runs),
					start_age=int(start_age),
					start_age_spouse=int(start_age_spouse),
					taxable_start=float(taxable_start),
					stock_total_return=float(stock_total_return),
					stock_dividend_yield=float(stock_dividend_yield),
					stock_turnover=float(stock_turnover),
					bond_return=float(bond_return),
					roth_start=float(roth_start),
					tda_start=float(tda_start),
					tda_spouse_start=float(tda_spouse_start),
					withdrawal_schedule=withdrawal_schedule,
					target_stock_pct=float(target_stock_pct),
					taxable_stock_basis_pct=float(taxable_stock_basis_pct),
					taxable_bond_basis_pct=float(taxable_bond_basis_pct),
					rmd_start_age=int(rmd_start_age),
					rmd_start_age_spouse=int(rmd_start_age_spouse),
					ss_income_annual=float(ss_income_input),
					ss_income_spouse_annual=float(ss_income_spouse_input),
					ss_cola=float(ss_cola),
					pension_income_annual=float(pension_income_input),
					pension_income_spouse_annual=float(pension_income_spouse_input),
					pension_cola=float(pension_cola),
					other_income_annual=float(other_income_input),
					filing_status=filing_status_key,
					use_itemized_deductions=bool(use_itemized),
					itemized_deduction_amount=float(itemized_deduction_input),
					roth_conversion_amount=float(roth_conversion_amount),
					roth_conversion_years=int(roth_conversion_years),
					roth_conversion_tax_source='taxable' if roth_conversion_tax_source == 'Taxable' else 'tda',
					taxable_log_drift=float(taxable_log_drift),
					taxable_log_volatility=float(taxable_log_volatility),
					bond_log_drift=float(bond_log_drift),
					bond_log_volatility=float(bond_log_volatility),
					life_expectancy_primary=int(life_expectancy_primary),
					life_expectancy_spouse=int(life_expectancy_spouse),
					inheritor_rate=float(inheritor_marginal_rate),
					years=int(years),
					guardrails_enabled=bool(guardrails_enabled),
					guardrail_lower=float(guardrail_lower),
					guardrail_upper=float(guardrail_upper),
					guardrail_target=float(guardrail_target),
					guardrail_inner_sims=int(guardrail_inner_sims),
					blended_mu=sim_params['blended_mu'],
					blended_sigma=sim_params['blended_sigma'],
					guardrail_max_spending_pct=float(guardrail_max_spending_pct),
				)
			store_distribution_results(mc_results, all_yearly_df, 'simulated')

	# ── Display results ──────────────────────────────────────────
	sim_mode = st.session_state.get('sim_mode')

	if sim_mode in ('simulated', 'historical_dist') and 'mc_percentile_rows' in st.session_state:
		pct_rows = st.session_state['mc_percentile_rows']
		percentile_df = pd.DataFrame(pct_rows)
		st.subheader('Distribution summary')
		num_sims = st.session_state.get('num_sims', 0)
		if sim_mode == 'historical_dist':
			st.caption(f'{num_sims} historical periods')
		else:
			st.caption(f'{num_sims} simulations')
		st.dataframe(percentile_df.style.format({
			'percentile': lambda x: f"{int(x)}th",
			'after_tax_end': currency_fmt,
			'total_taxes': currency_fmt,
			'effective_tax_rate': '{:.2%}'.format,
			'portfolio_cagr': '{:.2%}'.format,
			'roth_cagr': '{:.2%}'.format,
		}))
		pct_non_positive = st.session_state.get('mc_pct_non_positive', 0.0)
		st.caption(f"Percent of ending values ≤ 0: {pct_non_positive * 100:.1f}%")

		all_yearly = st.session_state['mc_all_yearly']
		run_ending_portfolios = all_yearly.groupby('run')['total_portfolio'].last()
		target_ending_value = st.number_input('Target ending portfolio value', value=500000.0, step=50000.0, key='target_ending_val')
		pct_at_or_above = float((run_ending_portfolios >= target_ending_value).mean()) * 100
		st.metric('Chance of ending with at least this amount', f"{pct_at_or_above:.1f}%")
		spending_pivot = all_yearly.pivot(index='year', columns='run', values='after_tax_spending')
		spending_pivot.columns = [f'Run {c}' for c in spending_pivot.columns]
		spending_csv = spending_pivot.to_csv()
		st.download_button('Download after-tax spending (all runs) as CSV', spending_csv, file_name='after_tax_spending_all_runs.csv', mime='text/csv')

		# Per-run spending summary: total and average annual spending across the retirement period
		st.subheader('Lifetime spending distribution (per run)')
		run_spending = all_yearly.groupby('run').agg(
			total_withdrawal_used=('withdrawal_used', 'sum'),
			total_after_tax_spending=('after_tax_spending', 'sum'),
			years_in_run=('year', 'count'),
		)
		run_spending['avg_annual_withdrawal'] = run_spending['total_withdrawal_used'] / run_spending['years_in_run']
		run_spending['avg_annual_after_tax_spending'] = run_spending['total_after_tax_spending'] / run_spending['years_in_run']
		spending_pctiles = [0, 10, 25, 50, 75, 90, 100]
		spending_pct_rows = []
		for p in spending_pctiles:
			spending_pct_rows.append({
				'percentile': p,
				'avg_annual_withdrawal': np.percentile(run_spending['avg_annual_withdrawal'], p),
				'avg_annual_after_tax_spending': np.percentile(run_spending['avg_annual_after_tax_spending'], p),
				'total_lifetime_withdrawal': np.percentile(run_spending['total_withdrawal_used'], p),
				'total_lifetime_after_tax_spending': np.percentile(run_spending['total_after_tax_spending'], p),
			})
		spending_pct_df = pd.DataFrame(spending_pct_rows)
		sched = sim_params.get('withdrawal_schedule', [])
		base_avg_wd = np.mean(sched) if sched else 0
		if base_avg_wd > 0:
			median_avg_wd = np.percentile(run_spending['avg_annual_withdrawal'], 50)
			pct_above = (median_avg_wd / base_avg_wd - 1) * 100
			direction = 'above' if pct_above >= 0 else 'below'
			st.caption(f'Base avg withdrawal target: {currency_fmt.format(base_avg_wd)} | '
					   f'Median avg annual withdrawal: {currency_fmt.format(median_avg_wd)} '
					   f'({abs(pct_above):.1f}% {direction} target)')
		st.dataframe(spending_pct_df.style.format({
			'percentile': lambda x: f"{int(x)}th",
			'avg_annual_withdrawal': currency_fmt,
			'avg_annual_after_tax_spending': currency_fmt,
			'total_lifetime_withdrawal': currency_fmt,
			'total_lifetime_after_tax_spending': currency_fmt,
		}))

		# Year-by-year median table (50th percentile across all simulations)
		st.subheader('Year-by-year median across all runs')
		st.caption('Each value is the 50th percentile across all simulation runs for that year — not a single run.')
		median_cols = ['age_p1', 'withdrawal_used', 'after_tax_spending', 'total_portfolio',
					   'end_taxable_total', 'end_tda_total', 'end_roth',
					   'total_taxes', 'rmd_total', 'withdraw_from_tda', 'withdraw_from_roth',
					   'withdraw_from_taxable_net', 'ss_income_total', 'ordinary_taxable_income',
					   'capital_gains', 'effective_tax_rate_calc']
		all_yearly['effective_tax_rate_calc'] = (
			all_yearly['total_taxes'] /
			(all_yearly['ordinary_taxable_income'] + all_yearly['capital_gains']).replace(0, np.nan)
		).fillna(0.0)
		yearly_median = all_yearly.groupby('year')[median_cols].median()
		yearly_median.columns = ['Age', 'Withdrawal Target', 'After-Tax Spending', 'Total Portfolio',
								 'Taxable', 'TDA', 'Roth',
								 'Total Taxes', 'RMDs', 'Withdraw TDA', 'Withdraw Roth',
								 'Withdraw Taxable', 'SS Income', 'Ordinary Income',
								 'Cap Gains', 'Eff Tax Rate']
		st.dataframe(yearly_median.style.format({
			'Age': '{:.0f}',
			'Total Portfolio': currency_fmt, 'Taxable': currency_fmt, 'TDA': currency_fmt, 'Roth': currency_fmt,
			'Total Taxes': currency_fmt, 'RMDs': currency_fmt,
			'Withdraw TDA': currency_fmt, 'Withdraw Roth': currency_fmt, 'Withdraw Taxable': currency_fmt,
			'SS Income': currency_fmt, 'Ordinary Income': currency_fmt, 'Cap Gains': currency_fmt,
			'Eff Tax Rate': '{:.2%}'.format,
			'Withdrawal Target': currency_fmt, 'After-Tax Spending': currency_fmt,
		}))

		st.subheader('Total portfolio value — percentile bands across all runs')
		port_band_options = {'0th (min)': 0.00, '5th': 0.05, '10th': 0.10, '25th': 0.25, '50th (median)': 0.50, '75th': 0.75, '90th': 0.90}
		port_band_defaults = ['0th (min)', '10th', '25th', '50th (median)', '75th', '90th']
		selected_port_bands = st.multiselect('Percentile bands to display', list(port_band_options.keys()), default=port_band_defaults, key='port_bands')
		if selected_port_bands:
			port_quantiles = [port_band_options[b] for b in selected_port_bands]
			port_pcts = all_yearly.groupby('year')['total_portfolio'].quantile(port_quantiles).unstack()
			port_pcts.columns = selected_port_bands
			interactive_line_chart(port_pcts, y_title='Portfolio Value')

		st.subheader('Account balances — median across all runs')
		acct_median = all_yearly.groupby('year')[['total_portfolio', 'end_taxable_total', 'end_tda_total', 'end_roth']].median()
		acct_median.columns = ['Total Portfolio', 'Taxable', 'Tax-deferred', 'Roth']
		interactive_line_chart(acct_median, y_title='Balance')

		st.subheader('Annual taxes — percentile bands across all runs')
		tax_pcts = all_yearly.groupby('year')['total_taxes'].quantile([0.00, 0.10, 0.25, 0.50, 0.75, 0.90]).unstack()
		tax_pcts.columns = ['0th (min)', '10th', '25th', '50th (median)', '75th', '90th']
		interactive_line_chart(tax_pcts, y_title='Taxes')

		# Guardrail-specific charts: withdrawal target and after-tax spending
		if 'withdrawal_used' in all_yearly.columns:
			st.subheader('Withdrawal target — percentile bands across all runs')
			wd_pcts = all_yearly.groupby('year')['withdrawal_used'].quantile([0.00, 0.10, 0.25, 0.50, 0.75, 0.90]).unstack()
			wd_pcts.columns = ['0th (min)', '10th', '25th', '50th (median)', '75th', '90th']
			wd_data_min = float(wd_pcts.values.min())
			wd_data_max = float(wd_pcts.values.max())
			wd_pad = max(wd_data_max - wd_data_min, float(withdraw_amount) * 0.1) * 0.15
			wd_y_min = wd_data_min - wd_pad
			wd_y_max = wd_data_max + wd_pad
			wd_long = wd_pcts.reset_index().melt('year', var_name='Percentile', value_name='Withdrawal')
			wd_chart = alt.Chart(wd_long).mark_line().encode(
				x=alt.X('year:Q', title='Year'),
				y=alt.Y('Withdrawal:Q', scale=alt.Scale(domain=[wd_y_min, wd_y_max]), title='Withdrawal ($)'),
				color=alt.Color('Percentile:N', sort=['0th (min)', '10th', '25th', '50th (median)', '75th', '90th'])
			).properties(height=400)
			st.altair_chart(wd_chart, use_container_width=True)

			st.subheader('After-tax spending — percentile bands across all runs')
			spend_pcts = all_yearly.groupby('year')['after_tax_spending'].quantile([0.00, 0.10, 0.25, 0.50, 0.75, 0.90]).unstack()
			spend_pcts.columns = ['0th (min)', '10th', '25th', '50th (median)', '75th', '90th']
			sp_data_min = float(spend_pcts.values.min())
			sp_data_max = float(spend_pcts.values.max())
			sp_pad = max(sp_data_max - sp_data_min, float(withdraw_amount) * 0.1) * 0.15
			sp_long = spend_pcts.reset_index().melt('year', var_name='Percentile', value_name='Spending')
			sp_chart = alt.Chart(sp_long).mark_line().encode(
				x=alt.X('year:Q', title='Year'),
				y=alt.Y('Spending:Q', scale=alt.Scale(domain=[sp_data_min - sp_pad, sp_data_max + sp_pad]), title='After-Tax Spending ($)'),
				color=alt.Color('Percentile:N', sort=['0th (min)', '10th', '25th', '50th (median)', '75th', '90th'])
			).properties(height=400)
			st.altair_chart(sp_chart, use_container_width=True)

		st.markdown('---')
		# Interactive run selector
		window_dates = st.session_state.get('window_start_dates', {})
		run_ids = sorted(all_yearly['run'].unique())
		run_ends = all_yearly.groupby('run')['total_portfolio'].last()
		median_val = run_ends.median()
		median_run_idx = int((run_ends - median_val).abs().idxmin())

		if sim_mode == 'historical_dist' and window_dates:
			options = {f"Median run (start {window_dates.get(median_run_idx, '?')})": median_run_idx}
			for rid in run_ids:
				label = f"Start {window_dates.get(rid, rid)}"
				options[label] = rid
			selected_label = st.selectbox('Select run to display below', list(options.keys()))
			selected_run_idx = options[selected_label]
			st.subheader(f'Single run detail — {selected_label}')
		elif sim_mode == 'simulated':
			run_avg_spending = all_yearly.groupby('run')['after_tax_spending'].mean()
			selected_pct = st.number_input('Percentile of avg annual spending to display', value=50, min_value=0, max_value=100, step=5)
			target_val = np.percentile(run_avg_spending, selected_pct)
			selected_run_idx = int((run_avg_spending - target_val).abs().idxmin())
			avg_val = run_avg_spending[selected_run_idx]
			st.subheader(f'Single run detail — {selected_pct}th percentile (Average Annual After-Tax Spending ${avg_val:,.0f})')
		else:
			selected_run_idx = median_run_idx
			st.subheader('Single representative run (run closest to median ending value)')
		st.caption('All tables and charts below are from one specific simulation run — internally consistent year by year.')

		selected_df = all_yearly[all_yearly['run'] == selected_run_idx].drop(columns=['run', 'total_portfolio']).reset_index(drop=True)
		st.session_state['sim_df'] = selected_df

	df = st.session_state.get('sim_df')
	if df is not None and sim_mode is not None:
		if sim_mode == 'historical':
			st.subheader('Historical simulation detail')

		last = df.iloc[-1]
		display_df = df.round(int(display_decimals)).copy()
		display_df['portfolio_return'] = df['portfolio_return']
		display_df['roth_return_used'] = df['roth_return_used']

		years_simulated = len(df)
		portfolio_growth_factor = (df['portfolio_return'] + 1.0).prod()
		roth_growth_factor = (df['roth_return_used'] + 1.0).prod()
		portfolio_cagr = (portfolio_growth_factor ** (1.0 / years_simulated) - 1.0) if years_simulated > 0 else 0.0
		roth_cagr = (roth_growth_factor ** (1.0 / years_simulated) - 1.0) if years_simulated > 0 else 0.0
		avg_annual_after_tax = df['after_tax_spending'].mean()
		first = df.iloc[0]
		beginning_portfolio = first['start_stocks_mv'] + first['start_bonds_mv'] + first['start_tda_p1'] + first['start_tda_p2'] + first['start_roth']
		after_tax_ending = (last['end_stocks_mv'] + last['end_bonds_mv'] + last['end_roth'] +
			last['end_tda_total'] * max(0.0, 1.0 - float(inheritor_marginal_rate)))
		c1, c2, c3, c4, c5 = st.columns(5)
		c1.metric('Beginning Portfolio', f"${beginning_portfolio:,.0f}")
		c2.metric('After-Tax Ending Value', f"${after_tax_ending:,.0f}")
		c3.metric('Portfolio CAGR', f"{portfolio_cagr:.2%}")
		c4.metric('Roth CAGR', f"{roth_cagr:.2%}")
		c5.metric('Avg Annual After-Tax Spending', f"${avg_annual_after_tax:,.0f}")

		st.subheader('Year-by-year detail — single run')
		st.caption(f'Years simulated: {len(df)} | This is one complete simulation path.')
		# Reorder columns: put after_tax_spending and withdrawal_used before portfolio_return
		detail_col_order = [
			'year', 'age_p1', 'age_p2', 'withdrawal_used', 'after_tax_spending', 'portfolio_return', 'roth_return_used',
			'start_stocks_mv', 'start_bonds_mv', 'start_stocks_basis', 'start_bonds_basis',
			'start_tda_p1', 'start_tda_p2', 'start_roth',
			'rmd_divisor_p1', 'rmd_divisor_p2', 'rmd_p1', 'rmd_p2', 'rmd_total',
			'withdraw_from_taxable_net', 'withdraw_from_tda', 'withdraw_from_roth',
			'rmd_excess_to_taxable', 'gross_sold_taxable_bonds', 'gross_sold_taxable_stocks',
			'ss_income_total', 'taxable_social_security', 'pension_income_total', 'other_income',
			'roth_conversion', 'roth_conversion_tax', 'roth_conversion_tax_source',
			'deduction_applied', 'ordinary_taxable_income', 'capital_gains',
			'ordinary_tax_total', 'capital_gains_tax', 'niit_tax', 'total_taxes',
			'marginal_ordinary_rate', 'marginal_cap_gains_rate',
			'end_stocks_mv', 'end_bonds_mv', 'end_stocks_basis', 'end_bonds_basis',
			'end_taxable_total', 'investment_return_dollars',
			'end_tda_p1', 'end_tda_p2', 'end_tda_total', 'end_roth',
		]
		detail_col_order = [c for c in detail_col_order if c in display_df.columns]
		display_df = display_df[detail_col_order]
		st.dataframe(display_df.style.format({
			'start_stocks_mv': currency_fmt, 'start_bonds_mv': currency_fmt, 'start_stocks_basis': currency_fmt, 'start_bonds_basis': currency_fmt,
			'start_tda_p1': currency_fmt, 'start_tda_p2': currency_fmt, 'start_roth': currency_fmt,
			'rmd_p1': currency_fmt, 'rmd_p2': currency_fmt, 'rmd_total': currency_fmt, 'withdraw_from_taxable_net': currency_fmt, 'withdraw_from_tda': currency_fmt, 'withdraw_from_roth': currency_fmt,
			'rmd_excess_to_taxable': currency_fmt,
			'gross_sold_taxable_bonds': currency_fmt, 'gross_sold_taxable_stocks': currency_fmt,
			'ss_income_total': currency_fmt, 'taxable_social_security': currency_fmt, 'pension_income_total': currency_fmt, 'other_income': currency_fmt,
			'roth_conversion': currency_fmt, 'roth_conversion_tax': currency_fmt,
			'deduction_applied': currency_fmt, 'ordinary_taxable_income': currency_fmt,
			'capital_gains': currency_fmt,
			'end_stocks_mv': currency_fmt, 'end_bonds_mv': currency_fmt, 'end_stocks_basis': currency_fmt, 'end_bonds_basis': currency_fmt,
			'end_taxable_total': currency_fmt, 'investment_return_dollars': currency_fmt,
			'end_tda_p1': currency_fmt, 'end_tda_p2': currency_fmt, 'end_tda_total': currency_fmt, 'end_roth': currency_fmt,
			'ordinary_tax_total': currency_fmt, 'capital_gains_tax': currency_fmt, 'niit_tax': currency_fmt, 'total_taxes': currency_fmt,
			'marginal_ordinary_rate': '{:.2%}'.format, 'marginal_cap_gains_rate': '{:.2%}'.format,
			'withdrawal_used': currency_fmt, 'after_tax_spending': currency_fmt,
			'portfolio_return': '{:.2%}'.format,
			'roth_return_used': '{:.2%}'.format
		}))
		csv_data = display_df.to_csv(index=False)
		st.download_button('Download single run detail as CSV', csv_data, file_name='single_run_detail.csv', mime='text/csv')

		currency_round = df.round(int(display_decimals))
		st.subheader('Where withdrawals came from — single run (stacked)')
		chart_df = currency_round[['year','withdraw_from_taxable_net','withdraw_from_tda','withdraw_from_roth']].set_index('year')
		chart_df.columns = ['Taxable', 'TDA', 'Roth']
		st.bar_chart(chart_df)

		st.subheader('After-tax spending over time — single run')
		spending_chart_data = df[['year', 'after_tax_spending']].copy()
		spending_chart_data['goal'] = [withdrawal_schedule[i] if i < len(withdrawal_schedule) else withdrawal_schedule[-1] for i in range(len(spending_chart_data))]
		spending_chart_data['status'] = spending_chart_data.apply(
			lambda r: 'Below goal' if r['after_tax_spending'] < r['goal'] else 'At or above goal', axis=1)
		spend_bar = alt.Chart(spending_chart_data).mark_bar().encode(
			x=alt.X('year:O', title='Year'),
			y=alt.Y('after_tax_spending:Q', title='After-Tax Spending ($)'),
			color=alt.Color('status:N',
				scale=alt.Scale(domain=['Below goal', 'At or above goal'], range=['#d62728', '#1f77b4']),
				legend=alt.Legend(title=''))
		).properties(height=400)
		st.altair_chart(spend_bar, use_container_width=True)

		st.subheader('Account balances over time — single run')
		bal_df = currency_round[['year','end_taxable_total','end_tda_total','end_roth']].set_index('year')
		bal_df.columns = ['Taxable', 'Tax-deferred', 'Roth']
		interactive_line_chart(bal_df, y_title='Balance')

		st.subheader('Taxes paid per year — single run')
		tax_df = currency_round[['year','ordinary_taxable_income','ordinary_tax_total','capital_gains','capital_gains_tax','niit_tax','total_taxes','deduction_applied','roth_conversion','roth_conversion_tax']].set_index('year')
		st.dataframe(tax_df.style.format({
			'ordinary_taxable_income': currency_fmt,
			'ordinary_tax_total': currency_fmt,
			'capital_gains': currency_fmt,
			'capital_gains_tax': currency_fmt,
			'niit_tax': currency_fmt,
			'total_taxes': currency_fmt,
			'deduction_applied': currency_fmt,
			'roth_conversion': currency_fmt,
			'roth_conversion_tax': currency_fmt
		}))
		effective_income = df['ordinary_taxable_income'] + df['capital_gains']
		effective_tax_rate = (df['total_taxes'] / effective_income.replace(0, np.nan)).fillna(0.0)
		mtr_df = pd.DataFrame({
			'year': df['year'],
			'taxable_income_total': effective_income,
			'total_taxes': df['total_taxes'],
			'effective_tax_rate': effective_tax_rate,
			'marginal_ordinary_rate': df['marginal_ordinary_rate'],
			'marginal_cap_gains_rate': df['marginal_cap_gains_rate'],
			'niit_tax': df['niit_tax'],
		}).set_index('year')
		st.subheader('Effective and marginal tax rates — single run')
		st.dataframe(mtr_df.style.format({
			'taxable_income_total': currency_fmt,
			'total_taxes': currency_fmt,
			'effective_tax_rate': '{:.2%}'.format,
			'marginal_ordinary_rate': '{:.2%}'.format,
			'marginal_cap_gains_rate': '{:.2%}'.format,
			'niit_tax': currency_fmt,
		}))
		rate_df = mtr_df[['effective_tax_rate','marginal_ordinary_rate','marginal_cap_gains_rate']].copy()
		rate_df.columns = ['Effective Rate', 'Marginal Ordinary', 'Marginal Cap Gains']
		interactive_line_chart(rate_df, y_title='Tax Rate', fmt='.1%')
		lifetime_taxes = df['total_taxes'].sum()
		lifetime_ordinary_tax = df['ordinary_tax_total'].sum()
		lifetime_cap_gains_tax = df['capital_gains_tax'].sum()
		st.metric('Total lifetime taxes paid', currency_fmt.format(lifetime_taxes))
		st.caption(f'Ordinary: {currency_fmt.format(lifetime_ordinary_tax)} | Capital gains/QD: {currency_fmt.format(lifetime_cap_gains_tax)}')

		st.markdown('---')
		st.write('Ending balances')
		last_row = currency_round.iloc[-1]
		taxable_total_end = last_row['end_stocks_mv'] + last_row['end_bonds_mv']
		st.write({'taxable_end': taxable_total_end, 'stocks_end': last_row['end_stocks_mv'], 'stocks_end_basis': last_row['end_stocks_basis'], 'bonds_end': last_row['end_bonds_mv'], 'bonds_end_basis': last_row['end_bonds_basis'], 'tda_end': last_row['end_tda_total'], 'roth_end': last_row['end_roth']})

		# Store latest summary in session for post-run saving
		last = df.iloc[-1]
		total_accounts = last['end_stocks_mv'] + last['end_bonds_mv'] + last['end_tda_total'] + last['end_roth']
		after_tax_end = float(last['end_stocks_mv'] + last['end_bonds_mv'])
		after_tax_end += float(last['end_roth'])
		after_tax_end += float(last['end_tda_total']) * max(0.0, 1.0 - inheritor_marginal_rate)
		summary_data = {
			'label': f"conversion ${roth_conversion_amount:,.0f} for {roth_conversion_years} yrs, taxes from {roth_conversion_tax_source}",
			'total_taxes': float(lifetime_taxes),
			'total_accounts': float(total_accounts),
			'taxable_end': float(last['end_stocks_mv'] + last['end_bonds_mv']),
			'tda_end': float(last['end_tda_total']),
			'roth_end': float(last['end_roth']),
			'inheritor_marginal_rate': float(inheritor_marginal_rate),
			'after_tax_end': after_tax_end,
			'portfolio_cagr': portfolio_cagr,
			'roth_cagr': roth_cagr,
			'sim_type': sim_mode,
		}
		if sim_mode in ('simulated', 'historical_dist'):
			summary_data['percentiles'] = st.session_state.get('mc_percentile_rows', [])
			summary_data['pct_non_positive_end'] = st.session_state.get('mc_pct_non_positive', 0.0)
		st.session_state['last_summary'] = summary_data
	elif sim_mode is None:
		st.info('Set inputs and click "Run simulation" to see results.')

	# ── Scenario saving ──────────────────────────────────────────
	scenario_name_default = st.session_state.get('last_summary', {}).get('name', 'Scenario 1')
	scenario_name = st.text_input('Scenario name to save', value=scenario_name_default)
	if 'scenario_summaries' not in st.session_state:
		st.session_state['scenario_summaries'] = []

	if st.button('Save scenario'):
		if 'last_summary' not in st.session_state:
			st.warning('Run a simulation first.')
		else:
			summary = dict(st.session_state['last_summary'])
			summary['name'] = scenario_name
			st.session_state['scenario_summaries'] = [s for s in st.session_state['scenario_summaries'] if s['name'] != scenario_name]
			if len(st.session_state['scenario_summaries']) >= 5:
				st.warning('Maximum of 5 scenarios saved. Remove one by reusing a name.')
			else:
				st.session_state['scenario_summaries'].append(summary)
				st.success(f"Saved scenario '{scenario_name}'.")

	if st.session_state.get('scenario_summaries'):
		st.markdown('### Saved scenarios')
		compare_rows = []
		for s in st.session_state['scenario_summaries']:
			compare_rows.append({
				'name': s['name'],
				'type': s.get('sim_type', 'historical'),
				'after_tax_end': s['after_tax_end'],
				'total_taxes': s['total_taxes'],
				'portfolio_cagr': s['portfolio_cagr'],
				'roth_cagr': s['roth_cagr'],
				'pct_ran_out': s.get('pct_non_positive_end', None),
			})
		compare_df = pd.DataFrame(compare_rows).set_index('name')
		st.dataframe(compare_df.style.format({
			'after_tax_end': currency_fmt,
			'total_taxes': currency_fmt,
			'portfolio_cagr': '{:.2%}'.format,
			'roth_cagr': '{:.2%}'.format,
			'pct_ran_out': lambda x: f'{x*100:.1f}%' if x is not None else 'N/A',
			'type': lambda x: x,
		}))
		# Show percentile detail for distribution scenarios
		for s in st.session_state['scenario_summaries']:
			if s.get('sim_type') == 'simulated' and s.get('percentiles'):
				st.markdown(f"**{s['name']} — percentile detail**")
				pct_df = pd.DataFrame(s['percentiles'])
				st.dataframe(pct_df.style.format({
					'percentile': lambda x: f"{int(x)}th",
					'after_tax_end': currency_fmt,
					'total_taxes': currency_fmt,
					'effective_tax_rate': '{:.2%}'.format,
					'portfolio_cagr': '{:.2%}'.format,
					'roth_cagr': '{:.2%}'.format,
				}))


if __name__ == '__main__':
	main()
