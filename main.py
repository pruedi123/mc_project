import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from typing import Dict, Optional, Sequence

st.set_page_config(page_title='Withdrawal + RMD Simulator', layout='wide')

# Load median purchasing power factors for pension real-value adjustment (years 1-40)
_pp_df = pd.read_excel('median_cpi_purchasing_power.xlsx')
PP_FACTORS = _pp_df['Median_Purchasing_Power'].tolist()

# Load monthly CPI factors for per-run historical purchasing power
_cpi_mo_df = pd.read_excel('cpi_mo_factors.xlsx')
CPI_MO_FACTORS = _cpi_mo_df['cpi_factors'].values

def compute_run_pp_factors(start_idx: int, years: int) -> list:
	"""Compute cumulative purchasing power factors for a historical run
	starting at CPI monthly index start_idx, for each year 1..years.
	Compounds 12 monthly CPI factors per year."""
	pp = []
	cum = 1.0
	for y in range(years):
		mo_start = start_idx + y * 12
		mo_end = mo_start + 12
		if mo_end <= len(CPI_MO_FACTORS):
			for m in range(mo_start, mo_end):
				cum *= CPI_MO_FACTORS[m]
		pp.append(cum)
	return pp

def interactive_line_chart(data_df, y_title='Value', fmt='$,.0f', height=400, zero_base=True):
	"""Convert a DataFrame (index=x, columns=series) to an interactive Altair line chart with tooltips."""
	long = data_df.reset_index().melt(data_df.index.name or 'index', var_name='Series', value_name='value')
	x_col = data_df.index.name or 'index'
	nearest = alt.selection_point(nearest=True, on='pointerover', fields=[x_col], empty=False)
	y_scale = alt.Scale(zero=True) if zero_base else alt.Scale(zero=False)
	base = alt.Chart(long).encode(
		x=alt.X(f'{x_col}:Q', title=x_col.replace('_', ' ').title()),
		y=alt.Y('value:Q', title=y_title, axis=alt.Axis(format=fmt), scale=y_scale),
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

def forward_success_rate(portfolio, remaining_schedule, scale_factor, blended_mu, blended_sigma, n_sims=200, income_schedule=None):
	"""Fast vectorized MC to estimate probability portfolio survives the remaining schedule.
	remaining_schedule is a list of base withdrawal amounts for each remaining year.
	scale_factor is a multiplier applied to every element of the schedule.
	income_schedule offsets spending (SS, pension, other) so only the net draw hits the portfolio.
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
		inc = income_schedule[y] if income_schedule is not None and y < len(income_schedule) else 0.0
		net_draw = max(0.0, remaining_schedule[y] * scale_factor - inc)
		balances *= growth_factors[:, y]
		balances -= net_draw
		balances = np.maximum(balances, 0.0)
	return float(np.mean(balances > 0))

def find_sustainable_scale_factor(portfolio, remaining_schedule, blended_mu, blended_sigma, target_success=0.85, n_sims=200, tol=0.005, income_schedule=None):
	"""Binary search for the scaling factor on the remaining withdrawal schedule
	that gives target_success survival rate.
	Returns a multiplier (e.g. 1.0 = base schedule, 0.85 = 85% of base, 1.2 = 120% of base).
	income_schedule offsets spending so only the net portfolio draw is used.
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
			inc = income_schedule[y] if income_schedule is not None and y < len(income_schedule) else 0.0
			net_draw = max(0.0, remaining_schedule[y] * scale - inc)
			balances *= growth_factors[:, y]
			balances -= net_draw
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

def build_scenario_params(base_params: dict, overrides: dict, stock_mu: float = 0.0, stock_sigma: float = 0.0, bond_mu: float = 0.0, bond_sigma: float = 0.0) -> dict:
	"""Return a copy of base_params with scenario-specific overrides applied."""
	params = dict(base_params)
	if 'spend_scale' in overrides:
		params['withdrawal_schedule'] = [v * overrides['spend_scale'] for v in params['withdrawal_schedule']]
	elif 'spend_flat' in overrides:
		params['withdrawal_schedule'] = [overrides['spend_flat']] * len(params['withdrawal_schedule'])
	if 'target_stock_pct' in overrides:
		params['target_stock_pct'] = overrides['target_stock_pct']
		if params.get('guardrails_enabled'):
			pct = overrides['target_stock_pct']
			params['blended_mu'] = pct * stock_mu + (1 - pct) * bond_mu
			params['blended_sigma'] = pct * stock_sigma + (1 - pct) * bond_sigma
	if 'roth_conversion_amount' in overrides:
		params['roth_conversion_amount'] = overrides['roth_conversion_amount']
	if 'roth_conversion_years' in overrides:
		params['roth_conversion_years'] = overrides['roth_conversion_years']
	if 'annuity_purchase' in overrides:
		params['taxable_start'] = params['taxable_start'] - overrides['annuity_purchase']
		income = overrides.get('annuity_annual_income', 0.0)
		cola = overrides.get('annuity_cola', 0.0)
		surv = overrides.get('annuity_survivor_pct', 0.0)
		if overrides.get('annuity_person') == 'Person 1':
			params['annuity_income_p1'] = income
			params['annuity_cola_p1'] = cola
			params['annuity_survivor_pct_p1'] = surv
		else:
			params['annuity_income_p2'] = income
			params['annuity_cola_p2'] = cola
			params['annuity_survivor_pct_p2'] = surv
		params['annuity_start_year'] = overrides.get('annuity_start_year', 1)
	if 'buyout_choice' in overrides:
		person = overrides.get('buyout_person', 'Person 1')
		if overrides['buyout_choice'] == 'Take lump sum':
			# Add lump sum to the appropriate TDA
			lump = overrides.get('buyout_lump_sum', 0.0)
			if person == 'Person 1':
				params['tda_start'] = params.get('tda_start', 0.0) + lump
			else:
				params['tda_spouse_start'] = params.get('tda_spouse_start', 0.0) + lump
		else:
			# Take annuity: add income stream
			income = overrides.get('buyout_annuity_income', 0.0)
			cola = overrides.get('buyout_annuity_cola', 0.0)
			surv = overrides.get('buyout_annuity_survivor_pct', 0.0)
			if person == 'Person 1':
				params['annuity_income_p1'] = params.get('annuity_income_p1', 0.0) + income
				params['annuity_cola_p1'] = cola
				params['annuity_survivor_pct_p1'] = surv
			else:
				params['annuity_income_p2'] = params.get('annuity_income_p2', 0.0) + income
				params['annuity_cola_p2'] = cola
				params['annuity_survivor_pct_p2'] = surv
	# Direct TDA adjustments (for pension buyout reversal)
	if 'tda_delta_p1' in overrides:
		params['tda_start'] = params.get('tda_start', 0.0) + overrides['tda_delta_p1']
	if 'tda_delta_p2' in overrides:
		params['tda_spouse_start'] = params.get('tda_spouse_start', 0.0) + overrides['tda_delta_p2']
	# Direct annuity income param overrides
	for key in ['annuity_income_p1', 'annuity_income_p2', 'annuity_cola_p1', 'annuity_cola_p2',
				'annuity_survivor_pct_p1', 'annuity_survivor_pct_p2', 'annuity_start_year']:
		if key in overrides:
			params[key] = overrides[key]
	return params

def auto_scenario_name(scenario_idx: int, overrides: dict, base_params: dict) -> str:
	"""Generate a descriptive name from what differs vs baseline."""
	if not overrides:
		return "Baseline"
	parts = []
	if 'spend_scale' in overrides:
		parts.append(f"Spend {overrides['spend_scale'] * 100:.0f}%")
	elif 'spend_flat' in overrides:
		parts.append(f"Spend ${overrides['spend_flat']:,.0f}")
	if 'target_stock_pct' in overrides:
		parts.append(f"{overrides['target_stock_pct'] * 100:.0f}% stocks")
	if 'roth_conversion_amount' in overrides:
		amt = overrides['roth_conversion_amount']
		yrs = overrides.get('roth_conversion_years', base_params.get('roth_conversion_years', 0))
		if amt == 0 or yrs == 0:
			parts.append("No Roth conv")
		else:
			parts.append(f"Roth ${amt / 1000:.0f}k x {yrs}yr")
	if 'annuity_purchase' in overrides:
		purchase = overrides['annuity_purchase']
		income = overrides.get('annuity_annual_income', 0)
		parts.append(f"Annuity ${purchase / 1000:.0f}k → ${income / 1000:.0f}k/yr")
	if 'buyout_choice' in overrides:
		choice = overrides['buyout_choice']
		person = overrides.get('buyout_person', 'Person 1')
		tag = "P1" if person == "Person 1" else "P2"
		if choice == 'Take lump sum':
			lump = overrides.get('buyout_lump_sum', 0)
			parts.append(f"Buyout {tag}: Lump ${lump / 1000:.0f}k")
		else:
			income = overrides.get('buyout_annuity_income', 0)
			parts.append(f"Buyout {tag}: Annuity ${income / 1000:.0f}k/yr")
	if 'tda_delta_p1' in overrides or 'tda_delta_p2' in overrides:
		ann = overrides.get('annuity_income_p1', overrides.get('annuity_income_p2', 0))
		parts.append(f"Take Annuity ${ann / 1000:.0f}k/yr")
	return " | ".join(parts) if parts else f"Scenario {scenario_idx}"

def compute_scenario_summary(name: str, results: list, all_yearly_df: pd.DataFrame, inheritor_rate: float) -> dict:
	"""Compute percentile summary for one scenario run. Returns a summary dict."""
	mc_df = pd.DataFrame(results)
	percentiles_list = [0, 10, 25, 50, 75, 90]
	summary_cols = ['after_tax_end', 'total_taxes', 'effective_tax_rate', 'portfolio_cagr', 'roth_cagr']
	pct_rows = []
	for p in percentiles_list:
		row = {'percentile': p}
		for col in summary_cols:
			row[col] = np.percentile(mc_df[col], p)
		pct_rows.append(row)
	pct_non_positive = float((mc_df['after_tax_end'] <= 0).mean())
	run_spending = all_yearly_df.groupby('run').agg(
		total_after_tax_spending=('after_tax_spending', 'sum'),
		years_in_run=('year', 'count'),
	)
	run_spending['avg_annual_after_tax_spending'] = run_spending['total_after_tax_spending'] / run_spending['years_in_run']
	spending_pct_rows = []
	for p in percentiles_list:
		spending_pct_rows.append({
			'percentile': p,
			'avg_annual_after_tax_spending': np.percentile(run_spending['avg_annual_after_tax_spending'], p),
			'total_lifetime_after_tax_spending': np.percentile(run_spending['total_after_tax_spending'], p),
		})
	return {
		'name': name,
		'percentile_rows': pct_rows,
		'pct_non_positive': pct_non_positive,
		'spending_percentiles': spending_pct_rows,
		'all_yearly_df': all_yearly_df,
		'num_sims': len(results),
	}

def run_monte_carlo(num_runs: int, years: int, inheritor_rate: float,
					taxable_log_drift: float, taxable_log_volatility: float,
					bond_log_drift: float, bond_log_volatility: float,
					**sim_params):
	"""Run lognormal MC simulations. sim_params passed through to simulate_withdrawals."""
	results = []
	all_yearly = []
	for run_idx in range(num_runs):
		rng = np.random.default_rng()
		taxable_series = sample_lognormal_returns(years, taxable_log_drift, taxable_log_volatility, rng)
		bond_series = sample_lognormal_returns(years, bond_log_drift, bond_log_volatility, rng)
		df_run = simulate_withdrawals(
			years=years,
			stock_return_series=taxable_series,
			bond_return_series=bond_series,
			**sim_params,
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
						 pension_cola_p1: float = 0.0,
						 pension_cola_p2: float = 0.0,
						 pension_survivor_pct_p1: float = 0.0,
						 pension_survivor_pct_p2: float = 0.0,
						 pp_factors: Optional[list] = None,
						 pp_factors_run: Optional[list] = None,
						 annuity_income_p1: float = 0.0,
						 annuity_income_p2: float = 0.0,
						 annuity_cola_p1: float = 0.0,
						 annuity_cola_p2: float = 0.0,
						 annuity_survivor_pct_p1: float = 0.0,
						 annuity_survivor_pct_p2: float = 0.0,
						 annuity_start_year: int = 1,
						 other_income_annual: float = 0.0,
						 filing_status: str = 'single',
						 use_itemized_deductions: bool = False,
						 itemized_deduction_amount: float = 0.0,
						 roth_conversion_amount: float = 0.0,
						 roth_conversion_tax_source: str = 'taxable',
						 roth_conversion_years: int = 0,
						 roth_conversion_source: str = 'person1',
						 ss_start_age_p1: int = 67,
						 ss_start_age_p2: int = 67,
						 state_tax_rate: float = 0.0,
						 state_exempt_retirement: bool = False,
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
						 guardrail_max_spending_pct: float = 0.0,
						 taxes_enabled: bool = True):
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

	# Build expected income schedule (SS + pension + other) for guardrail inner MC
	income_schedule = []
	for y in range(1, years + 1):
		age_p1 = start_age_primary + y - 1
		age_p2 = start_age_spouse + y - 1
		p1_alive = age_p1 <= life_expectancy_primary
		p2_alive = age_p2 <= life_expectancy_spouse
		ss_b1 = ss_income_annual * ((1 + ss_cola) ** (y - 1))
		ss_b2 = ss_income_spouse_annual * ((1 + ss_cola) ** (y - 1))
		if p1_alive and p2_alive:
			yr_ss = (ss_b1 if age_p1 >= ss_start_age_p1 else 0.0) + (ss_b2 if age_p2 >= ss_start_age_p2 else 0.0)
		elif p1_alive:
			yr_ss = max(ss_b1, ss_b2) if age_p1 >= ss_start_age_p1 else 0.0
		elif p2_alive:
			yr_ss = max(ss_b1, ss_b2) if age_p2 >= ss_start_age_p2 else 0.0
		else:
			yr_ss = 0.0
		pen_p1_nom = pension_income_annual * ((1 + pension_cola_p1) ** (y - 1))
		pen_p2_nom = pension_income_spouse_annual * ((1 + pension_cola_p2) ** (y - 1))
		if p1_alive and p2_alive:
			yr_pen_nom = pen_p1_nom + pen_p2_nom
		elif p1_alive:
			yr_pen_nom = pen_p1_nom + pen_p2_nom * pension_survivor_pct_p2
		elif p2_alive:
			yr_pen_nom = pen_p2_nom + pen_p1_nom * pension_survivor_pct_p1
		else:
			yr_pen_nom = 0.0
		pp = pp_factors[y - 1] if pp_factors and y <= len(pp_factors) else 1.0
		yr_pen_real = yr_pen_nom * pp
		# Annuity income (same logic as pension: COLA, survivor, PP-adjusted)
		if y >= annuity_start_year:
			ann_yrs = y - annuity_start_year
			ann_p1_nom = annuity_income_p1 * ((1 + annuity_cola_p1) ** ann_yrs)
			ann_p2_nom = annuity_income_p2 * ((1 + annuity_cola_p2) ** ann_yrs)
			if p1_alive and p2_alive:
				yr_ann_nom = ann_p1_nom + ann_p2_nom
			elif p1_alive:
				yr_ann_nom = ann_p1_nom + ann_p2_nom * annuity_survivor_pct_p2
			elif p2_alive:
				yr_ann_nom = ann_p2_nom + ann_p1_nom * annuity_survivor_pct_p1
			else:
				yr_ann_nom = 0.0
			yr_ann_real = yr_ann_nom * pp
		else:
			yr_ann_real = 0.0
		income_schedule.append(yr_ss + yr_pen_real + yr_ann_real + other_income_annual)

	# Guardrail: compute initial scaling factor via binary search
	if guardrails_enabled:
		total_portfolio_init = float(taxable_start) + float(tda_start) + float(tda_spouse_start) + float(roth_start)
		current_scale_factor = find_sustainable_scale_factor(
			total_portfolio_init, list(withdrawal_schedule), blended_mu, blended_sigma,
			guardrail_target, guardrail_inner_sims, income_schedule=income_schedule)
		# Apply max spending cap to scale factor
		if guardrail_max_spending_pct > 0:
			max_scale = 1.0 + guardrail_max_spending_pct / 100.0
			current_scale_factor = min(current_scale_factor, max_scale)
	else:
		current_scale_factor = 1.0

	for y in range(1, years+1):
		age_p1 = start_age_primary + y - 1
		age_p2 = start_age_spouse + y - 1
		primary_alive = age_p1 <= life_expectancy_primary
		spouse_alive = age_p2 <= life_expectancy_spouse

		# Inherited IRA rollover: merge deceased spouse's TDA into survivor's
		if not primary_alive and (tda1_stocks_mv + tda1_bonds_mv) > 0 and spouse_alive:
			tda2_stocks_mv += tda1_stocks_mv
			tda2_bonds_mv += tda1_bonds_mv
			tda1_stocks_mv = 0.0
			tda1_bonds_mv = 0.0
		if not spouse_alive and (tda2_stocks_mv + tda2_bonds_mv) > 0 and primary_alive:
			tda1_stocks_mv += tda2_stocks_mv
			tda1_bonds_mv += tda2_bonds_mv
			tda2_stocks_mv = 0.0
			tda2_bonds_mv = 0.0

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
			remaining_income = income_schedule[y-1:]
			if len(remaining_schedule) > 1 and total_portfolio_now > 0:
				sr = forward_success_rate(total_portfolio_now, remaining_schedule,
					current_scale_factor, blended_mu, blended_sigma, guardrail_inner_sims,
					income_schedule=remaining_income)
				if sr < guardrail_lower or sr > guardrail_upper:
					current_scale_factor = find_sustainable_scale_factor(
						total_portfolio_now, remaining_schedule, blended_mu, blended_sigma,
						guardrail_target, guardrail_inner_sims, income_schedule=remaining_income)
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

		# apply Roth conversion at start of year before growth
		if y <= roth_conversion_years:
			if roth_conversion_source == 'person1' and primary_alive:
				src_stocks, src_bonds = tda1_stocks_mv, tda1_bonds_mv
			elif roth_conversion_source == 'person2' and spouse_alive:
				src_stocks, src_bonds = tda2_stocks_mv, tda2_bonds_mv
			else:
				src_stocks, src_bonds = 0.0, 0.0
			conversion_gross = min(roth_conversion_amount, src_stocks + src_bonds)
		else:
			conversion_gross = 0.0
		if conversion_gross > 0:
			src_total = src_stocks + src_bonds
			src_stock_ratio = (src_stocks / src_total) if src_total > 0 else 0.5
			if roth_conversion_source == 'person1':
				tda1_stocks_mv -= conversion_gross * src_stock_ratio
				tda1_bonds_mv -= conversion_gross * (1 - src_stock_ratio)
			else:
				tda2_stocks_mv -= conversion_gross * src_stock_ratio
				tda2_bonds_mv -= conversion_gross * (1 - src_stock_ratio)
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
		# SS with start ages and survivor benefit rules
		ss_benefit_p1 = ss_income_annual * ((1 + ss_cola) ** (y - 1))
		ss_benefit_p2 = ss_income_spouse_annual * ((1 + ss_cola) ** (y - 1))
		if primary_alive and spouse_alive:
			ss_income_p1 = ss_benefit_p1 if age_p1 >= ss_start_age_p1 else 0.0
			ss_income_p2 = ss_benefit_p2 if age_p2 >= ss_start_age_p2 else 0.0
			ss_income = ss_income_p1 + ss_income_p2
		elif primary_alive and not spouse_alive:
			# Survivor gets the higher of own or deceased spouse's benefit
			ss_income = max(ss_benefit_p1, ss_benefit_p2) if age_p1 >= ss_start_age_p1 else 0.0
		elif spouse_alive and not primary_alive:
			ss_income = max(ss_benefit_p1, ss_benefit_p2) if age_p2 >= ss_start_age_p2 else 0.0
		else:
			ss_income = 0.0
		pen_nom_p1 = pension_income_annual * ((1 + pension_cola_p1) ** (y - 1))
		pen_nom_p2 = pension_income_spouse_annual * ((1 + pension_cola_p2) ** (y - 1))
		if primary_alive and spouse_alive:
			pension_income = pen_nom_p1 + pen_nom_p2
		elif primary_alive and not spouse_alive:
			pension_income = pen_nom_p1 + pen_nom_p2 * pension_survivor_pct_p2
		elif spouse_alive and not primary_alive:
			pension_income = pen_nom_p2 + pen_nom_p1 * pension_survivor_pct_p1
		else:
			pension_income = 0.0
		# Use per-run CPI factors if available (historical), otherwise median PP factors
		run_pp = pp_factors_run if pp_factors_run is not None else pp_factors
		pp_yr = run_pp[y - 1] if run_pp and y <= len(run_pp) else 1.0
		pension_income_real = pension_income * pp_yr
		# Annuity income (same structure as pension)
		if y >= annuity_start_year:
			ann_yrs = y - annuity_start_year
			ann_nom_p1 = annuity_income_p1 * ((1 + annuity_cola_p1) ** ann_yrs)
			ann_nom_p2 = annuity_income_p2 * ((1 + annuity_cola_p2) ** ann_yrs)
			if primary_alive and spouse_alive:
				annuity_income = ann_nom_p1 + ann_nom_p2
			elif primary_alive and not spouse_alive:
				annuity_income = ann_nom_p1 + ann_nom_p2 * annuity_survivor_pct_p2
			elif spouse_alive and not primary_alive:
				annuity_income = ann_nom_p2 + ann_nom_p1 * annuity_survivor_pct_p1
			else:
				annuity_income = 0.0
		else:
			annuity_income = 0.0
		annuity_income_real = annuity_income * pp_yr
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
			ordinary_pre_ss_base = interest + w_tda + pension_income + annuity_income + other_income
			ordinary_pre_ss_with_conv = ordinary_pre_ss_base + pending_roth_conversion
			cg_total = div + turnover_realized_gain + realized_gains

			if taxes_enabled:
				# With conversion
				t_ss = compute_taxable_social_security(ss_income, ordinary_pre_ss_with_conv, cg_total, filing_status_this_year)
				t_ordinary = max(0.0, ordinary_pre_ss_with_conv + t_ss - deduction)
				ord_tax = apply_brackets(t_ordinary, get_ordinary_brackets(filing_status_this_year))
				cg_tax = compute_capital_gains_tax(t_ordinary, cg_total, filing_status_this_year)
				total_tax = ord_tax + cg_tax

				# Without conversion (for delta — include state tax so conversion cost is accurate)
				t_ss_nc = compute_taxable_social_security(ss_income, ordinary_pre_ss_base, cg_total, filing_status_this_year)
				t_ordinary_nc = max(0.0, ordinary_pre_ss_base + t_ss_nc - deduction)
				ord_tax_nc = apply_brackets(t_ordinary_nc, get_ordinary_brackets(filing_status_this_year))
				cg_tax_nc = compute_capital_gains_tax(t_ordinary_nc, cg_total, filing_status_this_year)
				total_tax_nc = ord_tax_nc + cg_tax_nc
				total_tax_with = total_tax  # ord_tax + cg_tax (with conversion)
				if state_tax_rate > 0:
					if state_exempt_retirement:
						# IL-style: only investment income (interest + div + cap gains) is taxable at state level
						state_base_with = max(0.0, interest + cg_total)
						state_base_nc = state_base_with  # conversion doesn't affect investment income
					else:
						state_base_with = t_ordinary + cg_total
						state_base_nc = t_ordinary_nc + cg_total
					total_tax_with += state_base_with * state_tax_rate
					total_tax_nc += state_base_nc * state_tax_rate
				conv_tax_delta = max(0.0, total_tax_with - total_tax_nc)

				# NIIT
				niit_threshold = 200000 if filing_status_this_year == 'single' else 250000
				agi = ordinary_pre_ss_with_conv + t_ss + cg_total
				niit_base_val = max(0.0, agi - niit_threshold)
				net_inv = max(0.0, cg_total + interest)
				niit = 0.038 * min(niit_base_val, net_inv)
				total_tax += niit

				# State income tax
				if state_tax_rate > 0:
					if state_exempt_retirement:
						state_taxable = max(0.0, interest + cg_total)
					else:
						state_taxable = t_ordinary + cg_total
					s_tax = state_taxable * state_tax_rate
				else:
					s_tax = 0.0
				total_tax += s_tax

				marg_ord, marg_cg = get_marginal_rates(t_ordinary, cg_total, filing_status_this_year)
				if niit_base_val > 0 and cg_total > 0:
					marg_cg += 0.038
				if state_tax_rate > 0:
					if not state_exempt_retirement:
						marg_ord += state_tax_rate
					marg_cg += state_tax_rate
			else:
				t_ss = 0.0
				t_ordinary = 0.0
				ord_tax = 0.0
				cg_tax = 0.0
				total_tax = 0.0
				conv_tax_delta = 0.0
				niit = 0.0
				s_tax = 0.0
				marg_ord = 0.0
				marg_cg = 0.0

			# Roth conversion tax payment
			conv_net_to_roth = 0.0
			if pending_roth_conversion > 0:
				if roth_conversion_tax_source == 'taxable' and taxes_enabled:
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
				elif not taxes_enabled:
					conv_net_to_roth = pending_roth_conversion
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
				- rmd_excess + ss_income + pension_income_real + annuity_income_real + other_income
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
				'niit_tax': niit, 'state_tax': s_tax, 'total_taxes': total_tax,
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
			'pension_income_real': pension_income_real,
			'pension_erosion': pension_income - pension_income_real,
			'annuity_income_total': annuity_income,
			'annuity_income_real': annuity_income_real,
			'other_income': other_income,
			'roth_conversion': pending_roth_conversion,
			'roth_conversion_tax': chosen['roth_conversion_tax_delta'],
			'roth_conversion_tax_source': roth_conversion_tax_source,
			'ordinary_taxable_income': chosen['taxable_ordinary'],
			'ordinary_tax_total': chosen['ordinary_tax_total'],
			'capital_gains': chosen['cap_gains_total'],
			'capital_gains_tax': chosen['cap_gains_tax'],
			'niit_tax': chosen['niit_tax'],
			'state_tax': chosen['state_tax'],
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
					'mc_pct_non_positive', 'sim_df', 'sim_mode', 'mc_summaries', 'mc_percentiles',
					'multi_scenario_results', 'selected_scenario_idx', 'num_sims']:
			st.session_state.pop(key, None)
		st.success('Saved scenarios cleared.')

	with st.sidebar:
		st.header('Inputs')

		with st.expander('Scenario Comparison'):
			num_scenarios = st.number_input('Number of scenarios', min_value=1, max_value=4, value=1, step=1,
				help='Set up your baseline inputs in the sidebar sections below first (account balances, allocation, spending, etc.). '
				'Then add scenarios here to test variations. Each scenario overrides only what differs — everything else '
				'comes from the main sidebar. If using Pension Buyout, set to 1 for just lump vs annuity; set to 2+ to '
				'also test other changes (each override is crossed with both lump sum and annuity sides).')
			if num_scenarios > 1:
				st.caption('Scenario 1 = baseline (uses all inputs below). For each additional scenario, '
					'check the boxes below to override specific values. Anything not overridden stays the same as baseline.')
				scenario_overrides_ui = {}
				for i in range(2, num_scenarios + 1):
					st.markdown(f'**Scenario {i}**')
					spend_mode = st.radio(f'S{i} spending', ['Same as baseline', 'Scale by %', 'Set amount'],
						key=f'sc_spend_mode_{i}', horizontal=True,
						help='Same as baseline: no change. Scale by %: multiply all withdrawal amounts by a percentage '
						'(e.g. 80% = spend 20% less). Set amount: replace with a fixed annual amount.')
					sc_overrides = {}
					if spend_mode == 'Scale by %':
						sc_overrides['spend_scale'] = st.number_input(f'S{i} spending scale %',
							value=100.0, step=5.0, key=f'sc_spend_scale_{i}',
							help='100% = same as baseline. 80% = 20% less spending. 120% = 20% more spending. '
							'Applied to every year in the withdrawal schedule.') / 100.0
					elif spend_mode == 'Set amount':
						sc_overrides['spend_flat'] = st.number_input(f'S{i} flat annual spending',
							value=80000.0, step=5000.0, key=f'sc_spend_flat_{i}',
							help='Replaces the entire withdrawal schedule with this fixed amount every year.')
					stock_chk = st.checkbox(f'S{i} override stock %', key=f'sc_stock_chk_{i}',
						help='Test a different stock/bond allocation for this scenario.')
					if stock_chk:
						sc_overrides['target_stock_pct'] = st.slider(f'S{i} stock %',
							0, 100, 60, 5, key=f'sc_stock_{i}') / 100.0
					roth_chk = st.checkbox(f'S{i} override Roth conversions', key=f'sc_roth_chk_{i}',
						help='Test a different Roth conversion strategy for this scenario. '
						'Set amount to 0 and years to 0 for no conversions.')
					if roth_chk:
						sc_overrides['roth_conversion_amount'] = st.number_input(f'S{i} annual Roth conversion',
							value=0.0, step=10000.0, key=f'sc_roth_amt_{i}',
							help='Amount converted from TDA to Roth each year. Taxed as ordinary income in the year of conversion.')
						sc_overrides['roth_conversion_years'] = int(st.number_input(f'S{i} conversion years',
							value=0, min_value=0, max_value=100, key=f'sc_roth_yrs_{i}',
							help='Number of years to perform conversions, starting from year 1 of the simulation.'))
					annuity_chk = st.checkbox(f'S{i} buy annuity (from taxable)', key=f'sc_annuity_chk_{i}',
						help='Purchase an annuity using money from the taxable account. '
						'Reduces taxable balance and adds an income stream.')
					if annuity_chk:
						sc_overrides['annuity_purchase'] = st.number_input(f'S{i} annuity purchase price (from taxable)',
							value=200000.0, step=10000.0, key=f'sc_ann_purchase_{i}',
							help='Amount taken from taxable account to buy the annuity.')
						sc_overrides['annuity_annual_income'] = st.number_input(f'S{i} annuity annual income',
							value=12000.0, step=1000.0, key=f'sc_ann_income_{i}',
							help='Annual income received from the annuity.')
						sc_overrides['annuity_cola'] = st.number_input(f'S{i} annuity COLA',
							value=0.0, format="%.4f", key=f'sc_ann_cola_{i}',
							help='Annual cost-of-living adjustment on the annuity income. 0 = fixed payments.')
						sc_overrides['annuity_person'] = st.radio(f'S{i} annuity owner',
							['Person 1', 'Person 2'], horizontal=True, key=f'sc_ann_person_{i}')
						sc_overrides['annuity_survivor_pct'] = st.number_input(f'S{i} annuity survivor %',
							value=0.0, min_value=0.0, max_value=1.0, format="%.2f", step=0.05, key=f'sc_ann_surv_{i}',
							help='Fraction of annuity paid to survivor after owner dies. 1.0 = full benefit continues. 0 = payments stop at death.')
						sc_overrides['annuity_start_year'] = int(st.number_input(f'S{i} annuity income starts (year)',
							value=1, min_value=1, max_value=40, key=f'sc_ann_start_{i}',
							help='Simulation year when annuity payments begin. Year 1 = immediately.'))
					buyout_chk = st.checkbox(f'S{i} pension buyout (lump sum vs annuity)', key=f'sc_buyout_chk_{i}',
						help='Compare taking a one-time lump sum (rolled into TDA) vs receiving an annual pension/annuity. '
						'For a dedicated comparison, use the Pension Buyout section instead.')
					if buyout_chk:
						buyout_choice = st.radio(f'S{i} pension buyout choice',
							['Take lump sum', 'Take annuity'], horizontal=True, key=f'sc_buyout_choice_{i}',
							help='Take lump sum: amount is added to TDA. Take annuity: receive annual income instead.')
						sc_overrides['buyout_person'] = st.radio(f'S{i} buyout for',
							['Person 1', 'Person 2'], horizontal=True, key=f'sc_buyout_person_{i}')
						sc_overrides['buyout_lump_sum'] = st.number_input(f'S{i} lump sum amount (to TDA)',
							value=200000.0, step=10000.0, key=f'sc_buyout_lump_{i}',
							help='One-time amount rolled into the TDA (IRA/401k).')
						sc_overrides['buyout_annuity_income'] = st.number_input(f'S{i} annuity income alternative',
							value=12000.0, step=1000.0, key=f'sc_buyout_income_{i}',
							help='Annual income if you choose the annuity/pension option instead.')
						sc_overrides['buyout_annuity_cola'] = st.number_input(f'S{i} buyout annuity COLA',
							value=0.0, format="%.4f", key=f'sc_buyout_cola_{i}',
							help='Annual cost-of-living adjustment. 0 = fixed payments (purchasing power erodes with inflation).')
						sc_overrides['buyout_annuity_survivor_pct'] = st.number_input(f'S{i} buyout annuity survivor %',
							value=0.0, min_value=0.0, max_value=1.0, format="%.2f", step=0.05, key=f'sc_buyout_surv_{i}',
							help='Fraction of annuity paid to survivor. 1.0 = full benefit continues. 0 = stops at death.')
						sc_overrides['buyout_choice'] = buyout_choice
					scenario_overrides_ui[i] = sc_overrides
			else:
				scenario_overrides_ui = {}

		with st.expander('Ages & Timeline', expanded=True):
			start_age = st.number_input('Starting age (person 1)', min_value=18, max_value=120, value=65)
			start_age_spouse = st.number_input('Starting age (person 2)', min_value=18, max_value=120, value=60)
			life_expectancy_primary = st.number_input('Primary life expectancy (last age lived through)', min_value=int(start_age), max_value=120, value=84, step=1)
			life_expectancy_spouse = st.number_input('Spouse life expectancy (last age lived through)', min_value=int(start_age_spouse), max_value=120, value=89, step=1)

		with st.expander('Account Balances'):
			taxable_start = st.number_input('Taxable account starting balance', value=300000.0, step=1000.0)
			taxable_stock_basis_pct = st.number_input('Taxable stock basis % of market value', value=50.0, min_value=0.0, max_value=100.0, step=1.0) / 100.0
			taxable_bond_basis_pct = st.number_input('Taxable bond basis % of market value', value=100.0, min_value=0.0, max_value=100.0, step=1.0) / 100.0
			roth_start = st.number_input('Roth account starting balance', value=0.0, step=1000.0)
			tda_start = st.number_input('Tax-deferred account starting balance (IRA/401k) - person 1', value=700000.0, step=1000.0)
			tda_spouse_start = st.number_input('Tax-deferred account starting balance (IRA/401k) - person 2', value=0.0, step=1000.0)

		with st.expander('Allocation & Roth Conversions'):
			target_stock_pct = st.slider('Household target % in stocks', min_value=0, max_value=100, value=60, step=10) / 100.0
			roth_conversion_amount = st.number_input('Annual Roth conversion amount (from TDA)', value=0.0, step=1000.0)
			roth_conversion_years = st.number_input('Years to perform conversions', value=0, min_value=0, max_value=100, step=1)
			roth_conversion_source_tda = st.radio('Convert from', ['Person 1 TDA', 'Person 2 TDA'], horizontal=True)
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
					period_amount = st.number_input(f'Period {i+1} annual After-Tax Spending Goal', value=80000.0, step=1000.0, key=f'wd_amount_{i}')
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
			ss_income_input = st.number_input('Annual Social Security - person 1 (current year)', value=25000.0, step=1000.0)
			ss_start_age_p1 = st.number_input('SS start age - person 1', min_value=60, max_value=90, value=67, step=1)
			ss_income_spouse_input = st.number_input('Annual Social Security - person 2 (current year)', value=20000.0, step=1000.0)
			ss_start_age_p2 = st.number_input('SS start age - person 2', min_value=60, max_value=90, value=65, step=1)
			ss_cola = st.number_input('Social Security COLA', value=0.02, format="%.4f")
			st.caption('Survivor receives the higher of their own or deceased spouse\'s benefit')
			pension_income_input = st.number_input('Annual pension income - person 1', value=0.0, step=1000.0)
			pension_cola_p1 = st.number_input('Pension COLA - person 1', value=0.00, format="%.4f")
			pension_survivor_pct_p1 = st.number_input('Pension survivor % - person 1', value=0.0, min_value=0.0, max_value=1.0, format="%.2f", step=0.05,
				help='Fraction of person 1 pension paid to survivor after person 1 dies')
			pension_income_spouse_input = st.number_input('Annual pension income - person 2', value=20000.0, step=1000.0)
			pension_cola_p2 = st.number_input('Pension COLA - person 2', value=0.00, format="%.4f")
			pension_survivor_pct_p2 = st.number_input('Pension survivor % - person 2', value=0.0, min_value=0.0, max_value=1.0, format="%.2f", step=0.05,
				help='Fraction of person 2 pension paid to survivor after person 2 dies')
			other_income_input = st.number_input('Other ordinary income', value=0.0, step=1000.0)

		with st.expander('Pension Buyout (Lump Sum vs Annuity)'):
			pension_buyout_enabled = st.checkbox('Enable pension buyout comparison', value=False,
				help='Compare taking a lump sum (rolled into TDA) vs taking an annuity/pension income stream')
			if pension_buyout_enabled:
				pension_buyout_baseline = st.radio('Baseline (your choice)',
					['Take lump sum', 'Take annuity'],
					horizontal=True, key='pension_buyout_baseline')
				if pension_buyout_baseline == 'Take lump sum':
					st.caption('Baseline = lump sum added to TDA. Scenario 2 = annuity income instead.')
				else:
					st.caption('Baseline = annuity income. Scenario 2 = lump sum to TDA instead.')
				pension_buyout_person = st.radio('Buyout for', ['Person 1', 'Person 2'],
					horizontal=True, key='pension_buyout_person')
				pension_buyout_lump = st.number_input('Lump sum amount (rolled into TDA)',
					value=200000.0, step=10000.0, key='pension_buyout_lump')
				pension_buyout_income = st.number_input('Annuity income alternative (annual)',
					value=12000.0, step=1000.0, key='pension_buyout_income')
				pension_buyout_cola = st.number_input('Annuity COLA',
					value=0.0, format="%.4f", key='pension_buyout_cola')
				pension_buyout_survivor = st.number_input('Annuity survivor %',
					value=0.0, min_value=0.0, max_value=1.0, format="%.2f", step=0.05,
					key='pension_buyout_survivor',
					help='Fraction of annuity paid to survivor after owner dies')
			else:
				pension_buyout_baseline = 'Take lump sum'
				pension_buyout_person = 'Person 1'
				pension_buyout_lump = 0.0
				pension_buyout_income = 0.0
				pension_buyout_cola = 0.0
				pension_buyout_survivor = 0.0

		with st.expander('Tax Settings'):
			taxes_enabled = st.checkbox('Enable taxation', value=True, help='Uncheck to disable all taxes (useful for testing withdrawal mechanics)')
			filing_status_choice = st.radio('Filing status', ['Single', 'Married Filing Jointly'], horizontal=True, index=1)
			filing_status_key = 'single' if filing_status_choice == 'Single' else 'mfj'
			standard_deduction_display = 14600 if filing_status_key == 'single' else 29200
			use_itemized = st.checkbox('Use itemized deductions instead of standard', value=False)
			itemized_deduction_input = st.number_input('Itemized deduction amount', value=0.0, step=500.0)
			st.caption(f'Standard deduction used if not itemizing: ${standard_deduction_display:,.0f}')
			inheritor_marginal_rate = st.number_input(
				'Inheritor marginal tax rate on TDAs',
				value=0.35, min_value=0.0, max_value=0.50, format="%.4f")
			state_tax_rate = st.number_input('State income tax rate (flat)',
				value=0.05, min_value=0.0, max_value=0.15, format="%.4f", step=0.01)
			state_exempt_retirement = st.checkbox('Exempt retirement income from state tax (IL-style)',
				value=True)
			if state_exempt_retirement:
				st.caption('State tax applies only to investment income (interest, dividends, capital gains). SS, pensions, TDA withdrawals, and Roth conversions are exempt.')

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
				st.caption('Returns from LBM 100E (stocks) and LBM 100 F (bonds). Runs all historical periods as a distribution.')
				taxable_log_drift = 0.0
				taxable_log_volatility = 0.0
				bond_log_drift = 0.0
				bond_log_volatility = 0.0
				random_seed_input = 42
				seed_mode = 'Fixed seed'
			stock_dividend_yield = st.number_input('Stock dividend (qualified) yield', value=0.02, format="%.4f")
			stock_turnover = st.number_input('Stock turnover rate', value=0.10, format="%.4f")

		with st.expander('Withdrawal Guardrails'):
			guardrails_enabled = st.checkbox('Enable dynamic withdrawal guardrails', value=True)
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
	_beginning_portfolio = float(taxable_start) + float(tda_start) + float(tda_spouse_start) + float(roth_start)
	if pension_buyout_enabled:
		_lump_portfolio = _beginning_portfolio + float(pension_buyout_lump)
		if pension_buyout_baseline == 'Take lump sum':
			st.markdown(f"**Beginning portfolio (lump sum baseline): ${_lump_portfolio:,.0f}** | **Annuity alternative: ${_beginning_portfolio:,.0f} + ${pension_buyout_income:,.0f}/yr**")
		else:
			st.markdown(f"**Beginning portfolio (annuity baseline): ${_beginning_portfolio:,.0f} + ${pension_buyout_income:,.0f}/yr** | **Lump sum alternative: ${_lump_portfolio:,.0f}**")
	else:
		st.markdown(f"**Beginning portfolio: ${_beginning_portfolio:,.0f}**")

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
		pension_cola_p1=float(pension_cola_p1), pension_cola_p2=float(pension_cola_p2),
		pension_survivor_pct_p1=float(pension_survivor_pct_p1),
		pension_survivor_pct_p2=float(pension_survivor_pct_p2),
		pp_factors=PP_FACTORS,
		other_income_annual=float(other_income_input),
		filing_status=filing_status_key, use_itemized_deductions=bool(use_itemized),
		itemized_deduction_amount=float(itemized_deduction_input),
		roth_conversion_amount=float(roth_conversion_amount),
		roth_conversion_years=int(roth_conversion_years),
		roth_conversion_tax_source='taxable' if roth_conversion_tax_source == 'Taxable' else 'tda',
		roth_conversion_source='person1' if roth_conversion_source_tda == 'Person 1 TDA' else 'person2',
		ss_start_age_p1=int(ss_start_age_p1),
		ss_start_age_p2=int(ss_start_age_p2),
		state_tax_rate=float(state_tax_rate),
		state_exempt_retirement=bool(state_exempt_retirement),
		life_expectancy_primary=int(life_expectancy_primary),
		life_expectancy_spouse=int(life_expectancy_spouse),
		guardrails_enabled=bool(guardrails_enabled),
		guardrail_lower=float(guardrail_lower),
		guardrail_upper=float(guardrail_upper),
		guardrail_target=float(guardrail_target),
		guardrail_inner_sims=int(guardrail_inner_sims),
		guardrail_max_spending_pct=float(guardrail_max_spending_pct),
		taxes_enabled=bool(taxes_enabled),
	)

	# Compute stock/bond return parameters (needed for guardrails and scenario comparison)
	if return_mode == 'Simulated (lognormal)':
		stock_mu = float(taxable_log_drift)
		stock_sigma = float(taxable_log_volatility)
		bond_mu = float(bond_log_drift)
		bond_sigma = float(bond_log_volatility)
	else:
		mg_df = load_master_global()
		stock_factors = mg_df['LBM 100E'].dropna().values
		stock_log_rets = np.log(stock_factors)
		stock_mu = float(np.mean(stock_log_rets))
		stock_sigma = float(np.std(stock_log_rets))
		bond_log_returns = np.log(1.0 + load_bond_factors())
		bond_mu = float(np.mean(bond_log_returns))
		bond_sigma = float(np.std(bond_log_returns))
	if guardrails_enabled:
		blended_mu = target_stock_pct * stock_mu + (1 - target_stock_pct) * bond_mu
		blended_sigma = target_stock_pct * stock_sigma + (1 - target_stock_pct) * bond_sigma
		sim_params['blended_mu'] = blended_mu
		sim_params['blended_sigma'] = blended_sigma
	else:
		sim_params['blended_mu'] = 0.0
		sim_params['blended_sigma'] = 0.0

	def _load_scenario_to_session(scenario_summary, sim_mode_label):
		"""Load a scenario's data into standard session state keys for the detail display."""
		ay = scenario_summary['all_yearly_df']
		st.session_state['mc_percentile_rows'] = scenario_summary['percentile_rows']
		st.session_state['mc_pct_non_positive'] = scenario_summary['pct_non_positive']
		st.session_state['mc_all_yearly'] = ay
		st.session_state['num_sims'] = scenario_summary['num_sims']
		run_ends = ay.groupby('run')['total_portfolio'].last()
		median_val = run_ends.median()
		median_run_idx = int((run_ends - median_val).abs().idxmin())
		median_df = ay[ay['run'] == median_run_idx].drop(columns=['run', 'total_portfolio']).reset_index(drop=True)
		st.session_state['sim_df'] = median_df
		st.session_state['sim_mode'] = sim_mode_label

	# Pension buyout: set up baseline according to user's choice
	if pension_buyout_enabled:
		if pension_buyout_baseline == 'Take lump sum':
			# Baseline = lump sum added to TDA
			if pension_buyout_person == 'Person 1':
				sim_params['tda_start'] += float(pension_buyout_lump)
			else:
				sim_params['tda_spouse_start'] += float(pension_buyout_lump)
		else:
			# Baseline = annuity income stream
			if pension_buyout_person == 'Person 1':
				sim_params['annuity_income_p1'] = float(pension_buyout_income)
				sim_params['annuity_cola_p1'] = float(pension_buyout_cola)
				sim_params['annuity_survivor_pct_p1'] = float(pension_buyout_survivor)
			else:
				sim_params['annuity_income_p2'] = float(pension_buyout_income)
				sim_params['annuity_cola_p2'] = float(pension_buyout_cola)
				sim_params['annuity_survivor_pct_p2'] = float(pension_buyout_survivor)

	# Build scenario list
	if pension_buyout_enabled:
		if pension_buyout_baseline == 'Take lump sum':
			baseline_name = f"Take Lump Sum (${pension_buyout_lump / 1000:.0f}k)"
			# Scenario 2: reverse TDA addition + add annuity income
			alt_ovr = {}
			if pension_buyout_person == 'Person 1':
				alt_ovr['tda_delta_p1'] = -float(pension_buyout_lump)
				alt_ovr['annuity_income_p1'] = float(pension_buyout_income)
				alt_ovr['annuity_cola_p1'] = float(pension_buyout_cola)
				alt_ovr['annuity_survivor_pct_p1'] = float(pension_buyout_survivor)
			else:
				alt_ovr['tda_delta_p2'] = -float(pension_buyout_lump)
				alt_ovr['annuity_income_p2'] = float(pension_buyout_income)
				alt_ovr['annuity_cola_p2'] = float(pension_buyout_cola)
				alt_ovr['annuity_survivor_pct_p2'] = float(pension_buyout_survivor)
			alt_name = f"Take Annuity (${pension_buyout_income / 1000:.0f}k/yr)"
		else:
			baseline_name = f"Take Annuity (${pension_buyout_income / 1000:.0f}k/yr)"
			# Scenario 2: add lump sum to TDA + zero out annuity income
			alt_ovr = {}
			if pension_buyout_person == 'Person 1':
				alt_ovr['tda_delta_p1'] = float(pension_buyout_lump)
				alt_ovr['annuity_income_p1'] = 0.0
				alt_ovr['annuity_cola_p1'] = 0.0
				alt_ovr['annuity_survivor_pct_p1'] = 0.0
			else:
				alt_ovr['tda_delta_p2'] = float(pension_buyout_lump)
				alt_ovr['annuity_income_p2'] = 0.0
				alt_ovr['annuity_cola_p2'] = 0.0
				alt_ovr['annuity_survivor_pct_p2'] = 0.0
			alt_name = f"Take Lump Sum (${pension_buyout_lump / 1000:.0f}k)"
		all_scenarios = [({}, baseline_name)]
		all_scenarios.append((alt_ovr, alt_name))
		# Cross-product: each non-empty manual override generates two variants (baseline side + alt side)
		if num_scenarios > 1:
			for s_idx in range(2, num_scenarios + 1):
				ovr = scenario_overrides_ui.get(s_idx, {})
				if not ovr:
					continue  # skip empty overrides
				ovr_label = auto_scenario_name(s_idx, ovr, sim_params)
				# Variant on baseline side (override applied directly)
				all_scenarios.append((dict(ovr), f"{baseline_name} | {ovr_label}"))
				# Variant on alt side (merge alt_ovr + override)
				merged = dict(alt_ovr)
				merged.update(ovr)
				all_scenarios.append((merged, f"{alt_name} | {ovr_label}"))
		num_scenarios = len(all_scenarios)
	else:
		all_scenarios = [({}, 'Baseline')]  # scenario 1 = baseline, no overrides
		if num_scenarios > 1:
			for s_idx in range(2, num_scenarios + 1):
				ovr = scenario_overrides_ui.get(s_idx, {})
				all_scenarios.append((ovr, auto_scenario_name(s_idx, ovr, sim_params)))

	# Show scenario count so user knows what will run
	if pension_buyout_enabled:
		st.caption(f'Pension buyout will run **{len(all_scenarios)} scenarios**: {", ".join(name for _, name in all_scenarios)}')

	# Let user pick which scenario is the baseline for comparison
	if num_scenarios > 1:
		scenario_names = [name for _, name in all_scenarios]
		baseline_idx = st.selectbox('Baseline scenario (deltas compared against this)',
			range(len(scenario_names)), format_func=lambda i: scenario_names[i], index=0,
			key='baseline_scenario_idx')
		if baseline_idx != 0:
			# Move selected baseline to front
			all_scenarios.insert(0, all_scenarios.pop(baseline_idx))

	button_label = 'Run all scenarios' if num_scenarios > 1 else 'Run simulation'
	if st.button(button_label):
		sim_years = int(years)
		is_historical = return_mode == 'Historical (master_global_factors)'

		# Pre-load historical windows once (shared across scenarios)
		if is_historical:
			windows, window_start_dates = get_all_historical_windows(sim_years)
			n_windows = len(windows)

		def run_one_scenario(s_params, s_name, progress_placeholder):
			"""Run a single scenario and return (results, all_yearly_df)."""
			if is_historical:
				results = []
				all_yearly = []
				for run_idx, (stock_rets, bond_rets) in enumerate(windows):
					if run_idx % 50 == 0:
						progress_placeholder.progress(run_idx / n_windows,
							text=f'{s_name}: {run_idx}/{n_windows} periods')
					run_pp = compute_run_pp_factors(run_idx, sim_years)
					df_run = simulate_withdrawals(
						years=sim_years, stock_return_series=stock_rets,
						bond_return_series=bond_rets, pp_factors_run=run_pp, **s_params)
					df_run['total_portfolio'] = df_run['end_taxable_total'] + df_run['end_tda_total'] + df_run['end_roth']
					metrics = compute_summary_metrics(df_run, float(inheritor_marginal_rate))
					results.append(metrics)
					df_run['run'] = run_idx
					all_yearly.append(df_run)
				progress_placeholder.progress(1.0, text=f'{s_name}: complete')
				all_yearly_df = pd.concat(all_yearly, ignore_index=True)
				return results, all_yearly_df
			else:
				mc_results, all_yearly_df = run_monte_carlo(
					num_runs=int(monte_carlo_runs),
					years=sim_years,
					inheritor_rate=float(inheritor_marginal_rate),
					taxable_log_drift=float(taxable_log_drift),
					taxable_log_volatility=float(taxable_log_volatility),
					bond_log_drift=float(bond_log_drift),
					bond_log_volatility=float(bond_log_volatility),
					**s_params,
				)
				progress_placeholder.progress(1.0, text=f'{s_name}: complete')
				return mc_results, all_yearly_df

		if num_scenarios > 1:
			# Multi-scenario run
			multi_results = []
			scenario_bar = st.progress(0, text='Starting scenarios...')
			run_bar = st.progress(0, text='')
			for s_num, (ovr, s_name) in enumerate(all_scenarios):
				scenario_bar.progress(s_num / len(all_scenarios),
					text=f'Scenario {s_num + 1}/{len(all_scenarios)}: {s_name}')
				s_params = build_scenario_params(sim_params, ovr,
					stock_mu=stock_mu, stock_sigma=stock_sigma,
					bond_mu=bond_mu, bond_sigma=bond_sigma)
				results, all_yearly_df = run_one_scenario(s_params, s_name, run_bar)
				summary = compute_scenario_summary(s_name, results, all_yearly_df, float(inheritor_marginal_rate))
				multi_results.append(summary)
			scenario_bar.progress(1.0, text='All scenarios complete!')
			st.session_state['multi_scenario_results'] = multi_results
			st.session_state['selected_scenario_idx'] = 0
			# Load baseline into standard session state for detail display
			_load_scenario_to_session(multi_results[0], 'historical_dist' if is_historical else 'simulated')
			if is_historical:
				st.session_state['window_start_dates'] = {i: d.strftime('%Y-%m') for i, d in enumerate(window_start_dates)}
		else:
			# Single scenario run (backward compatible)
			run_bar = st.progress(0, text='Running...')
			results, all_yearly_df = run_one_scenario(sim_params, 'Baseline', run_bar)
			sim_mode_label = 'historical_dist' if is_historical else 'simulated'
			store_distribution_results(results, all_yearly_df, sim_mode_label)
			if is_historical:
				st.session_state['window_start_dates'] = {i: d.strftime('%Y-%m') for i, d in enumerate(window_start_dates)}
			st.session_state.pop('multi_scenario_results', None)

	# ── Multi-scenario comparison ────────────────────────────────
	if 'multi_scenario_results' in st.session_state:
		multi_results = st.session_state['multi_scenario_results']
		if len(multi_results) > 1:
			st.subheader('Scenario Comparison')
			compare_pct = st.selectbox('Compare at percentile', [0, 10, 25, 50, 75, 90], index=3)
			baseline = multi_results[0]
			baseline_row = next(r for r in baseline['percentile_rows'] if r['percentile'] == compare_pct)
			baseline_spend = next(r for r in baseline['spending_percentiles'] if r['percentile'] == compare_pct)
			comparison_rows = []
			for sc in multi_results:
				sc_row = next(r for r in sc['percentile_rows'] if r['percentile'] == compare_pct)
				sc_spend = next(r for r in sc['spending_percentiles'] if r['percentile'] == compare_pct)
				comparison_rows.append({
					'Scenario': sc['name'],
					'After-Tax Ending': sc_row['after_tax_end'],
					'vs Baseline': sc_row['after_tax_end'] - baseline_row['after_tax_end'],
					'Total Taxes': sc_row['total_taxes'],
					'Tax Delta': sc_row['total_taxes'] - baseline_row['total_taxes'],
					'Eff Tax Rate': sc_row['effective_tax_rate'],
					'Avg Annual Spending': sc_spend['avg_annual_after_tax_spending'],
					'Spend Delta': sc_spend['avg_annual_after_tax_spending'] - baseline_spend['avg_annual_after_tax_spending'],
					'% Depleted': sc['pct_non_positive'] * 100,
				})
			comp_df = pd.DataFrame(comparison_rows).set_index('Scenario')
			def _color_deltas(df):
				delta_cols = {'vs Baseline', 'Tax Delta', 'Spend Delta'}
				styles = pd.DataFrame('', index=df.index, columns=df.columns)
				for col in df.columns:
					if col in delta_cols:
						styles[col] = df[col].apply(
							lambda v: 'color: green' if isinstance(v, (int, float)) and v > 0
							else ('color: red' if isinstance(v, (int, float)) and v < 0 else ''))
				return styles
			st.dataframe(comp_df.style.format({
				'After-Tax Ending': currency_fmt,
				'vs Baseline': currency_fmt,
				'Total Taxes': currency_fmt,
				'Tax Delta': currency_fmt,
				'Eff Tax Rate': '{:.2%}'.format,
				'Avg Annual Spending': currency_fmt,
				'Spend Delta': currency_fmt,
				'% Depleted': '{:.1f}%'.format,
			}).apply(_color_deltas, axis=None))

			# Overlay charts
			st.subheader('Portfolio Value Comparison (median)')
			overlay_df = pd.DataFrame()
			for sc in multi_results:
				median_portfolio = sc['all_yearly_df'].groupby('year')['total_portfolio'].median()
				overlay_df[sc['name']] = median_portfolio
			interactive_line_chart(overlay_df, y_title='Portfolio Value (Median)', zero_base=False)

			st.subheader('After-Tax Spending Comparison (median)')
			spend_overlay = pd.DataFrame()
			for sc in multi_results:
				median_spend = sc['all_yearly_df'].groupby('year')['after_tax_spending'].median()
				spend_overlay[sc['name']] = median_spend
			interactive_line_chart(spend_overlay, y_title='After-Tax Spending (Median)', zero_base=False)

			# Scenario selector for detail drill-down
			st.markdown('---')
			scenario_names = [sc['name'] for sc in multi_results]
			current_idx = st.session_state.get('selected_scenario_idx', 0)
			selected_name = st.selectbox('Select scenario for detailed view', scenario_names,
				index=current_idx, key='scenario_detail_select')
			new_idx = scenario_names.index(selected_name)
			if new_idx != current_idx:
				st.session_state['selected_scenario_idx'] = new_idx
				sim_mode_label = st.session_state.get('sim_mode', 'historical_dist')
				_load_scenario_to_session(multi_results[new_idx], sim_mode_label)
				st.rerun()

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
		st.session_state['mc_spending_pct_rows'] = spending_pct_rows
		sched = sim_params.get('withdrawal_schedule', [])
		base_avg_wd = np.mean(sched) if sched else 0
		if base_avg_wd > 0:
			median_avg_spending = np.percentile(run_spending['avg_annual_after_tax_spending'], 50)
			pct_above = (median_avg_spending / base_avg_wd - 1) * 100
			direction = 'above' if pct_above >= 0 else 'below'
			st.caption(f'After-tax spending target: {currency_fmt.format(base_avg_wd)} | '
					   f'Median avg annual after-tax spending: {currency_fmt.format(median_avg_spending)} '
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
					   'withdraw_from_taxable_net', 'ss_income_total', 'pension_income_total',
					   'pension_income_real', 'pension_erosion', 'ordinary_taxable_income',
					   'capital_gains', 'effective_tax_rate_calc']
		all_yearly['effective_tax_rate_calc'] = (
			all_yearly['total_taxes'] /
			(all_yearly['ordinary_taxable_income'] + all_yearly['capital_gains']).replace(0, np.nan)
		).fillna(0.0)
		yearly_median = all_yearly.groupby('year')[median_cols].median()
		yearly_median.columns = ['Age', 'Withdrawal Target', 'After-Tax Spending', 'Total Portfolio',
								 'Taxable', 'TDA', 'Roth',
								 'Total Taxes', 'RMDs', 'Withdraw TDA', 'Withdraw Roth',
								 'Withdraw Taxable', 'SS Income', 'Pension (Nominal)',
								 'Pension (Real)', 'Pension Erosion', 'Ordinary Income',
								 'Cap Gains', 'Eff Tax Rate']
		st.dataframe(yearly_median.style.format({
			'Age': '{:.0f}',
			'Total Portfolio': currency_fmt, 'Taxable': currency_fmt, 'TDA': currency_fmt, 'Roth': currency_fmt,
			'Total Taxes': currency_fmt, 'RMDs': currency_fmt,
			'Withdraw TDA': currency_fmt, 'Withdraw Roth': currency_fmt, 'Withdraw Taxable': currency_fmt,
			'SS Income': currency_fmt, 'Pension (Nominal)': currency_fmt, 'Pension (Real)': currency_fmt, 'Pension Erosion': currency_fmt,
			'Ordinary Income': currency_fmt, 'Cap Gains': currency_fmt,
			'Eff Tax Rate': '{:.2%}'.format,
			'Withdrawal Target': currency_fmt, 'After-Tax Spending': currency_fmt,
		}))

		st.subheader('Total portfolio value — percentile bands across all runs')
		port_value_mode = st.radio('Portfolio value display', ['Pre-tax', 'After-tax', 'Both'], horizontal=True, index=0, key='port_value_mode')
		port_band_options = {'0th (min)': 0.00, '5th': 0.05, '10th': 0.10, '25th': 0.25, '50th (median)': 0.50, '75th': 0.75, '90th': 0.90}
		port_band_defaults = ['0th (min)', '10th', '25th', '50th (median)', '75th', '90th']
		selected_port_bands = st.multiselect('Percentile bands to display', list(port_band_options.keys()), default=port_band_defaults, key='port_bands')
		if selected_port_bands:
			port_quantiles = [port_band_options[b] for b in selected_port_bands]
			if port_value_mode in ('Pre-tax', 'Both'):
				if port_value_mode == 'Both':
					st.caption('Pre-tax portfolio value')
				port_pcts = all_yearly.groupby('year')['total_portfolio'].quantile(port_quantiles).unstack()
				port_pcts.columns = selected_port_bands
				interactive_line_chart(port_pcts, y_title='Portfolio Value (Pre-tax)')
			if port_value_mode in ('After-tax', 'Both'):
				inh_rate = float(inheritor_marginal_rate)
				all_yearly['after_tax_portfolio'] = (
					all_yearly['end_taxable_total'] + all_yearly['end_roth'] +
					all_yearly['end_tda_total'] * max(0.0, 1.0 - inh_rate))
				if port_value_mode == 'Both':
					st.caption(f'After-tax portfolio value (TDA taxed at {inh_rate:.0%} inheritor rate)')
				port_pcts_at = all_yearly.groupby('year')['after_tax_portfolio'].quantile(port_quantiles).unstack()
				port_pcts_at.columns = selected_port_bands
				interactive_line_chart(port_pcts_at, y_title='Portfolio Value (After-tax)')

		st.subheader('Account balances — median across all runs')
		inh_rate_acct = float(inheritor_marginal_rate)
		all_yearly['after_tax_total'] = (
			all_yearly['end_taxable_total'] + all_yearly['end_roth'] +
			all_yearly['end_tda_total'] * max(0.0, 1.0 - inh_rate_acct))
		acct_median = all_yearly.groupby('year')[['total_portfolio', 'after_tax_total', 'end_taxable_total', 'end_tda_total', 'end_roth']].median()
		acct_median.columns = ['Pre-tax Total', 'After-tax Total', 'Taxable', 'Tax-deferred', 'Roth']
		interactive_line_chart(acct_median, y_title='Balance')
		if inh_rate_acct > 0:
			st.caption(f'After-tax total discounts TDA balances at {inh_rate_acct:.0%} inheritor rate')

		st.subheader('Annual taxes — percentile bands across all runs')
		tax_pcts = all_yearly.groupby('year')['total_taxes'].quantile([0.00, 0.10, 0.25, 0.50, 0.75, 0.90]).unstack()
		tax_pcts.columns = ['0th (min)', '10th', '25th', '50th (median)', '75th', '90th']
		interactive_line_chart(tax_pcts, y_title='Taxes')

		# After-tax spending percentile bands
		if 'withdrawal_used' in all_yearly.columns:
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
			'ss_income_total', 'taxable_social_security', 'pension_income_total', 'pension_income_real', 'pension_erosion',
			'annuity_income_total', 'annuity_income_real', 'other_income',
			'roth_conversion', 'roth_conversion_tax', 'roth_conversion_tax_source',
			'deduction_applied', 'ordinary_taxable_income', 'capital_gains',
			'ordinary_tax_total', 'capital_gains_tax', 'niit_tax', 'state_tax', 'total_taxes',
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
			'ss_income_total': currency_fmt, 'taxable_social_security': currency_fmt, 'pension_income_total': currency_fmt, 'pension_income_real': currency_fmt, 'pension_erosion': currency_fmt,
			'annuity_income_total': currency_fmt, 'annuity_income_real': currency_fmt, 'other_income': currency_fmt,
			'roth_conversion': currency_fmt, 'roth_conversion_tax': currency_fmt,
			'deduction_applied': currency_fmt, 'ordinary_taxable_income': currency_fmt,
			'capital_gains': currency_fmt,
			'end_stocks_mv': currency_fmt, 'end_bonds_mv': currency_fmt, 'end_stocks_basis': currency_fmt, 'end_bonds_basis': currency_fmt,
			'end_taxable_total': currency_fmt, 'investment_return_dollars': currency_fmt,
			'end_tda_p1': currency_fmt, 'end_tda_p2': currency_fmt, 'end_tda_total': currency_fmt, 'end_roth': currency_fmt,
			'ordinary_tax_total': currency_fmt, 'capital_gains_tax': currency_fmt, 'niit_tax': currency_fmt, 'state_tax': currency_fmt, 'total_taxes': currency_fmt,
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
		spending_src = df[['year', 'after_tax_spending', 'total_taxes',
						   'ss_income_total', 'pension_income_real', 'annuity_income_real', 'other_income']].copy()
		spending_src['goal'] = [withdrawal_schedule[i] if i < len(withdrawal_schedule) else withdrawal_schedule[-1] for i in range(len(spending_src))]
		# Net portfolio withdrawals = after_tax_spending minus income streams (so stacked bars add up exactly)
		spending_src['Net Portfolio Withdrawals'] = (spending_src['after_tax_spending']
			- spending_src['ss_income_total'] - spending_src['pension_income_real']
			- spending_src['annuity_income_real'] - spending_src['other_income']).clip(lower=0)
		source_cols = ['Net Portfolio Withdrawals', 'Social Security', 'Pension (Real)', 'Annuity (Real)', 'Other Income']
		spending_src = spending_src.rename(columns={'ss_income_total': 'Social Security',
			'pension_income_real': 'Pension (Real)', 'annuity_income_real': 'Annuity (Real)', 'other_income': 'Other Income'})
		# Drop sources that are zero everywhere
		source_cols = [c for c in source_cols if spending_src[c].sum() > 0]
		src_long = spending_src[['year'] + source_cols].melt('year', var_name='Source', value_name='Amount')
		nearest = alt.selection_point(nearest=True, on='pointerover', fields=['year'], empty=False)
		stacked_bars = alt.Chart(src_long).mark_bar().encode(
			x=alt.X('year:O', title='Year'),
			y=alt.Y('Amount:Q', title='After-Tax Spending ($)', stack='zero'),
			color=alt.Color('Source:N', sort=source_cols, legend=alt.Legend(title='Source')),
			order=alt.Order('source_order:Q'),
			tooltip=[alt.Tooltip('year:O', title='Year'),
					 alt.Tooltip('Source:N'),
					 alt.Tooltip('Amount:Q', format='$,.0f')]
		).transform_calculate(
			source_order=' : '.join([f'datum.Source === "{s}" ? {i}' for i, s in enumerate(source_cols)]) + ' : 99'
		).properties(height=400)
		# Goal line
		goal_line = alt.Chart(spending_src).mark_line(color='black', strokeDash=[6, 3], strokeWidth=2).encode(
			x=alt.X('year:O'),
			y=alt.Y('goal:Q'),
			tooltip=[alt.Tooltip('year:O', title='Year'),
					 alt.Tooltip('goal:Q', title='Spending Goal', format='$,.0f')]
		)
		# Invisible points for total-level tooltip with full breakdown
		tip_fields = [alt.Tooltip('year:O', title='Year'),
					  alt.Tooltip('after_tax_spending:Q', title='Total Spending', format='$,.0f'),
					  alt.Tooltip('goal:Q', title='Goal', format='$,.0f')]
		for sc in source_cols:
			tip_fields.append(alt.Tooltip(f'{sc}:Q', title=sc, format='$,.0f'))
		tip_fields.append(alt.Tooltip('total_taxes:Q', title='Taxes Paid', format='$,.0f'))
		total_tip = alt.Chart(spending_src).mark_point(size=80, filled=True, opacity=0).encode(
			x=alt.X('year:O'),
			y=alt.Y('after_tax_spending:Q'),
			tooltip=tip_fields,
		).add_params(nearest)
		st.altair_chart((stacked_bars + goal_line + total_tip).interactive(), use_container_width=True)
		st.caption('Stacked bars show income sources. Dashed line = spending goal. Hover for details.')

		st.subheader('Account balances over time — single run')
		bal_df = currency_round[['year','end_taxable_total','end_tda_total','end_roth']].set_index('year')
		bal_df.columns = ['Taxable', 'Tax-deferred', 'Roth']
		interactive_line_chart(bal_df, y_title='Balance')

		st.subheader('Taxes paid per year — single run')
		tax_df = currency_round[['year','ordinary_taxable_income','ordinary_tax_total','capital_gains','capital_gains_tax','niit_tax','state_tax','total_taxes','deduction_applied','roth_conversion','roth_conversion_tax']].set_index('year')
		st.dataframe(tax_df.style.format({
			'ordinary_taxable_income': currency_fmt,
			'ordinary_tax_total': currency_fmt,
			'capital_gains': currency_fmt,
			'capital_gains_tax': currency_fmt,
			'niit_tax': currency_fmt,
			'state_tax': currency_fmt,
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
			'state_tax': df['state_tax'],
		}).set_index('year')
		st.subheader('Effective and marginal tax rates — single run')
		st.dataframe(mtr_df.style.format({
			'taxable_income_total': currency_fmt,
			'total_taxes': currency_fmt,
			'effective_tax_rate': '{:.2%}'.format,
			'marginal_ordinary_rate': '{:.2%}'.format,
			'marginal_cap_gains_rate': '{:.2%}'.format,
			'niit_tax': currency_fmt,
			'state_tax': currency_fmt,
		}))
		rate_df = mtr_df[['effective_tax_rate','marginal_ordinary_rate','marginal_cap_gains_rate']].copy()
		rate_df.columns = ['Effective Rate', 'Marginal Ordinary', 'Marginal Cap Gains']
		interactive_line_chart(rate_df, y_title='Tax Rate', fmt='.1%')
		lifetime_taxes = df['total_taxes'].sum()
		lifetime_ordinary_tax = df['ordinary_tax_total'].sum()
		lifetime_cap_gains_tax = df['capital_gains_tax'].sum()
		lifetime_state_tax = df['state_tax'].sum()
		st.metric('Total lifetime taxes paid', currency_fmt.format(lifetime_taxes))
		tax_breakdown = f'Ordinary: {currency_fmt.format(lifetime_ordinary_tax)} | Capital gains/QD: {currency_fmt.format(lifetime_cap_gains_tax)}'
		if lifetime_state_tax > 0:
			tax_breakdown += f' | State: {currency_fmt.format(lifetime_state_tax)}'
		st.caption(tax_breakdown)

		st.markdown('---')
		st.write('Ending balances')
		last_row = currency_round.iloc[-1]
		taxable_total_end = last_row['end_stocks_mv'] + last_row['end_bonds_mv']
		st.write({'taxable_end': taxable_total_end, 'stocks_end': last_row['end_stocks_mv'], 'stocks_end_basis': last_row['end_stocks_basis'], 'bonds_end': last_row['end_bonds_mv'], 'bonds_end_basis': last_row['end_bonds_basis'], 'tda_end': last_row['end_tda_total'], 'roth_end': last_row['end_roth']})

		# Store latest summary in session for post-run saving
		inh_rate_save = float(inheritor_marginal_rate)
		if sim_mode in ('simulated', 'historical_dist') and 'mc_all_yearly' in st.session_state:
			# Use median ending balances across all runs
			ay = st.session_state['mc_all_yearly']
			last_year_data = ay[ay['year'] == ay['year'].max()]
			med_taxable = float(last_year_data['end_taxable_total'].median())
			med_tda = float(last_year_data['end_tda_total'].median())
			med_roth = float(last_year_data['end_roth'].median())
			med_taxes = float(ay.groupby('run')['total_taxes'].sum().median())
			# Compute per-run totals first, then take median (median of sum, not sum of medians)
			run_pre_tax = last_year_data['end_taxable_total'] + last_year_data['end_tda_total'] + last_year_data['end_roth']
			run_after_tax = (last_year_data['end_taxable_total'] + last_year_data['end_roth'] +
				last_year_data['end_tda_total'] * max(0.0, 1.0 - inh_rate_save))
			pre_tax_total = float(run_pre_tax.median())
			after_tax_total = float(run_after_tax.median())
		else:
			last = df.iloc[-1]
			med_taxable = float(last['end_stocks_mv'] + last['end_bonds_mv'])
			med_tda = float(last['end_tda_total'])
			med_roth = float(last['end_roth'])
			med_taxes = float(lifetime_taxes)
			pre_tax_total = med_taxable + med_tda + med_roth
			after_tax_total = med_taxable + med_roth + med_tda * max(0.0, 1.0 - inh_rate_save)
		summary_data = {
			'label': f"conversion ${roth_conversion_amount:,.0f} for {roth_conversion_years} yrs, taxes from {roth_conversion_tax_source}",
			'total_taxes': med_taxes,
			'total_accounts': pre_tax_total,
			'taxable_end': med_taxable,
			'tda_end': med_tda,
			'roth_end': med_roth,
			'inheritor_marginal_rate': inh_rate_save,
			'after_tax_end': after_tax_total,
			'portfolio_cagr': portfolio_cagr,
			'roth_cagr': roth_cagr,
			'sim_type': sim_mode,
		}
		if sim_mode in ('simulated', 'historical_dist'):
			summary_data['percentiles'] = st.session_state.get('mc_percentile_rows', [])
			summary_data['pct_non_positive_end'] = st.session_state.get('mc_pct_non_positive', 0.0)
			summary_data['spending_percentiles'] = st.session_state.get('mc_spending_pct_rows', [])
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
		for s in st.session_state['scenario_summaries']:
			pcts = s.get('percentiles', [])
			if pcts:
				st.markdown(f"**{s['name']}**")
				pct_df = pd.DataFrame(pcts)
				pct_non_pos = s.get('pct_non_positive_end', 0.0)
				st.dataframe(pct_df.style.format({
					'percentile': lambda x: f"{int(x)}th",
					'after_tax_end': currency_fmt,
					'total_taxes': currency_fmt,
					'effective_tax_rate': '{:.2%}'.format,
					'portfolio_cagr': '{:.2%}'.format,
					'roth_cagr': '{:.2%}'.format,
				}))
				if pct_non_pos is not None:
					st.caption(f"Percent of ending values ≤ 0: {pct_non_pos * 100:.1f}%")
			spending_pcts = s.get('spending_percentiles', [])
			if spending_pcts:
				st.markdown(f"*{s['name']} — Lifetime spending distribution*")
				sp_df = pd.DataFrame(spending_pcts)
				st.dataframe(sp_df.style.format({
					'percentile': lambda x: f"{int(x)}th",
					'avg_annual_withdrawal': currency_fmt,
					'avg_annual_after_tax_spending': currency_fmt,
					'total_lifetime_withdrawal': currency_fmt,
					'total_lifetime_after_tax_spending': currency_fmt,
				}))


if __name__ == '__main__':
	main()
