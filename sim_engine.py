"""Simulation engine — data loading, MC runner, withdrawal simulation.

No Streamlit dependency.  Uses functools.lru_cache for data caching.
"""

import os
import functools
import numpy as np
import pandas as pd
from typing import Dict, Optional, Sequence

from tax_engine import (get_standard_deduction, get_ordinary_brackets,
	compute_taxable_social_security, apply_brackets, compute_capital_gains_tax,
	get_marginal_rates, compute_niit, compute_state_tax)

# ── Data loading ────────────────────────────────────────────────

_DATA_DIR = os.path.dirname(__file__)

# Load median purchasing power factors for pension real-value adjustment (years 1-40)
_pp_df = pd.read_excel(os.path.join(_DATA_DIR, 'median_cpi_purchasing_power.xlsx'))
PP_FACTORS = _pp_df['Median_Purchasing_Power'].tolist()

# Load monthly CPI factors for per-run historical purchasing power
_cpi_mo_df = pd.read_excel(os.path.join(_DATA_DIR, 'cpi_mo_factors.xlsx'))
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

@functools.lru_cache(maxsize=1)
def load_master_global():
	"""Load the full master_global_factors.xlsx and return the DataFrame."""
	path = os.path.join(_DATA_DIR, 'master_global_factors.xlsx')
	df = pd.read_excel(path)
	df['begin month'] = pd.to_datetime(df['begin month'])
	return df

@functools.lru_cache(maxsize=1)
def load_bond_factors():
	"""Load historical bond growth factors from master_global_factors.xlsx (LBM 100 F column).
	Returns an array of annual returns (growth_factor - 1)."""
	df = load_master_global()
	factors = df['LBM 100 F'].dropna().values
	return factors - 1.0  # convert growth factors to returns

@functools.lru_cache(maxsize=None)
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

def sample_lognormal_returns(years: int, drift: float, volatility: float, rng: np.random.Generator) -> Sequence[float]:
	"""Return `years` draws from a lognormal distribution built from the provided normal parameters."""
	log_returns = rng.normal(loc=drift, scale=volatility, size=years)
	return np.exp(log_returns) - 1

# ── Simulation helpers ──────────────────────────────────────────

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
	"""Return a dict age -> distribution period (divisor) using the IRS Uniform Lifetime Table."""
	table = {
		70:27.4,71:26.5,72:25.6,73:24.7,74:23.8,75:22.9,76:22.0,77:21.2,78:20.3,79:19.5,
		80:18.7,81:17.9,82:17.1,83:16.3,84:15.5,85:14.8,86:14.1,87:13.4,88:12.7,89:12.0,
		90:11.4,91:10.8,92:10.2,93:9.6,94:9.1,95:8.6,96:8.1,97:7.6,98:7.1,99:6.7,100:6.3,
		101:5.9,102:5.5,103:5.2,104:4.9,105:4.6,106:4.3,107:4.1,108:3.9,109:3.7,110:3.5,
		111:3.3,112:3.1,113:2.9,114:2.7,115:2.5,116:2.3,117:2.1,118:1.9,119:1.7,120:1.5
	}
	return table

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

def forward_success_rate(portfolio, remaining_schedule, scale_factor, blended_mu, blended_sigma, n_sims=200, income_schedule=None):
	"""Fast vectorized MC to estimate probability portfolio survives the remaining schedule."""
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
	that gives target_success survival rate."""
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

# ── Result processing ───────────────────────────────────────────

def store_distribution_results(results, all_yearly_df, sim_mode_label, ending_balance_goal=1.0):
	"""Process MC or historical distribution results.
	Returns a dict of results to be written into session state by the caller."""
	mc_df = pd.DataFrame(results)
	percentiles_list = [0, 10, 25, 50, 75, 90]
	summary_cols = ['after_tax_end', 'total_taxes', 'effective_tax_rate', 'portfolio_cagr', 'roth_cagr']
	pct_rows = []
	for p in percentiles_list:
		row = {'percentile': p}
		for col in summary_cols:
			row[col] = np.percentile(mc_df[col], p)
		pct_rows.append(row)
	goal = float(ending_balance_goal)
	pct_non_positive = float((mc_df['after_tax_end'] < goal).mean())
	run_ends = all_yearly_df.groupby('run')['total_portfolio'].last()
	median_val = run_ends.median()
	median_run_idx = int((run_ends - median_val).abs().idxmin())
	median_df = all_yearly_df[all_yearly_df['run'] == median_run_idx].drop(columns=['run', 'total_portfolio']).reset_index(drop=True)
	return {
		'mc_percentile_rows': pct_rows,
		'mc_pct_non_positive': pct_non_positive,
		'mc_all_yearly': all_yearly_df,
		'num_sims': len(results),
		'sim_df': median_df,
		'sim_mode': sim_mode_label,
	}

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

def compute_scenario_summary(name: str, results: list, all_yearly_df: pd.DataFrame,
							 inheritor_rate: float, ending_balance_goal: float = 1.0) -> dict:
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
	goal = float(ending_balance_goal)
	pct_non_positive = float((mc_df['after_tax_end'] < goal).mean())
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

# ── Monte Carlo runner ──────────────────────────────────────────

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

# ── Withdrawal waterfall + tax computation ──────────────────────

def try_gross_withdrawal(gross_target, snap_balances, year_income, tax_cfg):
	"""Run withdrawal waterfall + full tax computation for a given gross
	withdrawal from portfolio. Returns (net_after_tax_spending, result_dict).

	snap_balances: dict with stocks_mv, bonds_mv, stocks_basis, bonds_basis,
		tda1_stocks, tda1_bonds, tda2_stocks, tda2_bonds, roth_stocks, roth_bonds, total_rmd_cash
	year_income: dict with interest, div, turnover_realized_gain, ss_income,
		pension_nominal, pension_real, annuity_nominal, annuity_real,
		other_income, deduction, filing_status, pending_roth_conversion
	tax_cfg: dict with taxes_enabled, state_tax_rate, state_exempt_retirement,
		roth_conversion_tax_source
	"""
	s_mv = snap_balances['stocks_mv']
	b_mv = snap_balances['bonds_mv']
	s_basis = snap_balances['stocks_basis']
	b_basis = snap_balances['bonds_basis']
	t1s = snap_balances['tda1_stocks']
	t1b = snap_balances['tda1_bonds']
	t2s = snap_balances['tda2_stocks']
	t2b = snap_balances['tda2_bonds']
	rs = snap_balances['roth_stocks']
	rb = snap_balances['roth_bonds']
	total_rmd_cash = snap_balances['total_rmd_cash']

	interest = year_income['interest']
	div = year_income['div']
	turnover_realized_gain = year_income['turnover_realized_gain']
	ss_income = year_income['ss_income']
	pension_income = year_income['pension_nominal']
	pension_income_real = year_income['pension_real']
	annuity_income = year_income['annuity_nominal']
	annuity_income_real = year_income['annuity_real']
	other_income = year_income['other_income']
	deduction = year_income['deduction']
	filing_status_this_year = year_income['filing_status']
	pending_roth_conversion = year_income['pending_roth_conversion']

	taxes_enabled = tax_cfg['taxes_enabled']
	state_tax_rate = tax_cfg['state_tax_rate']
	state_exempt_retirement = tax_cfg['state_exempt_retirement']
	roth_conversion_tax_source = tax_cfg['roth_conversion_tax_source']

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
		agi = ordinary_pre_ss_with_conv + t_ss + cg_total
		net_inv = max(0.0, cg_total + interest)
		niit = compute_niit(agi, net_inv, filing_status_this_year)
		total_tax += niit

		# State income tax
		s_tax = compute_state_tax(t_ordinary, cg_total, interest, state_tax_rate, state_exempt_retirement)
		total_tax += s_tax

		marg_ord, marg_cg = get_marginal_rates(t_ordinary, cg_total, filing_status_this_year)
		niit_threshold = 200000 if filing_status_this_year == 'single' else 250000
		niit_base_val = max(0.0, agi - niit_threshold)
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

# ── Main simulation ─────────────────────────────────────────────

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
						 guardrail_max_spending_pct: float = -1.0,
						 taxes_enabled: bool = True,
						 investment_fee_bps: float = 0.0):
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
		if guardrail_max_spending_pct >= 0:
			max_scale = 1.0 + guardrail_max_spending_pct / 100.0
			current_scale_factor = min(current_scale_factor, max_scale)
	else:
		current_scale_factor = 1.0

	# Tax config dict (shared across all years — constant per run)
	tax_cfg = {
		'taxes_enabled': taxes_enabled,
		'state_tax_rate': state_tax_rate,
		'state_exempt_retirement': state_exempt_retirement,
		'roth_conversion_tax_source': roth_conversion_tax_source,
	}

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
					if guardrail_max_spending_pct >= 0:
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
		if investment_fee_bps > 0:
			fee_rate = investment_fee_bps / 10000.0
			stock_return_year = (1 + stock_return_year) * (1 - fee_rate) - 1
			bond_return_year = (1 + bond_return_year) * (1 - fee_rate) - 1
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
				annuity_income_yr = ann_nom_p1 + ann_nom_p2
			elif primary_alive and not spouse_alive:
				annuity_income_yr = ann_nom_p1 + ann_nom_p2 * annuity_survivor_pct_p2
			elif spouse_alive and not primary_alive:
				annuity_income_yr = ann_nom_p2 + ann_nom_p1 * annuity_survivor_pct_p1
			else:
				annuity_income_yr = 0.0
		else:
			annuity_income_yr = 0.0
		annuity_income_real = annuity_income_yr * pp_yr
		other_income = other_income_annual
		deduction = itemized_deduction_amount if use_itemized_deductions else get_standard_deduction(filing_status_this_year)

		# Snapshot mutable balances (post-growth, post-RMD) for iterative solve
		total_rmd_cash = rmd_p1 + rmd_p2
		snap_balances = {
			'stocks_mv': stocks_mv, 'bonds_mv': bonds_mv,
			'stocks_basis': stocks_basis, 'bonds_basis': bonds_basis,
			'tda1_stocks': tda1_stocks_mv, 'tda1_bonds': tda1_bonds_mv,
			'tda2_stocks': tda2_stocks_mv, 'tda2_bonds': tda2_bonds_mv,
			'roth_stocks': roth_stocks_mv, 'roth_bonds': roth_bonds_mv,
			'total_rmd_cash': total_rmd_cash,
		}
		year_income = {
			'interest': interest, 'div': div,
			'turnover_realized_gain': turnover_realized_gain,
			'ss_income': ss_income,
			'pension_nominal': pension_income, 'pension_real': pension_income_real,
			'annuity_nominal': annuity_income_yr, 'annuity_real': annuity_income_real,
			'other_income': other_income,
			'deduction': deduction,
			'filing_status': filing_status_this_year,
			'pending_roth_conversion': pending_roth_conversion,
		}

		# Binary search: find gross withdrawal that delivers the net spending target
		base_withdrawal_this_year = withdrawal_schedule[y-1]
		net_target = base_withdrawal_this_year * current_scale_factor

		base_net, base_result = try_gross_withdrawal(0.0, snap_balances, year_income, tax_cfg)

		if net_target <= 0 or base_net >= net_target:
			chosen = base_result
		else:
			max_available = (stocks_mv + bonds_mv +
				tda1_stocks_mv + tda1_bonds_mv +
				tda2_stocks_mv + tda2_bonds_mv +
				roth_stocks_mv + roth_bonds_mv + total_rmd_cash)
			lo = 0.0
			hi = min(net_target * 2.5, max_available)
			chosen = base_result
			for _ in range(40):
				mid = (lo + hi) / 2.0
				mid_net, mid_result = try_gross_withdrawal(mid, snap_balances, year_income, tax_cfg)
				if mid_net < net_target - 0.50:
					lo = mid
				elif mid_net > net_target + 0.50:
					hi = mid
				else:
					chosen = mid_result
					break
				chosen = mid_result
				if hi - lo < 1.0:
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
			'annuity_income_total': annuity_income_yr,
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
