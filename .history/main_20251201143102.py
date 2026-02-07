import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Optional, Sequence, List, Any, Tuple

st.set_page_config(page_title='Withdrawal + RMD Simulator', layout='wide')

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
					withdraw_amount: float,
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
					target_stock_pct: float,
					taxable_stock_basis_pct: float,
					taxable_bond_basis_pct: float,
					gross_up_withdrawals: bool,
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
					roth_log_drift: float,
					roth_log_volatility: float,
					life_expectancy_primary: int,
					life_expectancy_spouse: int,
					inheritor_rate: float,
					years: int,
					seed: Optional[int] = None) -> List[Dict[str, float]]:
	results = []
	rng = np.random.default_rng(seed)
	for _ in range(num_runs):
		taxable_series = sample_lognormal_returns(years, taxable_log_drift, taxable_log_volatility, rng)
		roth_series = sample_lognormal_returns(years, roth_log_drift, roth_log_volatility, rng)
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
		withdraw_amount=withdraw_amount,
		target_stock_pct=target_stock_pct,
		taxable_stock_basis_pct=taxable_stock_basis_pct,
		taxable_bond_basis_pct=taxable_bond_basis_pct,
		gross_up_withdrawals=gross_up_withdrawals,
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
		bond_return_series=taxable_series,
		roth_return_series=roth_series,
		life_expectancy_primary=life_expectancy_primary,
		life_expectancy_spouse=life_expectancy_spouse)
		metrics = compute_summary_metrics(df_run, inheritor_rate)
		results.append(metrics)
	return results
def sample_lognormal_returns(years: int, drift: float, volatility: float, rng: np.random.Generator) -> Sequence[float]:
	"""Return `years` draws from a lognormal distribution built from the provided normal parameters."""
	log_returns = rng.normal(loc=drift, scale=volatility, size=years)
	return np.exp(log_returns) - 1

def make_seed_provider(base_seed: Optional[int]):
	counter = 0
	def provider():
		nonlocal counter
		if base_seed is None:
			return None
		value = base_seed + counter
		counter += 1
		return value
	return provider

def evaluate_success_rate(withdraw_amount: float,
						  years: int,
						  mc_kwargs: Dict[str, Any],
						  mc_runs: int,
						  seed_provider) -> Tuple[float, pd.DataFrame]:
	if years <= 0:
		return 1.0, pd.DataFrame()
	seed = seed_provider()
	results = run_monte_carlo(
		num_runs=mc_runs,
		withdraw_amount=withdraw_amount,
		seed=seed,
		**mc_kwargs
	)
	df_results = pd.DataFrame(results)
	if df_results.empty:
		return 0.0, df_results
	success_rate = float((df_results['after_tax_end'] > 0.0).mean())
	return success_rate, df_results

def search_withdrawal_for_target(start_total: float,
								target_success: float,
								mc_kwargs: Dict[str, Any],
								years: int,
								mc_runs: int,
								tolerance: float,
								max_iterations: int,
								seed_provider,
								max_withdrawal: Optional[float] = None) -> Tuple[float, float]:
	high = max(start_total, 1.0)
	if max_withdrawal is not None:
		high = min(high, max_withdrawal)
	best_withdrawal = 0.0
	best_success = 0.0
	success_high, last_df = evaluate_success_rate(high, years, mc_kwargs, mc_runs, seed_provider)
	doubling = 0
	while success_high >= target_success and doubling < 10:
		if max_withdrawal is not None and high >= max_withdrawal:
			break
		high *= 1.5
		if max_withdrawal is not None:
			high = min(high, max_withdrawal)
		success_high, last_df = evaluate_success_rate(high, years, mc_kwargs, mc_runs, seed_provider)
		doubling += 1
	low = 0.0
	for _ in range(max_iterations):
		mid = (low + high) / 2.0
		success_mid, mid_df = evaluate_success_rate(mid, years, mc_kwargs, mc_runs, seed_provider)
		if success_mid >= target_success:
			best_withdrawal = mid
			best_success = success_mid
			low = mid
		else:
			high = mid
		if high - low <= tolerance:
			break
	if best_withdrawal == 0.0 and success_high >= target_success:
		best_withdrawal = high
		best_success = success_high
	return best_withdrawal, best_success

def sum_account_values(mc_kwargs: Dict[str, Any]) -> float:
	return float(mc_kwargs.get('taxable_start', 0.0) +
				 mc_kwargs.get('tda_start', 0.0) +
				 mc_kwargs.get('tda_spouse_start', 0.0) +
				 mc_kwargs.get('roth_start', 0.0))

def run_closed_loop_monte_carlo(initial_withdrawal: float,
								total_years: int,
								start_age: int,
								start_age_spouse: int,
								simulation_constants: Dict[str, Any],
								base_mc_kwargs: Dict[str, Any],
								mc_runs: int,
								tolerance: float,
								max_iterations: int,
								taxable_log_drift: float,
								taxable_log_volatility: float,
								roth_log_drift: float,
								roth_log_volatility: float,
								seed_base: Optional[int],
								hist_schedule: pd.DataFrame,
								hist_bucket_column: str,
								hist_roth_column: str,
								hist_start_row: int,
								spending_cap_multiplier: float) -> Dict[str, Any]:
	results: List[Dict[str, Any]] = []
	kwargs = dict(base_mc_kwargs)
	seed_provider = make_seed_provider(seed_base)
	actual_rng = np.random.default_rng(seed_base)
	current_withdrawal = initial_withdrawal
	last_success = 0.0
	max_permitted = max(initial_withdrawal * spending_cap_multiplier, initial_withdrawal)
	for year in range(1, total_years + 1):
		remaining_years = total_years - year
		row_idx = hist_start_row + (year - 1) * 12
		if row_idx >= len(hist_schedule):
			break
		stock_return = float(hist_schedule.iloc[row_idx][hist_bucket_column]) - 1.0
		bond_return = stock_return
		roth_return = float(hist_schedule.iloc[row_idx][hist_roth_column]) - 1.0
		df_year = simulate_withdrawals(
			start_age_primary=start_age,
			start_age_spouse=start_age_spouse,
			years=1,
			taxable_start=kwargs['taxable_start'],
			stock_total_return=simulation_constants['stock_total_return'],
			stock_dividend_yield=simulation_constants['stock_dividend_yield'],
			stock_turnover=simulation_constants['stock_turnover'],
			bond_return=simulation_constants['bond_return'],
			roth_start=kwargs['roth_start'],
			tda_start=kwargs['tda_start'],
			tda_spouse_start=kwargs['tda_spouse_start'],
			withdraw_amount=current_withdrawal,
			target_stock_pct=simulation_constants['target_stock_pct'],
			taxable_stock_basis_pct=simulation_constants['taxable_stock_basis_pct'],
			taxable_bond_basis_pct=simulation_constants['taxable_bond_basis_pct'],
			gross_up_withdrawals=simulation_constants['gross_up_withdrawals'],
			rmd_start_age=simulation_constants['rmd_start_age'],
			rmd_start_age_spouse=simulation_constants['rmd_start_age_spouse'],
			ss_income_annual=simulation_constants['ss_income_annual'],
			ss_income_spouse_annual=simulation_constants['ss_income_spouse_annual'],
			ss_cola=simulation_constants['ss_cola'],
			pension_income_annual=simulation_constants['pension_income_annual'],
			pension_income_spouse_annual=simulation_constants['pension_income_spouse_annual'],
			pension_cola=simulation_constants['pension_cola'],
			other_income_annual=simulation_constants['other_income_annual'],
			filing_status=simulation_constants['filing_status'],
			use_itemized_deductions=simulation_constants['use_itemized_deductions'],
			itemized_deduction_amount=simulation_constants['itemized_deduction_amount'],
			roth_conversion_amount=simulation_constants['roth_conversion_amount'],
			roth_conversion_years=simulation_constants['roth_conversion_years'],
			roth_conversion_tax_source=simulation_constants['roth_conversion_tax_source'],
			stock_return_series=[stock_return],
			bond_return_series=[bond_return],
			roth_return_series=[roth_return],
			life_expectancy_primary=simulation_constants['life_expectancy_primary'],
			life_expectancy_spouse=simulation_constants['life_expectancy_spouse']
		)
		if df_year.empty:
			break
		row = df_year.iloc[0].to_dict()
		kwargs['taxable_start'] = float(row['end_taxable_total'])
		kwargs['tda_start'] = float(row['end_tda_p1'])
		kwargs['tda_spouse_start'] = float(row['end_tda_p2'])
		kwargs['roth_start'] = float(row['end_roth'])
		year_withdrawal = current_withdrawal
		success_rate = last_success
		adjustment_note = 'unchanged'
		if remaining_years > 0:
			kwargs['years'] = remaining_years
			success_rate, _ = evaluate_success_rate(current_withdrawal, remaining_years, kwargs, mc_runs, seed_provider)
			if success_rate >= 0.90:
				start_total = sum_account_values(kwargs)
				new_withdrawal, _ = search_withdrawal_for_target(max(start_total, 1.0), 0.85, kwargs, remaining_years, mc_runs, tolerance, max_iterations, seed_provider, max_withdrawal=max_permitted)
				if new_withdrawal > 0 and abs(new_withdrawal - current_withdrawal) > 1e-6:
					current_withdrawal = new_withdrawal
					adjustment_note = '↑ to 85%'
			elif success_rate < 0.75:
				start_total = sum_account_values(kwargs)
				new_withdrawal, _ = search_withdrawal_for_target(max(start_total, 1.0), 0.85, kwargs, remaining_years, mc_runs, tolerance, max_iterations, seed_provider, max_withdrawal=max_permitted)
				if new_withdrawal > 0 and abs(new_withdrawal - current_withdrawal) > 1e-6:
					current_withdrawal = new_withdrawal
					adjustment_note = '↓ to 85%'
			last_success = success_rate
		results.append({
			'year': year,
			'withdrawal': year_withdrawal,
			'success_rate': success_rate,
			'adjustment': adjustment_note or 'unchanged',
			'hist_row': row_idx,
			'hist_period_start': hist_schedule.iloc[row_idx]['begin month'],
			'start_total': float(row['start_stocks_mv'] + row['start_bonds_mv'] + row['start_tda_p1'] + row['start_tda_p2'] + row['start_roth']),
			'end_total': float(row['end_accounts_total']),
			'portfolio_return': float(row['portfolio_return']),
			'roth_return': float(row['roth_return_used']),
			'total_taxes': float(row['total_taxes'])
		})
	return {
		'records': results,
		'final_withdrawal': current_withdrawal,
		'final_total': results[-1]['end_total'] if results else 0.0,
		'last_success_rate': last_success,
		'total_years': total_years,
		'iterations': mc_runs
	}
def build_percentile_rows(df: pd.DataFrame, percentiles: Sequence[int], columns: Sequence[str]) -> List[Dict[str, float]]:
	rows: List[Dict[str, float]] = []
	if df.empty:
		return rows
	for p in percentiles:
		row: Dict[str, float] = {'percentile': float(p)}
		for col in columns:
			if col in df:
				row[col] = float(np.percentile(df[col], p))
			else:
				row[col] = 0.0
		rows.append(row)
	return rows

def find_withdrawal_for_success(initial_total: float,
								target_success: float,
								finder_runs: int,
								tolerance: float,
								max_iterations: int,
								seed: Optional[int],
								mc_kwargs: Dict[str, Any]) -> Tuple[float, float, List[Dict[str, float]]]:
	"""Binary search the withdrawal amount that meets target success across Monte Carlo runs."""
	def evaluate(amount: float) -> Tuple[float, pd.DataFrame]:
		results = run_monte_carlo(
			num_runs=finder_runs,
			withdraw_amount=amount,
			seed=seed,
			**mc_kwargs
		)
		df_results = pd.DataFrame(results)
		if df_results.empty:
			return 0.0, df_results
		success_rate = (df_results['after_tax_end'] >= 1.0).mean()
		return float(success_rate), df_results

	high = max(initial_total, 1.0)
	success_high, df_last = evaluate(high)
	doubling_steps = 0
	while success_high >= target_success and doubling_steps < 10:
		high *= 1.5
		success_high, df_last = evaluate(high)
		doubling_steps += 1

	low = 0.0
	best_withdrawal = 0.0
	best_success = 0.0
	for _ in range(max_iterations):
		mid = (low + high) / 2.0
		success_mid, df_mid = evaluate(mid)
		if success_mid >= target_success:
			best_withdrawal = mid
			best_success = success_mid
			low = mid
		else:
			high = mid
		df_last = df_mid if not df_mid.empty else df_last
		if high - low <= tolerance:
			break

	if best_withdrawal == 0.0 and success_high >= target_success:
		best_withdrawal = high
		best_success = success_high

	percentiles = [0, 10, 25, 50, 75, 90]
	cols = ['after_tax_end', 'total_taxes', 'effective_tax_rate', 'portfolio_cagr', 'roth_cagr']
	percentile_rows = build_percentile_rows(df_last, percentiles, cols) if df_last is not None else []
	return best_withdrawal, best_success, percentile_rows

HISTORICAL_SOURCES = {
	'SPX': {
		'path': 'master_spx_factors.xlsx',
		'cols': {
		100: 'spx100e',
		90: 'spx90e',
		80: 'spx80e',
		70: 'spx70e',
		60: 'spx60e',
		50: 'spx50e',
		40: 'spx40e',
		30: 'spx30e',
		20: 'spx20e',
		10: 'spx10e',
		0: 'spx0e',
		},
		'roth': 'spx100e',
	},
	'Global': {
		'path': 'master_global_factors.xlsx',
		'cols': {
		100: 'LBM 100E',
		90: 'LBM 90E',
		80: 'LBM 80E',
		70: 'LBM 70E',
		60: 'LBM 60E',
		50: 'LBM 50E',
		40: 'LBM 40E',
		30: 'LBM 30E',
		20: 'LBM 20E ',
		10: 'LBM 10E',
		0: 'LBM 100 F',
		},
		'roth': 'LBM 100E',
	},
}

@st.cache_data
def load_historical_schedule(source_key: str) -> pd.DataFrame:
	info = HISTORICAL_SOURCES[source_key]
	df = pd.read_excel(info['path'], parse_dates=['begin month', 'end month'])
	df = df.sort_values('begin month').reset_index(drop=True)
	min_year = df['begin month'].dt.year.min()
	if min_year != 1927:
		offset = min_year - 1927
		df['begin month'] = df['begin month'] - pd.DateOffset(years=offset)
		df['end month'] = df['end month'] - pd.DateOffset(years=offset)
	return df

def choose_bucket_column(info: Dict[str, Any], target_pct: float) -> str:
	buckets = sorted(info['cols'].keys(), reverse=True)
	pct = int(target_pct * 100)
	for bucket in buckets:
		if pct >= bucket:
			return info['cols'][bucket].strip()
	return info['cols'][0].strip()

def build_historical_returns(df: pd.DataFrame, start_row: int, years: int, column: str) -> List[float]:
	return [float(df.iloc[start_row + i][column]) - 1.0 for i in range(years)]
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

def round_account(stock: float, bond: float) -> Tuple[float, float]:
	total = stock + bond
	if total <= 0:
		return 0.0, 0.0
	frac = stock / total
	frac = round(frac * 10) / 10.0
	frac = min(1.0, max(0.0, frac))
	stock_new = total * frac
	bond_new = total - stock_new
	return stock_new, bond_new


def rebalance_accounts(target_stock_pct: float,
						 taxable_stock_mv: float,
						 taxable_bond_mv: float,
						 taxable_stock_basis: float,
						 taxable_bond_basis: float,
						 tda1_mv: float,
						 tda2_mv: float,
						 roth_mv: float):
	"""Rebalance household portfolio so Roth carries stocks, TDAs carry bonds, and taxable meets the household target."""
	total_household = taxable_stock_mv + taxable_bond_mv + tda1_mv + tda2_mv + roth_mv
	if total_household <= 0:
		return (taxable_stock_mv, taxable_bond_mv, taxable_stock_basis, taxable_bond_basis,
				0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

	desired_stock_total = total_household * target_stock_pct
	desired_bond_total = total_household - desired_stock_total
	taxable_total = taxable_stock_mv + taxable_bond_mv
	tda_total = tda1_mv + tda2_mv

	def clamp(val: float, minimum: float, maximum: float) -> float:
		return max(minimum, min(val, maximum))

	taxable_stock = clamp(desired_stock_total - roth_mv, 0.0, taxable_total)
	remaining_taxable = taxable_total - taxable_stock
	taxable_bond = clamp(desired_bond_total - tda_total, 0.0, remaining_taxable)
	remaining_taxable -= taxable_bond

	if remaining_taxable > 0:
		current_stock_gap = desired_stock_total - (roth_mv + taxable_stock)
		if current_stock_gap > 0:
			additional_stock = min(remaining_taxable, current_stock_gap)
			taxable_stock += additional_stock
			remaining_taxable -= additional_stock
		taxable_bond += remaining_taxable

	taxable_stock, taxable_bond = round_account(taxable_stock, taxable_bond)
	tda1_stock, tda1_bond = 0.0, tda1_mv
	tda2_stock, tda2_bond = 0.0, tda2_mv
	roth_stock, roth_bond = roth_mv, 0.0

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
						 withdraw_amount: float,
						 target_stock_pct: float,
						 taxable_stock_basis_pct: float,
						 taxable_bond_basis_pct: float,
						 gross_up_withdrawals: bool = True,
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
						 roth_return_series: Optional[Sequence[float]] = None,
						 life_expectancy_primary: int = 120,
						 life_expectancy_spouse: int = 120):
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
	if roth_return_series is not None and len(roth_return_series) < years:
		raise ValueError('roth_return_series must have at least `years` entries')

	rows = []

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
		cap_gain_rate_for_grossup = 0.20

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
		roth_return_year = roth_return_series[y-1] if roth_return_series is not None else stock_return_year

		# grow TDA (per user mix) and Roth (per user mix)
		tda1_stocks_mv *= (1 + stock_return_year)
		tda1_bonds_mv *= (1 + bond_return_year)
		tda2_stocks_mv *= (1 + stock_return_year)
		tda2_bonds_mv *= (1 + bond_return_year)
		roth_stocks_mv *= (1 + roth_return_year)
		roth_bonds_mv *= (1 + roth_return_year)

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

		# Begin withdrawal processing
		wanted = withdraw_amount
		out_taxable_cash = 0.0
		withdraw_from_tda = rmd_p1 + rmd_p2
		withdraw_from_roth = 0.0
		gross_sold_taxable_stocks = 0.0
		gross_sold_taxable_bonds = 0.0
		realized_gains_from_sales = 0.0
		rmd_excess_to_taxable = 0.0

		def gross_for_net(net, mv, basis, cap_rate):
			# Solve S such that S - tax = net, tax = (S - basis*(S/mv))*cap_rate
			if mv <= 0 or net <= 0:
				return 0.0
			basis_ratio = basis / mv if mv > 0 else 0.0
			denom = 1.0 - cap_rate * (1.0 - basis_ratio)
			if denom <= 0:
				return mv
			S = net / denom
			return min(S, mv)

		total_rmd_cash = rmd_p1 + rmd_p2
		if total_rmd_cash > 0:
			if total_rmd_cash > wanted:
				excess = total_rmd_cash - wanted
				rmd_excess_to_taxable = excess
				total_mv = stocks_mv + bonds_mv
				if total_mv > 0:
					stocks_mv += excess * (stocks_mv / total_mv)
					bonds_mv += excess * (bonds_mv / total_mv)
					# assume excess increases basis proportionally (treated as deposit)
					if (stocks_basis + bonds_basis) > 0:
						stocks_basis += excess * (stocks_basis / (stocks_basis + bonds_basis))
						bonds_basis += excess * (bonds_basis / (stocks_basis + bonds_basis))
					else:
						stocks_basis += excess * 0.5
						bonds_basis += excess * 0.5
				else:
					stocks_mv += excess * 0.5
					bonds_mv += excess * 0.5
					stocks_basis += excess * 0.5
					bonds_basis += excess * 0.5
				remaining = 0.0
			else:
				remaining = wanted - total_rmd_cash
		else:
			remaining = wanted

		# withdraw from taxable (gross-up to deliver net if requested)
		net_needed = remaining
		if net_needed > 0 and bonds_mv > 0:
			S = gross_for_net(net_needed, bonds_mv, bonds_basis, cap_gain_rate_for_grossup) if gross_up_withdrawals else min(net_needed, bonds_mv)
			basis_sold = S * (bonds_basis / bonds_mv) if bonds_mv > 0 else 0.0
			realized = max(0.0, S - basis_sold)
			gross_sold_taxable_bonds += S
			realized_gains_from_sales += realized
			bonds_mv -= S
			bonds_basis -= basis_sold
			net_needed -= S if gross_up_withdrawals else S

		if net_needed > 1e-8 and stocks_mv > 0:
			S = gross_for_net(net_needed, stocks_mv, stocks_basis, cap_gain_rate_for_grossup) if gross_up_withdrawals else min(net_needed, stocks_mv)
			basis_sold = S * (stocks_basis / stocks_mv) if stocks_mv > 0 else 0.0
			realized = max(0.0, S - basis_sold)
			realized_gains_from_sales += realized
			gross_sold_taxable_stocks += S
			stocks_mv -= S
			stocks_basis -= basis_sold
			net_needed -= S if gross_up_withdrawals else S

		out_taxable_cash += (remaining - max(0.0, net_needed))

		if net_needed > 1e-8:
			total_tda_balance = tda1_stocks_mv + tda1_bonds_mv
			take_tda = min(net_needed, total_tda_balance)
			tda_stock_ratio = (tda1_stocks_mv / total_tda_balance) if total_tda_balance > 0 else 0.5
			tda1_stocks_mv -= take_tda * tda_stock_ratio
			tda1_bonds_mv -= take_tda * (1 - tda_stock_ratio)
			withdraw_from_tda += take_tda
			net_needed -= take_tda
		if net_needed > 1e-8:
			total_tda2_balance = tda2_stocks_mv + tda2_bonds_mv
			take_tda2 = min(net_needed, total_tda2_balance)
			tda_stock_ratio2 = (tda2_stocks_mv / total_tda2_balance) if total_tda2_balance > 0 else 0.5
			tda2_stocks_mv -= take_tda2 * tda_stock_ratio2
			tda2_bonds_mv -= take_tda2 * (1 - tda_stock_ratio2)
			withdraw_from_tda += take_tda2
			net_needed -= take_tda2
		if net_needed > 1e-8:
			total_roth_balance = roth_stocks_mv + roth_bonds_mv
			take_roth = min(net_needed, total_roth_balance)
			roth_stock_ratio = (roth_stocks_mv / total_roth_balance) if total_roth_balance > 0 else 1.0
			roth_stocks_mv -= take_roth * roth_stock_ratio
			roth_bonds_mv -= take_roth * (1 - roth_stock_ratio)
			withdraw_from_roth += take_roth
			net_needed -= take_roth

		filing_status_this_year = filing_status
		if filing_status == 'mfj' and (not spouse_alive or not primary_alive):
			filing_status_this_year = 'single'

		# Other income items (Social Security and pension grow with COLA)
		ss_income_p1 = ss_income_annual * ((1 + ss_cola) ** (y - 1)) if primary_alive else 0.0
		ss_income_p2 = ss_income_spouse_annual * ((1 + ss_cola) ** (y - 1)) if spouse_alive else 0.0
		ss_income = ss_income_p1 + ss_income_p2
		pension_income_p1 = pension_income_annual * ((1 + pension_cola) ** (y - 1)) if primary_alive else 0.0
		pension_income_p2 = pension_income_spouse_annual * ((1 + pension_cola) ** (y - 1)) if spouse_alive else 0.0
		pension_income = pension_income_p1 + pension_income_p2
		other_income = other_income_annual

		# Ordinary income before deductions (exclude cap gains/qualified dividends for stacking)
		ordinary_income_pre_ss_base = interest + withdraw_from_tda + pension_income + other_income
		ordinary_income_pre_ss_with_conv = ordinary_income_pre_ss_base + pending_roth_conversion
		cap_gains_total = div + turnover_realized_gain + realized_gains_from_sales

		deduction = itemized_deduction_amount if use_itemized_deductions else get_standard_deduction(filing_status_this_year)

		def compute_tax_bundle(ordinary_income_pre_ss_val, status):
			taxable_ss_val = compute_taxable_social_security(ss_income, ordinary_income_pre_ss_val, cap_gains_total, status)
			ordinary_income_total_val = ordinary_income_pre_ss_val + taxable_ss_val
			taxable_ordinary_val = max(0.0, ordinary_income_total_val - deduction)
			ordinary_tax_val = apply_brackets(taxable_ordinary_val, get_ordinary_brackets(status))
			cap_gains_tax_val = compute_capital_gains_tax(taxable_ordinary_val, cap_gains_total, status)
			total_val = ordinary_tax_val + cap_gains_tax_val
			return taxable_ss_val, taxable_ordinary_val, ordinary_tax_val, cap_gains_tax_val, total_val

		taxable_ss, taxable_ordinary, ordinary_tax_total, cap_gains_tax, total_taxes = compute_tax_bundle(ordinary_income_pre_ss_with_conv, filing_status_this_year)
		_, _, _, _, total_tax_without_conv = compute_tax_bundle(ordinary_income_pre_ss_base, filing_status_this_year)
		roth_conversion_tax_delta = max(0.0, total_taxes - total_tax_without_conv)
		# NIIT approximation
		niit_threshold = 200000 if filing_status_this_year == 'single' else 250000
		agi_approx = ordinary_income_pre_ss_with_conv + taxable_ss + cap_gains_total
		niit_base = max(0.0, agi_approx - niit_threshold)
		net_investment_income = max(0.0, cap_gains_total + interest + div)
		niit_tax = 0.038 * min(niit_base, net_investment_income)
		total_taxes += niit_tax
		marginal_ordinary_rate, marginal_cg_rate = get_marginal_rates(taxable_ordinary, cap_gains_total, filing_status_this_year)
		if niit_base > 0 and cap_gains_total > 0:
			marginal_cg_rate += 0.038

		# apply Roth conversion flow based on tax source
		if pending_roth_conversion > 0:
			if roth_conversion_tax_source == 'taxable':
				# pay taxes from taxable balances (no gross-up)
				tax_to_pay = roth_conversion_tax_delta
				pay_from_bonds = min(tax_to_pay, bonds_mv)
				if pay_from_bonds > 0 and bonds_mv > 0:
					bonds_basis_ratio = bonds_basis / bonds_mv if bonds_mv > 0 else 0.0
					bonds_basis -= pay_from_bonds * bonds_basis_ratio
					bonds_mv -= pay_from_bonds
				remaining_tax = tax_to_pay - pay_from_bonds
				pay_from_stocks = min(remaining_tax, stocks_mv)
				if pay_from_stocks > 0 and stocks_mv > 0:
					stocks_basis_ratio = stocks_basis / stocks_mv if stocks_mv > 0 else 0.0
					stocks_basis -= pay_from_stocks * stocks_basis_ratio
					stocks_mv -= pay_from_stocks
				remaining_tax -= pay_from_stocks
				# if taxable couldn't cover all tax, pay leftover out of conversion (behave like TDA pay)
				conversion_net_to_roth = max(0.0, pending_roth_conversion - remaining_tax)
			else:
				# taxes paid from conversion amount inside TDA, reducing what lands in Roth
				conversion_net_to_roth = max(0.0, pending_roth_conversion - roth_conversion_tax_delta)

			# deposit into Roth (favor stocks; bonds only if necessary)
			roth_stocks_mv += conversion_net_to_roth

		rows.append({
			'year': y,
			'portfolio_return': stock_return_year,
			'roth_return_used': roth_return_year,
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
			'withdraw_from_taxable_net': out_taxable_cash,
			'withdraw_from_tda': withdraw_from_tda,
			'withdraw_from_roth': withdraw_from_roth,
			'rmd_excess_to_taxable': rmd_excess_to_taxable,
			'gross_sold_taxable_bonds': gross_sold_taxable_bonds,
			'gross_sold_taxable_stocks': gross_sold_taxable_stocks,
			'ss_income_total': ss_income,
			'taxable_social_security': taxable_ss,
			'pension_income_total': pension_income,
			'other_income': other_income,
			'roth_conversion': pending_roth_conversion,
			'roth_conversion_tax': roth_conversion_tax_delta,
			'roth_conversion_tax_source': roth_conversion_tax_source,
			'ordinary_taxable_income': taxable_ordinary,
			'ordinary_tax_total': ordinary_tax_total,
			'capital_gains': cap_gains_total,
			'capital_gains_tax': cap_gains_tax,
			'niit_tax': niit_tax,
			'marginal_ordinary_rate': marginal_ordinary_rate,
			'marginal_cap_gains_rate': marginal_cg_rate,
			'deduction_applied': deduction,
			'total_taxes': total_taxes,
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
			'end_accounts_total': (stocks_mv + bonds_mv) + (tda1_stocks_mv + tda1_bonds_mv + tda2_stocks_mv + tda2_bonds_mv) + (roth_stocks_mv + roth_bonds_mv),
		})

	df = pd.DataFrame(rows)
	return df

def main():
	st.title('Withdrawal + RMD Simulator (30-year)')

	if st.button('Reset saved scenarios'):
		st.session_state.pop('scenario_summaries', None)
		st.session_state.pop('last_summary', None)
		st.success('Saved scenarios cleared.')

	with st.sidebar:
		st.header('Inputs')
		start_age = st.number_input('Starting age (person 1)', min_value=18, max_value=120, value=65)
		start_age_spouse = st.number_input('Starting age (person 2)', min_value=18, max_value=120, value=60)
		life_expectancy_primary = st.number_input('Primary life expectancy (last age lived through)', min_value=int(start_age), max_value=120, value=84, step=1)
		life_expectancy_spouse = st.number_input('Spouse life expectancy (last age lived through)', min_value=int(start_age_spouse), max_value=120, value=89, step=1)
		taxable_start = st.number_input('Taxable account starting balance', value=300000.0, step=1000.0)
		taxable_stock_basis_pct = st.number_input('Taxable stock basis % of market value', value=50.0, min_value=0.0, max_value=100.0, step=1.0) / 100.0
		taxable_bond_basis_pct = st.number_input('Taxable bond basis % of market value', value=100.0, min_value=0.0, max_value=100.0, step=1.0) / 100.0
		roth_start = st.number_input('Roth account starting balance', value=0.0, step=1000.0)
		tda_start = st.number_input('Tax-deferred account starting balance (IRA/401k) - person 1', value=400000.0, step=1000.0)
		tda_spouse_start = st.number_input('Tax-deferred account starting balance (IRA/401k) - person 2', value=300000.0, step=1000.0)
		st.markdown('**Household allocation target**')
		# household target allocation
		target_stock_pct = st.slider('Household target % in stocks', min_value=0, max_value=100, value=60, step=10) / 100.0
		roth_conversion_amount = st.number_input('Annual Roth conversion amount (from TDA)', value=0.0, step=1000.0)
		roth_conversion_years = st.number_input('Years to perform conversions', value=0, min_value=0, max_value=100, step=1)
		roth_conversion_tax_source = st.radio('Pay conversion taxes from', ['Taxable', 'TDA (reduce net conversion)'], horizontal=False)

		st.markdown('---')
		withdraw_amount = st.number_input('Desired annual withdrawal', value=40000.0, step=1000.0)
		rmd_start_age = st.number_input('RMD start age (person 1)', min_value=65, max_value=89, value=73)
		rmd_start_age_spouse = st.number_input('RMD start age (person 2)', min_value=65, max_value=90, value=73)

		st.markdown('Other income (all taxed as ordinary for now)')
		ss_income_input = st.number_input('Annual Social Security - person 1 (current year)', value=30000.0, step=1000.0)
		ss_income_spouse_input = st.number_input('Annual Social Security - person 2 (current year)', value=20000.0, step=1000.0)
		ss_cola = st.number_input('Social Security COLA', value=0.02, format="%.4f")
		pension_income_input = st.number_input('Annual pension income - person 1', value=10000.0, step=1000.0)
		pension_income_spouse_input = st.number_input('Annual pension income - person 2', value=0.0, step=1000.0)
		pension_cola = st.number_input('Pension COLA', value=0.00, format="%.4f")
		other_income_input = st.number_input('Other ordinary income', value=0.0, step=1000.0)

		st.markdown('Filing status & deductions')
		filing_status_choice = st.radio('Filing status', ['Single', 'Married Filing Jointly'], horizontal=True, index=1)
		filing_status_key = 'single' if filing_status_choice == 'Single' else 'mfj'
		standard_deduction_display = 14600 if filing_status_key == 'single' else 29200
		use_itemized = st.checkbox('Use itemized deductions instead of standard', value=False)
		itemized_deduction_input = st.number_input('Itemized deduction amount', value=0.0, step=500.0)
		st.caption(f'Standard deduction used if not itemizing: ${standard_deduction_display:,.0f}')
		inheritor_marginal_rate = st.number_input(
		'Inheritor marginal tax rate on TDAs',
		value=0.22,
		min_value=0.0,
		max_value=0.37,
		format="%.4f"
		)

		st.markdown('Expected annual returns and taxable details')
		st.caption('Taxable/TDA follow lognormal 7% drift / 12.7% vol; Roth uses 9.038% / 20.485% (log parameters).')
		stock_total_return = 0.0
		bond_return = 0.0
		taxable_log_drift = st.number_input('Taxable/TDA log drift (µ)', value=0.07, format="%.4f")
		taxable_log_volatility = st.number_input('Taxable/TDA log volatility (σ)', value=0.127, format="%.4f")
		roth_log_drift = 0.090382612
		roth_log_volatility = 0.204852769
		random_seed_input = st.number_input('Random seed for returns', value=42, step=1)
		seed_mode = st.radio('Seed mode', ['Random each run', 'Fixed seed'], horizontal=True, index=0)
		stock_dividend_yield = st.number_input('Stock dividend (qualified) yield', value=0.02, format="%.4f")
		stock_turnover = st.number_input('Stock turnover rate', value=0.10, format="%.4f")
		st.info('You can set stock/bond splits separately for Roth and tax-deferred in the balances section.')

		gross_up = st.checkbox('Gross-up taxable sales to deliver requested net withdrawal (recommended)', value=True)
		display_decimals = st.number_input('Decimal places for tables/charts', min_value=0, max_value=6, value=0, step=1)
		monte_carlo_runs = st.number_input('Monte Carlo runs', min_value=50, max_value=5000, value=500, step=50)

	years = max(1, max(life_expectancy_primary - start_age, life_expectancy_spouse - start_age_spouse) + 1)

	simulation_constants = {
		'stock_total_return': float(stock_total_return),
		'stock_dividend_yield': float(stock_dividend_yield),
		'stock_turnover': float(stock_turnover),
		'bond_return': float(bond_return),
		'target_stock_pct': float(target_stock_pct),
		'taxable_stock_basis_pct': float(taxable_stock_basis_pct),
		'taxable_bond_basis_pct': float(taxable_bond_basis_pct),
		'gross_up_withdrawals': bool(gross_up),
		'rmd_start_age': int(rmd_start_age),
		'rmd_start_age_spouse': int(rmd_start_age_spouse),
		'ss_income_annual': float(ss_income_input),
		'ss_income_spouse_annual': float(ss_income_spouse_input),
		'ss_cola': float(ss_cola),
		'pension_income_annual': float(pension_income_input),
		'pension_income_spouse_annual': float(pension_income_spouse_input),
		'pension_cola': float(pension_cola),
		'other_income_annual': float(other_income_input),
		'filing_status': filing_status_key,
		'use_itemized_deductions': bool(use_itemized),
		'itemized_deduction_amount': float(itemized_deduction_input),
		'roth_conversion_amount': float(roth_conversion_amount),
		'roth_conversion_years': int(roth_conversion_years),
		'roth_conversion_tax_source': 'taxable' if roth_conversion_tax_source == 'Taxable' else 'tda',
		'life_expectancy_primary': int(life_expectancy_primary),
		'life_expectancy_spouse': int(life_expectancy_spouse),
		'inheritor_rate': float(inheritor_marginal_rate),
		'taxable_log_drift': float(taxable_log_drift),
		'taxable_log_volatility': float(taxable_log_volatility),
		'roth_log_drift': float(roth_log_drift),
		'roth_log_volatility': float(roth_log_volatility),
	}
	initial_total = float(taxable_start) + float(tda_start) + float(tda_spouse_start) + float(roth_start)
	common_mc_kwargs = {
		**simulation_constants,
		'start_age': int(start_age),
		'start_age_spouse': int(start_age_spouse),
		'taxable_start': float(taxable_start),
		'tda_start': float(tda_start),
		'tda_spouse_start': float(tda_spouse_start),
		'roth_start': float(roth_start),
		'years': int(years),
	}
	seed_for_mc = int(random_seed_input) if seed_mode == 'Fixed seed' else None

	df = None
	if st.button('Run simulation'):
		stock_return_series = None
		bond_return_series = None
		sim_years = int(years)
		if seed_mode == 'Fixed seed':
			seed_value = int(random_seed_input)
		else:
			seed_value = np.random.default_rng().integers(0, 2**32 - 1)
		rng = np.random.default_rng(seed_value)
		taxable_return_series = sample_lognormal_returns(sim_years, float(taxable_log_drift), float(taxable_log_volatility), rng)
		roth_return_series = sample_lognormal_returns(sim_years, float(roth_log_drift), float(roth_log_volatility), rng)
		df = simulate_withdrawals(start_age_primary=int(start_age),
				start_age_spouse=int(start_age_spouse),
				years=sim_years,
				taxable_start=float(taxable_start),
				stock_total_return=float(stock_total_return),
				stock_dividend_yield=float(stock_dividend_yield),
				stock_turnover=float(stock_turnover),
				bond_return=float(bond_return),
				roth_start=float(roth_start),
				tda_start=float(tda_start),
				tda_spouse_start=float(tda_spouse_start),
				target_stock_pct=float(target_stock_pct),
				taxable_stock_basis_pct=float(taxable_stock_basis_pct),
				taxable_bond_basis_pct=float(taxable_bond_basis_pct),
				withdraw_amount=float(withdraw_amount),
				gross_up_withdrawals=bool(gross_up),
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
				stock_return_series=taxable_return_series,
				bond_return_series=taxable_return_series,
				roth_return_series=roth_return_series,
				life_expectancy_primary=int(life_expectancy_primary),
				life_expectancy_spouse=int(life_expectancy_spouse))

	currency_fmt_mc = f'${{:,.{int(display_decimals)}f}}'
	if st.button('Run Monte Carlo'):
		with st.spinner('Running Monte Carlo simulations...'):
			mc_results = run_monte_carlo(
				num_runs=int(monte_carlo_runs),
				withdraw_amount=float(withdraw_amount),
				seed=seed_for_mc,
				**common_mc_kwargs
			)
			if mc_results:
				mc_df = pd.DataFrame(mc_results)
				percentiles = [0, 10, 25, 50, 75, 90]
				cols = ['after_tax_end', 'total_taxes', 'effective_tax_rate', 'portfolio_cagr', 'roth_cagr']
				rows = []
				for p in percentiles:
					row = {'percentile': p}
					for col in cols:
						row[col] = np.percentile(mc_df[col], p)
					rows.append(row)
				st.session_state['mc_percentiles'] = rows
		st.session_state['mc_currency_fmt'] = f'${{:,.{int(display_decimals)}f}}'
		st.session_state['mc_pct_non_positive_end'] = (mc_df['after_tax_end'] <= 0).mean() if len(mc_df) > 0 else 0.0

	rows = st.session_state.get('mc_percentiles')
	pct_non_positive = st.session_state.get('mc_pct_non_positive_end', 0.0)
	if rows:
		percentile_df = pd.DataFrame(rows)
		currency_fmt_mc = st.session_state.get('mc_currency_fmt', currency_fmt_mc)
		st.subheader('Monte Carlo percentiles')
		st.dataframe(percentile_df.style.format({
		'percentile': lambda x: f"{int(x)}th" if pd.notnull(x) else '',
		'after_tax_end': currency_fmt_mc,
		'total_taxes': currency_fmt_mc,
		'effective_tax_rate': '{:.2%}'.format,
		'portfolio_cagr': '{:.2%}'.format,
		'roth_cagr': '{:.2%}'.format
		}))
	st.caption(f"Percent of ending values ≤0: {pct_non_positive * 100:.1f}%")

	mc_name = st.text_input('Monte Carlo scenario name', value=st.session_state.get('mc_name', 'MC Scenario 1'))
	st.session_state['mc_name'] = mc_name
	if rows and st.button('Save Monte Carlo scenario'):
		median_row = next((row for row in rows if row['percentile'] == 50), rows[2] if len(rows) > 2 else rows[-1])
		summary = {
		'name': mc_name,
		'after_tax_end': median_row['after_tax_end'],
		'total_taxes': median_row['total_taxes'],
		'effective_tax_rate': median_row['effective_tax_rate'],
		'portfolio_cagr': median_row['portfolio_cagr'],
		'roth_cagr': median_row['roth_cagr'],
		'percentile': median_row['percentile'],
		'percentiles': rows,
		'pct_non_positive_end': pct_non_positive,
		'description': (
				f"Convert ${roth_conversion_amount:,.0f} per year for {int(roth_conversion_years)} "
				f"years, taxes from {roth_conversion_tax_source}"
		),
		}
		st.session_state['mc_summaries'] = [s for s in st.session_state.get('mc_summaries', []) if s['name'] != mc_name]
		st.session_state['mc_summaries'].append(summary)
		st.success(f"Saved Monte Carlo scenario '{mc_name}'")

	st.subheader('Withdrawal finder')
	finder_target_success_pct = st.number_input('Target success rate (%)', value=95, min_value=50, max_value=99, step=1)
	finder_runs_input = st.number_input('Finder simulations', min_value=100, max_value=5000, value=1000, step=100)
	finder_tolerance = st.number_input('Finder tolerance ($)', min_value=10.0, value=1000.0, step=50.0, format="%.2f")
	finder_iterations = st.number_input('Finder iterations', min_value=5, max_value=25, value=12, step=1)
	if st.button('Find withdrawal for target success'):
		with st.spinner('Searching for the withdrawal amount...'):
			found_withdrawal, found_success, finder_percentiles = find_withdrawal_for_success(
				initial_total=initial_total,
				target_success=finder_target_success_pct / 100.0,
				finder_runs=int(finder_runs_input),
				tolerance=float(finder_tolerance),
				max_iterations=int(finder_iterations),
				seed=seed_for_mc,
				mc_kwargs=common_mc_kwargs)
			st.session_state['withdrawal_finder'] = {
				'withdrawal': found_withdrawal,
				'success_rate': found_success,
				'percentiles': finder_percentiles,
				'runs': int(finder_runs_input),
				'target_success_pct': finder_target_success_pct,
			}
	finder_result = st.session_state.get('withdrawal_finder')
	if finder_result:
		st.metric('Withdrawal for target success', f"${finder_result['withdrawal']:,.0f}")
		st.caption(
			f"{finder_result['success_rate'] * 100:.1f}% success over {finder_result['runs']} sims "
			f"(target {finder_result['target_success_pct']}%).")
		if finder_result['percentiles']:
			st.dataframe(pd.DataFrame(finder_result['percentiles']).style.format({
				'percentile': lambda x: f"{int(x)}th",
				'after_tax_end': '{:,.0f}'.format,
				'total_taxes': '{:,.0f}'.format,
				'effective_tax_rate': '{:.2%}'.format,
				'portfolio_cagr': '{:.2%}'.format,
				'roth_cagr': '{:.2%}'.format
			}))

	st.markdown('---')
	st.subheader('Historical audit')
	hist_source = st.selectbox('Historical factor source', list(HISTORICAL_SOURCES.keys()))
	hist_schedule = load_historical_schedule(hist_source)
	available_years = len(hist_schedule) // 12
	if available_years == 0:
		st.warning('No full 12-month historical periods available for this source.')
		hist_years = 0
		max_start_row = 0
	else:
		hist_years = st.number_input(
		'Historical horizon (years)',
		min_value=1,
		max_value=min(30, available_years),
		value=min(30, available_years),
		step=1,
		key='hist_horizon'
		)
		max_start_row = max(0, len(hist_schedule) - hist_years * 12)
	hist_start_row = st.number_input(
		'Historical start row index',
		min_value=0,
		max_value=max_start_row,
		value=0,
		step=1,
		key='hist_start_row'
	)
	info = HISTORICAL_SOURCES[hist_source]
	if hist_years > 0 and st.button('Run historical audit'):
		portfolio_col = choose_bucket_column(info, float(target_stock_pct))
		roth_col = info['roth']
		stock_series = build_historical_returns(hist_schedule, hist_start_row, hist_years, portfolio_col)
		roth_series = build_historical_returns(hist_schedule, hist_start_row, hist_years, roth_col)
		df_run = simulate_withdrawals(
		start_age_primary=int(start_age),
		start_age_spouse=int(start_age_spouse),
		years=int(hist_years),
		taxable_start=float(taxable_start),
		stock_total_return=float(stock_total_return),
		stock_dividend_yield=float(stock_dividend_yield),
		stock_turnover=float(stock_turnover),
		bond_return=float(bond_return),
		roth_start=float(roth_start),
		tda_start=float(tda_start),
		tda_spouse_start=float(tda_spouse_start),
		withdraw_amount=float(withdraw_amount),
		target_stock_pct=float(target_stock_pct),
		taxable_stock_basis_pct=float(taxable_stock_basis_pct),
		taxable_bond_basis_pct=float(taxable_bond_basis_pct),
		gross_up_withdrawals=bool(gross_up),
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
		stock_return_series=stock_series,
		bond_return_series=stock_series,
		roth_return_series=roth_series,
		life_expectancy_primary=int(life_expectancy_primary),
		life_expectancy_spouse=int(life_expectancy_spouse))
		st.session_state['historical_audit_results'] = df_run
		st.session_state['historical_audit_period'] = {
		'source': hist_source,
		'start_row': hist_start_row,
		'years': hist_years,
		'period_start': hist_schedule.iloc[hist_start_row]['begin month']
		}
	if 'historical_audit_results' in st.session_state:
		df_hist = st.session_state['historical_audit_results']
		meta = st.session_state.get('historical_audit_period', {})
		st.write(meta)
		st.dataframe(df_hist.style.format({
		'portfolio_return': '{:.2%}'.format,
		'roth_return_used': '{:.2%}'.format
		}))

	if hist_years > 0 and st.button('Run historical sweep'):
		sweep_results: List[Dict[str, Any]] = []
		if len(hist_schedule) >= hist_years * 12:
			for row in range(0, max_start_row + 1):
				if row + hist_years * 12 > len(hist_schedule):
					break
				stock_series = build_historical_returns(hist_schedule, row, hist_years, choose_bucket_column(info, float(target_stock_pct)))
				roth_series = build_historical_returns(hist_schedule, row, hist_years, info['roth'])
				df_run = simulate_withdrawals(
					start_age_primary=int(start_age),
					start_age_spouse=int(start_age_spouse),
					years=int(hist_years),
					taxable_start=float(taxable_start),
					stock_total_return=float(stock_total_return),
					stock_dividend_yield=float(stock_dividend_yield),
					stock_turnover=float(stock_turnover),
					bond_return=float(bond_return),
					roth_start=float(roth_start),
					tda_start=float(tda_start),
					tda_spouse_start=float(tda_spouse_start),
					withdraw_amount=float(withdraw_amount),
					target_stock_pct=float(target_stock_pct),
					taxable_stock_basis_pct=float(taxable_stock_basis_pct),
					taxable_bond_basis_pct=float(taxable_bond_basis_pct),
					gross_up_withdrawals=bool(gross_up),
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
					stock_return_series=stock_series,
					bond_return_series=stock_series,
					roth_return_series=roth_series,
					life_expectancy_primary=int(life_expectancy_primary),
					life_expectancy_spouse=int(life_expectancy_spouse))
				metrics = compute_summary_metrics(df_run, inheritor_marginal_rate)
				metrics.update({'start_row': row})
				sweep_results.append(metrics)
		if sweep_results:
			percentile_rows = []
			sweep_df = pd.DataFrame(sweep_results)
			for p in [0, 10, 25, 50, 75, 90]:
				row = {'percentile': p}
				for col in ['after_tax_end', 'total_taxes', 'effective_tax_rate', 'portfolio_cagr', 'roth_cagr']:
					row[col] = np.percentile(sweep_df[col], p)
				percentile_rows.append(row)
			st.session_state['historical_sweep'] = percentile_rows
	if 'historical_sweep' in st.session_state:
		sweep_df = pd.DataFrame(st.session_state['historical_sweep'])
		st.subheader('Historical sweep percentiles')
		st.dataframe(sweep_df.style.format({
		'percentile': lambda x: f"{int(x)}th",
		'after_tax_end': '{:,.0f}'.format,
		'total_taxes': '{:,.0f}'.format,
		'effective_tax_rate': '{:.2%}'.format,
		'portfolio_cagr': '{:.2%}'.format,
		'roth_cagr': '{:.2%}'.format
		}))

	if 'mc_summaries' not in st.session_state:
		st.session_state['mc_summaries'] = []

	if st.session_state['mc_summaries']:
		summary_rows = []
		for summary in st.session_state['mc_summaries']:
			summary_rows.append({
				'name': summary['name'],
				'after_tax_end': summary['after_tax_end'],
				'total_taxes': summary['total_taxes'],
				'effective_tax_rate': summary['effective_tax_rate'],
				'portfolio_cagr': summary['portfolio_cagr'],
				'roth_cagr': summary['roth_cagr'],
				'percentile': summary.get('percentile', 50),
				'pct_non_positive_end': summary.get('pct_non_positive_end', 0.0),
				'description': summary.get('description', '')
			})
		summary_df = pd.DataFrame(summary_rows).set_index('name')
		st.subheader('Saved Monte Carlo scenarios')
		st.dataframe(summary_df.style.format({
			'after_tax_end': currency_fmt_mc,
			'total_taxes': currency_fmt_mc,
			'effective_tax_rate': '{:.2%}'.format,
			'portfolio_cagr': '{:.2%}'.format,
			'roth_cagr': '{:.2%}'.format,
			'percentile': lambda x: f"{int(x)}th",
			'pct_non_positive_end': '{:.1%}'.format,
			'description': lambda x: x
		}))
		pct_style = {
		'percentile': lambda x: f"{int(x)}th" if pd.notnull(x) else '',
		'after_tax_end': currency_fmt_mc,
		'total_taxes': currency_fmt_mc,
		'effective_tax_rate': '{:.2%}'.format,
		'portfolio_cagr': '{:.2%}'.format,
		'roth_cagr': '{:.2%}'.format
		}
		for summary in st.session_state['mc_summaries']:
			desc = summary.get('description')
			header = f"**{summary['name']} percentile table**"
			if desc:
				header += f" — {desc}"
			st.markdown(header)
			percentile_table = pd.DataFrame(summary.get('percentiles', []))
			if not percentile_table.empty:
				st.dataframe(percentile_table.style.format(pct_style))
			else:
				st.write('No percentile data saved for this scenario.')


	st.markdown('---')
	st.subheader('Closed-loop Monte Carlo withdrawals')
	hist_source_closed = st.selectbox('Historical source for the closed-loop path', list(HISTORICAL_SOURCES.keys()), key='closed_loop_source')
	max_closed_loop_start = st.session_state.get('closed_loop_hist_max', 0)
	if max_closed_loop_start == 0:
		st.caption('Historical data will load when you press "Run closed-loop Monte Carlo".')
	closed_loop_start_row = st.number_input(
		'Historical start row index for closed-loop path',
		min_value=0,
		max_value=max_closed_loop_start,
		value=st.session_state.get('closed_loop_hist_start', 0),
		step=1,
		key='closed_loop_hist_start'
	)
	closed_loop_runs = st.number_input('Closed-loop Monte Carlo runs per iteration', min_value=100, max_value=5000, value=100, step=100)
	closed_loop_tolerance = st.number_input('Withdrawal tolerance ($)', min_value=10.0, value=500.0, step=50.0, format="%.2f")
	closed_loop_iterations = st.number_input('Binary search iterations for adjustments', min_value=5, max_value=25, value=5, step=1)
	spending_cap_pct = st.slider('Max spending cap over goal (%)', min_value=0, max_value=100, value=25, step=5)
	spending_cap_multiplier = 1.0 + spending_cap_pct / 100.0
	if st.button('Run closed-loop Monte Carlo'):
		with st.spinner('Running dynamic withdrawal strategy...'):
			if 'closed_loop_schedule' not in st.session_state or st.session_state.get('closed_loop_schedule_source') != hist_source_closed:
				hist_schedule = load_historical_schedule(hist_source_closed)
				st.session_state['closed_loop_schedule'] = hist_schedule
				st.session_state['closed_loop_schedule_source'] = hist_source_closed
				st.session_state['closed_loop_hist_max'] = max(0, len(hist_schedule) - int(years) * 12)
			else:
				hist_schedule = st.session_state['closed_loop_schedule']
			hist_info = HISTORICAL_SOURCES[hist_source_closed]
			hist_bucket_column = choose_bucket_column(hist_info, float(target_stock_pct))
			hist_roth_column = hist_info['roth'].strip()
			if len(hist_schedule) < years * 12:
				st.warning('Not enough historical months to cover the full horizon; the path will stop early.')
			closed_loop_result = run_closed_loop_monte_carlo(
				initial_withdrawal=float(withdraw_amount),
				total_years=int(years),
				start_age=int(start_age),
				start_age_spouse=int(start_age_spouse),
				simulation_constants=simulation_constants,
				base_mc_kwargs=dict(common_mc_kwargs),
				mc_runs=int(closed_loop_runs),
				tolerance=float(closed_loop_tolerance),
				max_iterations=int(closed_loop_iterations),
				taxable_log_drift=float(taxable_log_drift),
				taxable_log_volatility=float(taxable_log_volatility),
				roth_log_drift=float(roth_log_drift),
				roth_log_volatility=float(roth_log_volatility),
				seed_base=seed_for_mc,
				hist_schedule=hist_schedule,
				hist_bucket_column=hist_bucket_column,
				hist_roth_column=hist_roth_column,
				hist_start_row=int(closed_loop_start_row),
				spending_cap_multiplier=float(spending_cap_multiplier)
			)
			st.session_state['closed_loop_results'] = closed_loop_result

	if 'closed_loop_results' in st.session_state:
		cl_res = st.session_state['closed_loop_results']
		if cl_res.get('records'):
			st.metric('Final ending total', f"${cl_res['final_total']:,.0f}")
			st.metric('Final withdrawal', f"${cl_res['final_withdrawal']:,.0f}")
			st.caption(f"Last evaluated success rate: {cl_res['last_success_rate']:.1%}")
			closed_df = pd.DataFrame(cl_res['records']).set_index('year')
			st.dataframe(closed_df.style.format({
				'withdrawal': '{:,.0f}'.format,
				'success_rate': '{:.1%}'.format,
				'adjustment': lambda x: x,
				'hist_row': '{:,.0f}'.format,
				'hist_period_start': lambda x: x.strftime('%Y-%m') if not pd.isna(x) else '',
				'start_total': '{:,.0f}'.format,
				'end_total': '{:,.0f}'.format,
				'portfolio_return': '{:.2%}'.format,
				'roth_return': '{:.2%}'.format,
				'total_taxes': '{:,.0f}'.format
			}))

	if df is not None:
		last = df.iloc[-1]
		currency_fmt = f'${{:,.{int(display_decimals)}f}}'
		display_df = df.round(int(display_decimals)).copy()
		display_df['portfolio_return'] = df['portfolio_return']
		display_df['roth_return_used'] = df['roth_return_used']

		years_simulated = len(df)
		portfolio_growth_factor = (df['portfolio_return'] + 1.0).prod()
		roth_growth_factor = (df['roth_return_used'] + 1.0).prod()
		portfolio_cagr = (portfolio_growth_factor ** (1.0 / years_simulated) - 1.0) if years_simulated > 0 else 0.0
		roth_cagr = (roth_growth_factor ** (1.0 / years_simulated) - 1.0) if years_simulated > 0 else 0.0
		c1, c2 = st.columns(2)
		c1.metric('Portfolio CAGR', f"{portfolio_cagr:.2%}")
		c2.metric('Roth CAGR', f"{roth_cagr:.2%}")

		st.subheader('Year-by-year table')
		st.caption(f'Years simulated: {len(df)}')
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
				'end_accounts_total': currency_fmt,
				'ordinary_tax_total': currency_fmt, 'capital_gains_tax': currency_fmt, 'niit_tax': currency_fmt, 'total_taxes': currency_fmt,
				'marginal_ordinary_rate': '{:.2%}'.format, 'marginal_cap_gains_rate': '{:.2%}'.format,
				'portfolio_return': '{:.2%}'.format,
				'roth_return_used': '{:.2%}'.format
		}))

		currency_round = df.round(int(display_decimals))
		st.subheader('Where withdrawals came from (stacked)')
		chart_df = currency_round[['year','withdraw_from_taxable_net','withdraw_from_tda','withdraw_from_roth']].set_index('year')
		st.area_chart(chart_df)

		st.subheader('Account balances over time')
		bal_df = currency_round[['year','end_taxable_total','end_tda_total','end_roth','end_accounts_total']].set_index('year')
		st.line_chart(bal_df)

		st.subheader('Taxes paid per year')
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
		st.subheader('Effective and marginal tax rates')
		st.dataframe(mtr_df.style.format({
		'taxable_income_total': currency_fmt,
		'total_taxes': currency_fmt,
		'effective_tax_rate': '{:.2%}'.format,
		'marginal_ordinary_rate': '{:.2%}'.format,
		'marginal_cap_gains_rate': '{:.2%}'.format,
		'niit_tax': currency_fmt,
		}))
		st.line_chart(mtr_df[['effective_tax_rate','marginal_ordinary_rate','marginal_cap_gains_rate']])
		lifetime_taxes = df['total_taxes'].sum()
		lifetime_ordinary_tax = df['ordinary_tax_total'].sum()
		lifetime_cap_gains_tax = df['capital_gains_tax'].sum()
		st.metric('Total lifetime taxes paid', currency_fmt.format(lifetime_taxes))
		st.caption(f'Ordinary: {currency_fmt.format(lifetime_ordinary_tax)} | Capital gains/QD: {currency_fmt.format(lifetime_cap_gains_tax)}')

		# Store latest summary in session for post-run saving
		last = df.iloc[-1]
		total_accounts = last['end_stocks_mv'] + last['end_bonds_mv'] + last['end_tda_total'] + last['end_roth']
		after_tax_end = float(last['end_stocks_mv'] + last['end_bonds_mv'])
		after_tax_end += float(last['end_roth'])
		after_tax_end += float(last['end_tda_total']) * max(0.0, 1.0 - inheritor_marginal_rate)
		st.session_state['last_summary'] = {
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
		}

		st.markdown('---')
		st.write('Ending balances')
		last_row = currency_round.iloc[-1]
		taxable_total_end = last_row['end_stocks_mv'] + last_row['end_bonds_mv']
		st.write({'taxable_end': taxable_total_end, 'stocks_end': last_row['end_stocks_mv'], 'stocks_end_basis': last_row['end_stocks_basis'], 'bonds_end': last_row['end_bonds_mv'], 'bonds_end_basis': last_row['end_bonds_basis'], 'tda_end': last_row['end_tda_total'], 'roth_end': last_row['end_roth']})
	else:
		st.info('Set inputs and click "Run simulation" to see results.')

	# Scenario saving/comparison (up to 5), available after a run
	scenario_name_default = st.session_state.get('last_summary', {}).get('name', 'Scenario 1')
	scenario_name = st.text_input('Scenario name to save', value=scenario_name_default)
	if 'scenario_summaries' not in st.session_state:
		st.session_state['scenario_summaries'] = []

	if st.button('Save latest run as scenario (max 5)'):
		if 'last_summary' not in st.session_state:
			st.warning('Run a simulation first.')
		else:
			summary = dict(st.session_state['last_summary'])
			summary['name'] = scenario_name
			# replace if same name exists
			st.session_state['scenario_summaries'] = [s for s in st.session_state['scenario_summaries'] if s['name'] != scenario_name]
			if len(st.session_state['scenario_summaries']) >= 5:
				st.warning('Maximum of 5 scenarios saved. Remove one by reusing a name.')
			else:
				st.session_state['scenario_summaries'].append(summary)
				st.success(f"Saved scenario '{scenario_name}'.")

	if st.session_state['scenario_summaries']:
		st.markdown('### Saved scenarios (up to 5)')
		compare_df = pd.DataFrame(st.session_state['scenario_summaries']).set_index('name')
		st.dataframe(compare_df.style.format({
		'total_accounts': currency_fmt if 'currency_fmt' in locals() else '${:,.0f}',
		'total_taxes': currency_fmt if 'currency_fmt' in locals() else '${:,.0f}',
		'taxable_end': currency_fmt if 'currency_fmt' in locals() else '${:,.0f}',
		'tda_end': currency_fmt if 'currency_fmt' in locals() else '${:,.0f}',
		'roth_end': currency_fmt if 'currency_fmt' in locals() else '${:,.0f}',
		'after_tax_end': currency_fmt if 'currency_fmt' in locals() else '${:,.0f}',
		'inheritor_marginal_rate': '{:.2%}'.format,
		'portfolio_cagr': '{:.2%}'.format,
		'roth_cagr': '{:.2%}'.format
		}))


if __name__ == '__main__':
	main()
