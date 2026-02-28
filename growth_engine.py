"""Dollar growth distribution — pure investment compounding, no withdrawals/taxes.

No Streamlit dependency.  Reuses historical windows and lognormal draws from sim_engine.
"""

import numpy as np
from sim_engine import get_all_historical_windows, sample_lognormal_returns


def dollar_growth_distribution(
	stock_pct: float,
	years: int,
	mode: str = 'historical',
	stock_drift: float = 0.0,
	stock_vol: float = 0.0,
	bond_drift: float = 0.0,
	bond_vol: float = 0.0,
	num_runs: int = 1000,
	fee_bps: float = 0.0,
) -> dict:
	"""Compound $1 over *years* at a given stock/bond allocation across many paths.

	Parameters
	----------
	stock_pct : float  – equity weight (e.g. 0.60)
	years     : int    – plan horizon
	mode      : str    – 'historical' or 'simulated'
	fee_bps   : float  – annual advisory fee in basis points (applied after returns)

	Returns
	-------
	dict with keys: ending_values, percentiles, cagrs, median_cagr, num_runs
	"""
	annual_fee = fee_bps / 10_000.0
	bond_pct = 1.0 - stock_pct

	if mode == 'historical':
		windows, _ = get_all_historical_windows(years)
		n = len(windows)
		ending = np.empty(n)
		for i, (stock_rets, bond_rets) in enumerate(windows):
			val = 1.0
			for y in range(years):
				val *= 1.0 + stock_pct * stock_rets[y] + bond_pct * bond_rets[y] - annual_fee
			ending[i] = val
	else:
		rng = np.random.default_rng()
		ending = np.empty(num_runs)
		for i in range(num_runs):
			s_rets = sample_lognormal_returns(years, stock_drift, stock_vol, rng)
			b_rets = sample_lognormal_returns(years, bond_drift, bond_vol, rng)
			val = 1.0
			for y in range(years):
				val *= 1.0 + stock_pct * s_rets[y] + bond_pct * b_rets[y] - annual_fee
			ending[i] = val

	cagrs = ending ** (1.0 / years) - 1.0
	pct_keys = [0, 10, 25, 50, 75, 90, 100]
	percentiles = {p: float(np.percentile(ending, p)) for p in pct_keys}

	return {
		'ending_values': ending,
		'percentiles': percentiles,
		'cagrs': cagrs,
		'median_cagr': float(np.median(cagrs)),
		'num_runs': len(ending),
	}


def dollar_growth_by_year(
	stock_pct: float,
	max_years: int,
	mode: str = 'historical',
	stock_drift: float = 0.0,
	stock_vol: float = 0.0,
	bond_drift: float = 0.0,
	bond_vol: float = 0.0,
	num_runs: int = 1000,
	fee_bps: float = 0.0,
) -> np.ndarray:
	"""Compound $1 and record its value at every year across many paths.

	Returns
	-------
	np.ndarray of shape (n_paths, max_years) where element [i, y] is the
	cumulative value of $1 at the end of year y+1 along path i.
	"""
	annual_fee = fee_bps / 10_000.0
	bond_pct = 1.0 - stock_pct

	if mode == 'historical':
		windows, _ = get_all_historical_windows(max_years)
		n = len(windows)
		grid = np.empty((n, max_years))
		for i, (stock_rets, bond_rets) in enumerate(windows):
			val = 1.0
			for y in range(max_years):
				val *= 1.0 + stock_pct * stock_rets[y] + bond_pct * bond_rets[y] - annual_fee
				grid[i, y] = val
	else:
		rng = np.random.default_rng()
		grid = np.empty((num_runs, max_years))
		for i in range(num_runs):
			s_rets = sample_lognormal_returns(max_years, stock_drift, stock_vol, rng)
			b_rets = sample_lognormal_returns(max_years, bond_drift, bond_vol, rng)
			val = 1.0
			for y in range(max_years):
				val *= 1.0 + stock_pct * s_rets[y] + bond_pct * b_rets[y] - annual_fee
				grid[i, y] = val

	return grid
