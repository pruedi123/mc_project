import streamlit as st
import pandas as pd
import numpy as np

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
	"""Rebalance household portfolio (taxable + two TDAs + Roth) to target stock %, rounded to nearest 10% per account."""
	total_household = taxable_stock_mv + taxable_bond_mv + tda1_mv + tda2_mv + roth_mv
	if total_household <= 0:
		return (taxable_stock_mv, taxable_bond_mv, taxable_stock_basis, taxable_bond_basis,
				0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

	desired_stock = total_household * target_stock_pct
	remaining = desired_stock

	# Roth: fund stocks up to remaining desired
	roth_stock = min(roth_mv, remaining)
	remaining -= roth_stock
	roth_bond = roth_mv - roth_stock

	# Taxable: favor stocks
	taxable_total = taxable_stock_mv + taxable_bond_mv
	taxable_stock = min(taxable_total, remaining)
	remaining -= taxable_stock
	taxable_bond = taxable_total - taxable_stock

	# TDA 1: fill next
	tda1_stock = min(tda1_mv, remaining)
	remaining -= tda1_stock
	tda1_bond = tda1_mv - tda1_stock

	# TDA 2: remaining stock allocation
	tda2_stock = min(tda2_mv, remaining)
	remaining -= tda2_stock
	tda2_bond = tda2_mv - tda2_stock

	def round_account(stock, bond):
		total = stock + bond
		if total <= 0:
			return 0.0, 0.0
		frac = stock / total
		frac = round(frac * 10) / 10.0
		frac = min(1.0, max(0.0, frac))
		stock_new = total * frac
		bond_new = total - stock_new
		return stock_new, bond_new

	taxable_stock, taxable_bond = round_account(taxable_stock, taxable_bond)
	tda1_stock, tda1_bond = round_account(tda1_stock, tda1_bond)
	tda2_stock, tda2_bond = round_account(tda2_stock, tda2_bond)
	roth_stock, roth_bond = round_account(roth_stock, roth_bond)

	# adjust taxable basis proportionally
	taxable_basis_total = taxable_stock_basis + taxable_bond_basis
	if taxable_stock + taxable_bond > 0 and taxable_basis_total > 0:
		taxable_stock_basis = taxable_basis_total * (taxable_stock / (taxable_stock + taxable_bond))
		taxable_bond_basis = taxable_basis_total - taxable_stock_basis
	else:
		taxable_stock_basis = taxable_stock_basis
		taxable_bond_basis = taxable_bond_basis

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
						 roth_conversion_years: int = 0):
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

		# apply Roth conversion at start of year before growth
		conversion_gross = min(roth_conversion_amount, tda1_stocks_mv + tda1_bonds_mv) if y <= roth_conversion_years else 0.0
		if conversion_gross > 0:
			total_tda_balance = tda1_stocks_mv + tda1_bonds_mv
			tda_stock_ratio = (tda1_stocks_mv / total_tda_balance) if total_tda_balance > 0 else 0.5
			tda1_stocks_mv -= conversion_gross * tda_stock_ratio
			tda1_bonds_mv -= conversion_gross * (1 - tda_stock_ratio)
			# temporarily park converted amount; taxes handled after computing tax delta
			pending_roth_conversion = conversion_gross
		else:
			pending_roth_conversion = 0.0

		# grow TDA (per user mix) and Roth (per user mix)
		tda1_stocks_mv *= (1 + stock_total_return)
		tda1_bonds_mv *= (1 + bond_return)
		tda2_stocks_mv *= (1 + stock_total_return)
		tda2_bonds_mv *= (1 + bond_return)
		roth_stocks_mv *= (1 + stock_total_return)
		roth_bonds_mv *= (1 + bond_return)

		# Stocks: split total return into price appreciation and dividend yield
		price_return = stock_total_return - stock_dividend_yield
		# apply price appreciation
		stocks_mv *= (1 + price_return)

		# dividends (qualified) - reinvest gross; tax computed later
		div = stocks_mv * stock_dividend_yield
		stocks_mv += div
		stocks_basis += div

		# Bonds: accrue interest, tax handled later
		interest = bonds_mv * bond_return
		bonds_mv += interest
		bonds_basis += interest

		# Stock turnover: track realized gains for tax; ignore drag in balances for now (assume immediate reinvest)
		turnover_sale = stocks_mv * stock_turnover
		turnover_basis_sold = stocks_basis * stock_turnover
		turnover_realized_gain = max(0.0, turnover_sale - turnover_basis_sold)
		# rebalance basis to reflect sale and repurchase at current market value
		stocks_mv = stocks_mv  # unchanged net of round trip
		stocks_basis = stocks_basis - turnover_basis_sold + turnover_sale

		# compute RMD for each spouse if applicable
		tda1_stocks_mv, tda1_bonds_mv, rmd_p1, divisor_p1 = process_rmd(tda1_stocks_mv, tda1_bonds_mv, age_p1, rmd_start_age, table)
		tda2_stocks_mv, tda2_bonds_mv, rmd_p2, divisor_p2 = process_rmd(tda2_stocks_mv, tda2_bonds_mv, age_p2, rmd_start_age_spouse, table)

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

		# Other income items (Social Security and pension grow with COLA)
		ss_income_p1 = ss_income_annual * ((1 + ss_cola) ** (y - 1))
		ss_income_p2 = ss_income_spouse_annual * ((1 + ss_cola) ** (y - 1))
		ss_income = ss_income_p1 + ss_income_p2
		pension_income_p1 = pension_income_annual * ((1 + pension_cola) ** (y - 1))
		pension_income_p2 = pension_income_spouse_annual * ((1 + pension_cola) ** (y - 1))
		pension_income = pension_income_p1 + pension_income_p2
		other_income = other_income_annual

		# Ordinary income before deductions (exclude cap gains/qualified dividends for stacking)
		ordinary_income_pre_ss_base = interest + withdraw_from_tda + pension_income + other_income
		ordinary_income_pre_ss_with_conv = ordinary_income_pre_ss_base + pending_roth_conversion
		cap_gains_total = div + turnover_realized_gain + realized_gains_from_sales

		deduction = itemized_deduction_amount if use_itemized_deductions else get_standard_deduction(filing_status)

		def compute_tax_bundle(ordinary_income_pre_ss_val):
			taxable_ss_val = compute_taxable_social_security(ss_income, ordinary_income_pre_ss_val, cap_gains_total, filing_status)
			ordinary_income_total_val = ordinary_income_pre_ss_val + taxable_ss_val
			taxable_ordinary_val = max(0.0, ordinary_income_total_val - deduction)
			ordinary_tax_val = apply_brackets(taxable_ordinary_val, get_ordinary_brackets(filing_status))
			cap_gains_tax_val = compute_capital_gains_tax(taxable_ordinary_val, cap_gains_total, filing_status)
			total_val = ordinary_tax_val + cap_gains_tax_val
			return taxable_ss_val, taxable_ordinary_val, ordinary_tax_val, cap_gains_tax_val, total_val

		taxable_ss, taxable_ordinary, ordinary_tax_total, cap_gains_tax, total_taxes = compute_tax_bundle(ordinary_income_pre_ss_with_conv)
		_, _, _, _, total_tax_without_conv = compute_tax_bundle(ordinary_income_pre_ss_base)
		roth_conversion_tax_delta = max(0.0, total_taxes - total_tax_without_conv)
		# NIIT approximation
		niit_threshold = 200000 if filing_status == 'single' else 250000
		agi_approx = ordinary_income_pre_ss_with_conv + taxable_ss + cap_gains_total
		niit_base = max(0.0, agi_approx - niit_threshold)
		net_investment_income = max(0.0, cap_gains_total + interest + div)
		niit_tax = 0.038 * min(niit_base, net_investment_income)
		total_taxes += niit_tax
		marginal_ordinary_rate, marginal_cg_rate = get_marginal_rates(taxable_ordinary, cap_gains_total, filing_status)
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
			'end_tda_p1': tda1_stocks_mv + tda1_bonds_mv,
			'end_tda_p2': tda2_stocks_mv + tda2_bonds_mv,
			'end_tda_total': (tda1_stocks_mv + tda1_bonds_mv + tda2_stocks_mv + tda2_bonds_mv),
			'end_roth': roth_stocks_mv + roth_bonds_mv,
		})

	df = pd.DataFrame(rows)
	return df

def main():
	st.title('Withdrawal + RMD Simulator (30-year)')

	with st.sidebar:
		st.header('Inputs')
		start_age = st.number_input('Starting age (person 1)', min_value=18, max_value=120, value=65)
		start_age_spouse = st.number_input('Starting age (person 2)', min_value=18, max_value=120, value=60)
		years = st.number_input('Years to simulate', min_value=1, max_value=100, value=30)
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
		rmd_start_age = st.number_input('RMD start age (person 1)', min_value=65, max_value=90, value=73)
		rmd_start_age_spouse = st.number_input('RMD start age (person 2)', min_value=65, max_value=90, value=73)

		st.markdown('Other income (all taxed as ordinary for now)')
		ss_income_input = st.number_input('Annual Social Security - person 1 (current year)', value=40000.0, step=1000.0)
		ss_income_spouse_input = st.number_input('Annual Social Security - person 2 (current year)', value=0.0, step=1000.0)
		ss_cola = st.number_input('Social Security COLA', value=0.02, format="%.4f")
		pension_income_input = st.number_input('Annual pension income - person 1', value=24000.0, step=1000.0)
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

		st.markdown('Expected annual returns and taxable details')
		stock_total_return = st.number_input('Stock total return', value=0.10, format="%.4f")
		stock_dividend_yield = st.number_input('Stock dividend (qualified) yield', value=0.02, format="%.4f")
		stock_turnover = st.number_input('Stock turnover rate', value=0.10, format="%.4f")
		bond_return = st.number_input('Bond interest yield', value=0.04, format="%.4f")
		st.info('You can set stock/bond splits separately for Roth and tax-deferred in the balances section.')

		gross_up = st.checkbox('Gross-up taxable sales to deliver requested net withdrawal (recommended)', value=True)
		display_decimals = st.number_input('Decimal places for tables/charts', min_value=0, max_value=6, value=0, step=1)

	df = None
	if st.button('Run simulation'):
		df = simulate_withdrawals(start_age_primary=int(start_age),
								  start_age_spouse=int(start_age_spouse),
								  years=int(years),
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
								  roth_conversion_tax_source='taxable' if roth_conversion_tax_source == 'Taxable' else 'tda')

		currency_fmt = f'${{:,.{int(display_decimals)}f}}'

		st.subheader('Year-by-year table')
		st.dataframe(df.round(int(display_decimals)).style.format({
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
			'end_taxable_total': currency_fmt,
			'end_tda_p1': currency_fmt, 'end_tda_p2': currency_fmt, 'end_tda_total': currency_fmt, 'end_roth': currency_fmt,
			'ordinary_tax_total': currency_fmt, 'capital_gains_tax': currency_fmt, 'niit_tax': currency_fmt, 'total_taxes': currency_fmt,
			'marginal_ordinary_rate': '{:.2%}'.format, 'marginal_cap_gains_rate': '{:.2%}'.format
		}))

	if df is not None:
		currency_round = df.round(int(display_decimals))

		st.subheader('Where withdrawals came from (stacked)')
		chart_df = currency_round[['year','withdraw_from_taxable_net','withdraw_from_tda','withdraw_from_roth']].set_index('year')
		st.area_chart(chart_df)

		st.subheader('Account balances over time')
		bal_df = currency_round[['year','end_taxable_total','end_tda_total','end_roth']].set_index('year')
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
		st.session_state['last_summary'] = {
			'label': f"conversion ${roth_conversion_amount:,.0f} for {roth_conversion_years} yrs, taxes from {roth_conversion_tax_source}",
			'total_taxes': float(lifetime_taxes),
			'total_accounts': float(total_accounts),
			'taxable_end': float(last['end_stocks_mv'] + last['end_bonds_mv']),
			'tda_end': float(last['end_tda_total']),
			'roth_end': float(last['end_roth']),
		}

		st.markdown('---')
		st.write('Ending balances')
		last = currency_round.iloc[-1]
		taxable_total_end = last['end_stocks_mv'] + last['end_bonds_mv']
		st.write({'taxable_end': taxable_total_end, 'stocks_end': last['end_stocks_mv'], 'stocks_end_basis': last['end_stocks_basis'], 'bonds_end': last['end_bonds_mv'], 'bonds_end_basis': last['end_bonds_basis'], 'tda_end': last['end_tda_total'], 'roth_end': last['end_roth']})
	else:
		st.info('Set inputs and click \"Run simulation\" to see results.')

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
			'roth_end': currency_fmt if 'currency_fmt' in locals() else '${:,.0f}'
		}))


if __name__ == '__main__':
	main()
