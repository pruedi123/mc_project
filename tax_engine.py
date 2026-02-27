"""Pure tax math functions — no Streamlit dependency."""


def get_standard_deduction(filing_status: str, tax_law: str = 'tcja') -> float:
	if tax_law == 'pre_tcja':
		# Pre-TCJA: standard deduction + personal exemptions (2026 projected)
		# Single: $8,300 std + $5,300 exemption = $13,600
		# MFJ: $16,600 std + 2 × $5,300 exemptions = $27,200
		return 13600 if filing_status == 'single' else 27200
	# 2024 TCJA standard deductions
	return 14600 if filing_status == 'single' else 29200

def get_ordinary_brackets(filing_status: str, tax_law: str = 'tcja'):
	if tax_law == 'pre_tcja':
		# Pre-TCJA brackets (2026 projected, inflation-adjusted from 2017 law)
		if filing_status == 'single':
			return [
				(0, 0.10),
				(11925, 0.15),
				(48475, 0.25),
				(117150, 0.28),
				(244725, 0.33),
				(531200, 0.35),
				(533400, 0.396),
			]
		else:
			return [
				(0, 0.10),
				(23850, 0.15),
				(96950, 0.25),
				(198450, 0.28),
				(303350, 0.33),
				(531200, 0.35),
				(601050, 0.396),
			]
	# 2024 TCJA ordinary income brackets (taxable income after deductions)
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

def get_capital_gains_brackets(filing_status: str, tax_law: str = 'tcja'):
	if tax_law == 'pre_tcja':
		# Pre-TCJA: 0% up to top of 15% bracket, 15% up to top of 35%, 20% above
		if filing_status == 'single':
			return [
				(0, 0.00),
				(48475, 0.15),
				(533400, 0.20),
			]
		else:
			return [
				(0, 0.00),
				(96950, 0.15),
				(601050, 0.20),
			]
	# 2024 TCJA LTCG/QD thresholds (stacked on top of ordinary taxable income)
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

def compute_capital_gains_tax(ordinary_taxable: float, cap_gains: float, filing_status: str, tax_law: str = 'tcja') -> float:
	if cap_gains <= 0:
		return 0.0
	brackets = get_capital_gains_brackets(filing_status, tax_law)
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

def get_marginal_rates(taxable_ordinary: float, cap_gains: float, filing_status: str, tax_law: str = 'tcja'):
	# marginal ordinary = bracket rate for next ordinary dollar
	ordinary_brackets = get_ordinary_brackets(filing_status, tax_law)
	marginal_ordinary = 0.0
	for start, rate in ordinary_brackets:
		if taxable_ordinary >= start:
			marginal_ordinary = rate
		else:
			break

	# marginal cap gains rate given stacking (use top rate hit in cap gains bands)
	cg_brackets = get_capital_gains_brackets(filing_status, tax_law)
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

def bracket_ceiling(filing_status: str, target_rate: float, tax_law: str = 'tcja') -> float:
	"""Return the taxable income where the bracket at `target_rate` ends.

	E.g. for MFJ 22% → 201050 (income above this enters the 24% bracket).
	For the top rate returns float('inf').
	If the target rate doesn't exist in the current tax law (e.g. 22% under
	pre-TCJA), returns 0.0 so that bracket-fill conversions are skipped.
	"""
	brackets = get_ordinary_brackets(filing_status, tax_law)
	for i, (start, rate) in enumerate(brackets):
		if abs(rate - target_rate) < 1e-9:
			if i + 1 < len(brackets):
				return float(brackets[i + 1][0])
			return float('inf')
	return 0.0


def compute_niit(agi: float, net_investment_income: float, filing_status: str) -> float:
	"""Compute Net Investment Income Tax (3.8% surtax)."""
	niit_threshold = 200000 if filing_status == 'single' else 250000
	niit_base_val = max(0.0, agi - niit_threshold)
	return 0.038 * min(niit_base_val, net_investment_income)

def compute_state_tax(taxable_ordinary: float, cap_gains: float, interest: float,
					  state_rate: float, exempt_retirement: bool) -> float:
	"""Compute flat state income tax. If exempt_retirement, only investment income is taxed."""
	if state_rate <= 0:
		return 0.0
	if exempt_retirement:
		state_taxable = max(0.0, interest + cap_gains)
	else:
		state_taxable = taxable_ordinary + cap_gains
	return state_taxable * state_rate

def compute_irmaa(magi: float, filing_status: str, num_medicare_persons: int) -> float:
	"""Compute annual IRMAA surcharges (Medicare Part B + Part D) based on MAGI.

	Each Medicare-eligible person pays their own surcharge; for MFJ both use
	the joint MAGI threshold.  Returns total annual household surcharge.
	2024 brackets (treated as real dollars, consistent with project convention).
	"""
	if num_medicare_persons <= 0:
		return 0.0
	# (upper_threshold, annual_surcharge_per_person)
	# surcharge = (Part B + Part D monthly surcharge) × 12
	if filing_status == 'single':
		tiers = [
			(103000, 0),
			(129000, 994),
			(161000, 2496),
			(193000, 4003),
			(500000, 5510),
		]
	else:  # MFJ
		tiers = [
			(206000, 0),
			(258000, 994),
			(322000, 2496),
			(386000, 4003),
			(750000, 5510),
		]
	top_surcharge = 6004
	surcharge = top_surcharge
	for threshold, amount in tiers:
		if magi <= threshold:
			surcharge = amount
			break
	return surcharge * num_medicare_persons
