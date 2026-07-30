"""Pure tax math functions — no Streamlit dependency.

Current-law figures are tax year 2026 (Rev. Proc. 2025-32), reflecting the
One Big Beautiful Bill Act (OBBBA, July 2025): TCJA-style rates are now
permanent, so the old "TCJA sunset" scenario is a counterfactual what-if
only.  All amounts are treated as real (today's) dollars per project
convention.
"""

# ── 2026 constants (OBBBA / Rev. Proc. 2025-32) ─────────────────

# Additional standard deduction for age 65+ (per person; not available to itemizers)
AGED_EXTRA_DEDUCTION_SINGLE = 2050
AGED_EXTRA_DEDUCTION_MFJ = 1650  # per spouse 65+

# OBBBA "senior bonus" deduction: $6,000 per person 65+, tax years 2025-2028,
# phased out at 6% of MAGI above the threshold. Available to itemizers too.
SENIOR_BONUS_PER_PERSON = 6000
SENIOR_BONUS_MAGI_THRESHOLD_SINGLE = 75000
SENIOR_BONUS_MAGI_THRESHOLD_MFJ = 150000
SENIOR_BONUS_LAST_YEAR = 2028

# SECURE 2.0 indexed QCD annual limit, per person (2026)
QCD_ANNUAL_LIMIT = 111000


def get_standard_deduction(filing_status: str, tax_law: str = 'tcja') -> float:
	if tax_law == 'pre_tcja':
		# Pre-TCJA counterfactual: standard deduction + personal exemptions
		# (2026 projected from 2017 law).  Retained as a what-if scenario only —
		# under OBBBA current rates are permanent.
		# Single: $8,300 std + $5,300 exemption = $13,600
		# MFJ: $16,600 std + 2 × $5,300 exemptions = $27,200
		return 13600 if filing_status == 'single' else 27200
	# 2026 standard deductions (OBBBA)
	return 16100 if filing_status == 'single' else 32200

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
	# 2026 ordinary income brackets (taxable income after deductions)
	if filing_status == 'single':
		return [
			(0, 0.10),
			(12400, 0.12),
			(50400, 0.22),
			(105700, 0.24),
			(201775, 0.32),
			(256225, 0.35),
			(640600, 0.37),
		]
	else:
		return [
			(0, 0.10),
			(24800, 0.12),
			(100800, 0.22),
			(211400, 0.24),
			(403550, 0.32),
			(512450, 0.35),
			(768700, 0.37),
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
	# 2026 LTCG/QD thresholds (stacked on top of ordinary taxable income)
	if filing_status == 'single':
		return [
			(0, 0.00),
			(49450, 0.15),
			(545500, 0.20),
		]
	else:  # married filing jointly
		return [
			(0, 0.00),
			(98900, 0.15),
			(613700, 0.20),
		]

def compute_taxable_social_security(ss_income: float, other_income: float, cap_gains: float, filing_status: str) -> float:
	# Simplified provisional income calculation; treats all other income (ordinary + gains) as part of provisional.
	# Thresholds are fixed nominal amounts in law (never inflation-indexed);
	# holding them constant in this real-dollar model is slightly favorable to the client.
	base = 25000 if filing_status == 'single' else 32000
	max_base = 34000 if filing_status == 'single' else 44000
	provisional = other_income + cap_gains + 0.5 * ss_income

	if provisional <= base:
		return 0.0
	if provisional <= max_base:
		# Middle tier is capped at 50% of benefits (IRS worksheet line 16 vs 17)
		return min(0.5 * (provisional - base), 0.5 * ss_income)
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

	E.g. for MFJ 22% → 211400 (income above this enters the 24% bracket).
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
	"""Compute Net Investment Income Tax (3.8% surtax). Thresholds are fixed in law (not indexed)."""
	niit_threshold = 200000 if filing_status == 'single' else 250000
	niit_base_val = max(0.0, agi - niit_threshold)
	return 0.038 * min(niit_base_val, net_investment_income)


def compute_total_taxes(ordinary_pre_ss: float, cg_total: float, ss_income: float,
						interest: float, earned_income: float, other_income: float,
						base_deduction: float, filing_status: str,
						tax_law: str = 'tcja', num_65: int = 0,
						senior_bonus_active: bool = False, itemizing: bool = False,
						state_rate: float = 0.0, state_exempt_retirement: bool = False) -> dict:
	"""Consolidated federal + NIIT + state tax computation for one year.

	ordinary_pre_ss: all ordinary income except taxable Social Security
		(TDA withdrawals, interest, pension, annuity, other, earned, conversions),
		already net of any capital-loss deduction (may be negative).
	cg_total: net capital gains + qualified dividends (before deduction offset).
	num_65: count of living household members aged 65+ (drives aged extra
		deduction and OBBBA senior bonus).
	senior_bonus_active: whether the calendar year is within the OBBBA
		senior-deduction window (2025-2028).

	Returns dict with t_ss, agi, deduction, t_ordinary, cg_taxable,
	ord_tax, cg_tax, niit, state_tax, total.
	"""
	t_ss = compute_taxable_social_security(ss_income, max(0.0, ordinary_pre_ss), cg_total, filing_status)
	agi = ordinary_pre_ss + t_ss + cg_total

	# Deduction: base (standard or itemized) + aged 65+ extra + OBBBA senior bonus
	deduction = base_deduction
	if num_65 > 0:
		if not itemizing:
			per_person = AGED_EXTRA_DEDUCTION_SINGLE if filing_status == 'single' else AGED_EXTRA_DEDUCTION_MFJ
			deduction += num_65 * per_person
		if senior_bonus_active and tax_law == 'tcja':
			thr = SENIOR_BONUS_MAGI_THRESHOLD_SINGLE if filing_status == 'single' else SENIOR_BONUS_MAGI_THRESHOLD_MFJ
			bonus = num_65 * SENIOR_BONUS_PER_PERSON - 0.06 * max(0.0, agi - thr)
			deduction += max(0.0, bonus)

	income_before_deduction = ordinary_pre_ss + t_ss
	t_ordinary = max(0.0, income_before_deduction - deduction)
	# Deduction not absorbed by ordinary income offsets capital gains
	# (taxable income = AGI - deduction, and gains stack on top of ordinary)
	unused_deduction = max(0.0, deduction - max(0.0, income_before_deduction))
	cg_taxable = max(0.0, cg_total - unused_deduction)

	ord_tax = apply_brackets(t_ordinary, get_ordinary_brackets(filing_status, tax_law))
	cg_tax = compute_capital_gains_tax(t_ordinary, cg_taxable, filing_status, tax_law)

	# NIIT: net investment income is NOT reduced by the standard deduction
	niit = compute_niit(agi, max(0.0, cg_total + interest), filing_status)

	# State income tax (flat-rate approximation).  Social Security is excluded —
	# the large majority of states (incl. IL) do not tax SS benefits.
	state_tax = 0.0
	if state_rate > 0:
		if state_exempt_retirement:
			# IL-style: retirement income (TDA/pension/annuity/conversions) exempt,
			# but investment income and wages/other income are taxed.
			state_taxable = max(0.0, interest + cg_total + earned_income + other_income)
		else:
			state_taxable = max(0.0, t_ordinary - t_ss) + cg_taxable
		state_tax = state_taxable * state_rate

	total = ord_tax + cg_tax + niit + state_tax
	return {
		't_ss': t_ss,
		'agi': agi,
		'deduction': deduction,
		't_ordinary': t_ordinary,
		'cg_taxable': cg_taxable,
		'ord_tax': ord_tax,
		'cg_tax': cg_tax,
		'niit': niit,
		'state_tax': state_tax,
		'total': total,
	}


def compute_state_tax(taxable_ordinary: float, cap_gains: float, interest: float,
					  state_rate: float, exempt_retirement: bool) -> float:
	"""Legacy flat state income tax helper (superseded by compute_total_taxes)."""
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
	2026 brackets (treated as real dollars, consistent with project convention).
	Surcharge = (Part B + Part D monthly surcharge) × 12 per person.
	"""
	if num_medicare_persons <= 0:
		return 0.0
	# (upper_threshold, annual_surcharge_per_person)
	if filing_status == 'single':
		tiers = [
			(109000, 0),
			(137000, 1148),
			(171000, 2885),
			(205000, 4620),
			(500000, 6355),
		]
	else:  # MFJ
		tiers = [
			(218000, 0),
			(274000, 1148),
			(342000, 2885),
			(410000, 4620),
			(750000, 6355),
		]
	top_surcharge = 6936
	surcharge = top_surcharge
	for threshold, amount in tiers:
		if magi <= threshold:
			surcharge = amount
			break
	return surcharge * num_medicare_persons
