import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt

from sim_engine import (
	PP_FACTORS, compute_run_pp_factors, load_master_global, load_bond_factors,
	get_all_historical_windows, compute_summary_metrics, build_scenario_params,
	auto_scenario_name, compute_scenario_summary, run_monte_carlo,
	store_distribution_results, simulate_withdrawals, sample_lognormal_returns,
	ss_adjustment_factor, ss_back_calculate_fra_benefit, ss_breakeven_table,
	compute_ss_benefits,
)
from ui_inputs import render_sidebar, save_results_to_json, load_plan_results, get_plans_with_results
from growth_engine import dollar_growth_distribution, dollar_growth_by_year

st.set_page_config(page_title='Withdrawal + RMD Simulator', layout='wide')

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

_OUTCOME_CFG = [
	(0,   'Worst case',    'Markets tank'),
	(10,  'Below average', 'Weak markets'),
	(25,  'Modest',        'Slower growth'),
	(50,  'Typical',       'Most likely'),
	(75,  'Above average', 'Good markets'),
	(100, 'Best case',     'Strong markets'),
]
_BAR_COLORS = ['#94a3b8', '#7cafc8', '#5bb8c0', '#3bbfb0', '#2ec4a0', '#22c990']

def _spending_outcomes_chart(data_series, target=None, currency_fmt='${:,.0f}'):
	"""Horizontal bar chart: compressed x-axis, gradient colors, red target line, two-line labels."""
	plt.rcParams['font.family'] = 'sans-serif'
	values = [float(np.percentile(data_series, p)) for p, _, _ in _OUTCOME_CFG]
	labels = [lbl for _, lbl, _ in _OUTCOME_CFG]
	subs = [sub for _, _, sub in _OUTCOME_CFG]
	n = len(values)
	y_pos = list(range(n))

	fig, ax = plt.subplots(figsize=(4, 1.8), dpi=100)
	fig.patch.set_facecolor('white')
	ax.set_facecolor('white')

	# Compressed x-axis: start near the min so differences are visually meaningful
	val_min, val_max = min(values), max(values)
	val_range = val_max - val_min if val_max > val_min else val_max * 0.1
	x_min = val_min - val_range * 0.4
	if target and target > 0:
		x_min = min(x_min, target - val_range * 0.4)
	x_max = val_max + val_range * 0.5  # room for value labels

	# Draw bars starting from x_min (left edge)
	bar_widths = [v - x_min for v in values]
	bar_height = 0.52
	ax.barh(y_pos, bar_widths, left=x_min, height=bar_height, color=_BAR_COLORS,
			edgecolor='none', zorder=2, linewidth=0)

	# Remove all spines and gridlines
	for spine in ax.spines.values():
		spine.set_visible(False)
	ax.grid(False)

	# Value labels just outside each bar
	for i, v in enumerate(values):
		ax.text(v + val_range * 0.04, i, currency_fmt.format(v),
				va='center', ha='left', fontsize=5, fontweight='bold', color='#334155')

	# Y-axis: bold label + lighter gray subtitle
	ax.set_yticks(y_pos)
	ax.set_yticklabels([''] * n)
	for i, (lbl, sub) in enumerate(zip(labels, subs)):
		ax.text(x_min - val_range * 0.03, i - 0.13, lbl, va='center', ha='right',
				fontsize=5, fontweight='bold', color='#1e293b')
		ax.text(x_min - val_range * 0.03, i + 0.18, sub, va='center', ha='right',
				fontsize=4, color='#94a3b8')

	# Red dashed target line with label above the chart
	if target and target > 0:
		ax.axvline(x=target, color='#ef4444', linestyle='--', linewidth=1.2, zorder=3)
		target_k = target / 1000
		target_label = f'Your ${target_k:.0f}K target' if target_k == int(target_k) else f'Your ${target_k:,.1f}K target'
		ax.text(target, -0.6, target_label, va='bottom', ha='center',
				fontsize=5, color='#ef4444', fontweight='bold')

	ax.set_xlim(x_min, x_max)
	ax.xaxis.set_visible(False)
	ax.tick_params(left=False)
	ax.invert_yaxis()
	fig.subplots_adjust(left=0.18, right=0.90, top=0.92, bottom=0.02)
	st.pyplot(fig, use_container_width=False)
	plt.close(fig)

def _three_card_summary(data_series, target, currency_fmt='${:,.0f}'):
	"""3-card layout: Income Floor / Likely Lifestyle / Strong Markets with styled cards."""
	floor_val = float(np.percentile(data_series, 0))
	typical_val = float(np.percentile(data_series, 50))
	strong_val = float(np.percentile(data_series, 90))

	def _pct_vs_target(val):
		if target and target > 0:
			diff = (val / target - 1) * 100
			if diff >= 0:
				return f'<span style="color:#16a34a;font-size:0.85em;">+{diff:.0f}% vs target</span>'
			else:
				return f'<span style="color:#dc2626;font-size:0.85em;">{diff:.0f}% vs target</span>'
		return ''

	def _card_colors(val):
		if target and target > 0 and val >= target:
			return '#b8860b', '#fdf6e3'  # warm beige: at/above target
		return '#ef4444', '#fef2f2'      # red/pink: below target

	cards = [
		{'icon': '\U0001f6e1\ufe0f', 'title': 'Income Floor', 'val': floor_val,
		 'sub': 'Worst case across all runs', 'border': _card_colors(floor_val)[0], 'bg': _card_colors(floor_val)[1]},
		{'icon': '\U0001f3af', 'title': 'Likely Lifestyle', 'val': typical_val,
		 'sub': 'Median outcome', 'border': _card_colors(typical_val)[0], 'bg': _card_colors(typical_val)[1]},
		{'icon': '\U0001f680', 'title': 'Strong Markets', 'val': strong_val,
		 'sub': '90th percentile upside', 'border': _card_colors(strong_val)[0], 'bg': _card_colors(strong_val)[1]},
	]

	card_htmls = []
	for card in cards:
		pct_line = _pct_vs_target(card['val'])
		card_htmls.append(
			f'<div style="flex:1; border:1px solid {card["border"]}; border-left:4px solid {card["border"]}; '
			f'border-radius:8px; padding:16px 14px; background:{card["bg"]}; text-align:center;">'
			f'<div style="font-size:1.5em; margin-bottom:4px;">{card["icon"]}</div>'
			f'<div style="font-weight:600; color:#374151; margin-bottom:2px;">{card["title"]}</div>'
			f'<div style="font-size:1.6em; font-weight:700; color:#111827;">{currency_fmt.format(card["val"])}</div>'
			f'{pct_line}'
			f'<div style="color:#6b7280; font-size:0.82em; margin-top:4px;">{card["sub"]}</div>'
			f'</div>'
		)
	st.markdown(
		f'<div style="display:flex; gap:12px; align-items:stretch; max-width:50%;">{"".join(card_htmls)}</div>',
		unsafe_allow_html=True,
	)

def _render_client_view(success_rate, spending_pct_rows, percentile_rows,
						all_yearly, currency_fmt='${:,.0f}',
						has_goal_breakdown=False, active_goals=None,
						base_schedule=None, run_spending=None,
						inheritor_marginal_rate=0.0,
						withdrawal_schedule=None, window_start_dates=None,
						funded_goals=None):
	"""Simplified client-facing results view: verdict, spending, legacy, and median portfolio chart."""

	# ── Section 1: The Verdict ──
	pct = success_rate * 100
	num_sims = st.session_state.get('num_sims', 0)
	successes = int(round(success_rate * num_sims))
	if pct >= 90:
		color, bg = '#16a34a', '#f0fdf4'
		interp = 'Your plan is well-funded across a wide range of market conditions.'
	elif pct >= 75:
		color, bg = '#ca8a04', '#fefce8'
		interp = 'Your plan succeeds in most scenarios but has some vulnerability to sustained poor markets.'
	else:
		color, bg = '#dc2626', '#fef2f2'
		interp = 'Your plan faces meaningful risk. Consider adjusting spending or timing.'
	st.markdown(
		f'<div style="border:2px solid {color}; border-radius:12px; padding:24px 28px; '
		f'background:{bg}; text-align:center; max-width:60%; margin-bottom:20px;">'
		f'<div style="font-size:1.8em; font-weight:700; color:{color}; margin-bottom:6px;">'
		f'Your plan succeeds in {successes:,} out of {num_sims:,} simulations</div>'
		f'<div style="color:#374151; font-size:1.05em;">{interp}</div></div>',
		unsafe_allow_html=True,
	)

	# Spending success rate
	spending_success = st.session_state.get('mc_spending_success_rate')
	spending_target = st.session_state.get('mc_spending_target', 0)
	if spending_success is not None and spending_target > 0:
		spend_pct = spending_success * 100
		spend_count = int(round(spending_success * num_sims))
		st.markdown(
			f'<div style="color:#374151; font-size:0.95em; text-align:center; max-width:60%; margin-bottom:16px;">'
			f'{spend_count:,} of {num_sims:,} simulations ({spend_pct:.0f}%) maintained average annual spending '
			f'at or above the ${spending_target:,.0f} target</div>',
			unsafe_allow_html=True,
		)

	# ── Section 2: Your Spending (separated by goal when applicable) ──
	st.caption('All spending figures are after taxes.')
	if has_goal_breakdown and active_goals:
		# Base spending
		run_base = all_yearly.groupby('run').agg(
			total_base_spending=('base_after_tax_spending', 'sum'),
			years_in_run=('year', 'count'),
		)
		run_base['avg'] = run_base['total_base_spending'] / run_base['years_in_run']
		base_target = float(np.mean(base_schedule)) if base_schedule else 0
		st.markdown(f'#### Base Spending — {currency_fmt.format(base_target)}/yr target' if base_target > 0
			else '#### Base Spending')
		_client_spending_cards(run_base['avg'], base_target, currency_fmt)

		# Per-goal spending
		for gi, (g_label, g_amount, g_begin, g_end, *_gx) in enumerate(active_goals):
			col_spending = f'goal_{gi}_after_tax_spending'
			if col_spending not in all_yearly.columns:
				continue
			goal_years = all_yearly[all_yearly['year'].between(g_begin, g_end)]
			if goal_years.empty:
				continue
			run_goal = goal_years.groupby('run').agg(
				total_goal_spending=(col_spending, 'sum'),
				years_in_goal=('year', 'count'),
			)
			run_goal['avg'] = run_goal['total_goal_spending'] / run_goal['years_in_goal']
			goal_name = g_label or f'Goal {gi + 1}'
			num_years = g_end - g_begin + 1
			st.markdown(f'#### {goal_name} — {currency_fmt.format(g_amount)}/yr, years {g_begin}\u2013{g_end} ({num_years} yrs)')
			_client_spending_cards(run_goal['avg'], float(g_amount), currency_fmt)
	else:
		st.markdown('#### Your Spending')
		_client_spending_cards(run_spending['avg_annual_after_tax_spending'] if run_spending is not None
			else pd.Series([r['avg_annual_after_tax_spending'] for r in spending_pct_rows]),
			0, currency_fmt)

	# Separately funded goal cards (always shown when funded goals exist)
	if funded_goals and 'goal_taxable_balance' in all_yearly.columns:
		for fg in funded_goals:
			num_goal_yrs = fg['end_year'] - fg['begin_year'] + 1
			total_spending = fg['annual_amount'] * num_goal_yrs
			st.markdown(f"#### {fg['label']} — {currency_fmt.format(fg['annual_amount'])}/yr, "
				f"years {fg['begin_year']}\u2013{fg['end_year']} ({num_goal_yrs} yrs)")
			last_goal_year = fg['end_year']
			last_yr_data = all_yearly[all_yearly['year'] == min(last_goal_year, all_yearly['year'].max())]
			if not last_yr_data.empty:
				surplus_50 = float(np.percentile(
					last_yr_data['goal_taxable_balance'] + last_yr_data['goal_tda_balance'], 50))
			else:
				surplus_50 = 0.0
			cards = [
				('\U0001f4b0', 'Set Aside Today', fg['total_cost'],
				 f"Funds {currency_fmt.format(total_spending)} total",
				 '#b8860b', '#fdf6e3'),
				('\u2705', 'Annual Spending', fg['annual_amount'],
				 f"Guaranteed {num_goal_yrs} yrs ({fg['fund_stock_pct']}% stocks)",
				 '#16a34a', '#f0fdf4'),
				('\U0001f4c8', 'Median Surplus', surplus_50,
				 f"Goal account balance after yr {last_goal_year}",
				 '#2563eb', '#eff6ff'),
			]
			_cards_html(cards, currency_fmt)

	# ── Section 3: What You'll Leave Behind ──
	st.markdown('#### What You\'ll Leave Behind')
	end_0 = next(r for r in percentile_rows if r['percentile'] == 0)['after_tax_end']
	end_50 = next(r for r in percentile_rows if r['percentile'] == 50)['after_tax_end']
	end_90 = next(r for r in percentile_rows if r['percentile'] == 90)['after_tax_end']
	legacy_cards = [
		('\U0001f6e1\ufe0f', 'If Markets Struggle', end_0, 'Worst-case ending balance', '#ef4444', '#fef2f2'),
		('\U0001f3af', 'Most Likely', end_50, 'Median ending balance', '#b8860b', '#fdf6e3'),
		('\U0001f680', 'If Markets Do Well', end_90, '90th percentile ending', '#16a34a', '#f0fdf4'),
	]
	_cards_html(legacy_cards, currency_fmt)

	# ── Section 4: Median portfolio chart with pre/after-tax toggle ──
	st.markdown('#### Portfolio Over Time')
	_rc1, _rc2 = st.columns([1, 3])
	with _rc1:
		port_tax_mode = st.radio('Show portfolio value as', ['Pre-tax', 'After-tax'],
			horizontal=True, key='client_port_tax_mode', index=1)
	if port_tax_mode == 'After-tax':
		inh_rate = float(inheritor_marginal_rate)
		at_col = '_cv_after_tax_portfolio'
		all_yearly[at_col] = (all_yearly['end_taxable_total'] + all_yearly['end_roth']
			+ all_yearly['end_tda_total'] * max(0.0, 1.0 - inh_rate))
		median_port = all_yearly.groupby('year')[at_col].median()
		y_label = f'After-Tax Portfolio (TDA taxed at {inh_rate:.0%})'
	else:
		median_port = all_yearly.groupby('year')['total_portfolio'].median()
		y_label = 'Pre-Tax Portfolio'
	ending_val = float(median_port.iloc[-1])
	with _rc2:
		st.metric(f'Median Ending Value ({port_tax_mode})', currency_fmt.format(ending_val))
	chart_df = pd.DataFrame({'Year': median_port.index, 'Portfolio Value': median_port.values})
	chart = alt.Chart(chart_df).mark_line(strokeWidth=3, color='#2563eb').encode(
		x=alt.X('Year:Q', title='Year'),
		y=alt.Y('Portfolio Value:Q', title=y_label, axis=alt.Axis(format='$,.0f')),
		tooltip=[alt.Tooltip('Year:Q'), alt.Tooltip('Portfolio Value:Q', format='$,.0f')],
	).properties(height=350).interactive()
	st.altair_chart(chart, use_container_width=True)

	# ── Section 5: Worst historical stress tests ──
	if window_start_dates and withdrawal_schedule:
		_stress_periods = [('1929-09', 'September 1929 (Great Depression)'),
						   ('1966-01', 'January 1966 (Stagflation Era)')]
		# Find run indices matching these start dates
		date_to_run = {d: int(k) for k, d in window_start_dates.items()}
		matched = [(date_to_run[code], label) for code, label in _stress_periods
				   if code in date_to_run]
		if matched:
			st.markdown('#### What If You Had the Worst Timing?')
			st.caption(
				'Even when average spending meets your target over the full horizon, '
				'some years may require flexibility. These two historical periods '
				'represent the toughest stretches for retirees.')
			for run_idx, period_label in matched:
				run_raw = all_yearly[all_yearly['run'] == run_idx]
				if run_raw.empty:
					continue
				stress_df = pd.DataFrame({
					'Year': run_raw['year'].values,
					'Spending': run_raw['after_tax_spending'].values,
					'Target': [withdrawal_schedule[i] if i < len(withdrawal_schedule)
							   else withdrawal_schedule[-1] for i in range(len(run_raw))],
				})
				# Build spending summary broken out by base + goals
				st.markdown(f'**Starting {period_label}**')
				if has_goal_breakdown and active_goals and 'base_after_tax_spending' in run_raw.columns:
					avg_base = float(run_raw['base_after_tax_spending'].mean())
					st.caption(f'Base spending: {currency_fmt.format(avg_base)}/yr average')
					for gi, (g_label, g_amount, g_begin, g_end, *_gx) in enumerate(active_goals):
						col = f'goal_{gi}_after_tax_spending'
						if col in run_raw.columns:
							goal_years = run_raw[run_raw['year'].between(g_begin, g_end)]
							if not goal_years.empty:
								avg_goal = float(goal_years[col].mean())
								name = g_label or f'Goal {gi + 1}'
								st.caption(f'{name}: {currency_fmt.format(avg_goal)}/yr average')
				else:
					avg_spending = float(stress_df['Spending'].mean())
					st.caption(f'Average spending: {currency_fmt.format(avg_spending)}/yr')
				stress_df['_color'] = stress_df.apply(
					lambda r: 'Below target' if r['Spending'] < r['Target'] else 'At or above target', axis=1)
				stress_df['Shortfall'] = (stress_df['Target'] - stress_df['Spending']).clip(lower=0)
				spend_chart = alt.Chart(stress_df).mark_bar().encode(
					x=alt.X('Year:O', title='Year'),
					y=alt.Y('Spending:Q', title='After-Tax Spending'),
					color=alt.Color('_color:N', scale=alt.Scale(
						domain=['Below target', 'At or above target'],
						range=['#ef4444', '#60a5fa']),
						legend=None),
					tooltip=[alt.Tooltip('Year:O'), alt.Tooltip('Spending:Q', format='$,.0f'),
							 alt.Tooltip('Target:Q', format='$,.0f'),
							 alt.Tooltip('Shortfall:Q', format='$,.0f')],
				).properties(height=250)
				goal_line = alt.Chart(stress_df).mark_line(
					color='#ef4444', strokeDash=[6, 3], strokeWidth=2
				).encode(x=alt.X('Year:O'), y=alt.Y('Target:Q'))
				st.altair_chart((spend_chart + goal_line).interactive(), use_container_width=True)
			st.caption('Blue bars = after-tax spending at or above target. Red bars = below target. Dashed line = spending target.')


def _client_spending_cards(data_series, target, currency_fmt='${:,.0f}'):
	"""3 spending cards for client view: 0th / 50th / 90th percentile (matches advisor view)."""
	floor_val = float(np.percentile(data_series, 0))
	typical_val = float(np.percentile(data_series, 50))
	strong_val = float(np.percentile(data_series, 90))

	def _pick_colors(val):
		if target and target > 0 and val >= target:
			return '#b8860b', '#fdf6e3'
		return '#ef4444', '#fef2f2'

	cards = [
		('\U0001f6e1\ufe0f', 'If Markets Struggle', floor_val, 'Worst case across all runs',
		 *_pick_colors(floor_val)),
		('\U0001f3af', 'Most Likely', typical_val, 'Median outcome',
		 *_pick_colors(typical_val)),
		('\U0001f680', 'If Markets Do Well', strong_val, '90th percentile',
		 *_pick_colors(strong_val)),
	]
	_cards_html(cards, currency_fmt)


def _cards_html(cards, currency_fmt='${:,.0f}'):
	"""Render a list of (icon, title, value, subtitle, border_color, bg_color) as 3 styled cards."""
	html_parts = []
	for icon, title, val, sub, border, bg in cards:
		html_parts.append(
			f'<div style="flex:1; border:1px solid {border}; border-left:4px solid {border}; '
			f'border-radius:8px; padding:16px 14px; background:{bg}; text-align:center;">'
			f'<div style="font-size:1.5em; margin-bottom:4px;">{icon}</div>'
			f'<div style="font-weight:600; color:#374151; margin-bottom:2px;">{title}</div>'
			f'<div style="font-size:1.6em; font-weight:700; color:#111827;">{currency_fmt.format(val)}</div>'
			f'<div style="color:#6b7280; font-size:0.82em; margin-top:4px;">{sub}</div>'
			f'</div>'
		)
	st.markdown(
		f'<div style="display:flex; gap:12px; align-items:stretch; max-width:50%;">{"".join(html_parts)}</div>',
		unsafe_allow_html=True,
	)


def _display_comparison(scenarios, currency_fmt, key_suffix=''):
	"""Display a scenario comparison table and overlay charts.
	Each scenario dict has: name, percentile_rows, spending_percentiles, pct_non_positive,
	median_yearly_portfolio (list), median_yearly_spending (list), median_yearly_years (list)."""
	compare_pct = st.selectbox('Compare at percentile', [0, 10, 25, 50, 75, 90], index=3,
		key=f'compare_pct_{key_suffix}')
	baseline = scenarios[0]
	baseline_row = next(r for r in baseline['percentile_rows'] if r['percentile'] == compare_pct)
	baseline_spend = next(r for r in baseline['spending_percentiles'] if r['percentile'] == compare_pct)
	comparison_rows = []
	for sc in scenarios:
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
			'% Below Goal': sc['pct_non_positive'] * 100,
		})
	comp_df = pd.DataFrame(comparison_rows)
	# Ensure unique scenario names for Styler compatibility
	name_counts = {}
	unique_names = []
	for name in comp_df['Scenario']:
		name_counts[name] = name_counts.get(name, 0) + 1
		if name_counts[name] > 1:
			unique_names.append(f"{name} ({name_counts[name]})")
		else:
			unique_names.append(name)
	comp_df['Scenario'] = unique_names
	comp_df = comp_df.set_index('Scenario')
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
		'% Below Goal': '{:.1f}%'.format,
	}).apply(_color_deltas, axis=None))

	# Overlay charts
	st.subheader('Portfolio Value Comparison (median)')
	overlay_df = pd.DataFrame()
	for sc in scenarios:
		overlay_df[sc['name']] = pd.Series(sc['median_yearly_portfolio'],
			index=sc['median_yearly_years'])
	interactive_line_chart(overlay_df, y_title='Portfolio Value (Median)', zero_base=False)

	st.subheader('After-Tax Spending Comparison (median)')
	spend_overlay = pd.DataFrame()
	for sc in scenarios:
		spend_overlay[sc['name']] = pd.Series(sc['median_yearly_spending'],
			index=sc['median_yearly_years'])
	interactive_line_chart(spend_overlay, y_title='After-Tax Spending (Median)', zero_base=False)

def main():
	st.title('Withdrawal + RMD Simulator (30-year)')

	if st.button('Reset saved scenarios'):
		for key in ['scenario_summaries', 'last_summary', 'mc_percentile_rows', 'mc_all_yearly',
					'mc_pct_non_positive', 'sim_df', 'sim_mode', 'mc_summaries', 'mc_percentiles',
					'multi_scenario_results', 'selected_scenario_idx', 'num_sims',
					'saved_plan_comparison', 'ss_analysis_results',
					'ss_analysis_grid', 'ss_analysis_meta']:
			st.session_state.pop(key, None)
		st.success('Saved scenarios cleared.')

	# ── Render all sidebar inputs ────────────────────────────────
	inputs = render_sidebar()

	# Unpack commonly used values
	start_age = inputs['start_age']
	start_age_spouse = inputs['start_age_spouse']
	life_expectancy_primary = inputs['life_expectancy_primary']
	life_expectancy_spouse = inputs['life_expectancy_spouse']
	taxable_start = inputs['taxable_start']
	taxable_stock_basis_pct = inputs['taxable_stock_basis_pct']
	taxable_bond_basis_pct = inputs['taxable_bond_basis_pct']
	roth_start = inputs['roth_start']
	tda_start = inputs['tda_start']
	tda_spouse_start = inputs['tda_spouse_start']
	target_stock_pct = inputs['target_stock_pct']
	roth_conversion_mode_ui = inputs['roth_conversion_mode']
	roth_conversion_amount = inputs['roth_conversion_amount']
	roth_bracket_fill_rate = inputs['roth_bracket_fill_rate']
	roth_conversion_years = inputs['roth_conversion_years']
	roth_conversion_source_tda = inputs['roth_conversion_source_tda']
	roth_conversion_tax_source = inputs['roth_conversion_tax_source']
	withdrawal_schedule_inputs = inputs['withdrawal_schedule_inputs']
	rmd_start_age = inputs['rmd_start_age']
	rmd_start_age_spouse = inputs['rmd_start_age_spouse']
	ending_balance_goal = inputs['ending_balance_goal']
	add_goal_inputs = inputs['add_goal_inputs']
	ss_income_input = inputs['ss_income']
	ss_start_age_p1 = inputs['ss_start_age_p1']
	ss_income_spouse_input = inputs['ss_income_spouse']
	ss_start_age_p2 = inputs['ss_start_age_p2']
	ss_fra_age_p1 = inputs['ss_fra_age_p1']
	ss_fra_age_p2 = inputs['ss_fra_age_p2']
	ss_cola = inputs['ss_cola']
	pension_income_input = inputs['pension_income']
	pension_cola_p1 = inputs['pension_cola_p1']
	pension_survivor_pct_p1 = inputs['pension_survivor_pct_p1']
	pension_income_spouse_input = inputs['pension_income_spouse']
	pension_cola_p2 = inputs['pension_cola_p2']
	pension_survivor_pct_p2 = inputs['pension_survivor_pct_p2']
	other_income_input = inputs['other_income']
	earned_income_input = inputs['earned_income']
	earned_income_years = inputs['earned_income_years']
	qcd_annual = inputs['qcd_annual']
	pension_buyout_enabled = inputs['pension_buyout_enabled']
	pension_buyout_baseline = inputs['pension_buyout_baseline']
	pension_buyout_person = inputs['pension_buyout_person']
	pension_buyout_lump = inputs['pension_buyout_lump']
	pension_buyout_income = inputs['pension_buyout_income']
	pension_buyout_cola = inputs['pension_buyout_cola']
	pension_buyout_survivor = inputs['pension_buyout_survivor']
	inheritance_enabled = inputs['inheritance_enabled']
	inheritance_year = inputs['inheritance_year']
	inheritance_taxable_amount = inputs['inheritance_taxable_amount']
	inheritance_ira_amount = inputs['inheritance_ira_amount']
	taxes_enabled = inputs['taxes_enabled']
	filing_status_key = inputs['filing_status_key']
	use_itemized = inputs['use_itemized']
	itemized_deduction_input = inputs['itemized_deduction']
	inheritor_marginal_rate = inputs['inheritor_marginal_rate']
	state_tax_rate = inputs['state_tax_rate']
	state_exempt_retirement = inputs['state_exempt_retirement']
	tcja_sunset = inputs['tcja_sunset']
	tcja_sunset_year = inputs['tcja_sunset_year']
	irmaa_enabled = inputs['irmaa_enabled']
	prefer_tda_before_taxable = inputs['prefer_tda_before_taxable']
	return_mode = inputs['return_mode']
	stock_total_return = inputs['stock_total_return']
	bond_return = inputs['bond_return']
	taxable_log_drift = inputs['taxable_log_drift']
	taxable_log_volatility = inputs['taxable_log_volatility']
	bond_log_drift = inputs['bond_log_drift']
	bond_log_volatility = inputs['bond_log_volatility']
	stock_dividend_yield = inputs['stock_dividend_yield']
	stock_turnover = inputs['stock_turnover']
	investment_fee_bps = inputs['investment_fee_bps']
	guardrails_enabled = inputs['guardrails_enabled']
	guardrail_lower = inputs['guardrail_lower']
	guardrail_upper = inputs['guardrail_upper']
	guardrail_target = inputs['guardrail_target']
	guardrail_inner_sims = inputs['guardrail_inner_sims']
	guardrail_max_spending_pct = inputs['guardrail_max_spending_pct']
	flex_goal_min_pct = inputs['flex_goal_min_pct']
	base_is_essential = inputs['base_is_essential']
	display_decimals = inputs['display_decimals']
	monte_carlo_runs = inputs['monte_carlo_runs']
	num_scenarios = inputs['num_scenarios']
	scenario_overrides_ui = inputs['scenario_overrides_ui']

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
	base_withdrawal_schedule = list(withdrawal_schedule)  # save before goals
	# Build per-year goal schedule and layer goals on top of base withdrawals
	# "Need" goals go into goal_schedule (unscaled by guardrails)
	# "Want" goals are added to base (scaled by guardrails)
	goal_schedule = [0.0] * years
	flex_goal_schedule = [0.0] * years
	flex_capped_base_schedule = [0.0] * years
	flex_cap_max_schedule = [0.0] * years
	has_flex_caps = False
	funded_goals = []
	goal_liquidation_schedule = [0.0] * years
	for g_label, g_amount, g_begin, g_end, g_priority, *g_rest in add_goal_inputs:
		g_cap = g_rest[0] if g_rest else -1.0
		g_fund_sep = g_rest[1] if len(g_rest) > 1 else False
		g_fund_taxable = g_rest[2] if len(g_rest) > 2 else 0.0
		g_fund_tda1 = g_rest[3] if len(g_rest) > 3 else 0.0
		g_fund_tda2 = g_rest[4] if len(g_rest) > 4 else 0.0
		g_fund_stock_pct = g_rest[5] if len(g_rest) > 5 else 60
		if g_fund_sep and g_amount > 0:
			funded_goals.append({
				'label': g_label, 'annual_amount': g_amount,
				'begin_year': g_begin, 'end_year': g_end,
				'fund_taxable': g_fund_taxable, 'fund_tda1': g_fund_tda1, 'fund_tda2': g_fund_tda2,
				'fund_stock_pct': g_fund_stock_pct,
			})
			for y_idx in range(g_begin - 1, min(g_end, years)):
				goal_liquidation_schedule[y_idx] += g_amount
			continue  # skip normal goal schedule logic
		if g_amount > 0:
			for y_idx in range(g_begin - 1, min(g_end, years)):
				withdrawal_schedule[y_idx] += g_amount
				if g_priority in ('Essential', 'Need'):
					goal_schedule[y_idx] += g_amount
				elif g_priority == 'Flexible':
					flex_goal_schedule[y_idx] += g_amount
					if g_cap >= 0:
						has_flex_caps = True
						flex_capped_base_schedule[y_idx] += g_amount
						flex_cap_max_schedule[y_idx] += g_amount * (1.0 + g_cap / 100.0)
	withdraw_amount = withdrawal_schedule[0]  # for backward-compat references
	# Filter _add_goal_inputs for display to exclude funded-separately goals
	_active_goal_inputs = [g for g in add_goal_inputs if not (len(g) > 6 and g[6])]
	st.session_state['_add_goal_inputs'] = _active_goal_inputs
	st.session_state['_base_withdrawal_schedule'] = base_withdrawal_schedule
	st.session_state['_withdrawal_schedule'] = list(withdrawal_schedule)

	# ── Calculate set-aside cost for separately funded goals ──
	goal_taxable_start = 0.0
	goal_tda1_start = 0.0
	goal_tda2_start = 0.0
	if funded_goals:
		is_hist_mode = return_mode == 'Historical (master_global_factors)'
		for fg in funded_goals:
			grid = dollar_growth_by_year(
				stock_pct=fg['fund_stock_pct'] / 100.0,
				max_years=int(fg['end_year']),
				mode='historical' if is_hist_mode else 'simulated',
				stock_drift=float(inputs['taxable_log_drift']),
				stock_vol=float(inputs['taxable_log_volatility']),
				bond_drift=float(inputs['bond_log_drift']),
				bond_vol=float(inputs['bond_log_volatility']),
				num_runs=int(monte_carlo_runs),
				fee_bps=float(investment_fee_bps),
			)
			total_cost = 0.0
			fg['year_costs'] = []
			for yr in range(int(fg['begin_year']), int(fg['end_year']) + 1):
				growth_0 = float(np.percentile(grid[:, yr - 1], 0))
				cost_per_dollar = 1.0 / growth_0 if growth_0 > 0 else float('inf')
				set_aside = fg['annual_amount'] * cost_per_dollar
				total_cost += set_aside
				fg['year_costs'].append({
					'year': yr, 'growth_0': growth_0, 'cost_per_dollar': cost_per_dollar,
					'goal': fg['annual_amount'], 'set_aside': set_aside,
				})
			fg['total_cost'] = total_cost
			# Auto-fill sources if all zero
			source_sum = fg['fund_taxable'] + fg['fund_tda1'] + fg['fund_tda2']
			if source_sum == 0:
				# Auto-fill: taxable first, then TDA P1, then TDA P2
				remaining = total_cost
				fg['fund_taxable'] = min(remaining, float(taxable_start) - goal_taxable_start)
				remaining -= fg['fund_taxable']
				if remaining > 0:
					fg['fund_tda1'] = min(remaining, float(tda_start) - goal_tda1_start)
					remaining -= fg['fund_tda1']
				if remaining > 0:
					fg['fund_tda2'] = min(remaining, float(tda_spouse_start) - goal_tda2_start)
			goal_taxable_start += fg['fund_taxable']
			goal_tda1_start += fg['fund_tda1']
			goal_tda2_start += fg['fund_tda2']
		st.session_state['_funded_goals'] = funded_goals

	# Adjusted starting balances (primary reduced by set-aside)
	adjusted_taxable_start = float(taxable_start) - goal_taxable_start
	adjusted_tda_start = float(tda_start) - goal_tda1_start
	adjusted_tda_spouse_start = float(tda_spouse_start) - goal_tda2_start

	if funded_goals:
		total_set_aside = sum(fg['total_cost'] for fg in funded_goals)
		st.info(f"**Separately funded goals:** ${total_set_aside:,.0f} set aside from portfolio "
			f"(Taxable: ${goal_taxable_start:,.0f} | TDA P1: ${goal_tda1_start:,.0f} | TDA P2: ${goal_tda2_start:,.0f}). "
			f"Primary portfolio after set-aside: ${adjusted_taxable_start + adjusted_tda_start + adjusted_tda_spouse_start + float(roth_start):,.0f}")

	currency_fmt = f'${{:,.{int(display_decimals)}f}}'

	# Common simulation parameters (used by all modes)
	goal_tda_total = goal_tda1_start + goal_tda2_start
	# Weighted-average stock allocation for goal shadow accounts
	goal_total_start = goal_taxable_start + goal_tda_total
	if goal_total_start > 0 and funded_goals:
		goal_stock_pct = sum(
			(fg['fund_taxable'] + fg['fund_tda1'] + fg['fund_tda2']) * fg['fund_stock_pct'] / 100.0
			for fg in funded_goals
		) / goal_total_start
	else:
		goal_stock_pct = float(target_stock_pct)
	sim_params = dict(
		start_age_primary=int(start_age), start_age_spouse=int(start_age_spouse),
		taxable_start=adjusted_taxable_start, stock_total_return=float(stock_total_return),
		stock_dividend_yield=float(stock_dividend_yield), stock_turnover=float(stock_turnover),
		investment_fee_bps=float(investment_fee_bps),
		bond_return=float(bond_return), roth_start=float(roth_start),
		tda_start=adjusted_tda_start, tda_spouse_start=adjusted_tda_spouse_start,
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
		roth_conversion_mode={'None': 'none', 'Fixed amount': 'fixed', 'Fill to bracket': 'bracket_fill'}[roth_conversion_mode_ui],
		roth_bracket_fill_rate=float(roth_bracket_fill_rate),
		ss_start_age_p1=int(ss_start_age_p1),
		ss_start_age_p2=int(ss_start_age_p2),
		ss_fra_age_p1=int(ss_fra_age_p1),
		ss_fra_age_p2=int(ss_fra_age_p2),
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
		goal_schedule=goal_schedule if any(g > 0 for g in goal_schedule) else None,
		flex_goal_schedule=flex_goal_schedule if any(g > 0 for g in flex_goal_schedule) else None,
		flex_goal_min_pct=float(flex_goal_min_pct),
		base_is_essential=bool(base_is_essential),
		flex_capped_base_schedule=flex_capped_base_schedule if has_flex_caps else None,
		flex_cap_max_schedule=flex_cap_max_schedule if has_flex_caps else None,
		inheritance_enabled=bool(inheritance_enabled),
		inheritance_year=int(inheritance_year),
		inheritance_taxable_amount=float(inheritance_taxable_amount),
		inheritance_ira_amount=float(inheritance_ira_amount),
		tcja_sunset=bool(tcja_sunset),
		tcja_sunset_year=int(tcja_sunset_year),
		qcd_annual=float(qcd_annual),
		earned_income_annual=float(earned_income_input),
		earned_income_years=int(earned_income_years),
		irmaa_enabled=bool(irmaa_enabled),
		prefer_tda_before_taxable=bool(prefer_tda_before_taxable),
		goal_taxable_start=goal_taxable_start,
		goal_tda_start=goal_tda_total,
		goal_tda_p1_fraction=goal_tda1_start / goal_tda_total if goal_tda_total > 0 else 0.0,
		goal_liquidation_schedule=goal_liquidation_schedule if any(g > 0 for g in goal_liquidation_schedule) else None,
		goal_stock_pct=goal_stock_pct,
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
		if return_mode != 'Simulated (lognormal)':
			# Compute sigma from actual blended portfolio log returns — captures real correlation
			n = min(len(stock_log_rets), len(bond_log_returns))
			blended_log_rets = target_stock_pct * stock_log_rets[:n] + (1 - target_stock_pct) * bond_log_returns[:n]
			blended_sigma = float(np.std(blended_log_rets))
		else:
			# No raw return series available; use variance formula with assumed correlation
			rho = 0.10
			w_s, w_b = target_stock_pct, 1 - target_stock_pct
			blended_sigma = float(np.sqrt(
				(w_s * stock_sigma) ** 2 + (w_b * bond_sigma) ** 2
				+ 2 * w_s * w_b * stock_sigma * bond_sigma * rho
			))
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
		# Clear stale SS claiming age analysis (depends on baseline plan inputs)
		st.session_state.pop('ss_analysis_grid', None)
		st.session_state.pop('ss_analysis_meta', None)
		st.session_state.pop('ss_analysis_results', None)

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
				_base_sched_ms = st.session_state.get('_base_withdrawal_schedule', [])
				_spend_target_ms = _base_sched_ms[0] if _base_sched_ms else 0.0
				summary = compute_scenario_summary(s_name, results, all_yearly_df,
					float(inheritor_marginal_rate), float(ending_balance_goal),
					spending_target=_spend_target_ms)
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
			_base_sched = st.session_state.get('_base_withdrawal_schedule', [])
			_spending_target = _base_sched[0] if _base_sched else 0.0
			dist_results = store_distribution_results(results, all_yearly_df, sim_mode_label,
				float(ending_balance_goal), spending_target=_spending_target)
			for k, v in dist_results.items():
				st.session_state[k] = v
			if is_historical:
				st.session_state['window_start_dates'] = {i: d.strftime('%Y-%m') for i, d in enumerate(window_start_dates)}
			st.session_state.pop('multi_scenario_results', None)

		# Store sim config for spending finder
		st.session_state['_sim_params'] = sim_params
		st.session_state['_sim_years'] = sim_years
		st.session_state['_is_historical'] = is_historical
		st.session_state['_monte_carlo_runs'] = int(monte_carlo_runs)
		st.session_state['_inheritor_marginal_rate'] = float(inheritor_marginal_rate)
		st.session_state['_ending_balance_goal'] = float(ending_balance_goal)

	# ── Multi-scenario comparison ────────────────────────────────
	if 'multi_scenario_results' in st.session_state:
		multi_results = st.session_state['multi_scenario_results']
		if len(multi_results) > 1:
			st.subheader('Scenario Comparison')
			# Convert multi_results to the common format for _display_comparison
			comparison_scenarios = []
			for sc in multi_results:
				median_by_year = sc['all_yearly_df'].groupby('year')[['total_portfolio', 'after_tax_spending']].median()
				comparison_scenarios.append({
					'name': sc['name'],
					'percentile_rows': sc['percentile_rows'],
					'spending_percentiles': sc['spending_percentiles'],
					'pct_non_positive': sc['pct_non_positive'],
					'median_yearly_years': median_by_year.index.tolist(),
					'median_yearly_portfolio': median_by_year['total_portfolio'].tolist(),
					'median_yearly_spending': median_by_year['after_tax_spending'].tolist(),
				})
			_display_comparison(comparison_scenarios, currency_fmt, key_suffix='multi')

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

		all_yearly = st.session_state['mc_all_yearly']

		# ── Decompose spending into base + per-goal using proportional allocation ──
		_goals = st.session_state.get('_add_goal_inputs', [])
		_active_goals = [g for g in _goals if len(g) >= 4 and g[1] > 0]
		_base_schedule = st.session_state.get('_base_withdrawal_schedule', [])
		_combined_schedule = st.session_state.get('_withdrawal_schedule', [])
		if _base_schedule and _combined_schedule:
			# Allocate actual spending to goals, capping each at its max contribution.
			# Essential goals: capped at target (never scale up).
			# Flexible goals with cap: capped at target * (1 + cap/100).
			# Flexible goals without cap: proportional (no ceiling).
			# Base gets the remainder.
			goal_spending_cols = []
			goal_wd_cols = []
			for gi, (g_label, g_amount, g_begin, g_end, *_gx) in enumerate(_active_goals):
				g_priority = _gx[0] if _gx else 'Essential'
				g_cap = _gx[1] if len(_gx) > 1 else -1.0
				# Determine per-year ceiling for this goal's after-tax share
				if g_priority in ('Essential', 'Need'):
					goal_ceiling = g_amount  # essential: never exceeds target
				elif g_cap >= 0:
					goal_ceiling = g_amount * (1.0 + g_cap / 100.0)
				else:
					goal_ceiling = None  # no ceiling
				def _goal_pct(yr, _amt=g_amount, _bg=g_begin, _ed=g_end):
					if yr < 1 or yr > len(_combined_schedule):
						return 0.0
					if _bg <= yr <= _ed and _combined_schedule[yr - 1] > 0:
						return _amt / _combined_schedule[yr - 1]
					return 0.0
				col_sp = f'goal_{gi}_after_tax_spending'
				col_wd = f'goal_{gi}_withdrawal_used'
				all_yearly[col_sp] = (
					all_yearly['after_tax_spending'] * all_yearly['year'].map(_goal_pct))
				all_yearly[col_wd] = (
					all_yearly['withdrawal_used'] * all_yearly['year'].map(_goal_pct))
				if goal_ceiling is not None:
					all_yearly[col_sp] = all_yearly[col_sp].clip(upper=goal_ceiling)
					all_yearly[col_wd] = all_yearly[col_wd].clip(upper=goal_ceiling)
				goal_spending_cols.append(col_sp)
				goal_wd_cols.append(col_wd)
			# Base = total minus all goal allocations
			all_yearly['base_after_tax_spending'] = (
				all_yearly['after_tax_spending'] - all_yearly[goal_spending_cols].sum(axis=1)).clip(lower=0)
			all_yearly['base_withdrawal_used'] = (
				all_yearly['withdrawal_used'] - all_yearly[goal_wd_cols].sum(axis=1)).clip(lower=0)

		# Compute spending percentiles (used by PDF report and scenario comparison)
		run_spending = all_yearly.groupby('run').agg(
			total_withdrawal_used=('withdrawal_used', 'sum'),
			total_after_tax_spending=('after_tax_spending', 'sum'),
			years_in_run=('year', 'count'),
		)
		run_spending['avg_annual_withdrawal'] = run_spending['total_withdrawal_used'] / run_spending['years_in_run']
		run_spending['avg_annual_after_tax_spending'] = run_spending['total_after_tax_spending'] / run_spending['years_in_run']
		spending_pct_rows = []
		for p in [0, 10, 25, 50, 75, 90, 100]:
			spending_pct_rows.append({
				'percentile': p,
				'avg_annual_withdrawal': np.percentile(run_spending['avg_annual_withdrawal'], p),
				'avg_annual_after_tax_spending': np.percentile(run_spending['avg_annual_after_tax_spending'], p),
				'total_lifetime_withdrawal': np.percentile(run_spending['total_withdrawal_used'], p),
				'total_lifetime_after_tax_spending': np.percentile(run_spending['total_after_tax_spending'], p),
			})
		st.session_state['mc_spending_pct_rows'] = spending_pct_rows

		has_goal_breakdown = 'base_after_tax_spending' in all_yearly.columns and _active_goals

		# ── View mode toggle ──
		view_mode = st.radio('Results view', ['Client View', 'Advisor View'],
			horizontal=True, key='results_view_mode', index=0)

		if view_mode == 'Client View':
			_render_client_view(
				success_rate=1.0 - st.session_state.get('mc_pct_non_positive', 0.0),
				spending_pct_rows=spending_pct_rows,
				percentile_rows=pct_rows,
				all_yearly=all_yearly,
				currency_fmt=currency_fmt,
				has_goal_breakdown=has_goal_breakdown,
				active_goals=_active_goals,
				base_schedule=_base_schedule,
				run_spending=run_spending,
				inheritor_marginal_rate=float(inheritor_marginal_rate),
				withdrawal_schedule=withdrawal_schedule,
				window_start_dates=st.session_state.get('window_start_dates'),
				funded_goals=st.session_state.get('_funded_goals'),
			)
		else:
			# Base spending cards (always shown)
			spending_pctiles = [0, 10, 25, 50, 75, 90, 100]
			if has_goal_breakdown:
				run_base = all_yearly.groupby('run').agg(
					total_base_wd=('base_withdrawal_used', 'sum'),
					total_base_spending=('base_after_tax_spending', 'sum'),
					years_in_run=('year', 'count'),
				)
				run_base['avg_annual_base_wd'] = run_base['total_base_wd'] / run_base['years_in_run']
				run_base['avg_annual_base_spending'] = run_base['total_base_spending'] / run_base['years_in_run']
				base_sched_avg = np.mean(_base_schedule) if _base_schedule else 0
				base_spending_series = run_base['avg_annual_base_spending']
			else:
				sched = sim_params.get('withdrawal_schedule', [])
				base_sched_avg = np.mean(sched) if sched else 0
				base_spending_series = run_spending['avg_annual_after_tax_spending']

			base_target_str = currency_fmt.format(base_sched_avg) if base_sched_avg > 0 else ''
			if base_target_str:
				st.markdown(f'#### Base spending — {base_target_str}/yr target')
			else:
				st.markdown('#### Base spending')
			_three_card_summary(base_spending_series, base_sched_avg, currency_fmt)
			if st.checkbox('Show detailed table', value=False, key='show_base_table'):
				if has_goal_breakdown:
					base_pct_rows = []
					for p in spending_pctiles:
						base_pct_rows.append({
							'percentile': p,
							'avg_annual_withdrawal': np.percentile(run_base['avg_annual_base_wd'], p),
							'avg_annual_after_tax_spending': np.percentile(run_base['avg_annual_base_spending'], p),
							'total_lifetime_withdrawal': np.percentile(run_base['total_base_wd'], p),
							'total_lifetime_after_tax_spending': np.percentile(run_base['total_base_spending'], p),
						})
					st.dataframe(pd.DataFrame(base_pct_rows).style.format({
						'percentile': lambda x: f"{int(x)}th",
						'avg_annual_withdrawal': currency_fmt,
						'avg_annual_after_tax_spending': currency_fmt,
						'total_lifetime_withdrawal': currency_fmt,
						'total_lifetime_after_tax_spending': currency_fmt,
					}))
				else:
					st.dataframe(pd.DataFrame(spending_pct_rows).style.format({
						'percentile': lambda x: f"{int(x)}th",
						'avg_annual_withdrawal': currency_fmt,
						'avg_annual_after_tax_spending': currency_fmt,
						'total_lifetime_withdrawal': currency_fmt,
						'total_lifetime_after_tax_spending': currency_fmt,
					}))

			# Per-goal spending (only when additional goals are active)
			if has_goal_breakdown and st.checkbox('Show per-goal spending breakdown', value=True, key='show_per_goal_cards'):
				for gi, (g_label, g_amount, g_begin, g_end, *_gx) in enumerate(_active_goals):
					col_spending = f'goal_{gi}_after_tax_spending'
					col_wd = f'goal_{gi}_withdrawal_used'
					if col_spending not in all_yearly.columns:
						continue
					num_years = g_end - g_begin + 1
					goal_name = g_label or f'Goal {gi + 1}'
					st.markdown(f'#### {goal_name} — {currency_fmt.format(g_amount)}/yr, years {g_begin}–{g_end} ({num_years} yrs)')
					goal_years = all_yearly[all_yearly['year'].between(g_begin, g_end)]
					if goal_years.empty:
						continue
					run_goal = goal_years.groupby('run').agg(
						total_goal_wd=(col_wd, 'sum'),
						total_goal_spending=(col_spending, 'sum'),
						years_in_goal=('year', 'count'),
					)
					run_goal['avg_annual_goal_wd'] = run_goal['total_goal_wd'] / run_goal['years_in_goal']
					run_goal['avg_annual_goal_spending'] = run_goal['total_goal_spending'] / run_goal['years_in_goal']
					_three_card_summary(run_goal['avg_annual_goal_spending'],
										 g_amount, currency_fmt)
					if st.checkbox('Show detailed table', value=False, key=f'show_goal_{gi}_table'):
						goal_pct_rows = []
						for p in spending_pctiles:
							goal_pct_rows.append({
								'percentile': p,
								'avg_annual_gross': np.percentile(run_goal['avg_annual_goal_wd'], p),
								'avg_annual_after_tax': np.percentile(run_goal['avg_annual_goal_spending'], p),
								f'total_gross ({num_years} yrs)': np.percentile(run_goal['total_goal_wd'], p),
								f'total_after_tax ({num_years} yrs)': np.percentile(run_goal['total_goal_spending'], p),
							})
						st.dataframe(pd.DataFrame(goal_pct_rows).style.format({
							'percentile': lambda x: f"{int(x)}th",
							'avg_annual_gross': currency_fmt,
							'avg_annual_after_tax': currency_fmt,
							f'total_gross ({num_years} yrs)': currency_fmt,
							f'total_after_tax ({num_years} yrs)': currency_fmt,
						}))

			# ── Separately Funded Goals ──
			_funded = st.session_state.get('_funded_goals', [])
			if _funded and 'goal_taxable_balance' in all_yearly.columns:
				st.markdown('---')
				st.subheader('Separately Funded Goals')
				for fg in _funded:
					num_goal_yrs = fg['end_year'] - fg['begin_year'] + 1
					total_spending = fg['annual_amount'] * num_goal_yrs
					st.markdown(f"#### {fg['label']} — {currency_fmt.format(fg['annual_amount'])}/yr, "
						f"years {fg['begin_year']}–{fg['end_year']} ({num_goal_yrs} yrs)")

					# 3-card summary: Cost Today / Guaranteed Spending / Median Surplus
					last_goal_year = fg['end_year']
					last_yr_data = all_yearly[all_yearly['year'] == min(last_goal_year, all_yearly['year'].max())]
					if not last_yr_data.empty:
						surplus_0 = float(np.percentile(
							last_yr_data['goal_taxable_balance'] + last_yr_data['goal_tda_balance'], 0))
						surplus_50 = float(np.percentile(
							last_yr_data['goal_taxable_balance'] + last_yr_data['goal_tda_balance'], 50))
						surplus_90 = float(np.percentile(
							last_yr_data['goal_taxable_balance'] + last_yr_data['goal_tda_balance'], 90))
					else:
						surplus_0 = surplus_50 = surplus_90 = 0.0
					cards = [
						{'icon': '\U0001f4b0', 'title': 'Set Aside Today',
						 'val': fg['total_cost'], 'sub': f"Funds {currency_fmt.format(total_spending)} total",
						 'border': '#b8860b', 'bg': '#fdf6e3'},
						{'icon': '\u2705', 'title': 'Annual Spending',
						 'val': fg['annual_amount'], 'sub': f"Guaranteed {num_goal_yrs} yrs ({fg['fund_stock_pct']}% stocks)",
						 'border': '#16a34a', 'bg': '#f0fdf4'},
						{'icon': '\U0001f4c8', 'title': 'Median Surplus',
						 'val': surplus_50, 'sub': f"Goal account balance after yr {last_goal_year}",
						 'border': '#2563eb', 'bg': '#eff6ff'},
					]
					card_htmls = []
					for card in cards:
						card_htmls.append(
							f'<div style="flex:1; border:1px solid {card["border"]}; border-left:4px solid {card["border"]}; '
							f'border-radius:8px; padding:16px 14px; background:{card["bg"]}; text-align:center;">'
							f'<div style="font-size:1.5em; margin-bottom:4px;">{card["icon"]}</div>'
							f'<div style="font-weight:600; color:#374151; margin-bottom:2px;">{card["title"]}</div>'
							f'<div style="font-size:1.6em; font-weight:700; color:#111827;">{currency_fmt.format(card["val"])}</div>'
							f'<div style="color:#6b7280; font-size:0.82em; margin-top:4px;">{card["sub"]}</div>'
							f'</div>')
					st.markdown(
						f'<div style="display:flex; gap:12px; align-items:stretch; max-width:60%;">{"".join(card_htmls)}</div>',
						unsafe_allow_html=True)

					# Source breakdown
					src_parts = []
					if fg['fund_taxable'] > 0:
						src_parts.append(f"Taxable: {currency_fmt.format(fg['fund_taxable'])}")
					if fg['fund_tda1'] > 0:
						src_parts.append(f"TDA P1: {currency_fmt.format(fg['fund_tda1'])}")
					if fg['fund_tda2'] > 0:
						src_parts.append(f"TDA P2: {currency_fmt.format(fg['fund_tda2'])}")
					source_total = fg['fund_taxable'] + fg['fund_tda1'] + fg['fund_tda2']
					if src_parts:
						st.caption(f"Sources: {' | '.join(src_parts)} = {currency_fmt.format(source_total)}")
					if abs(source_total - fg['total_cost']) > 1.0:
						st.warning(f"Source total ({currency_fmt.format(source_total)}) does not match "
							f"calculated cost ({currency_fmt.format(fg['total_cost'])}). "
							f"Difference: {currency_fmt.format(source_total - fg['total_cost'])}")

					# Surplus distribution table
					if st.checkbox('Show cost & surplus detail', value=False, key=f'show_funded_goal_detail_{fg["label"]}'):
						# Cost breakdown table
						if fg.get('year_costs'):
							st.caption('Set-aside cost by goal year')
							cost_df = pd.DataFrame(fg['year_costs']).set_index('year')
							st.dataframe(cost_df.style.format({
								'growth_0': '${:,.4f}', 'cost_per_dollar': '${:,.4f}',
								'goal': currency_fmt, 'set_aside': currency_fmt,
							}))
						# Surplus percentile table
						if not last_yr_data.empty:
							st.caption(f'Goal account surplus distribution after year {last_goal_year}')
							surplus_vals = last_yr_data['goal_taxable_balance'] + last_yr_data['goal_tda_balance']
							surplus_rows = []
							for p in [0, 10, 25, 50, 75, 90, 100]:
								surplus_rows.append({'percentile': p, 'surplus': float(np.percentile(surplus_vals, p))})
							st.dataframe(pd.DataFrame(surplus_rows).style.format({
								'percentile': lambda x: f"{int(x)}th", 'surplus': currency_fmt,
							}))

				# Shadow account balance over time (all funded goals combined)
				st.markdown('#### Goal Account Balances Over Time')
				goal_pcts_data = {}
				for p_label, p_val in [('0th', 0), ('25th', 25), ('50th', 50), ('75th', 75), ('90th', 90)]:
					goal_pcts_data[p_label] = all_yearly.groupby('year').apply(
						lambda g: float(np.percentile(
							g['goal_taxable_balance'] + g['goal_tda_balance'], p_val)))
				goal_pcts_df = pd.DataFrame(goal_pcts_data)
				interactive_line_chart(goal_pcts_df, y_title='Goal Account Balance')

			ending_goal = float(st.session_state.get('ending_balance_goal', 1.0))
			run_ending_portfolios = all_yearly.groupby('run')['total_portfolio'].last()
			target_ending_value = st.number_input('Explore: chance of ending above...', value=ending_goal, step=50000.0, key='target_ending_val')
			pct_at_or_above = float((run_ending_portfolios >= target_ending_value).mean()) * 100
			st.metric('Chance of ending with at least this amount', f"{pct_at_or_above:.1f}%")
			spending_pivot = all_yearly.pivot(index='year', columns='run', values='after_tax_spending')
			spending_pivot.columns = [f'Run {c}' for c in spending_pivot.columns]
			spending_csv = spending_pivot.to_csv()
			st.download_button('Download after-tax spending (all runs) as CSV', spending_csv, file_name='after_tax_spending_all_runs.csv', mime='text/csv')

			# Year-by-year median table (50th percentile across all simulations)
			st.subheader('Year-by-year median across all runs')
			st.caption('Each value is the 50th percentile across all simulation runs for that year — not a single run.')
			median_cols = ['age_p1', 'withdrawal_used', 'after_tax_spending']
			if has_goal_breakdown:
				col_names = ['Age', 'Withdrawal Target (Total)', 'After-Tax Spending (Total)']
			else:
				col_names = ['Age', 'Withdrawal Target', 'After-Tax Spending']
			col_formats = {}
			# Insert base + per-goal columns when goals are active
			if has_goal_breakdown:
				median_cols += ['base_after_tax_spending']
				col_names += ['Base Spending']
				col_formats['Base Spending'] = currency_fmt
				for gi, (g_label, g_amount, g_begin, g_end, *_gx) in enumerate(_active_goals):
					col_key = f'goal_{gi}_after_tax_spending'
					if col_key in all_yearly.columns:
						median_cols.append(col_key)
						goal_col_name = g_label or f'Goal {gi + 1}'
						col_names.append(goal_col_name)
						col_formats[goal_col_name] = currency_fmt
			median_cols += ['total_portfolio',
						   'end_taxable_total', 'end_tda_total', 'end_roth']
			col_names += ['Total Portfolio', 'Taxable', 'TDA', 'Roth']
			# Add goal account columns if present
			if 'goal_taxable_balance' in all_yearly.columns and st.session_state.get('_funded_goals'):
				median_cols += ['goal_taxable_balance', 'goal_tda_balance', 'goal_liquidation']
				col_names += ['Goal Taxable', 'Goal TDA', 'Goal Liquidation']
				col_formats['Goal Taxable'] = currency_fmt
				col_formats['Goal TDA'] = currency_fmt
				col_formats['Goal Liquidation'] = currency_fmt
			median_cols += ['total_taxes', 'rmd_total', 'withdraw_from_tda', 'withdraw_from_roth',
						   'withdraw_from_taxable_net', 'ss_income_total', 'pension_income_total',
						   'pension_income_real', 'pension_erosion', 'ordinary_taxable_income',
						   'capital_gains', 'effective_tax_rate_calc']
			col_names += ['Total Taxes', 'RMDs', 'Withdraw TDA', 'Withdraw Roth',
						  'Withdraw Taxable', 'SS Income', 'Pension (Nominal)',
						  'Pension (Real)', 'Pension Erosion', 'Ordinary Income',
						  'Cap Gains', 'Eff Tax Rate']
			all_yearly['effective_tax_rate_calc'] = (
				all_yearly['total_taxes'] /
				(all_yearly['ordinary_taxable_income'] + all_yearly['capital_gains']).replace(0, np.nan)
			).fillna(0.0)
			yearly_median = all_yearly.groupby('year')[median_cols].median()
			yearly_median.columns = col_names
			_wd_col = 'Withdrawal Target (Total)' if has_goal_breakdown else 'Withdrawal Target'
			_sp_col = 'After-Tax Spending (Total)' if has_goal_breakdown else 'After-Tax Spending'
			col_formats.update({
				'Age': '{:.0f}',
				_wd_col: currency_fmt, _sp_col: currency_fmt,
				'Total Portfolio': currency_fmt, 'Taxable': currency_fmt, 'TDA': currency_fmt, 'Roth': currency_fmt,
				'Total Taxes': currency_fmt, 'RMDs': currency_fmt,
				'Withdraw TDA': currency_fmt, 'Withdraw Roth': currency_fmt, 'Withdraw Taxable': currency_fmt,
				'SS Income': currency_fmt, 'Pension (Nominal)': currency_fmt, 'Pension (Real)': currency_fmt, 'Pension Erosion': currency_fmt,
				'Ordinary Income': currency_fmt, 'Cap Gains': currency_fmt,
				'Eff Tax Rate': '{:.2%}'.format,
			})
			st.dataframe(yearly_median.style.format(col_formats))

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

			st.subheader('Tax rates at selected percentile — all runs')
			rate_pctile = st.selectbox('Percentile', [10, 25, 50, 75, 90], index=2, key='tax_rate_pctile')
			q = rate_pctile / 100.0
			rate_cols = ['effective_tax_rate_calc', 'marginal_ordinary_rate', 'marginal_cap_gains_rate']
			rate_at_pctile = all_yearly.groupby('year')[rate_cols].quantile(q)
			rate_at_pctile.columns = ['Effective Rate', 'Marginal Ordinary', 'Marginal Cap Gains']
			st.dataframe(rate_at_pctile.style.format({
				'Effective Rate': '{:.2%}'.format,
				'Marginal Ordinary': '{:.2%}'.format,
				'Marginal Cap Gains': '{:.2%}'.format,
			}))
			interactive_line_chart(rate_at_pctile, y_title='Tax Rate', fmt='.1%')

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

		# Always compute median run sim_df (needed for last_summary CAGR in both views)
		run_ends = all_yearly.groupby('run')['total_portfolio'].last()
		median_val = run_ends.median()
		median_run_idx = int((run_ends - median_val).abs().idxmin())
		median_df = all_yearly[all_yearly['run'] == median_run_idx].drop(columns=['run', 'total_portfolio']).reset_index(drop=True)
		st.session_state['sim_df'] = median_df

		# Resolve view_mode for use outside this block
		view_mode = st.session_state.get('results_view_mode', 'Client View')

		if view_mode == 'Advisor View':
			st.markdown('---')
			# Interactive run selector
			window_dates = st.session_state.get('window_start_dates', {})
			run_ids = sorted(all_yearly['run'].unique())

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

	# Make view_mode accessible outside the mc block
	view_mode = st.session_state.get('results_view_mode', 'Client View')

	df = st.session_state.get('sim_df')
	if df is not None and sim_mode is not None:
		# Compute CAGR values unconditionally (needed by last_summary in both views)
		years_simulated = len(df)
		portfolio_growth_factor = (df['portfolio_return'] + 1.0).prod()
		roth_growth_factor = (df['roth_return_used'] + 1.0).prod()
		portfolio_cagr = (portfolio_growth_factor ** (1.0 / years_simulated) - 1.0) if years_simulated > 0 else 0.0
		roth_cagr = (roth_growth_factor ** (1.0 / years_simulated) - 1.0) if years_simulated > 0 else 0.0

	if df is not None and sim_mode is not None and view_mode == 'Advisor View':
		if sim_mode == 'historical':
			st.subheader('Historical simulation detail')

		last = df.iloc[-1]
		display_df = df.round(int(display_decimals)).copy()
		display_df['portfolio_return'] = df['portfolio_return']
		display_df['roth_return_used'] = df['roth_return_used']

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
		# Include base and per-goal spending columns if available
		_sr_goals = st.session_state.get('_add_goal_inputs', [])
		_sr_active_goals = [g for g in _sr_goals if len(g) >= 4 and g[1] > 0]
		_goal_detail_cols = []
		_goal_detail_fmts = {}
		if 'base_after_tax_spending' in display_df.columns:
			_goal_detail_cols.append('base_after_tax_spending')
			_goal_detail_fmts['base_after_tax_spending'] = currency_fmt
			for gi, (g_label, g_amount, g_begin, g_end, *_gx) in enumerate(_sr_active_goals):
				col_key = f'goal_{gi}_after_tax_spending'
				if col_key in display_df.columns:
					_goal_detail_cols.append(col_key)
					_goal_detail_fmts[col_key] = currency_fmt
		detail_col_order = [
			'year', 'age_p1', 'age_p2', 'withdrawal_used', 'after_tax_spending',
			] + _goal_detail_cols + [
			'portfolio_return', 'roth_return_used',
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
			'goal_taxable_balance', 'goal_tda_balance', 'goal_liquidation',
		]
		detail_col_order = [c for c in detail_col_order if c in display_df.columns]
		display_df = display_df[detail_col_order]
		_detail_fmts = {
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
			'goal_taxable_balance': currency_fmt, 'goal_tda_balance': currency_fmt, 'goal_liquidation': currency_fmt,
			'ordinary_tax_total': currency_fmt, 'capital_gains_tax': currency_fmt, 'niit_tax': currency_fmt, 'state_tax': currency_fmt, 'total_taxes': currency_fmt,
			'marginal_ordinary_rate': '{:.2%}'.format, 'marginal_cap_gains_rate': '{:.2%}'.format,
			'withdrawal_used': currency_fmt, 'after_tax_spending': currency_fmt,
			'portfolio_return': '{:.2%}'.format,
			'roth_return_used': '{:.2%}'.format
		}
		_detail_fmts.update(_goal_detail_fmts)
		st.dataframe(display_df.style.format(_detail_fmts))
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

	# Store latest summary in session for post-run saving (runs in both views)
	if df is not None and sim_mode is not None:
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
			med_taxes = float(df['total_taxes'].sum())
			pre_tax_total = med_taxable + med_tda + med_roth
			after_tax_total = med_taxable + med_roth + med_tda * max(0.0, 1.0 - inh_rate_save)
		summary_data = {
			'label': f"Roth fill→{roth_bracket_fill_rate*100:.0f}% for {roth_conversion_years} yrs" if roth_conversion_mode_ui == 'Fill to bracket'
			else f"conversion ${roth_conversion_amount:,.0f} for {roth_conversion_years} yrs, taxes from {roth_conversion_tax_source}",
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

	# ── Compare Saved Plans (Advisor View only) ──────────────────
	client = st.session_state.get('client_select', '')
	if view_mode == 'Advisor View' and client and client != '-- New Client --':
		plans_with_results = get_plans_with_results(client)
		if plans_with_results:
			st.markdown('---')
			st.subheader('Compare Saved Plans')
			selected_plans = st.multiselect('Select plans to compare',
				plans_with_results, key='compare_plans_select')
			if st.button('Compare', key='compare_plans_btn') and len(selected_plans) >= 2:
				comparison_scenarios = []
				for plan_name in selected_plans:
					r = load_plan_results(client, plan_name)
					if r is None:
						st.warning(f'Could not load results for {plan_name}')
						continue
					comparison_scenarios.append({
						'name': plan_name,
						'percentile_rows': r['percentile_rows'],
						'spending_percentiles': r['spending_percentiles'],
						'pct_non_positive': r['pct_non_positive'],
						'median_yearly_years': r['median_yearly']['years'],
						'median_yearly_portfolio': r['median_yearly']['portfolio'],
						'median_yearly_spending': r['median_yearly']['spending'],
					})
				if len(comparison_scenarios) >= 2:
					st.session_state['saved_plan_comparison'] = comparison_scenarios
			if st.session_state.get('saved_plan_comparison'):
				_display_comparison(st.session_state['saved_plan_comparison'],
					currency_fmt, key_suffix='saved')
		else:
			if sim_mode is not None:
				st.caption('No saved plans with simulation results yet. Run a simulation and click "Save Inputs" to save results.')

	# ── SS Claiming Age Analysis (Advisor View only) ─────────────
	if view_mode == 'Advisor View' and sim_mode is not None and 'mc_percentile_rows' in st.session_state:
		st.markdown('---')
		st.subheader('Social Security Claiming Age Analysis')

		is_couple = (filing_status_key == 'married_filing_jointly')
		has_p1_ss = float(ss_income_input) > 0
		has_p2_ss = float(ss_income_spouse_input) > 0
		_fra_p1 = int(ss_fra_age_p1)
		_fra_p2 = int(ss_fra_age_p2)
		p1_ages = list(range(max(62, int(start_age)), 71)) if has_p1_ss and int(start_age) <= 70 else []
		p2_ages = list(range(max(62, int(start_age_spouse)), 71)) if has_p2_ss and int(start_age_spouse) <= 70 else []

		if has_p1_ss and p1_ages:
			_fra_benefit_p1 = ss_back_calculate_fra_benefit(float(ss_income_input), int(ss_start_age_p1), _fra_p1)
		if has_p2_ss and p2_ages:
			_fra_benefit_p2 = ss_back_calculate_fra_benefit(float(ss_income_spouse_input), int(ss_start_age_p2), _fra_p2)

		def _ss_delta_fmt(v):
			if abs(v) < 1:
				return 'Baseline'
			return f'+${v:,.0f}' if v > 0 else f'-${abs(v):,.0f}'

		_run_grid = is_couple and len(p1_ages) > 0 and len(p2_ages) > 0

		if not p1_ages and not p2_ages:
			st.caption('No Social Security claiming ages to analyze (past age 70 or no SS benefit entered).')

		elif _run_grid:
			# ── Married: all P1 × P2 combinations ──
			n_combos = len(p1_ages) * len(p2_ages)
			st.markdown(
				f"**Person 1:** FRA benefit ${_fra_benefit_p1:,.0f}/yr (ages {p1_ages[0]}\u2013{p1_ages[-1]}) &nbsp;|&nbsp; "
				f"**Person 2:** FRA benefit ${_fra_benefit_p2:,.0f}/yr (ages {p2_ages[0]}\u2013{p2_ages[-1]})")
			st.caption(f'{n_combos} combinations will be simulated. '
				f'Baseline: Person 1 at {int(ss_start_age_p1)}, Person 2 at {int(ss_start_age_p2)}.')

			with st.expander('Preview: benefits at each claiming age'):
				_prev1, _prev2 = st.columns(2)
				with _prev1:
					st.markdown('**Person 1**')
					_prev_rows_p1 = []
					for ca in p1_ages:
						_ssb = compute_ss_benefits(
							_fra_benefit_p1 * ss_adjustment_factor(ca, _fra_p1), ca, _fra_p1,
							float(ss_income_spouse_input), int(ss_start_age_p2), _fra_p2,
							filing_status_key,
							sim_start_age_p1=int(start_age), sim_start_age_p2=int(start_age_spouse))
						_own = _ssb['ss_own_p1']
						_topup = _ssb['ss_spousal_topup_p1']
						_prev_rows_p1.append({'Age': ca, 'Own': f'${_own:,.0f}',
							'Spousal': f'${_topup:,.0f}' if _topup > 0 else '—',
							'Total': f'${_own + _topup:,.0f}'})
					st.table(pd.DataFrame(_prev_rows_p1).set_index('Age'))
				with _prev2:
					st.markdown('**Person 2**')
					_prev_rows_p2 = []
					for ca in p2_ages:
						_ssb = compute_ss_benefits(
							float(ss_income_input), int(ss_start_age_p1), _fra_p1,
							_fra_benefit_p2 * ss_adjustment_factor(ca, _fra_p2), ca, _fra_p2,
							filing_status_key,
							sim_start_age_p1=int(start_age), sim_start_age_p2=int(start_age_spouse))
						_own = _ssb['ss_own_p2']
						_topup = _ssb['ss_spousal_topup_p2']
						_prev_rows_p2.append({'Age': ca, 'Own': f'${_own:,.0f}',
							'Spousal': f'${_topup:,.0f}' if _topup > 0 else '—',
							'Total': f'${_own + _topup:,.0f}'})
					st.table(pd.DataFrame(_prev_rows_p2).set_index('Age'))
				st.caption('Spousal top-up = max(0, 50% of other\'s PIA − own PIA), reduced for early claiming. '
					'Shown with other person at their current claiming age.')

			if st.button('Run Claiming Age Analysis', key='run_ss_analysis'):
				sim_years = int(years)
				is_historical = return_mode == 'Historical (master_global_factors)'
				grid_results = []
				st.session_state.pop('ss_analysis_results', None)

				if is_historical:
					all_hist_windows, _ = get_all_historical_windows(sim_years)
					n_windows = len(all_hist_windows)
					target_per = max(50, 20000 // n_combos)
					stride = max(1, n_windows // target_per)
					sampled_indices = list(range(0, n_windows, stride))
				else:
					mc_runs = max(25, min(200, 10000 // n_combos))

				progress = st.progress(0, text='SS analysis: starting...')
				combo_idx = 0
				for p1_ca in p1_ages:
					p1_adj = _fra_benefit_p1 * ss_adjustment_factor(p1_ca, _fra_p1)
					for p2_ca in p2_ages:
						progress.progress(combo_idx / n_combos,
							text=f'P1={p1_ca}, P2={p2_ca} ({combo_idx+1}/{n_combos})')
						p2_adj = _fra_benefit_p2 * ss_adjustment_factor(p2_ca, _fra_p2)

						sp = dict(sim_params)
						sp['ss_income_annual'] = p1_adj
						sp['ss_start_age_p1'] = p1_ca
						sp['ss_income_spouse_annual'] = p2_adj
						sp['ss_start_age_p2'] = p2_ca

						if is_historical:
							results = []
							_run_spending = []
							for idx in sampled_indices:
								stock_rets, bond_rets = all_hist_windows[idx]
								run_pp = compute_run_pp_factors(idx, sim_years)
								df_run = simulate_withdrawals(
									years=sim_years, stock_return_series=stock_rets,
									bond_return_series=bond_rets, pp_factors_run=run_pp, **sp)
								df_run['total_portfolio'] = (df_run['end_taxable_total']
									+ df_run['end_tda_total'] + df_run['end_roth'])
								results.append(compute_summary_metrics(df_run, float(inheritor_marginal_rate)))
								_run_spending.append(df_run['after_tax_spending'].mean())
							mc_df = pd.DataFrame(results)
							median_spending = float(np.median(_run_spending))
						else:
							results, _ay = run_monte_carlo(
								num_runs=mc_runs, years=sim_years,
								inheritor_rate=float(inheritor_marginal_rate),
								taxable_log_drift=float(taxable_log_drift),
								taxable_log_volatility=float(taxable_log_volatility),
								bond_log_drift=float(bond_log_drift),
								bond_log_volatility=float(bond_log_volatility),
								**sp)
							mc_df = pd.DataFrame(results)
							median_spending = float(_ay.groupby('run')['after_tax_spending'].mean().median())
							del _ay

						grid_results.append({
							'p1_age': p1_ca, 'p2_age': p2_ca,
							'p1_benefit': p1_adj, 'p2_benefit': p2_adj,
							'median_ending': float(np.percentile(mc_df['after_tax_end'], 50)),
							'median_spending': median_spending,
						})
						combo_idx += 1

				progress.progress(1.0, text='SS analysis: complete!')
				st.session_state['ss_analysis_grid'] = grid_results
				st.session_state['ss_analysis_meta'] = {
					'p1_ages': p1_ages, 'p2_ages': p2_ages,
					'fra_benefit_p1': _fra_benefit_p1, 'fra_benefit_p2': _fra_benefit_p2,
					'fra_p1': _fra_p1, 'fra_p2': _fra_p2,
					'baseline_p1': int(ss_start_age_p1), 'baseline_p2': int(ss_start_age_p2),
				}

			if 'ss_analysis_grid' in st.session_state:
				grid = st.session_state['ss_analysis_grid']
				meta = st.session_state['ss_analysis_meta']
				grid_df = pd.DataFrame(grid)

				# Find baseline — match current inputs, or closest combo
				bl_p1, bl_p2 = meta['baseline_p1'], meta['baseline_p2']
				bl_row = next((r for r in grid if r['p1_age'] == bl_p1 and r['p2_age'] == bl_p2), None)
				if bl_row is None:
					bl_row = min(grid, key=lambda r: abs(r['p1_age'] - bl_p1) + abs(r['p2_age'] - bl_p2))
				bl_ending = bl_row['median_ending']
				bl_spending = bl_row['median_spending']

				grid_df['ending_delta'] = grid_df['median_ending'] - bl_ending
				grid_df['spending_delta'] = grid_df['median_spending'] - bl_spending

				def _ss_age_label(age, fra_b, fra_a):
					return f"{age} (${fra_b * ss_adjustment_factor(age, fra_a):,.0f})"

				p1_labels = [_ss_age_label(a, meta['fra_benefit_p1'], meta['fra_p1']) for a in meta['p1_ages']]
				p2_labels = [_ss_age_label(a, meta['fra_benefit_p2'], meta['fra_p2']) for a in meta['p2_ages']]

				# Ending balance grids: absolute values then delta
				st.markdown('**Median Ending Balance**')
				end_abs_pivot = grid_df.pivot(index='p1_age', columns='p2_age', values='median_ending')
				end_abs_pivot.index = p1_labels
				end_abs_pivot.columns = p2_labels
				end_abs_pivot.index.name = 'Person 1 ↓  /  Person 2 →'
				st.dataframe(end_abs_pivot.style.format('${:,.0f}')
					.background_gradient(cmap='RdYlGn', axis=None))

				st.markdown(f'**Median Ending Balance vs Baseline** (P1={bl_row["p1_age"]}, P2={bl_row["p2_age"]})')
				end_pivot = grid_df.pivot(index='p1_age', columns='p2_age', values='ending_delta')
				end_pivot.index = p1_labels
				end_pivot.columns = p2_labels
				end_pivot.index.name = 'Person 1 ↓  /  Person 2 →'
				st.dataframe(end_pivot.style.format(_ss_delta_fmt)
					.background_gradient(cmap='RdYlGn', axis=None))

				# Spending grids: absolute values then delta
				st.markdown('**Median Annual Spending**')
				spend_abs_pivot = grid_df.pivot(index='p1_age', columns='p2_age', values='median_spending')
				spend_abs_pivot.index = p1_labels
				spend_abs_pivot.columns = p2_labels
				spend_abs_pivot.index.name = 'Person 1 ↓  /  Person 2 →'
				st.dataframe(spend_abs_pivot.style.format('${:,.0f}')
					.background_gradient(cmap='RdYlGn', axis=None))

				st.markdown(f'**Median Annual Spending vs Baseline** (P1={bl_row["p1_age"]}, P2={bl_row["p2_age"]})')
				spend_pivot = grid_df.pivot(index='p1_age', columns='p2_age', values='spending_delta')
				spend_pivot.index = p1_labels
				spend_pivot.columns = p2_labels
				spend_pivot.index.name = 'Person 1 ↓  /  Person 2 →'
				st.dataframe(spend_pivot.style.format(_ss_delta_fmt)
					.background_gradient(cmap='RdYlGn', axis=None))

				# Best combination
				best_end = grid_df.loc[grid_df['median_ending'].idxmax()]
				best_spend = grid_df.loc[grid_df['median_spending'].idxmax()]
				st.markdown(f"**Highest ending balance:** P1 at {int(best_end['p1_age'])}, "
					f"P2 at {int(best_end['p2_age'])} "
					f"({_ss_delta_fmt(best_end['ending_delta'])} vs baseline)")
				st.markdown(f"**Highest spending:** P1 at {int(best_spend['p1_age'])}, "
					f"P2 at {int(best_spend['p2_age'])} "
					f"({_ss_delta_fmt(best_spend['spending_delta'])} vs baseline)")

				# Breakeven tables side by side
				st.subheader('Cumulative SS Benefits by Age')
				_be1, _be2 = st.columns(2)
				with _be1:
					st.markdown('**Person 1**')
					be1 = pd.DataFrame(ss_breakeven_table(meta['fra_benefit_p1'], meta['fra_p1'],
						claiming_ages=meta['p1_ages'])).rename(columns={
						'claiming_age': 'Claim Age', 'annual_benefit': 'Annual',
						'cumulative_75': 'By 75', 'cumulative_80': 'By 80',
						'cumulative_85': 'By 85', 'cumulative_90': 'By 90',
						'cumulative_95': 'By 95'}).set_index('Claim Age')
					st.dataframe(be1.style.format('${:,.0f}'))
				with _be2:
					st.markdown('**Person 2**')
					be2 = pd.DataFrame(ss_breakeven_table(meta['fra_benefit_p2'], meta['fra_p2'],
						claiming_ages=meta['p2_ages'])).rename(columns={
						'claiming_age': 'Claim Age', 'annual_benefit': 'Annual',
						'cumulative_75': 'By 75', 'cumulative_80': 'By 80',
						'cumulative_85': 'By 85', 'cumulative_90': 'By 90',
						'cumulative_95': 'By 95'}).set_index('Claim Age')
					st.dataframe(be2.style.format('${:,.0f}'))

		else:
			# ── Single filer or only one person has SS ──
			if p1_ages:
				_person_label = 'Person 1'
				_fra_benefit = _fra_benefit_p1
				_claim_ages = p1_ages
				_fra = _fra_p1
				_ss_claim_age = int(ss_start_age_p1)
				_ss_benefit = float(ss_income_input)
			else:
				_person_label = 'Person 2'
				_fra_benefit = _fra_benefit_p2
				_claim_ages = p2_ages
				_fra = _fra_p2
				_ss_claim_age = int(ss_start_age_p2)
				_ss_benefit = float(ss_income_spouse_input)

			st.markdown(f"**Current input:** ${_ss_benefit:,.0f}/yr at age {_ss_claim_age} &nbsp;|&nbsp; "
				f"**Back-calculated FRA benefit (age {_fra}):** ${_fra_benefit:,.0f}/yr")

			with st.expander('Preview: benefit at each claiming age'):
				st.table(pd.DataFrame([
					{'Claiming Age': ca, 'Adjustment': f'{ss_adjustment_factor(ca, _fra):.2%}',
					 'Annual Benefit': f'${_fra_benefit * ss_adjustment_factor(ca, _fra):,.0f}'}
					for ca in _claim_ages]).set_index('Claiming Age'))

			if st.button('Run Claiming Age Analysis', key='run_ss_analysis'):
				sim_years = int(years)
				is_historical = return_mode == 'Historical (master_global_factors)'
				ss_analysis_results = []
				st.session_state.pop('ss_analysis_grid', None)
				st.session_state.pop('ss_analysis_meta', None)

				if is_historical:
					hist_windows, _ = get_all_historical_windows(sim_years)

				n_ages = len(_claim_ages)
				progress_bar = st.progress(0, text='SS analysis: starting...')
				for i, ca in enumerate(_claim_ages):
					progress_bar.progress(i / n_ages, text=f'SS analysis: age {ca}...')
					adj_benefit = _fra_benefit * ss_adjustment_factor(ca, _fra)
					sp = dict(sim_params)
					if _person_label == 'Person 1':
						sp['ss_income_annual'] = adj_benefit
						sp['ss_start_age_p1'] = ca
					else:
						sp['ss_income_spouse_annual'] = adj_benefit
						sp['ss_start_age_p2'] = ca

					if is_historical:
						results = []
						all_yearly = []
						for run_idx, (stock_rets, bond_rets) in enumerate(hist_windows):
							run_pp = compute_run_pp_factors(run_idx, sim_years)
							df_run = simulate_withdrawals(
								years=sim_years, stock_return_series=stock_rets,
								bond_return_series=bond_rets, pp_factors_run=run_pp, **sp)
							df_run['total_portfolio'] = df_run['end_taxable_total'] + df_run['end_tda_total'] + df_run['end_roth']
							results.append(compute_summary_metrics(df_run, float(inheritor_marginal_rate)))
							df_run['run'] = run_idx
							all_yearly.append(df_run)
						all_yearly_df = pd.concat(all_yearly, ignore_index=True)
					else:
						mc_runs = min(int(monte_carlo_runs), 200)
						results, all_yearly_df = run_monte_carlo(
							num_runs=mc_runs, years=sim_years,
							inheritor_rate=float(inheritor_marginal_rate),
							taxable_log_drift=float(taxable_log_drift),
							taxable_log_volatility=float(taxable_log_volatility),
							bond_log_drift=float(bond_log_drift),
							bond_log_volatility=float(bond_log_volatility),
							**sp)

					_base_sched_ss = st.session_state.get('_base_withdrawal_schedule', [])
					_spend_target_ss = _base_sched_ss[0] if _base_sched_ss else 0.0
					summary = compute_scenario_summary(
						f'Age {ca} (${adj_benefit:,.0f})', results, all_yearly_df,
						float(inheritor_marginal_rate), float(ending_balance_goal),
						spending_target=_spend_target_ss)
					# Store claiming age and benefit for display
					summary['_claiming_age'] = ca
					summary['_annual_benefit'] = adj_benefit
					_ayd = summary['all_yearly_df']
					_inh_rate = float(inheritor_marginal_rate)
					_ayd['after_tax_portfolio'] = (_ayd['end_taxable_total'] + _ayd['end_roth']
						+ _ayd['end_tda_total'] * max(0.0, 1.0 - _inh_rate))
					median_by_year = _ayd.groupby('year')[['total_portfolio', 'after_tax_portfolio', 'after_tax_spending']].median()
					summary['median_yearly_years'] = median_by_year.index.tolist()
					summary['median_yearly_portfolio'] = median_by_year['total_portfolio'].tolist()
					summary['median_yearly_portfolio_at'] = median_by_year['after_tax_portfolio'].tolist()
					summary['median_yearly_spending'] = median_by_year['after_tax_spending'].tolist()
					# Pre-tax ending: median of last-year total_portfolio across runs
					_run_ends = _ayd.groupby('run')['total_portfolio'].last()
					summary['_pretax_ending_p50'] = float(np.median(_run_ends))
					del summary['all_yearly_df']
					ss_analysis_results.append(summary)

				progress_bar.progress(1.0, text='SS analysis: complete!')
				st.session_state['ss_analysis_results'] = ss_analysis_results

			if st.session_state.get('ss_analysis_results'):
				sa = st.session_state['ss_analysis_results']
				# Find baseline row matching current claiming age
				bl_idx = next((i for i, r in enumerate(sa) if r['_claiming_age'] == _ss_claim_age),
					min(range(len(sa)), key=lambda i: abs(sa[i]['_claiming_age'] - _ss_claim_age)))
				bl = sa[bl_idx]
				bl_ending_at = next(r for r in bl['percentile_rows'] if r['percentile'] == 50)['after_tax_end']
				bl_ending_pt = bl.get('_pretax_ending_p50', bl_ending_at)
				bl_spending = next(r for r in bl['spending_percentiles'] if r['percentile'] == 50)['avg_annual_after_tax_spending']

				# Delta table
				table_rows = []
				for r in sa:
					p50_end_at = next(row for row in r['percentile_rows'] if row['percentile'] == 50)['after_tax_end']
					p50_end_pt = r.get('_pretax_ending_p50', p50_end_at)
					p50_spend = next(row for row in r['spending_percentiles'] if row['percentile'] == 50)['avg_annual_after_tax_spending']
					table_rows.append({
						'Claiming Age': r['_claiming_age'],
						'Annual Benefit': r['_annual_benefit'],
						'Pre-Tax Ending': p50_end_pt,
						'Pre-Tax \u0394': p50_end_pt - bl_ending_pt,
						'After-Tax Ending': p50_end_at,
						'After-Tax \u0394': p50_end_at - bl_ending_at,
						'Annual Spending': p50_spend,
						'Spending \u0394': p50_spend - bl_spending,
						'% Below Goal': r['pct_non_positive'] * 100,
					})
				delta_df = pd.DataFrame(table_rows).set_index('Claiming Age')
				def _delta_color(val):
					if isinstance(val, (int, float)):
						if val > 0: return 'color: green'
						if val < 0: return 'color: red'
					return ''
				st.dataframe(delta_df.style.format({
					'Annual Benefit': '${:,.0f}',
					'Pre-Tax Ending': '${:,.0f}',
					'Pre-Tax \u0394': _ss_delta_fmt,
					'After-Tax Ending': '${:,.0f}',
					'After-Tax \u0394': _ss_delta_fmt,
					'Annual Spending': '${:,.0f}',
					'Spending \u0394': _ss_delta_fmt,
					'% Below Goal': '{:.1f}%',
				}).map(_delta_color, subset=['Pre-Tax \u0394', 'After-Tax \u0394', 'Spending \u0394']))

				# Portfolio overlay charts
				st.subheader('Pre-Tax Portfolio Comparison (median)')
				overlay = pd.DataFrame()
				for r in sa:
					overlay[r['name']] = pd.Series(r['median_yearly_portfolio'],
						index=r['median_yearly_years'])
				interactive_line_chart(overlay, y_title='Pre-Tax Portfolio (Median)', zero_base=False)

				if all('median_yearly_portfolio_at' in r for r in sa):
					st.subheader('After-Tax Portfolio Comparison (median)')
					overlay_at = pd.DataFrame()
					for r in sa:
						overlay_at[r['name']] = pd.Series(r['median_yearly_portfolio_at'],
							index=r['median_yearly_years'])
					interactive_line_chart(overlay_at, y_title='After-Tax Portfolio (Median)', zero_base=False)

				# Spending overlay chart
				st.subheader('After-Tax Spending Comparison (median)')
				spend_overlay = pd.DataFrame()
				for r in sa:
					spend_overlay[r['name']] = pd.Series(r['median_yearly_spending'],
						index=r['median_yearly_years'])
				interactive_line_chart(spend_overlay, y_title='After-Tax Spending (Median)', zero_base=False)

				# Breakeven table
				st.subheader('Cumulative SS Benefits by Age')
				be_table = ss_breakeven_table(_fra_benefit, _fra, claiming_ages=_claim_ages)
				be_df = pd.DataFrame(be_table).rename(columns={
					'claiming_age': 'Claim Age', 'annual_benefit': 'Annual Benefit',
					'cumulative_75': 'By Age 75', 'cumulative_80': 'By Age 80',
					'cumulative_85': 'By Age 85', 'cumulative_90': 'By Age 90',
					'cumulative_95': 'By Age 95',
				}).set_index('Claim Age')
				st.dataframe(be_df.style.format('${:,.0f}'))

	# ── Dollar Growth Analysis (Advisor View only) ─────────────
	if view_mode == 'Advisor View' and sim_mode is not None:
		st.markdown('---')
		with st.expander('Dollar Growth Analysis'):
			st.caption('What does $1 grow to over the plan horizon at a given allocation? '
				'Pure compounding — no withdrawals, taxes, or RMDs.')
			_growth_alloc = st.slider('Stock allocation', 0, 100,
				int(round(float(target_stock_pct) * 100)), 5,
				format='%d%%', key='growth_stock_pct')
			if st.button('Run Growth Analysis', key='run_growth_btn'):
				is_hist = return_mode == 'Historical (master_global_factors)'
				sim_years = int(years)
				with st.spinner('Computing growth distribution...'):
					result = dollar_growth_distribution(
						stock_pct=_growth_alloc / 100.0,
						years=sim_years,
						mode='historical' if is_hist else 'simulated',
						stock_drift=float(taxable_log_drift),
						stock_vol=float(taxable_log_volatility),
						bond_drift=float(bond_log_drift),
						bond_vol=float(bond_log_volatility),
						num_runs=int(monte_carlo_runs),
						fee_bps=float(investment_fee_bps),
					)
				st.session_state['_growth_result'] = result
				st.session_state['_growth_years'] = sim_years
				st.session_state['_growth_alloc'] = _growth_alloc

			if '_growth_result' in st.session_state:
				gr = st.session_state['_growth_result']
				_gy = st.session_state['_growth_years']
				_ga = st.session_state['_growth_alloc']
				pct = gr['percentiles']
				st.markdown(f"**{_ga}% stock / {100 - _ga}% bond &nbsp;·&nbsp; {_gy} years "
					f"&nbsp;·&nbsp; {gr['num_runs']} paths &nbsp;·&nbsp; "
					f"Median CAGR {gr['median_cagr']:.2%}**")
				pct_df = pd.DataFrame([
					{'Percentile': f'{p}th', '$1 grows to': pct[p],
					 '$1 buys': 1.0 / pct[p] if pct[p] != 0 else float('inf'),
					 'CAGR': f"{float(np.percentile(gr['cagrs'], p)):.2%}"}
					for p in [0, 10, 25, 50, 75, 90, 100]
				]).set_index('Percentile')
				st.dataframe(pct_df.style.format({'$1 grows to': '${:,.2f}', '$1 buys': '${:,.2f}'}))

				# Histogram of ending values
				hist_df = pd.DataFrame({'Ending Value of $1': gr['ending_values']})
				hist_chart = alt.Chart(hist_df).mark_bar(opacity=0.7, color='#3bbfb0').encode(
					alt.X('Ending Value of $1:Q', bin=alt.Bin(maxbins=50), title='Ending Value of $1'),
					alt.Y('count()', title='Count'),
				).properties(height=300)
				median_rule = alt.Chart(pd.DataFrame({'x': [pct[50]]})).mark_rule(
					color='red', strokeDash=[4, 4], strokeWidth=2).encode(x='x:Q')
				st.altair_chart(hist_chart + median_rule, use_container_width=True)

			# ── Goal Funding Analysis ────────────────────────────
			st.markdown('---')
			st.markdown('#### Goal Funding Analysis')
			st.caption('How much must you set aside **today** to fund a future goal? '
				'Uses the worst-case (0th percentile) growth path for the most conservative estimate.')
			gf_c1, gf_c2, gf_c3 = st.columns(3)
			with gf_c1:
				gf_amount = st.number_input('Annual goal amount ($)', min_value=0,
					value=100_000, step=10_000, key='gf_goal_amount')
			with gf_c2:
				gf_start = st.number_input('Start year', min_value=1,
					value=28, step=1, key='gf_start_year')
			with gf_c3:
				gf_end = st.number_input('End year', min_value=1,
					value=30, step=1, key='gf_end_year')

			if st.button('Run Goal Funding', key='run_goal_funding_btn'):
				if gf_end < gf_start:
					st.error('End year must be ≥ start year.')
				else:
					is_hist = return_mode == 'Historical (master_global_factors)'
					with st.spinner('Computing year-by-year growth paths...'):
						grid = dollar_growth_by_year(
							stock_pct=_growth_alloc / 100.0,
							max_years=int(gf_end),
							mode='historical' if is_hist else 'simulated',
							stock_drift=float(taxable_log_drift),
							stock_vol=float(taxable_log_volatility),
							bond_drift=float(bond_log_drift),
							bond_vol=float(bond_log_volatility),
							num_runs=int(monte_carlo_runs),
							fee_bps=float(investment_fee_bps),
						)
					st.session_state['_gf_grid'] = grid
					st.session_state['_gf_params'] = (gf_amount, gf_start, gf_end, _growth_alloc)

			if '_gf_grid' in st.session_state:
				grid = st.session_state['_gf_grid']
				gf_amount, gf_start, gf_end, gf_alloc = st.session_state['_gf_params']
				rows = []
				total = 0.0
				for yr in range(int(gf_start), int(gf_end) + 1):
					growth_0 = float(np.percentile(grid[:, yr - 1], 0))
					cost_per_dollar = 1.0 / growth_0 if growth_0 > 0 else float('inf')
					set_aside = gf_amount * cost_per_dollar
					total += set_aside
					rows.append({
						'Year': yr,
						'$1 Grows To (0th pctl)': growth_0,
						'Cost per $1': cost_per_dollar,
						'Goal': gf_amount,
						'Set Aside Today': set_aside,
					})
				gf_df = pd.DataFrame(rows).set_index('Year')
				st.dataframe(gf_df.style.format({
					'$1 Grows To (0th pctl)': '${:,.4f}',
					'Cost per $1': '${:,.4f}',
					'Goal': '${:,.0f}',
					'Set Aside Today': '${:,.0f}',
				}))
				st.metric('Total Set Aside Today', f'${total:,.0f}')

	# ── Spending Finder ─────────────────────────────────────────
	if sim_mode is not None and '_sim_params' in st.session_state:
		st.markdown('---')
		with st.expander('Spending Finder'):
			st.caption('Find the spending level that gives you a target probability of meeting your spending goal.')
			col_target, col_guess, col_tol = st.columns(3)
			with col_target:
				spend_find_target = st.number_input('Target spending success %', min_value=50, max_value=99,
					value=90, step=1, key='spend_find_target')
			with col_guess:
				_base_sched_default = st.session_state.get('_base_withdrawal_schedule', [])
				_default_guess = _base_sched_default[0] if _base_sched_default else 60000.0
				spend_find_guess = st.number_input('Starting guess ($)', min_value=0.0,
					value=_default_guess, step=5000.0, key='spend_find_guess')
			with col_tol:
				spend_find_tol = st.number_input('Tolerance ($)', min_value=500, max_value=10000,
					value=1000, step=500, key='spend_find_tol')

			if st.button('Find Spending Level', key='run_spend_finder'):
				stored_params = st.session_state['_sim_params']
				sim_years = st.session_state['_sim_years']
				is_hist = st.session_state['_is_historical']
				mc_runs = st.session_state['_monte_carlo_runs']
				inheritor_rate = st.session_state['_inheritor_marginal_rate']
				base_sched = st.session_state.get('_base_withdrawal_schedule', [])
				original_spending = base_sched[0] if base_sched else 0.0
				target_pct = spend_find_target / 100.0
				tol = float(spend_find_tol)

				if original_spending <= 0:
					st.warning('No spending target found. Run a simulation first.')
				else:
					def _run_with_spending(spend_amt):
						"""Run simulation with a scaled withdrawal schedule and return spending success rate."""
						test_params = dict(stored_params)
						orig_sched = list(test_params['withdrawal_schedule'])
						scale = spend_amt / original_spending if original_spending > 0 else 1.0
						test_params['withdrawal_schedule'] = [v * scale for v in orig_sched]
						if is_hist:
							windows, _ = get_all_historical_windows(sim_years)
							results = []
							all_yearly = []
							for run_idx, (stock_rets, bond_rets) in enumerate(windows):
								run_pp = compute_run_pp_factors(run_idx, sim_years)
								df_run = simulate_withdrawals(
									years=sim_years, stock_return_series=stock_rets,
									bond_return_series=bond_rets, pp_factors_run=run_pp, **test_params)
								df_run['total_portfolio'] = df_run['end_taxable_total'] + df_run['end_tda_total'] + df_run['end_roth']
								metrics = compute_summary_metrics(df_run, inheritor_rate)
								results.append(metrics)
								df_run['run'] = run_idx
								all_yearly.append(df_run)
							all_yearly_df = pd.concat(all_yearly, ignore_index=True)
						else:
							results, all_yearly_df = run_monte_carlo(
								num_runs=mc_runs, years=sim_years, **test_params)
						run_avg = all_yearly_df.groupby('run')['after_tax_spending'].mean()
						return float((run_avg >= spend_amt).mean())

					# Test the user's guess first
					guess = float(spend_find_guess) if spend_find_guess > 0 else original_spending
					progress = st.progress(0, text=f'Testing guess ${guess:,.0f}...')
					guess_rate = _run_with_spending(guess)
					iterations = [{'Iteration': 0, 'Spending': f'${guess:,.0f}',
						'Success Rate': f'{guess_rate*100:.1f}%', 'Range': 'initial guess'}]

					if abs(guess_rate - target_pct) <= 0.01:
						progress.progress(1.0, text='Complete')
						st.success(f'Your guess of {guess:,.0f} achieves '
							f'{guess_rate*100:.0f}% spending success (target: {spend_find_target}%).')
					else:
						has_guess = spend_find_guess > 0 and spend_find_guess != original_spending
						# Set search bounds based on guess result
						if guess_rate >= target_pct:
							# Guess is too conservative — search upward
							lo = guess
							if has_guess:
								# Tight bounds: +20% above guess
								hi = guess * 1.2
							else:
								hi = guess * 2.0
							# Expand hi until success drops below target
							for _ in range(5):
								r = _run_with_spending(hi)
								if r < target_pct:
									break
								hi = lo + (hi - lo) * 1.5 if has_guess else hi * 1.5
						else:
							# Guess is too aggressive — search downward
							if has_guess:
								# Tight bounds: -20% below guess
								lo = guess * 0.8
							else:
								lo = 0.0
							hi = guess
							# Shrink lo until success exceeds target
							if has_guess:
								for _ in range(5):
									r = _run_with_spending(lo)
									if r >= target_pct:
										break
									lo = hi - (hi - lo) * 1.5

						max_iter = 15
						for i in range(max_iter):
							mid = (lo + hi) / 2.0
							progress.progress((i + 1) / max_iter, text=f'Iteration {i+1}: trying ${mid:,.0f}...')
							rate = _run_with_spending(mid)
							iterations.append({'Iteration': i + 1, 'Spending': f'${mid:,.0f}',
								'Success Rate': f'{rate*100:.1f}%', 'Range': f'${lo:,.0f} – ${hi:,.0f}'})
							if rate >= target_pct:
								lo = mid
							else:
								hi = mid
							if hi - lo < tol:
								break

						progress.progress(1.0, text='Complete')
						result_spending = round((lo + hi) / 2.0 / 1000) * 1000
						diff = result_spending - original_spending

						if diff > 0:
							st.success(f'You can increase spending to {result_spending:,.0f} '
								f'(+{diff:,.0f}/yr) and still achieve a {spend_find_target}% spending success rate.')
						else:
							st.warning(f'To achieve a {spend_find_target}% spending success rate, '
								f'reduce spending to {result_spending:,.0f} ({abs(diff):,.0f}/yr less).')
						st.caption(f'Original spending: {original_spending:,.0f}')
						st.dataframe(pd.DataFrame(iterations), use_container_width=True, hide_index=True)

	# ── Client PDF Report ────────────────────────────────────────
	if sim_mode is not None and 'mc_percentile_rows' in st.session_state:
		st.markdown('---')
		st.subheader('Client Report')
		if st.button('Generate Client Report (PDF)', key='gen_pdf_btn'):
			import importlib
			import pdf_report
			importlib.reload(pdf_report)
			with st.spinner('Building PDF report...'):
				st.session_state['_pdf_report_bytes'] = pdf_report.generate_report(dict(st.session_state))

		if '_pdf_report_bytes' in st.session_state:
			import base64
			b64 = base64.b64encode(st.session_state['_pdf_report_bytes']).decode()
			st.markdown(
				f'<iframe src="data:application/pdf;base64,{b64}" '
				f'width="100%" height="800" type="application/pdf"></iframe>',
				unsafe_allow_html=True,
			)
			client = st.session_state.get('client_select', 'client')
			plan = st.session_state.get('save_file_name', 'report')
			filename = f"{client}_{plan}_report.pdf".replace(' ', '_').replace(',', '')
			st.download_button('Download PDF Report', st.session_state['_pdf_report_bytes'],
							   file_name=filename, mime='application/pdf', key='download_pdf_btn')


if __name__ == '__main__':
	main()
