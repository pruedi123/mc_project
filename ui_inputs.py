"""Sidebar widgets, save/load, client management — all Streamlit UI for inputs."""

import streamlit as st
import os
import re
import json
import shutil
import glob as globmod
from datetime import datetime

# ── Save / Load infrastructure ──────────────────────────────────

SAVES_DIR = os.path.expanduser('~/RWM/Current Client Plans')

def _is_cloud() -> bool:
	"""Detect if running on Streamlit Cloud (no local client plans folder)."""
	return not os.path.exists(SAVES_DIR)

# Keys that map directly to session_state widget keys for save/load
_SAVEABLE_KEYS = [
	'start_age', 'start_age_spouse', 'life_expectancy_primary', 'life_expectancy_spouse',
	'taxable_start', 'taxable_stock_basis_pct', 'taxable_bond_basis_pct',
	'roth_start', 'tda_start', 'tda_spouse_start',
	'target_stock_pct', 'roth_conversion_mode', 'roth_conversion_amount', 'roth_conversion_years',
	'roth_conversion_source_tda', 'roth_conversion_tax_source', 'roth_bracket_fill_rate',
	'num_withdrawal_periods',
	'rmd_start_age', 'rmd_start_age_spouse', 'ending_balance_goal', 'num_add_goals',
	'ss_income', 'ss_start_age_p1', 'ss_fra_age_p1', 'ss_income_spouse', 'ss_start_age_p2', 'ss_fra_age_p2', 'ss_cola',
	'pension_income', 'pension_cola_p1', 'pension_survivor_pct_p1',
	'pension_income_spouse', 'pension_cola_p2', 'pension_survivor_pct_p2',
	'other_income',
	'pension_buyout_enabled', 'pension_buyout_baseline', 'pension_buyout_person',
	'pension_buyout_lump', 'pension_buyout_income', 'pension_buyout_cola', 'pension_buyout_survivor',
	'taxes_enabled', 'filing_status', 'use_itemized', 'itemized_deduction',
	'inheritor_marginal_rate', 'state_tax_rate', 'state_exempt_retirement',
	'return_mode', 'taxable_log_drift', 'taxable_log_volatility',
	'bond_log_drift', 'bond_log_volatility', 'random_seed', 'seed_mode',
	'stock_dividend_yield', 'stock_turnover', 'investment_fee_bps',
	'guardrails_enabled', 'guardrail_lower', 'guardrail_upper', 'guardrail_target',
	'guardrail_inner_sims', 'guardrail_max_spending_pct',
	'flex_goal_min_pct',
	'inheritance_enabled', 'inheritance_year', 'inheritance_taxable_amount', 'inheritance_ira_amount',
	'tcja_sunset', 'tcja_sunset_year',
	'qcd_annual', 'earned_income', 'earned_income_years',
	'irmaa_enabled', 'prefer_tda_before_taxable',
	'display_decimals', 'monte_carlo_runs',
	'num_scenarios',
]

def _get_client_dirs():
	"""Return sorted list of client subfolder names."""
	os.makedirs(SAVES_DIR, exist_ok=True)
	return sorted([d for d in os.listdir(SAVES_DIR)
					if os.path.isdir(os.path.join(SAVES_DIR, d)) and not d.startswith('.')])

def _build_client_folder(last: str, first: str, identifier: str = '') -> str:
	"""Build client folder name like 'Smith, John' or 'Smith, John (Portland)'."""
	name = f'{last}, {first}'
	if identifier:
		name += f' ({identifier})'
	return name

def _default_plan_name(last: str, first: str, identifier: str, existing: list) -> str:
	"""Generate a default plan name like 'smith_john' or 'smith_john_portland', auto-incrementing if needed."""
	parts = [last.lower(), first.lower()]
	if identifier:
		parts.append(identifier.lower())
	base = '_'.join(parts).replace(' ', '_')
	if base not in existing:
		return base
	n = 2
	while f'{base}_{n}' in existing:
		n += 1
	return f'{base}_{n}'

def _get_saved_files(client: str):
	"""Return sorted list of .json filenames (without extension) for a client, excluding _results companion files."""
	client_dir = os.path.join(SAVES_DIR, client)
	os.makedirs(client_dir, exist_ok=True)
	files = globmod.glob(os.path.join(client_dir, '*.json'))
	return sorted([os.path.splitext(os.path.basename(f))[0] for f in files
					if not os.path.basename(f).endswith('_results.json')])

def _save_inputs_to_json(client: str, name: str):
	"""Collect current widget values from session_state and write to Current Client Plans/{client}/{name}.json."""
	data = {}
	for k in _SAVEABLE_KEYS:
		if k in st.session_state:
			data[k] = st.session_state[k]
	# Save withdrawal period details (dynamic keys)
	n_periods = int(st.session_state.get('num_withdrawal_periods', 1))
	periods = []
	for i in range(n_periods):
		p = {'amount': st.session_state.get(f'wd_amount_{i}')}
		if f'wd_end_{i}' in st.session_state:
			p['end_year'] = st.session_state[f'wd_end_{i}']
		periods.append(p)
	data['periods'] = periods
	# Save additional spending goals (dynamic keys)
	n_goals = int(st.session_state.get('num_add_goals', 0))
	add_goals = []
	for g in range(n_goals):
		add_goals.append({
			'label': st.session_state.get(f'add_goal_label_{g}', ''),
			'amount': st.session_state.get(f'add_goal_amount_{g}', 0.0),
			'begin': st.session_state.get(f'add_goal_begin_{g}', 1),
			'end': st.session_state.get(f'add_goal_end_{g}', 1),
			'priority': st.session_state.get(f'add_goal_priority_{g}', 'Need'),
			'cap': st.session_state.get(f'add_goal_cap_{g}', -1.0),
			'fund_sep': st.session_state.get(f'add_goal_fund_sep_{g}', False),
			'fund_taxable': st.session_state.get(f'add_goal_fund_taxable_{g}', 0.0),
			'fund_tda1': st.session_state.get(f'add_goal_fund_tda1_{g}', 0.0),
			'fund_tda2': st.session_state.get(f'add_goal_fund_tda2_{g}', 0.0),
			'fund_stock_pct': st.session_state.get(f'add_goal_fund_stock_pct_{g}', 60),
		})
	data['add_goals'] = add_goals
	# Save scenario override UI keys (dynamic)
	n_sc = int(st.session_state.get('num_scenarios', 1))
	if n_sc > 1:
		sc_overrides = {}
		for s_idx in range(2, n_sc + 1):
			sc_data = {}
			for suffix in ['spend_mode', 'spend_scale', 'spend_flat', 'stock_chk', 'stock',
							'roth_chk', 'roth_mode', 'roth_amt', 'roth_yrs', 'roth_bracket',
							'annuity_chk', 'ann_purchase',
							'ann_income', 'ann_cola', 'ann_person', 'ann_surv', 'ann_start',
							'buyout_chk', 'buyout_choice', 'buyout_person', 'buyout_lump',
							'buyout_income', 'buyout_cola', 'buyout_surv',
							'le_chk', 'le_p1', 'le_p2']:
				key = f'sc_{suffix}_{s_idx}'
				if key in st.session_state:
					sc_data[key] = st.session_state[key]
			sc_overrides[str(s_idx)] = sc_data
		data['scenario_overrides'] = sc_overrides
	client_dir = os.path.join(SAVES_DIR, client)
	os.makedirs(client_dir, exist_ok=True)
	path = os.path.join(client_dir, f'{name}.json')

	# --- Timestamped backup & diff ---
	old_data = None
	is_overwrite = os.path.exists(path)
	if is_overwrite:
		with open(path) as f:
			old_data = json.load(f)
		backup_dir = os.path.join(client_dir, 'backups')
		os.makedirs(backup_dir, exist_ok=True)
		stamp = datetime.now().strftime('%Y-%m-%d_%H%M')
		backup_name = f'{name}.{stamp}.bak.json'
		shutil.copy2(path, os.path.join(backup_dir, backup_name))

	with open(path, 'w') as f:
		json.dump(data, f, indent=2, default=str)

	# --- Audit log ---
	action = 'overwrite' if is_overwrite else 'save'
	stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
	changes = ''
	if old_data is not None:
		diffs = []
		all_keys = sorted(set(list(old_data.keys()) + list(data.keys())))
		skip = {'periods', 'scenario_overrides'}
		for k in all_keys:
			if k in skip:
				continue
			old_val = old_data.get(k)
			new_val = data.get(k)
			if old_val != new_val:
				diffs.append(f'{k}: {old_val!r} -> {new_val!r}')
		# Check periods
		if old_data.get('periods') != data.get('periods'):
			diffs.append('periods: changed')
		if old_data.get('scenario_overrides') != data.get('scenario_overrides'):
			diffs.append('scenario_overrides: changed')
		changes = '; '.join(diffs) if diffs else 'no changes'
	else:
		changes = 'new file'
	log_path = os.path.join(client_dir, 'audit.log')
	with open(log_path, 'a') as lf:
		lf.write(f'[{stamp}] {action} {name}.json | {changes}\n')

	return path


def _collect_inputs_as_json() -> str:
	"""Collect current widget values from session_state and return as JSON string (for download on cloud)."""
	data = {}
	for k in _SAVEABLE_KEYS:
		if k in st.session_state:
			data[k] = st.session_state[k]
	n_periods = int(st.session_state.get('num_withdrawal_periods', 1))
	periods = []
	for i in range(n_periods):
		p = {'amount': st.session_state.get(f'wd_amount_{i}')}
		if f'wd_end_{i}' in st.session_state:
			p['end_year'] = st.session_state[f'wd_end_{i}']
		periods.append(p)
	data['periods'] = periods
	n_goals = int(st.session_state.get('num_add_goals', 0))
	add_goals = []
	for g in range(n_goals):
		add_goals.append({
			'label': st.session_state.get(f'add_goal_label_{g}', ''),
			'amount': st.session_state.get(f'add_goal_amount_{g}', 0.0),
			'begin': st.session_state.get(f'add_goal_begin_{g}', 1),
			'end': st.session_state.get(f'add_goal_end_{g}', 1),
			'priority': st.session_state.get(f'add_goal_priority_{g}', 'Need'),
			'cap': st.session_state.get(f'add_goal_cap_{g}', -1.0),
			'fund_sep': st.session_state.get(f'add_goal_fund_sep_{g}', False),
			'fund_taxable': st.session_state.get(f'add_goal_fund_taxable_{g}', 0.0),
			'fund_tda1': st.session_state.get(f'add_goal_fund_tda1_{g}', 0.0),
			'fund_tda2': st.session_state.get(f'add_goal_fund_tda2_{g}', 0.0),
			'fund_stock_pct': st.session_state.get(f'add_goal_fund_stock_pct_{g}', 60),
		})
	data['add_goals'] = add_goals
	n_sc = int(st.session_state.get('num_scenarios', 1))
	if n_sc > 1:
		sc_overrides = {}
		for s_idx in range(2, n_sc + 1):
			sc_data = {}
			for suffix in ['spend_mode', 'spend_scale', 'spend_flat', 'stock_chk', 'stock',
							'roth_chk', 'roth_mode', 'roth_amt', 'roth_yrs', 'roth_bracket',
							'annuity_chk', 'ann_purchase',
							'ann_income', 'ann_cola', 'ann_person', 'ann_surv', 'ann_start',
							'buyout_chk', 'buyout_choice', 'buyout_person', 'buyout_lump',
							'buyout_income', 'buyout_cola', 'buyout_surv',
							'le_chk', 'le_p1', 'le_p2']:
				key = f'sc_{suffix}_{s_idx}'
				if key in st.session_state:
					sc_data[key] = st.session_state[key]
			sc_overrides[str(s_idx)] = sc_data
		data['scenario_overrides'] = sc_overrides
	return json.dumps(data, indent=2, default=str)

def _load_inputs_from_json(client: str, name: str):
	"""Read Current Client Plans/{client}/{name}.json and set values into session_state."""
	path = os.path.join(SAVES_DIR, client, f'{name}.json')
	with open(path) as f:
		data = json.load(f)
	for k in _SAVEABLE_KEYS:
		if k in data:
			st.session_state[k] = data[k]
	# Restore withdrawal periods
	if 'periods' in data:
		for i, p in enumerate(data['periods']):
			if 'amount' in p and p['amount'] is not None:
				st.session_state[f'wd_amount_{i}'] = p['amount']
			if 'end_year' in p:
				st.session_state[f'wd_end_{i}'] = p['end_year']
	# Restore additional spending goals
	if 'add_goals' in data:
		for g, goal in enumerate(data['add_goals']):
			st.session_state[f'add_goal_label_{g}'] = goal.get('label', '')
			st.session_state[f'add_goal_amount_{g}'] = goal.get('amount', 0.0)
			st.session_state[f'add_goal_begin_{g}'] = goal.get('begin', 1)
			st.session_state[f'add_goal_end_{g}'] = goal.get('end', 1)
			# Backward compat: map old Need/Want to Essential/Flexible
			raw_priority = goal.get('priority', 'Essential')
			if raw_priority == 'Need': raw_priority = 'Essential'
			elif raw_priority == 'Want': raw_priority = 'Flexible'
			st.session_state[f'add_goal_priority_{g}'] = raw_priority
			st.session_state[f'add_goal_cap_{g}'] = goal.get('cap', -1.0)
			st.session_state[f'add_goal_fund_sep_{g}'] = goal.get('fund_sep', False)
			st.session_state[f'add_goal_fund_taxable_{g}'] = goal.get('fund_taxable', 0.0)
			st.session_state[f'add_goal_fund_tda1_{g}'] = goal.get('fund_tda1', 0.0)
			st.session_state[f'add_goal_fund_tda2_{g}'] = goal.get('fund_tda2', 0.0)
			st.session_state[f'add_goal_fund_stock_pct_{g}'] = goal.get('fund_stock_pct', 60)
	# Restore scenario overrides
	if 'scenario_overrides' in data:
		for s_idx_str, sc_data in data['scenario_overrides'].items():
			for key, val in sc_data.items():
				st.session_state[key] = val

def save_results_to_json(client: str, name: str):
	"""Save pre-aggregated simulation results alongside a plan's input file.
	Returns the path written, or '' if no results are available."""
	import numpy as np
	pct_rows = st.session_state.get('mc_percentile_rows')
	all_yearly = st.session_state.get('mc_all_yearly')
	if pct_rows is None or all_yearly is None:
		return ''
	median_by_year = all_yearly.groupby('year')[['total_portfolio', 'after_tax_spending']].median()
	data = {
		'saved_at': datetime.now().isoformat(timespec='seconds'),
		'sim_mode': st.session_state.get('sim_mode', 'simulated'),
		'num_sims': int(st.session_state.get('num_sims', 0)),
		'percentile_rows': pct_rows,
		'spending_percentiles': st.session_state.get('mc_spending_pct_rows', []),
		'pct_non_positive': float(st.session_state.get('mc_pct_non_positive', 0.0)),
		'median_yearly': {
			'years': median_by_year.index.tolist(),
			'portfolio': median_by_year['total_portfolio'].tolist(),
			'spending': median_by_year['after_tax_spending'].tolist(),
		},
	}
	client_dir = os.path.join(SAVES_DIR, client)
	os.makedirs(client_dir, exist_ok=True)
	path = os.path.join(client_dir, f'{name}_results.json')
	with open(path, 'w') as f:
		json.dump(data, f, indent=2, default=str)
	return path

def load_plan_results(client: str, name: str):
	"""Load saved simulation results for a plan. Returns dict or None if not found."""
	path = os.path.join(SAVES_DIR, client, f'{name}_results.json')
	if not os.path.exists(path):
		return None
	with open(path) as f:
		return json.load(f)

def get_plans_with_results(client: str):
	"""Return list of plan names that have companion _results.json files (and a matching input file)."""
	client_dir = os.path.join(SAVES_DIR, client)
	if not os.path.isdir(client_dir):
		return []
	plans = []
	for f in sorted(os.listdir(client_dir)):
		if f.endswith('_results.json'):
			plan_name = f[:-len('_results.json')]
			if os.path.exists(os.path.join(client_dir, f'{plan_name}.json')):
				plans.append(plan_name)
	return plans

# ── Sidebar rendering ───────────────────────────────────────────

def _render_save_load_section():
	"""Render Save / Load Inputs expander."""
	with st.expander('Save / Load Inputs'):
		client_dirs = _get_client_dirs()

		# Handle pending rename before selectbox renders
		if st.session_state.get('_pending_rename'):
			old_folder, new_folder = st.session_state.pop('_pending_rename')
			old_path = os.path.join(SAVES_DIR, old_folder)
			new_path = os.path.join(SAVES_DIR, new_folder)
			if os.path.exists(new_path):
				st.error(f'Client "{new_folder}" already exists.')
			else:
				os.rename(old_path, new_path)
				st.session_state['client_select'] = new_folder
				client_dirs = _get_client_dirs()  # refresh after rename

		# Choose existing client or enter new
		if client_dirs:
			client_options = ['-- New Client --'] + client_dirs
			selected = st.selectbox('Select client', client_options, key='client_select')
		else:
			selected = '-- New Client --'

		if selected == '-- New Client --':
			col_last, col_first = st.columns(2)
			with col_last:
				last_s = st.text_input('Last name', value='', key='client_last',
									placeholder='e.g. Smith').strip()
			with col_first:
				first_s = st.text_input('First name', value='', key='client_first',
									placeholder='e.g. John').strip()
			id_s = st.text_input('Identifier (optional)', value='', key='client_id',
										placeholder='e.g. Portland').strip()
			if last_s and first_s:
				client = _build_client_folder(last_s, first_s, id_s)
			else:
				client = ''
				last_s = first_s = id_s = ''
				st.info('Enter last and first name to save or load plans.')
		else:
			client = selected
			# Parse last, first, identifier back from folder name
			m = re.match(r'^(.+),\s+(.+?)(?:\s+\((.+)\))?$', client)
			if m:
				last_s, first_s, id_s = m.group(1), m.group(2), m.group(3) or ''
			else:
				last_s = first_s = id_s = ''

			# --- Rename Client ---
			with st.popover('Rename Client'):
				new_last = st.text_input('Last name', value=last_s, key='rename_last')
				new_first = st.text_input('First name', value=first_s, key='rename_first')
				new_id = st.text_input('Identifier (optional)', value=id_s, key='rename_id')
				new_last_s, new_first_s, new_id_s = new_last.strip(), new_first.strip(), new_id.strip()
				if new_last_s and new_first_s:
					new_folder = _build_client_folder(new_last_s, new_first_s, new_id_s)
					if new_folder != client:
						if st.button('Rename'):
							st.session_state['_pending_rename'] = (client, new_folder)
							st.rerun()
					else:
						st.caption('No changes.')
				else:
					st.warning('Last and first name are required.')

		if client:
			if _is_cloud():
				# ── Cloud mode: download instead of save to disk ──
				default_plan = _default_plan_name(last_s, first_s, id_s, []) if last_s and first_s else 'my_plan'
				save_name = st.text_input('Plan name', value=default_plan, key='save_file_name')
				json_str = _collect_inputs_as_json()
				st.download_button(
					'Download Plan JSON',
					data=json_str,
					file_name=f'{save_name.strip() or default_plan}.json',
					mime='application/json',
				)
				st.caption('Save this file to iCloud Drive > RWM > Current Client Plans > (client folder) to sync with your Mac.')
			else:
				# ── Local mode: save to disk ──
				saved_files = _get_saved_files(client)
				# Handle pending load (set before widgets render)
				if st.session_state.get('_pending_load'):
					load_client, load_name = st.session_state.pop('_pending_load')
					_load_inputs_from_json(load_client, load_name)
					st.session_state['save_file_name'] = load_name
				# Auto-generate default plan name from client name
				default_plan = _default_plan_name(last_s, first_s, id_s, saved_files) if last_s and first_s else 'my_plan'
				save_name = st.text_input('Plan name', value=default_plan, key='save_file_name')
				if st.button('Save Inputs'):
					if save_name.strip():
						path = _save_inputs_to_json(client, save_name.strip())
						results_path = save_results_to_json(client, save_name.strip())
						if results_path:
							st.success(f'Saved inputs + results to {client}/{save_name.strip()}')
						else:
							st.success(f'Saved inputs to {client}/{os.path.basename(path)} (no simulation results yet)')
					else:
						st.warning('Enter a plan name.')
				if saved_files:
					load_choice = st.selectbox('Load saved plan', saved_files, key='load_file_choice')
					col_load, col_del = st.columns(2)
					with col_load:
						if st.button('Load'):
							st.session_state['_pending_load'] = (client, load_choice)
							st.rerun()
					with col_del:
						if st.button('Delete'):
							os.remove(os.path.join(SAVES_DIR, client, f'{load_choice}.json'))
							# Also remove companion results file if it exists
							results_path = os.path.join(SAVES_DIR, client, f'{load_choice}_results.json')
							if os.path.exists(results_path):
								os.remove(results_path)
							# Remove client folder if empty
							client_dir = os.path.join(SAVES_DIR, client)
							if not os.listdir(client_dir):
								os.rmdir(client_dir)
							st.rerun()
				else:
					st.caption('No saved plans for this client yet.')

		# Upload plan JSON file
		st.divider()
		uploaded = st.file_uploader('Upload a plan JSON file', type=['json'], key='upload_plan_file')
		if uploaded is not None and not st.session_state.get('_upload_processed'):
			import json as _json
			try:
				data = _json.load(uploaded)
				st.session_state['_pending_upload'] = data
				st.session_state['_upload_processed'] = True
				st.rerun()
			except Exception as e:
				st.error(f'Error loading plan: {e}')
		# Reset upload processed flag when file uploader is cleared
		if uploaded is None:
			st.session_state.pop('_upload_processed', None)

def _render_scenario_section():
	"""Render Scenario Comparison expander. Returns (num_scenarios, scenario_overrides_ui)."""
	with st.expander('Scenario Comparison'):
		num_scenarios = st.number_input('Number of scenarios', min_value=1, max_value=4, value=1, step=1, key='num_scenarios',
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
					'Set to None for no conversions.')
				if roth_chk:
					sc_roth_mode = st.radio(f'S{i} Roth mode',
						['None', 'Fixed amount', 'Fill to bracket'],
						horizontal=True, key=f'sc_roth_mode_{i}')
					sc_overrides['roth_conversion_mode'] = sc_roth_mode
					if sc_roth_mode == 'Fixed amount':
						sc_overrides['roth_conversion_amount'] = st.number_input(f'S{i} annual Roth conversion',
							value=0.0, step=10000.0, key=f'sc_roth_amt_{i}',
							help='Amount converted from TDA to Roth each year. Taxed as ordinary income in the year of conversion.')
					elif sc_roth_mode == 'Fill to bracket':
						sc_bracket_label = st.selectbox(f'S{i} fill up to bracket',
							list(_BRACKET_RATE_OPTIONS.keys()), index=2, key=f'sc_roth_bracket_{i}')
						sc_overrides['roth_bracket_fill_rate'] = _BRACKET_RATE_OPTIONS[sc_bracket_label]
					if sc_roth_mode != 'None':
						sc_overrides['roth_conversion_years'] = int(st.number_input(f'S{i} conversion years',
							value=0, min_value=0, max_value=100, key=f'sc_roth_yrs_{i}',
							help='Number of years to perform conversions, starting from year 1 of the simulation.'))
					else:
						sc_overrides['roth_conversion_years'] = 0
				annuity_chk = st.checkbox(f'S{i} buy annuity', key=f'sc_annuity_chk_{i}',
					help='Purchase an annuity using money from one of your accounts. '
					'Reduces that account balance and adds an income stream.')
				if annuity_chk:
					sc_overrides['annuity_purchase'] = st.number_input(f'S{i} annuity purchase price',
						value=500000.0, step=10000.0, key=f'sc_ann_purchase_{i}',
						help='Amount taken from the selected account to buy the annuity.')
					sc_overrides['annuity_fund_source'] = st.radio(f'S{i} fund from',
						['Taxable', 'TDA Person 1', 'TDA Person 2'],
						horizontal=True, key=f'sc_ann_source_{i}', index=1,
						help='Which account the purchase price is deducted from.')
					sc_overrides['annuity_annual_income'] = st.number_input(f'S{i} annuity annual income',
						value=36000.0, step=1000.0, key=f'sc_ann_income_{i}',
						help='Annual income received from the annuity.')
					sc_overrides['annuity_cola'] = st.number_input(f'S{i} annuity COLA',
						value=0.0, format="%.4f", key=f'sc_ann_cola_{i}',
						help='Annual cost-of-living adjustment on the annuity income. 0 = fixed payments.')
					sc_overrides['annuity_person'] = st.radio(f'S{i} annuity owner',
						['Person 1', 'Person 2'], horizontal=True, key=f'sc_ann_person_{i}')
					sc_overrides['annuity_survivor_pct'] = st.number_input(f'S{i} annuity survivor %',
						value=1.0, min_value=0.0, max_value=1.0, format="%.2f", step=0.05, key=f'sc_ann_surv_{i}',
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
				le_chk = st.checkbox(f'S{i} override life expectancy', key=f'sc_le_chk_{i}',
					help='Test early death scenarios. Overrides life expectancy for one or both people.')
				if le_chk:
					_le_min_p1 = int(st.session_state.get('start_age', 65))
					_le_min_p2 = int(st.session_state.get('start_age_spouse', 60))
					sc_overrides['life_expectancy_primary'] = int(st.number_input(
						f'S{i} Person 1 life expectancy', value=84,
						min_value=_le_min_p1, max_value=120,
						step=1, key=f'sc_le_p1_{i}',
						help='Last age P1 lives through in this scenario.'))
					sc_overrides['life_expectancy_spouse'] = int(st.number_input(
						f'S{i} Person 2 life expectancy', value=89,
						min_value=_le_min_p2, max_value=120,
						step=1, key=f'sc_le_p2_{i}',
						help='Last age P2 lives through in this scenario.'))
				scenario_overrides_ui[i] = sc_overrides
		else:
			scenario_overrides_ui = {}
	return num_scenarios, scenario_overrides_ui

def _render_ages_section():
	"""Render Ages & Timeline expander. Returns dict of age values."""
	with st.expander('Ages & Timeline'):
		start_age = st.number_input('Starting age (person 1)', min_value=18, max_value=120, value=65, key='start_age')
		start_age_spouse = st.number_input('Starting age (person 2)', min_value=18, max_value=120, value=60, key='start_age_spouse')
		life_expectancy_primary = st.number_input('Primary life expectancy (last age lived through)', min_value=int(start_age), max_value=120, value=84, step=1, key='life_expectancy_primary')
		life_expectancy_spouse = st.number_input('Spouse life expectancy (last age lived through)', min_value=int(start_age_spouse), max_value=120, value=89, step=1, key='life_expectancy_spouse')
	return {
		'start_age': start_age,
		'start_age_spouse': start_age_spouse,
		'life_expectancy_primary': life_expectancy_primary,
		'life_expectancy_spouse': life_expectancy_spouse,
	}

def _render_accounts_section():
	"""Render Account Balances expander. Returns dict of account values."""
	with st.expander('Account Balances'):
		taxable_start = st.number_input('Taxable account starting balance', value=300000.0, step=1000.0, key='taxable_start')
		taxable_stock_basis_pct = st.number_input('Taxable stock basis % of market value', value=50.0, min_value=0.0, max_value=100.0, step=1.0, key='taxable_stock_basis_pct') / 100.0
		taxable_bond_basis_pct = st.number_input('Taxable bond basis % of market value', value=100.0, min_value=0.0, max_value=100.0, step=1.0, key='taxable_bond_basis_pct') / 100.0
		roth_start = st.number_input('Roth account starting balance', value=0.0, step=1000.0, key='roth_start')
		tda_start = st.number_input('Tax-deferred account starting balance (IRA/401k) - person 1', value=400000.0, step=1000.0, key='tda_start')
		tda_spouse_start = st.number_input('Tax-deferred account starting balance (IRA/401k) - person 2', value=300000.0, step=1000.0, key='tda_spouse_start')
	return {
		'taxable_start': taxable_start,
		'taxable_stock_basis_pct': taxable_stock_basis_pct,
		'taxable_bond_basis_pct': taxable_bond_basis_pct,
		'roth_start': roth_start,
		'tda_start': tda_start,
		'tda_spouse_start': tda_spouse_start,
	}

def _render_inheritance_section():
	"""Render Expected Inheritance expander. Returns dict of inheritance values."""
	with st.expander('Expected Inheritance'):
		inheritance_enabled = st.checkbox('Enable expected inheritance', value=False,
			help='Model a one-time inheritance received during the simulation. '
			'Inherited taxable assets get a stepped-up (100%) cost basis. '
			'Inherited IRAs are subject to the SECURE Act 10-year drawdown rule.',
			key='inheritance_enabled')
		if inheritance_enabled:
			inheritance_year = st.number_input('Year of inheritance (simulation year)',
				min_value=1, max_value=40, value=10, step=1, key='inheritance_year',
				help='Simulation year when the inheritance is received.')
			inheritance_taxable_amount = st.number_input('Inherited taxable assets',
				value=0.0, min_value=0.0, step=10000.0, key='inheritance_taxable_amount',
				help='Amount received as taxable assets (stocks/bonds). Gets a stepped-up cost basis equal to market value — no embedded gains.')
			inheritance_ira_amount = st.number_input('Inherited IRA amount',
				value=0.0, min_value=0.0, step=10000.0, key='inheritance_ira_amount',
				help='Amount received as an inherited IRA (pre-tax). Subject to SECURE Act 10-year drawdown: '
				'distributed evenly over 10 years as ordinary income.')
		else:
			inheritance_year = 10
			inheritance_taxable_amount = 0.0
			inheritance_ira_amount = 0.0
	return {
		'inheritance_enabled': inheritance_enabled,
		'inheritance_year': inheritance_year,
		'inheritance_taxable_amount': inheritance_taxable_amount,
		'inheritance_ira_amount': inheritance_ira_amount,
	}

_BRACKET_RATE_OPTIONS = {
	'10%': 0.10, '12%': 0.12, '22%': 0.22, '24%': 0.24,
	'32%': 0.32, '35%': 0.35, '37%': 0.37,
}

def _render_allocation_section():
	"""Render Allocation & Roth Conversions expander. Returns dict of allocation values."""
	with st.expander('Allocation & Roth Conversions'):
		target_stock_pct = st.slider('Household target % in stocks', min_value=0, max_value=100, value=60, step=10, key='target_stock_pct') / 100.0
		prefer_tda_before_taxable = st.checkbox('Prefer TDA withdrawals before taxable',
			value=False, key='prefer_tda_before_taxable',
			help='Default waterfall: taxable bonds → stocks → TDA → Roth. '
			'When checked: TDA → taxable bonds → stocks → Roth. '
			'Useful to deplete TDA faster (reduce future RMDs) or take advantage of low tax brackets.')
		roth_conversion_mode = st.radio('Roth conversion mode',
			['None', 'Fixed amount', 'Fill to bracket'],
			horizontal=True, key='roth_conversion_mode',
			help='None = no conversions. Fixed = convert a set dollar amount each year. '
			'Fill to bracket = automatically convert enough to fill up to a chosen tax bracket ceiling.')
		roth_conversion_amount = 0.0
		roth_bracket_fill_rate = 0.22
		if roth_conversion_mode == 'Fixed amount':
			roth_conversion_amount = st.number_input('Annual Roth conversion amount (from TDA)', value=0.0, step=1000.0, key='roth_conversion_amount')
		elif roth_conversion_mode == 'Fill to bracket':
			bracket_label = st.selectbox('Fill up to bracket',
				list(_BRACKET_RATE_OPTIONS.keys()), index=2, key='roth_bracket_fill_rate',
				help='Convert enough each year to fill taxable income up to the top of this bracket.')
			roth_bracket_fill_rate = _BRACKET_RATE_OPTIONS[bracket_label]
		if roth_conversion_mode != 'None':
			roth_conversion_years = st.number_input('Years to perform conversions', value=0, min_value=0, max_value=100, step=1, key='roth_conversion_years')
			roth_conversion_source_tda = st.radio('Convert from', ['Person 1 TDA', 'Person 2 TDA'], horizontal=True, key='roth_conversion_source_tda')
			roth_conversion_tax_source = st.radio('Pay conversion taxes from', ['Taxable', 'TDA (reduce net conversion)'], horizontal=False, key='roth_conversion_tax_source')
		else:
			roth_conversion_years = 0
			roth_conversion_source_tda = 'Person 1 TDA'
			roth_conversion_tax_source = 'Taxable'
	return {
		'target_stock_pct': target_stock_pct,
		'prefer_tda_before_taxable': prefer_tda_before_taxable,
		'roth_conversion_mode': roth_conversion_mode,
		'roth_conversion_amount': roth_conversion_amount,
		'roth_bracket_fill_rate': roth_bracket_fill_rate,
		'roth_conversion_years': roth_conversion_years,
		'roth_conversion_source_tda': roth_conversion_source_tda,
		'roth_conversion_tax_source': roth_conversion_tax_source,
	}

def _render_withdrawal_section(horizon):
	"""Render Withdrawal Schedule expander. Returns (withdrawal_schedule_inputs, rmd_start_age, rmd_start_age_spouse, ending_balance_goal)."""
	with st.expander('Withdrawal Schedule'):
		num_withdrawal_periods = st.number_input('Number of withdrawal periods', value=1, min_value=1, max_value=10, step=1, key='num_withdrawal_periods')
		withdrawal_schedule_inputs = []
		period_start = 1
		for i in range(int(num_withdrawal_periods)):
			is_last = (i == int(num_withdrawal_periods) - 1)
			if is_last:
				period_end = horizon
				st.markdown(f'**Period {i+1}:** years {period_start}–{period_end}')
				period_amount = st.number_input(f'Period {i+1} annual After-Tax Spending Goal', value=60000.0, step=1000.0, key=f'wd_amount_{i}')
			else:
				max_end = horizon - (int(num_withdrawal_periods) - 1 - i)
				default_end = min(period_start + 4, max_end)
				period_end = st.number_input(
					f'Period {i+1}: years {period_start} through',
					value=default_end, min_value=period_start, max_value=max_end, step=1, key=f'wd_end_{i}')
				period_amount = st.number_input(f'Period {i+1} Annual After-Tax Spend Goal', value=40000.0, step=1000.0, key=f'wd_amount_{i}')
			num_years = int(period_end) - period_start + 1
			withdrawal_schedule_inputs.append((num_years, float(period_amount)))
			period_start = int(period_end) + 1
		rmd_start_age = st.number_input('RMD start age (person 1)', min_value=65, max_value=89, value=73, key='rmd_start_age')
		rmd_start_age_spouse = st.number_input('RMD start age (person 2)', min_value=65, max_value=90, value=73, key='rmd_start_age_spouse')
		ending_balance_goal = st.number_input('Ending balance goal (legacy target)', value=0.0, min_value=0.0, step=50000.0, key='ending_balance_goal',
			help='Success = ending portfolio ≥ this amount. Use this as your legacy target — money left in the portfolio avoids the tax drag of withdrawing and re-gifting. $1 = simply not depleted.')
	return {
		'withdrawal_schedule_inputs': withdrawal_schedule_inputs,
		'rmd_start_age': rmd_start_age,
		'rmd_start_age_spouse': rmd_start_age_spouse,
		'ending_balance_goal': ending_balance_goal,
	}

def _render_add_goals_section(horizon):
	"""Render Additional Spending Goals expander. Returns list of (label, amount, begin, end, priority)."""
	with st.expander('Additional Spending Goals'):
		st.caption('Extra spending layered on top of base withdrawals (e.g. long-term care, travel, home repair)')
		num_add_goals = st.number_input('Number of additional goals', value=0, min_value=0, max_value=10, step=1, key='num_add_goals')
		add_goal_inputs = []
		# Defaults for goal 0: Long-term care in last 3 years; goal 1: Legacy in final year
		_ltc_default_begin = max(1, horizon - 2)
		_goal_defaults = {
			0: {'label': 'Long-term care', 'amount': 100000.0, 'begin': _ltc_default_begin, 'end': horizon, 'priority': 'Flexible', 'cap': 50.0},
		}
		for g in range(int(num_add_goals)):
			st.markdown(f'**Goal {g+1}**')
			gd = _goal_defaults.get(g, {'label': '', 'amount': 0.0, 'begin': 1, 'end': horizon, 'priority': 'Essential', 'cap': -1.0})
			g_label = st.text_input(f'Goal {g+1} label', value=gd['label'], key=f'add_goal_label_{g}')
			g_amount = st.number_input(f'Goal {g+1} annual amount', value=gd['amount'], step=1000.0, key=f'add_goal_amount_{g}')
			g_begin = st.number_input(f'Goal {g+1} begin year', value=gd['begin'], min_value=1, max_value=horizon, step=1, key=f'add_goal_begin_{g}')
			g_end = st.number_input(f'Goal {g+1} end year', value=gd['end'], min_value=1, max_value=horizon, step=1, key=f'add_goal_end_{g}')
			priority_idx = ['Essential', 'Flexible'].index(gd['priority']) if gd['priority'] in ('Essential', 'Flexible') else 0
			g_priority = st.selectbox(f'Goal {g+1} priority', ['Essential', 'Flexible'], index=priority_idx, key=f'add_goal_priority_{g}',
				help='Essential = funded at full target even if markets are down; Flexible = adjusted with base spending when portfolio is under pressure')
			g_cap = st.number_input(f'Goal {g+1} spending cap (% above target, -1=no cap)',
				value=gd['cap'], min_value=-1.0, max_value=200.0, format="%.0f", step=10.0,
				help='Cap how much this goal can increase in good markets. 0 = never exceed target. 50 = up to 150% of target. -1 = no cap (unlimited).',
				key=f'add_goal_cap_{g}')
			g_fund_sep = st.checkbox(f'Fund separately', value=False, key=f'add_goal_fund_sep_{g}',
				help='Set aside money today to fully fund this goal. Removes it from the withdrawal plan and tracks it in shadow accounts.')
			g_fund_taxable = 0.0
			g_fund_tda1 = 0.0
			g_fund_tda2 = 0.0
			g_fund_stock_pct = 60
			if g_fund_sep:
				g_fund_stock_pct = st.slider(f'Goal {g+1} stock allocation %', 0, 100, 60, 5,
					key=f'add_goal_fund_stock_pct_{g}',
					help='Stock/bond allocation for the separately funded goal account. '
					'Lower allocations reduce worst-case growth, increasing the set-aside cost.')
				st.caption('Source of funds (leave all at 0 to auto-fill from taxable when sim runs)')
				fc1, fc2, fc3 = st.columns(3)
				with fc1:
					g_fund_taxable = st.number_input(f'From Taxable', value=0.0, min_value=0.0,
						step=5000.0, key=f'add_goal_fund_taxable_{g}')
				with fc2:
					g_fund_tda1 = st.number_input(f'From TDA P1', value=0.0, min_value=0.0,
						step=5000.0, key=f'add_goal_fund_tda1_{g}')
				with fc3:
					g_fund_tda2 = st.number_input(f'From TDA P2', value=0.0, min_value=0.0,
						step=5000.0, key=f'add_goal_fund_tda2_{g}')
			add_goal_inputs.append((g_label, float(g_amount), int(g_begin), int(g_end), g_priority, float(g_cap),
				bool(g_fund_sep), float(g_fund_taxable), float(g_fund_tda1), float(g_fund_tda2), int(g_fund_stock_pct)))
	return add_goal_inputs

def _render_income_section():
	"""Render Other Income expander. Returns dict of income values."""
	with st.expander('Other Income'):
		ss_income_input = st.number_input('Annual Social Security - person 1 (current year)', value=0.0, step=1000.0, key='ss_income')
		ss_start_age_p1 = st.number_input('SS start age - person 1', min_value=60, max_value=90, value=67, step=1, key='ss_start_age_p1')
		ss_fra_age_p1 = st.selectbox('SS full retirement age - person 1', [66, 67], index=1, key='ss_fra_age_p1')
		ss_income_spouse_input = st.number_input('Annual Social Security - person 2 (current year)', value=0.0, step=1000.0, key='ss_income_spouse')
		ss_start_age_p2 = st.number_input('SS start age - person 2', min_value=60, max_value=90, value=67, step=1, key='ss_start_age_p2')
		ss_fra_age_p2 = st.selectbox('SS full retirement age - person 2', [66, 67], index=1, key='ss_fra_age_p2')
		ss_cola = st.number_input('Social Security COLA', value=0.0, format="%.4f", key='ss_cola')
		st.caption('Enter each person\'s own worker benefit from their SSA statement. '
			'Spousal top-up and survivor benefits are computed automatically for married filers.')
		pension_income_input = st.number_input('Annual pension income - person 1', value=0.0, step=1000.0, key='pension_income')
		pension_cola_p1 = st.number_input('Pension COLA - person 1', value=0.00, format="%.4f", key='pension_cola_p1')
		pension_survivor_pct_p1 = st.number_input('Pension survivor % - person 1', value=0.0, min_value=0.0, max_value=1.0, format="%.2f", step=0.05,
			help='Fraction of person 1 pension paid to survivor after person 1 dies', key='pension_survivor_pct_p1')
		pension_income_spouse_input = st.number_input('Annual pension income - person 2', value=0.0, step=1000.0, key='pension_income_spouse')
		pension_cola_p2 = st.number_input('Pension COLA - person 2', value=0.00, format="%.4f", key='pension_cola_p2')
		pension_survivor_pct_p2 = st.number_input('Pension survivor % - person 2', value=0.0, min_value=0.0, max_value=1.0, format="%.2f", step=0.05,
			help='Fraction of person 2 pension paid to survivor after person 2 dies', key='pension_survivor_pct_p2')
		other_income_input = st.number_input('Other ordinary income', value=0.0, step=1000.0, key='other_income')
		st.markdown('---')
		st.caption('**Earned Income** (part-time work in early retirement)')
		earned_income_input = st.number_input('Annual earned income', value=0.0, step=1000.0, key='earned_income',
			help='Annual earned income (W-2 or self-employment) included as ordinary taxable income. '
			'FICA taxes are not modeled separately.')
		earned_income_years = st.number_input('Years of earned income', value=0, min_value=0, max_value=40, step=1,
			key='earned_income_years',
			help='Number of years from the start of simulation during which earned income is received.')
		st.markdown('---')
		st.caption('**Qualified Charitable Distributions (QCDs)**')
		qcd_annual = st.number_input('Annual QCD amount (from IRA, age 70+)', value=0.0, step=1000.0, key='qcd_annual',
			help='QCDs go directly from your IRA to charity. They satisfy RMD requirements but are excluded '
			'from taxable income. Available at age 70\u00bd (modeled as 70). Max $105,000/year per person.')
	return {
		'ss_income': ss_income_input,
		'ss_start_age_p1': ss_start_age_p1,
		'ss_fra_age_p1': ss_fra_age_p1,
		'ss_income_spouse': ss_income_spouse_input,
		'ss_start_age_p2': ss_start_age_p2,
		'ss_fra_age_p2': ss_fra_age_p2,
		'ss_cola': ss_cola,
		'pension_income': pension_income_input,
		'pension_cola_p1': pension_cola_p1,
		'pension_survivor_pct_p1': pension_survivor_pct_p1,
		'pension_income_spouse': pension_income_spouse_input,
		'pension_cola_p2': pension_cola_p2,
		'pension_survivor_pct_p2': pension_survivor_pct_p2,
		'other_income': other_income_input,
		'earned_income': earned_income_input,
		'earned_income_years': earned_income_years,
		'qcd_annual': qcd_annual,
	}

def _render_pension_buyout_section():
	"""Render Pension Buyout expander. Returns dict of buyout values."""
	with st.expander('Pension Buyout (Lump Sum vs Annuity)'):
		pension_buyout_enabled = st.checkbox('Enable pension buyout comparison', value=False,
			help='Compare taking a lump sum (rolled into TDA) vs taking an annuity/pension income stream', key='pension_buyout_enabled')
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
	return {
		'pension_buyout_enabled': pension_buyout_enabled,
		'pension_buyout_baseline': pension_buyout_baseline,
		'pension_buyout_person': pension_buyout_person,
		'pension_buyout_lump': pension_buyout_lump,
		'pension_buyout_income': pension_buyout_income,
		'pension_buyout_cola': pension_buyout_cola,
		'pension_buyout_survivor': pension_buyout_survivor,
	}

def _render_tax_section():
	"""Render Tax Settings expander. Returns dict of tax values."""
	with st.expander('Tax Settings'):
		taxes_enabled = st.checkbox('Enable taxation', value=True, help='Uncheck to disable all taxes (useful for testing withdrawal mechanics)', key='taxes_enabled')
		filing_status_choice = st.radio('Filing status', ['Single', 'Married Filing Jointly'], horizontal=True, index=1, key='filing_status')
		filing_status_key = 'single' if filing_status_choice == 'Single' else 'mfj'
		standard_deduction_display = 14600 if filing_status_key == 'single' else 29200
		use_itemized = st.checkbox('Use itemized deductions instead of standard', value=False, key='use_itemized')
		itemized_deduction_input = st.number_input('Itemized deduction amount', value=0.0, step=500.0, key='itemized_deduction')
		st.caption(f'Standard deduction used if not itemizing: ${standard_deduction_display:,.0f}')
		inheritor_marginal_rate = st.number_input(
			'Inheritor marginal tax rate on TDAs',
			value=0.35, min_value=0.0, max_value=0.50, format="%.4f", key='inheritor_marginal_rate')
		state_tax_rate = st.number_input('State income tax rate (flat)',
			value=0.05, min_value=0.0, max_value=0.15, format="%.4f", step=0.01, key='state_tax_rate')
		state_exempt_retirement = st.checkbox('Exempt retirement income from state tax (IL-style)',
			value=True, key='state_exempt_retirement')
		if state_exempt_retirement:
			st.caption('State tax applies only to investment income (interest, dividends, capital gains). SS, pensions, TDA withdrawals, and Roth conversions are exempt.')
		st.markdown('---')
		tcja_sunset = st.checkbox('Model TCJA sunset (brackets revert to pre-2018 law)',
			value=False, key='tcja_sunset',
			help='After the sunset year, tax brackets revert from TCJA (10/12/22/24/32/35/37%) '
			'to pre-TCJA (10/15/25/28/33/35/39.6%). Standard deduction also decreases. '
			'If using bracket-fill Roth conversions, note that TCJA-specific rates (12%, 22%, 24%) '
			'do not exist under pre-TCJA law and conversions will be skipped for those years.')
		if tcja_sunset:
			tcja_sunset_year = st.number_input('TCJA sunset begins (simulation year)',
				value=2, min_value=1, max_value=40, step=1, key='tcja_sunset_year',
				help='Simulation year when TCJA provisions expire. Year 2 = second year of simulation.')
		else:
			tcja_sunset_year = 2
		st.markdown('---')
		irmaa_enabled = st.checkbox('Model Medicare IRMAA surcharges',
			value=False, key='irmaa_enabled',
			help='Adds Medicare Part B + Part D income-related surcharges for persons age 65+. '
			'Based on MAGI from 2 years prior (surcharges start year 3 of simulation). '
			'2024 brackets: MFJ surcharges begin at $206K MAGI; single at $103K. '
			'Max surcharge ~$6,004/person/year.')
	return {
		'taxes_enabled': taxes_enabled,
		'filing_status_choice': filing_status_choice,
		'filing_status_key': filing_status_key,
		'use_itemized': use_itemized,
		'itemized_deduction': itemized_deduction_input,
		'inheritor_marginal_rate': inheritor_marginal_rate,
		'state_tax_rate': state_tax_rate,
		'state_exempt_retirement': state_exempt_retirement,
		'tcja_sunset': tcja_sunset,
		'tcja_sunset_year': tcja_sunset_year,
		'irmaa_enabled': irmaa_enabled,
	}

def _render_return_section():
	"""Render Return Assumptions expander. Returns dict of return values."""
	with st.expander('Return Assumptions'):
		return_mode = st.radio('Return mode', ['Simulated (lognormal)', 'Historical (master_global_factors)'], horizontal=False, index=1, key='return_mode')
		stock_total_return = 0.0
		bond_return = 0.0
		if return_mode == 'Simulated (lognormal)':
			st.caption('Both stocks and bonds use lognormal draws.')
			taxable_log_drift = st.number_input('Stock log drift (µ)', value=0.09038261, format="%.8f", key='taxable_log_drift')
			taxable_log_volatility = st.number_input('Stock log volatility (σ)', value=0.20485277, format="%.8f", key='taxable_log_volatility')
			bond_log_drift = st.number_input('Bond log drift (µ)', value=0.0172918, format="%.8f", key='bond_log_drift')
			bond_log_volatility = st.number_input('Bond log volatility (σ)', value=0.04796435, format="%.8f", key='bond_log_volatility')
			random_seed_input = st.number_input('Random seed for returns', value=42, step=1, key='random_seed')
			seed_mode = st.radio('Seed mode', ['Random each run', 'Fixed seed'], horizontal=True, index=0, key='seed_mode')
		else:
			st.caption('Returns from LBM 100E (stocks) and LBM 100 F (bonds). Runs all historical periods as a distribution.')
			taxable_log_drift = 0.0
			taxable_log_volatility = 0.0
			bond_log_drift = 0.0
			bond_log_volatility = 0.0
			random_seed_input = 42
			seed_mode = 'Fixed seed'
		stock_dividend_yield = st.number_input('Stock dividend (qualified) yield', value=0.02, format="%.4f", key='stock_dividend_yield')
		stock_turnover = st.number_input('Stock turnover rate', value=0.10, format="%.4f", key='stock_turnover')
		investment_fee_bps = st.number_input('Investment fee (basis points)', value=0.0, min_value=0.0, max_value=100.0, step=5.0, key='investment_fee_bps')
	return {
		'return_mode': return_mode,
		'stock_total_return': stock_total_return,
		'bond_return': bond_return,
		'taxable_log_drift': taxable_log_drift,
		'taxable_log_volatility': taxable_log_volatility,
		'bond_log_drift': bond_log_drift,
		'bond_log_volatility': bond_log_volatility,
		'random_seed_input': random_seed_input,
		'seed_mode': seed_mode,
		'stock_dividend_yield': stock_dividend_yield,
		'stock_turnover': stock_turnover,
		'investment_fee_bps': investment_fee_bps,
	}

def _render_guardrail_section():
	"""Render Withdrawal Guardrails expander. Returns dict of guardrail values."""
	with st.expander('Withdrawal Guardrails'):
		guardrails_enabled = st.checkbox('Enable dynamic withdrawal guardrails', value=True, key='guardrails_enabled')
		if guardrails_enabled:
			st.caption('At each year, checks forward MC survival rate. If outside the dead band, resets withdrawal to the target success rate.')
			guardrail_lower = st.number_input('Lower guardrail (reduce spending if below)', value=0.75, min_value=0.50, max_value=0.95, format="%.2f", step=0.05, key='guardrail_lower')
			guardrail_upper = st.number_input('Upper guardrail (increase spending if above)', value=0.90, min_value=0.50, max_value=0.99, format="%.2f", step=0.05, key='guardrail_upper')
			guardrail_target = st.number_input('Target success rate (reset to)', value=0.85, min_value=0.50, max_value=0.99, format="%.2f", step=0.05, key='guardrail_target')
			guardrail_inner_sims = st.number_input('Inner MC simulations per check', value=200, min_value=50, max_value=1000, step=50, key='guardrail_inner_sims')
			guardrail_max_spending_pct = st.number_input(
				'Max spending cap (% above base withdrawal, -1=no cap, 0=no increase)',
				value=50.0, min_value=-1.0, max_value=200.0, format="%.0f", step=10.0,
				help='0 = spending can never exceed base target. 50 = up to 150% of base. -1 = unlimited.',
				key='guardrail_max_spending_pct')
			flex_goal_min_pct = st.number_input(
				'Flexible goal minimum %', value=50.0, min_value=0.0, max_value=100.0,
				format="%.0f", step=10.0,
				help='Floor for flexible goal cuts. 50% = flexible goals can be cut by at most half before base spending is reduced.',
				key='flex_goal_min_pct') / 100.0
			base_is_essential = False
		else:
			guardrail_lower = 0.75
			guardrail_upper = 0.90
			guardrail_target = 0.85
			guardrail_inner_sims = 200
			guardrail_max_spending_pct = -1.0
			flex_goal_min_pct = 0.5
			base_is_essential = False
	return {
		'guardrails_enabled': guardrails_enabled,
		'guardrail_lower': guardrail_lower,
		'guardrail_upper': guardrail_upper,
		'guardrail_target': guardrail_target,
		'guardrail_inner_sims': guardrail_inner_sims,
		'guardrail_max_spending_pct': guardrail_max_spending_pct,
		'flex_goal_min_pct': flex_goal_min_pct,
		'base_is_essential': base_is_essential,
	}

def _render_sim_settings():
	"""Render Simulation Settings expander. Returns dict of sim settings."""
	with st.expander('Simulation Settings'):
		display_decimals = st.number_input('Decimal places for tables/charts', min_value=0, max_value=6, value=0, step=1, key='display_decimals')
		monte_carlo_runs = st.number_input('Monte Carlo runs', min_value=50, max_value=5000, value=1000, step=50, key='monte_carlo_runs')
	return {
		'display_decimals': display_decimals,
		'monte_carlo_runs': monte_carlo_runs,
	}

def _apply_pending_upload():
	"""If a JSON was uploaded, apply its values to session state before widgets render."""
	data = st.session_state.pop('_pending_upload', None)
	if data is None:
		return
	for k in _SAVEABLE_KEYS:
		if k in data:
			st.session_state[k] = data[k]
	if 'periods' in data:
		for i, p in enumerate(data['periods']):
			if 'amount' in p and p['amount'] is not None:
				st.session_state[f'wd_amount_{i}'] = p['amount']
			if 'end_year' in p:
				st.session_state[f'wd_end_{i}'] = p['end_year']
	if 'add_goals' in data:
		for g, goal in enumerate(data['add_goals']):
			st.session_state[f'add_goal_label_{g}'] = goal.get('label', '')
			st.session_state[f'add_goal_amount_{g}'] = goal.get('amount', 0.0)
			st.session_state[f'add_goal_begin_{g}'] = goal.get('begin', 1)
			st.session_state[f'add_goal_end_{g}'] = goal.get('end', 1)
			raw_priority = goal.get('priority', 'Essential')
			if raw_priority == 'Need': raw_priority = 'Essential'
			elif raw_priority == 'Want': raw_priority = 'Flexible'
			st.session_state[f'add_goal_priority_{g}'] = raw_priority
			st.session_state[f'add_goal_cap_{g}'] = goal.get('cap', -1.0)
			st.session_state[f'add_goal_fund_sep_{g}'] = goal.get('fund_sep', False)
			st.session_state[f'add_goal_fund_taxable_{g}'] = goal.get('fund_taxable', 0.0)
			st.session_state[f'add_goal_fund_tda1_{g}'] = goal.get('fund_tda1', 0.0)
			st.session_state[f'add_goal_fund_tda2_{g}'] = goal.get('fund_tda2', 0.0)
			st.session_state[f'add_goal_fund_stock_pct_{g}'] = goal.get('fund_stock_pct', 60)
	if 'scenario_overrides' in data:
		for s_idx_str, sc_data in data['scenario_overrides'].items():
			for key, val in sc_data.items():
				st.session_state[key] = val


def render_sidebar():
	"""Render all sidebar inputs. Returns structured dict of all values."""
	_apply_pending_upload()
	with st.sidebar:
		st.header('Inputs')
		_render_save_load_section()
		num_scenarios, scenario_overrides_ui = _render_scenario_section()
		ages = _render_ages_section()
		accounts = _render_accounts_section()
		inheritance = _render_inheritance_section()
		allocation = _render_allocation_section()
		horizon = max(1, max(int(ages['life_expectancy_primary']) - int(ages['start_age']),
							  int(ages['life_expectancy_spouse']) - int(ages['start_age_spouse'])) + 1)
		withdrawals = _render_withdrawal_section(horizon)
		add_goal_inputs = _render_add_goals_section(horizon)
		income = _render_income_section()
		buyout = _render_pension_buyout_section()
		tax = _render_tax_section()
		returns = _render_return_section()
		guardrails = _render_guardrail_section()
		sim = _render_sim_settings()

	return {
		'num_scenarios': num_scenarios,
		'scenario_overrides_ui': scenario_overrides_ui,
		**ages,
		**accounts,
		**inheritance,
		**allocation,
		**withdrawals,
		'add_goal_inputs': add_goal_inputs,
		**income,
		**buyout,
		**tax,
		**returns,
		**guardrails,
		**sim,
		'horizon': horizon,
	}
