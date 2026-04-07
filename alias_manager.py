"""Local Streamlit app for managing client aliases.

Run with: streamlit run alias_manager.py
"""

import json
import os
import sys
sys.path.insert(0, os.path.expanduser('~/RWM'))
import streamlit as st
from pseudonymize import (
    init_aliases, add_client, delete_client, rename_client, list_aliases,
    resolve_alias, _load_alias_map, CLIENT_DIR,
)

st.set_page_config(page_title="Client Alias Manager", layout="centered")
st.title("Client Alias Manager")

# ── Add New Client ───────────────────────────────────────────
st.header("Add New Client")
real_name = st.text_input("Client name (e.g. Smith, Bob & Mary)", key="new_client")

if st.button("Generate Alias", type="primary"):
    if real_name.strip():
        alias = add_client(real_name.strip())
        st.success(f"**Alias: {alias}**")
        st.info(f'Tell Claude: "New client **{alias}**" and then give the plan details.')
    else:
        st.warning("Enter a client name first.")

# ── Client List ──────────────────────────────────────────────
st.divider()
st.header("All Client Aliases")

show_real = st.checkbox("Show real names", value=False)

entries = list_aliases(reveal=show_real)
if entries:
    for e in entries:
        alias = e['alias']
        labels = e.get('labels', [])
        plans = e['plans']

        if show_real:
            col1, col2 = st.columns([1, 1])
            col1.markdown(f"**{alias}**")
            col2.markdown(f'<span style="color: #1E90FF; font-size: 1.1rem;">{e.get("real_name", "—")}</span>', unsafe_allow_html=True)
        else:
            st.markdown(f"**{alias}**")

        if labels:
            # Get the actual plan files to read inputs
            real_name_for_plans = e.get('real_name') or _load_alias_map().get(alias, '')
            plan_dir = os.path.join(CLIENT_DIR, real_name_for_plans)
            plan_files = sorted([
                f for f in os.listdir(plan_dir)
                if f.endswith('.json')
                and not f.endswith('_results.json')
                and not f.endswith('_plan.json')
            ]) if os.path.isdir(plan_dir) else []

            for i, label in enumerate(labels):
                plan_data = None
                if i < len(plan_files):
                    try:
                        with open(os.path.join(plan_dir, plan_files[i])) as fh:
                            plan_data = json.load(fh)
                    except Exception:
                        pass

                if plan_data:
                    with st.expander(f"📄 {label}"):
                        total = (plan_data.get('taxable_start', 0)
                                 + plan_data.get('tda_start', 0)
                                 + plan_data.get('tda_spouse_start', 0)
                                 + plan_data.get('roth_start', 0))
                        periods = plan_data.get('periods', [])
                        spending = periods[0].get('amount', 0) if periods else 0

                        c1, c2 = st.columns(2)
                        with c1:
                            st.markdown("**People**")
                            p1_label = plan_data.get('person1_label', '')
                            p2_label = plan_data.get('person2_label', '')
                            p1_name = f" ({p1_label})" if p1_label else ""
                            p2_name = f" ({p2_label})" if p2_label else ""
                            st.write(f"Person 1{p1_name}: age {plan_data.get('start_age', '—')}, LE {plan_data.get('life_expectancy_primary', '—')}")
                            st.write(f"Person 2{p2_name}: age {plan_data.get('start_age_spouse', '—')}, LE {plan_data.get('life_expectancy_spouse', '—')}")

                            st.markdown("**Accounts**")
                            st.write(f"Taxable: ${plan_data.get('taxable_start', 0):,.0f}")
                            st.write(f"TDA P1: ${plan_data.get('tda_start', 0):,.0f}")
                            st.write(f"TDA P2: ${plan_data.get('tda_spouse_start', 0):,.0f}")
                            st.write(f"Roth: ${plan_data.get('roth_start', 0):,.0f}")
                            st.write(f"**Total: ${total:,.0f}**")

                        with c2:
                            st.markdown("**Allocation & Spending**")
                            st.write(f"Stocks: {plan_data.get('target_stock_pct', 60)}%")
                            st.write(f"Spending: ${spending:,.0f}/yr")
                            cap = plan_data.get('guardrail_max_spending_pct', 50)
                            st.write(f"Spending cap: {cap:.0f}%")

                            st.markdown("**Social Security**")
                            ss1 = plan_data.get('ss_income', 0)
                            ss2 = plan_data.get('ss_income_spouse', 0)
                            if ss1 > 0:
                                st.write(f"P1: ${ss1:,.0f} at age {plan_data.get('ss_start_age_p1', '—')}")
                            else:
                                st.write("P1: None")
                            if ss2 > 0:
                                st.write(f"P2: ${ss2:,.0f} at age {plan_data.get('ss_start_age_p2', '—')}")
                            else:
                                st.write("P2: None")

                            pen1 = plan_data.get('pension_income', 0)
                            pen2 = plan_data.get('pension_income_spouse', 0)
                            if pen1 > 0 or pen2 > 0:
                                st.markdown("**Pensions**")
                                if pen1 > 0:
                                    st.write(f"P1: ${pen1:,.0f}/yr")
                                if pen2 > 0:
                                    st.write(f"P2: ${pen2:,.0f}/yr")
                else:
                    st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;📄 {label}")
        elif plans > 0:
            st.caption(f"    {plans} plan(s)")
        else:
            st.caption("    No plans yet")
else:
    st.info("No aliases yet. Add a client above or click Sync below.")

# ── Edit Client Name ─────────────────────────────────────────
st.divider()
st.header("Edit Client Name")

alias_map_edit = _load_alias_map()
edit_options = sorted(alias_map_edit.keys())
if edit_options:
    edit_alias = st.selectbox("Select client to edit", edit_options, key="edit_alias")
    current_name = alias_map_edit.get(edit_alias, '')
    new_name = st.text_input("Corrected name", value=current_name, key="edit_name")

    if st.button("Save Name", type="primary"):
        if new_name.strip() and new_name.strip() != current_name:
            rename_client(edit_alias, new_name.strip())
            st.success(f"Updated **{edit_alias}** → {new_name.strip()}")
            st.rerun()
        elif new_name.strip() == current_name:
            st.info("Name is unchanged.")
        else:
            st.warning("Name cannot be empty.")
else:
    st.caption("No clients to edit.")

# ── Delete Client ────────────────────────────────────────────
st.divider()
st.header("Delete Client")

alias_map = _load_alias_map()
alias_options = sorted(alias_map.keys())
if alias_options:
    selected_alias = st.selectbox("Select client to delete", alias_options, key="delete_alias")
    delete_files = st.checkbox("Also delete all saved plan files", value=False)

    if f"confirm_delete_{selected_alias}" not in st.session_state:
        st.session_state[f"confirm_delete_{selected_alias}"] = False

    if st.button("Delete", type="secondary"):
        st.session_state[f"confirm_delete_{selected_alias}"] = True

    if st.session_state.get(f"confirm_delete_{selected_alias}", False):
        st.warning(f"Are you sure you want to delete **{selected_alias}**"
                   + (" and all their plan files?" if delete_files else "?"))
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Yes, delete", type="primary"):
                delete_client(selected_alias, delete_files=delete_files)
                st.session_state[f"confirm_delete_{selected_alias}"] = False
                st.success(f"Deleted {selected_alias}.")
                st.rerun()
        with col2:
            if st.button("Cancel"):
                st.session_state[f"confirm_delete_{selected_alias}"] = False
                st.rerun()
else:
    st.caption("No clients to delete.")

# ── Sync ─────────────────────────────────────────────────────
st.divider()
if st.button("Sync aliases with client folders"):
    alias_map = init_aliases()
    st.success(f"Synced {len(alias_map)} clients.")
    st.rerun()
