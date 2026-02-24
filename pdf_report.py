"""
Client-facing PDF report for the Monte Carlo retirement withdrawal simulator.
Uses fpdf2 for PDF generation and matplotlib for static charts.
"""
import io
import numpy as np
import pandas as pd
from datetime import date
from fpdf import FPDF
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Helpers ─────────────────────────────────────────────────────

def _fmt_dollar(v):
    """Format a number as $X,XXX with no decimals."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '-'
    return f"${v:,.0f}"

def _fmt_pct(v, decimals=1):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '-'
    return f"{v * 100:.{decimals}f}%"

def _fig_to_png_bytes(fig, dpi=150):
    """Render a matplotlib figure to PNG bytes and close it."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    buf.seek(0)
    return buf


# ── Data extraction ─────────────────────────────────────────────

def _gather_report_data(ss: dict) -> dict:
    """Pull all needed data from session state dict into a clean report dict."""
    data = {}

    # Client info
    data['client'] = ss.get('client_select', 'Client')
    data['plan_name'] = ss.get('save_file_name', '')
    data['date'] = date.today().strftime('%B %d, %Y')

    # Ages / horizon
    data['start_age'] = int(ss.get('start_age', 65))
    data['start_age_spouse'] = int(ss.get('start_age_spouse', 0))
    data['life_expectancy'] = int(ss.get('life_expectancy_primary', 84))
    data['life_expectancy_spouse'] = int(ss.get('life_expectancy_spouse', 89))
    years_p1 = data['life_expectancy'] - data['start_age']
    years_p2 = data['life_expectancy_spouse'] - data['start_age_spouse'] if data['start_age_spouse'] > 0 else 0
    data['years'] = max(years_p1, years_p2) + 1

    # Portfolio
    data['taxable_start'] = float(ss.get('taxable_start', 0))
    data['tda_start'] = float(ss.get('tda_start', 0))
    data['tda_spouse_start'] = float(ss.get('tda_spouse_start', 0))
    data['roth_start'] = float(ss.get('roth_start', 0))
    data['total_portfolio_start'] = data['taxable_start'] + data['tda_start'] + data['tda_spouse_start'] + data['roth_start']
    data['stock_pct'] = float(ss.get('target_stock_pct', 60))

    # Income sources
    data['ss_income'] = float(ss.get('ss_income', 0))
    data['ss_start_age_p1'] = int(ss.get('ss_start_age_p1', 67))
    data['ss_income_spouse'] = float(ss.get('ss_income_spouse', 0))
    data['ss_start_age_p2'] = int(ss.get('ss_start_age_p2', 65))
    data['pension_income'] = float(ss.get('pension_income', 0))
    data['pension_income_spouse'] = float(ss.get('pension_income_spouse', 0))
    data['other_income'] = float(ss.get('other_income', 0))

    # Withdrawal schedule (reconstruct period ranges the same way main.py does)
    num_periods = int(ss.get('num_withdrawal_periods', 1))
    periods = []
    period_start = 1
    for i in range(num_periods):
        amt = float(ss.get(f'wd_amount_{i}', 0))
        is_last = (i == num_periods - 1)
        if is_last:
            period_end = data['years']
        else:
            period_end = int(ss.get(f'wd_end_{i}', period_start))
        periods.append((amt, period_start, period_end))
        period_start = period_end + 1
    data['withdrawal_periods'] = periods

    # Simulation settings
    data['num_sims'] = int(ss.get('num_sims', 0))
    data['sim_mode'] = ss.get('sim_mode', '')
    data['return_mode'] = ss.get('return_mode', 'Lognormal (Monte Carlo)')
    data['taxes_enabled'] = bool(ss.get('taxes_enabled', True))
    data['guardrails_enabled'] = bool(ss.get('guardrails_enabled', False))
    data['investment_fee_bps'] = float(ss.get('investment_fee_bps', 0))
    data['guardrail_max_spending_pct'] = float(ss.get('guardrail_max_spending_pct', 0))
    data['filing_status'] = ss.get('filing_status', 'Married Filing Jointly')

    # Results
    data['pct_rows'] = ss.get('mc_percentile_rows', [])
    data['pct_non_positive'] = float(ss.get('mc_pct_non_positive', 0))
    data['spending_pct_rows'] = ss.get('mc_spending_pct_rows', [])
    data['all_yearly'] = ss.get('mc_all_yearly', None)
    data['sim_df'] = ss.get('sim_df', None)  # median single run

    # Multi-scenario
    data['multi_results'] = ss.get('multi_scenario_results', None)

    return data


# ── Chart functions ─────────────────────────────────────────────

_COLORS = {
    'band_fill': '#d4e6f1',
    'median_line': '#2471a3',
    'bad_line': '#cb4335',
    'good_line': '#27ae60',
    'ss': '#5dade2',
    'pension': '#58d68d',
    'portfolio': '#af7ac5',
    'other': '#f0b27a',
    'taxes': '#e74c3c',
}

def _chart_portfolio_bands(all_yearly: pd.DataFrame) -> io.BytesIO:
    """Portfolio value over time with 10th/50th/90th percentile bands."""
    grouped = all_yearly.groupby('year')['total_portfolio']
    p10 = grouped.quantile(0.10)
    p50 = grouped.quantile(0.50)
    p90 = grouped.quantile(0.90)
    years = p50.index.values

    fig, ax = plt.subplots(figsize=(7.5, 3.5))
    ax.fill_between(years, p10.values, p90.values, alpha=0.25, color=_COLORS['band_fill'], label='10th–90th range')
    ax.plot(years, p10.values, color=_COLORS['bad_line'], linewidth=1, alpha=0.7, label='10th (bad)')
    ax.plot(years, p50.values, color=_COLORS['median_line'], linewidth=2, label='50th (typical)')
    ax.plot(years, p90.values, color=_COLORS['good_line'], linewidth=1, alpha=0.7, label='90th (good)')
    ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
    ax.set_xlabel('Year', fontsize=9)
    ax.set_ylabel('Portfolio Value', fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M' if abs(x) >= 1e6 else f'${x/1e3:.0f}K'))
    ax.legend(fontsize=7, loc='upper right')
    ax.set_title('Range of Portfolio Outcomes', fontsize=11, fontweight='bold')
    fig.tight_layout()
    return _fig_to_png_bytes(fig)


def _chart_spending_sources(sim_df: pd.DataFrame) -> io.BytesIO:
    """Stacked area showing where income comes from for the median run."""
    df = sim_df.copy()
    # Compute net portfolio withdrawal as spending minus other income sources
    df['Net Portfolio'] = (df['after_tax_spending']
                           - df['ss_income_total']
                           - df.get('pension_income_real', df.get('pension_income_total', 0))
                           - df.get('annuity_income_real', pd.Series(0, index=df.index))
                           - df['other_income']).clip(lower=0)
    pension_col = 'pension_income_real' if 'pension_income_real' in df.columns else 'pension_income_total'

    sources = {}
    if df['ss_income_total'].sum() > 0:
        sources['Social Security'] = df['ss_income_total'].values
    if df[pension_col].sum() > 0:
        sources['Pension'] = df[pension_col].values
    if 'annuity_income_real' in df.columns and df['annuity_income_real'].sum() > 0:
        sources['Annuity'] = df['annuity_income_real'].values
    if df['other_income'].sum() > 0:
        sources['Other Income'] = df['other_income'].values
    if df['Net Portfolio'].sum() > 0:
        sources['Portfolio Withdrawals'] = df['Net Portfolio'].values

    colors_map = {
        'Social Security': _COLORS['ss'],
        'Pension': _COLORS['pension'],
        'Portfolio Withdrawals': _COLORS['portfolio'],
        'Other Income': _COLORS['other'],
        'Annuity': '#f7dc6f',
    }

    years = df['year'].values if 'year' in df.columns else np.arange(len(df))

    fig, ax = plt.subplots(figsize=(7.5, 3.5))
    if sources:
        labels = list(sources.keys())
        values = np.array([sources[k] for k in labels])
        colors = [colors_map.get(k, '#cccccc') for k in labels]
        ax.stackplot(years, *values, labels=labels, colors=colors, alpha=0.85)
        ax.legend(fontsize=7, loc='upper right')
    ax.set_xlabel('Year', fontsize=9)
    ax.set_ylabel('Annual Income', fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'${x/1e3:.0f}K'))
    ax.set_title('Where Your Money Comes From (Median Run)', fontsize=11, fontweight='bold')
    fig.tight_layout()
    return _fig_to_png_bytes(fig)


def _chart_scenario_comparison(multi_results: list) -> io.BytesIO:
    """Grouped bar comparing scenarios: success rate and median ending portfolio."""
    names = [s['name'] for s in multi_results]
    success = [round((1 - s['pct_non_positive']) * 100, 1) for s in multi_results]

    # Median ending portfolio
    ending = []
    for s in multi_results:
        row50 = next((r for r in s['percentile_rows'] if r['percentile'] == 50), None)
        ending.append(row50['after_tax_end'] if row50 else 0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3))
    x = np.arange(len(names))
    bar_w = 0.5

    # Success rate
    bars1 = ax1.bar(x, success, bar_w, color=_COLORS['median_line'], alpha=0.8)
    ax1.set_ylabel('Success Rate (%)', fontsize=9)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=7, rotation=15, ha='right')
    ax1.set_ylim(0, 105)
    for bar, val in zip(bars1, success):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.0f}%',
                ha='center', va='bottom', fontsize=8)

    # Ending portfolio
    bars2 = ax2.bar(x, [v/1000 for v in ending], bar_w, color=_COLORS['good_line'], alpha=0.8)
    ax2.set_ylabel('Median Ending ($K)', fontsize=9)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, fontsize=7, rotation=15, ha='right')
    for bar, val in zip(bars2, ending):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, _fmt_dollar(val),
                ha='center', va='bottom', fontsize=7)

    fig.suptitle('Scenario Comparison', fontsize=11, fontweight='bold')
    fig.tight_layout()
    return _fig_to_png_bytes(fig)


# ── PDF class ───────────────────────────────────────────────────

class RetirementReportPDF(FPDF):
    def __init__(self, client_name='Client', report_date=''):
        super().__init__()
        self.client_name = client_name
        self.report_date = report_date
        self.set_auto_page_break(auto=True, margin=20)

    def header(self):
        if self.page_no() == 1:
            return  # custom first page header
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 6, f'{self.client_name} - Retirement Plan Report', align='L')
        self.cell(0, 6, self.report_date, align='R', new_x='LMARGIN', new_y='NEXT')
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')

    def section_title(self, title):
        self.set_font('Helvetica', 'B', 14)
        self.set_text_color(36, 113, 163)
        self.cell(0, 10, title, new_x='LMARGIN', new_y='NEXT')
        self.set_draw_color(36, 113, 163)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def body_text(self, text, size=10):
        self.set_font('Helvetica', '', size)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 5.5, text)
        self.ln(2)

    def bold_text(self, text, size=10):
        self.set_font('Helvetica', 'B', size)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 5.5, text)
        self.ln(1)

    def simple_table(self, headers, rows, col_widths=None, header_fill=(36, 113, 163)):
        """Render a simple table with header row and data rows."""
        if col_widths is None:
            col_widths = [190 / len(headers)] * len(headers)

        # Header
        self.set_font('Helvetica', 'B', 9)
        self.set_fill_color(*header_fill)
        self.set_text_color(255, 255, 255)
        for i, h in enumerate(headers):
            align = 'L' if i == 0 else 'R'
            self.cell(col_widths[i], 7, h, border=1, align=align, fill=True)
        self.ln()

        # Data rows
        self.set_font('Helvetica', '', 9)
        self.set_text_color(40, 40, 40)
        fill = False
        for row in rows:
            if fill:
                self.set_fill_color(235, 245, 251)
            else:
                self.set_fill_color(255, 255, 255)
            for i, cell in enumerate(row):
                align = 'L' if i == 0 else 'R'
                self.cell(col_widths[i], 6.5, str(cell), border=1, align=align, fill=True)
            self.ln()
            fill = not fill
        self.ln(3)


# ── Page builders ───────────────────────────────────────────────

def _build_page_1(pdf: RetirementReportPDF, data: dict):
    """Page 1: The Big Picture - key assumptions and headline success rate."""
    pdf.add_page()

    # Title block
    pdf.set_font('Helvetica', 'B', 22)
    pdf.set_text_color(36, 113, 163)
    pdf.ln(10)
    pdf.cell(0, 12, 'Retirement Plan Report', align='C', new_x='LMARGIN', new_y='NEXT')
    pdf.ln(2)
    pdf.set_font('Helvetica', '', 12)
    pdf.set_text_color(80, 80, 80)
    client_display = data['client'].replace(',', ', ') if ',' in data['client'] else data['client']
    pdf.cell(0, 8, f"Prepared for {client_display}", align='C', new_x='LMARGIN', new_y='NEXT')
    pdf.cell(0, 8, data['date'], align='C', new_x='LMARGIN', new_y='NEXT')
    pdf.ln(8)

    # Success rate highlight box
    success_rate = (1 - data['pct_non_positive']) * 100
    pdf.set_fill_color(234, 250, 241) if success_rate >= 80 else pdf.set_fill_color(253, 237, 236)
    box_y = pdf.get_y()
    pdf.rect(30, box_y, 150, 28, style='F')
    pdf.set_draw_color(39, 174, 96) if success_rate >= 80 else pdf.set_draw_color(203, 67, 53)
    pdf.rect(30, box_y, 150, 28, style='D')

    pdf.set_xy(30, box_y + 4)
    pdf.set_font('Helvetica', 'B', 18)
    pdf.set_text_color(39, 174, 96) if success_rate >= 80 else pdf.set_text_color(203, 67, 53)
    pdf.cell(150, 10, f"{success_rate:.0f}% Success Rate", align='C', new_x='LMARGIN', new_y='NEXT')
    pdf.set_xy(30, box_y + 16)
    pdf.set_font('Helvetica', '', 9)
    pdf.set_text_color(80, 80, 80)
    mode_label = 'historical periods' if 'Historical' in data.get('return_mode', '') else 'simulations'
    pdf.cell(150, 6, f"Your portfolio lasted the full plan in {success_rate:.0f}% of {data['num_sims']:,} {mode_label}",
             align='C')
    pdf.ln(18)

    # Key assumptions section
    pdf.section_title('Your Plan at a Glance')

    # Build assumption lines
    lines = []
    lines.append(f"Starting age: {data['start_age']}")
    if data['start_age_spouse'] > 0:
        lines.append(f"Spouse starting age: {data['start_age_spouse']}")
    last_to_die_age = max(data['life_expectancy'], data['life_expectancy_spouse'])
    lines.append(f"Planning through age: {last_to_die_age} ({data['years']} years)")

    lines.append(f"Beginning portfolio: {_fmt_dollar(data['total_portfolio_start'])}")

    acct_parts = []
    if data['taxable_start'] > 0:
        acct_parts.append(f"Taxable {_fmt_dollar(data['taxable_start'])}")
    tda_total = data['tda_start'] + data['tda_spouse_start']
    if tda_total > 0:
        acct_parts.append(f"Tax-Deferred {_fmt_dollar(tda_total)}")
    if data['roth_start'] > 0:
        acct_parts.append(f"Roth {_fmt_dollar(data['roth_start'])}")
    if acct_parts:
        lines.append(f"  Accounts: {', '.join(acct_parts)}")

    lines.append(f"Investment mix: {data['stock_pct']:.0f}% stocks / {100 - data['stock_pct']:.0f}% bonds")

    # Withdrawal schedule
    if data['withdrawal_periods']:
        for i, (amt, start_yr, end_yr) in enumerate(data['withdrawal_periods']):
            if amt > 0:
                lines.append(f"Annual spending target (period {i+1}): {_fmt_dollar(amt)}/year, years {start_yr}-{end_yr}")
    if data['guardrails_enabled']:
        cap_pct = data.get('guardrail_max_spending_pct', 0)
        if cap_pct > 0:
            lines.append(f"  Spending cap: {cap_pct:.0f}% above base target")

    # Income sources
    if data['ss_income'] > 0:
        lines.append(f"Social Security (Person 1): {_fmt_dollar(data['ss_income'])}/year starting at age {data['ss_start_age_p1']}")
    if data['ss_income_spouse'] > 0:
        lines.append(f"Social Security (Person 2): {_fmt_dollar(data['ss_income_spouse'])}/year starting at age {data['ss_start_age_p2']}")
    if data['pension_income'] > 0:
        lines.append(f"Pension (Person 1): {_fmt_dollar(data['pension_income'])}/year")
    if data['pension_income_spouse'] > 0:
        lines.append(f"Pension (Person 2): {_fmt_dollar(data['pension_income_spouse'])}/year")
    if data['other_income'] > 0:
        lines.append(f"Other income: {_fmt_dollar(data['other_income'])}/year")

    # Return methodology (read from sidebar input, not sim_mode)
    return_mode = data.get('return_mode', '')
    if 'Historical' in return_mode:
        lines.append(f"Return method: Historical distribution ({data['num_sims']} rolling periods)")
    else:
        lines.append(f"Return method: Simulated lognormal ({data['num_sims']:,} runs)")

    if data['taxes_enabled']:
        lines.append(f"Tax filing status: {data['filing_status']}")
    if data.get('investment_fee_bps', 0) > 0:
        lines.append(f"Investment fees: {data['investment_fee_bps']:.0f} basis points")
    if data['guardrails_enabled']:
        lines.append("Dynamic spending guardrails: Enabled")

    for line in lines:
        pdf.body_text(line)

    # Footnote
    pdf.ln(4)
    pdf.set_font('Helvetica', 'I', 8)
    pdf.set_text_color(130, 130, 130)
    pdf.multi_cell(0, 4, 'All values are in today\'s dollars (adjusted for inflation). '
                   'This analysis is based on simulated market returns and is not a guarantee of future results.')


def _build_page_2(pdf: RetirementReportPDF, data: dict):
    """Page 2: Range of Outcomes - 10th/50th/90th table + portfolio bands chart."""
    pdf.add_page()
    pdf.section_title('Range of Outcomes')
    pdf.body_text(
        'No one can predict exactly what will happen with investments. The table below shows '
        'three possible outcomes based on our simulations: a bad scenario (10th percentile - only '
        '10% of outcomes were worse), a typical scenario (50th - the middle outcome), and a good '
        'scenario (90th percentile - only 10% did better).'
    )

    pct_rows = data['pct_rows']
    spend_rows = data['spending_pct_rows']

    # Build 3-row table: Bad / Typical / Good
    headers = ['Outcome', 'Ending Portfolio', 'Total Taxes', 'Eff. Tax Rate', 'Avg Annual Spending']
    col_widths = [30, 40, 35, 30, 55]
    rows = []
    label_map = {10: 'Bad (10th)', 50: 'Typical (50th)', 90: 'Good (90th)'}

    for pctile in [10, 50, 90]:
        prow = next((r for r in pct_rows if r['percentile'] == pctile), None)
        srow = next((r for r in spend_rows if r['percentile'] == pctile), None)
        if prow:
            avg_spend = _fmt_dollar(srow.get('avg_annual_after_tax_spending', 0)) if srow else '-'
            rows.append([
                label_map.get(pctile, f'{pctile}th'),
                _fmt_dollar(prow['after_tax_end']),
                _fmt_dollar(prow['total_taxes']),
                _fmt_pct(prow['effective_tax_rate']),
                avg_spend,
            ])

    pdf.simple_table(headers, rows, col_widths)

    # Portfolio bands chart
    if data['all_yearly'] is not None and len(data['all_yearly']) > 0:
        chart_buf = _chart_portfolio_bands(data['all_yearly'])
        pdf.image(chart_buf, x=15, w=180)

    pdf.ln(3)
    pdf.set_font('Helvetica', 'I', 8)
    pdf.set_text_color(130, 130, 130)
    pdf.multi_cell(0, 4, 'The shaded area shows the range of outcomes between the 10th and 90th '
                   'percentiles. The blue line shows the typical (median) path.')


def _build_page_3(pdf: RetirementReportPDF, data: dict):
    """Page 3: Where Your Money Comes From - median year-by-year table + chart."""
    pdf.add_page()
    pdf.section_title('Where Your Money Comes From')
    pdf.body_text(
        'This shows the typical (median) path for your retirement income. Each row represents '
        'a year, showing how much comes from Social Security, pensions, portfolio withdrawals, '
        'and how much you have remaining.'
    )

    all_yearly = data['all_yearly']
    sim_df = data['sim_df']

    if all_yearly is not None and len(all_yearly) > 0:
        # Build median year-by-year from all runs
        median_by_year = all_yearly.groupby('year').agg({
            'age_p1': 'first',
            'ss_income_total': 'median',
            'pension_income_real': 'median' if 'pension_income_real' in all_yearly.columns else 'first',
            'after_tax_spending': 'median',
            'total_portfolio': 'median',
            'withdrawal_used': 'median',
            'total_taxes': 'median',
        }).reset_index()

        # Determine row skip to fit on page (aim for ~20 rows max)
        n_years = len(median_by_year)
        if n_years > 25:
            step = 3
        elif n_years > 15:
            step = 2
        else:
            step = 1

        headers = ['Year', 'Age', 'Soc Sec', 'Pension', 'Spending', 'Taxes', 'Portfolio']
        col_widths = [18, 16, 28, 28, 30, 28, 42]
        rows = []
        for i in range(0, n_years, step):
            r = median_by_year.iloc[i]
            rows.append([
                str(int(r['year'])),
                str(int(r['age_p1'])),
                _fmt_dollar(r['ss_income_total']),
                _fmt_dollar(r.get('pension_income_real', 0)),
                _fmt_dollar(r['after_tax_spending']),
                _fmt_dollar(r['total_taxes']),
                _fmt_dollar(r['total_portfolio']),
            ])
        # Always include last year
        if n_years > 1 and (n_years - 1) % step != 0:
            r = median_by_year.iloc[-1]
            rows.append([
                str(int(r['year'])),
                str(int(r['age_p1'])),
                _fmt_dollar(r['ss_income_total']),
                _fmt_dollar(r.get('pension_income_real', 0)),
                _fmt_dollar(r['after_tax_spending']),
                _fmt_dollar(r['total_taxes']),
                _fmt_dollar(r['total_portfolio']),
            ])

        pdf.simple_table(headers, rows, col_widths)

        if step > 1:
            pdf.set_font('Helvetica', 'I', 8)
            pdf.set_text_color(130, 130, 130)
            pdf.cell(0, 4, f'Showing every {_ordinal(step)} year. All values are medians across simulations.',
                     new_x='LMARGIN', new_y='NEXT')
            pdf.ln(3)

    # Income sources chart (from median single run)
    if sim_df is not None and len(sim_df) > 0:
        chart_buf = _chart_spending_sources(sim_df)
        pdf.image(chart_buf, x=15, w=180)


def _build_page_4(pdf: RetirementReportPDF, data: dict):
    """Page 4: Scenario Comparison (only if multiple scenarios)."""
    multi = data.get('multi_results')
    if not multi or len(multi) < 2:
        return

    pdf.add_page()
    pdf.section_title('Scenario Comparison')
    pdf.body_text(
        'We tested different approaches to see how they affect your retirement. '
        'Here\'s how each scenario compares.'
    )

    headers = ['Scenario', 'Success Rate', 'Ending Portfolio', 'Avg Spending', 'Total Taxes']
    col_widths = [50, 28, 38, 38, 36]
    rows = []
    for sc in multi:
        success = f"{(1 - sc['pct_non_positive']) * 100:.0f}%"
        row50 = next((r for r in sc['percentile_rows'] if r['percentile'] == 50), None)
        end_val = _fmt_dollar(row50['after_tax_end']) if row50 else '-'
        taxes = _fmt_dollar(row50['total_taxes']) if row50 else '-'
        spend50 = next((r for r in sc['spending_percentiles'] if r['percentile'] == 50), None)
        avg_spend = _fmt_dollar(spend50.get('avg_annual_after_tax_spending', 0)) if spend50 else '-'
        rows.append([sc['name'], success, end_val, avg_spend, taxes])

    pdf.simple_table(headers, rows, col_widths)

    # Chart
    chart_buf = _chart_scenario_comparison(multi)
    pdf.image(chart_buf, x=15, w=180)


def _ordinal(n):
    if n == 2:
        return '2nd'
    if n == 3:
        return '3rd'
    return f'{n}th'


# ── Entry point ─────────────────────────────────────────────────

def generate_report(session_state: dict) -> bytes:
    """Build the full client PDF report and return as bytes."""
    data = _gather_report_data(session_state)

    pdf = RetirementReportPDF(
        client_name=data['client'],
        report_date=data['date'],
    )

    _build_page_1(pdf, data)
    _build_page_2(pdf, data)
    _build_page_3(pdf, data)
    _build_page_4(pdf, data)

    return bytes(pdf.output())
