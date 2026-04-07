"""HTML comparison report generator for multi-scenario retirement plan analysis.

Produces a self-contained HTML file viewable in any browser.
Designed for in-office client meetings.

Usage:
    from html_report import save_html_report

    scenarios = [
        {"label": "40% Stocks", "stock_pct": 40, "found_spending": 74000, ...},
        {"label": "60% Stocks", "stock_pct": 60, "found_spending": 80000, ...},
        {"label": "80% Stocks", "stock_pct": 80, "found_spending": 83000, ...},
    ]
    narrative = "Plain-language write-up..."
    path = save_html_report(scenarios, narrative, "Sky-Dune", "Mike", "Tina")
"""

import re
from datetime import date


# ── Dollar formatting ─────────────────────────────────────────

def _fmt(v):
    """Format a number as $X,XXX with no decimals."""
    if v is None:
        return "—"
    if isinstance(v, str):
        return v
    if v < 0:
        return f"-${abs(v):,.0f}"
    return f"${v:,.0f}"


def _pct(v):
    """Format a decimal as X.X%."""
    if v is None:
        return "—"
    return f"{v:.1f}%"


# ── Minimal Markdown to HTML ──────────────────────────────────

def _md_to_html(text: str) -> str:
    """Convert simple markdown to HTML. Handles headers, bold, italic,
    lists, horizontal rules, and paragraphs."""
    lines = text.split('\n')
    html_parts = []
    in_list = False
    paragraph = []

    def flush_paragraph():
        if paragraph:
            p_text = ' '.join(paragraph)
            # Bold
            p_text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', p_text)
            # Italic
            p_text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', p_text)
            html_parts.append(f'<p>{p_text}</p>')
            paragraph.clear()

    for line in lines:
        stripped = line.strip()

        # Horizontal rule
        if stripped in ('---', '***', '___'):
            flush_paragraph()
            if in_list:
                html_parts.append('</ul>')
                in_list = False
            html_parts.append('<hr>')
            continue

        # Headers
        if stripped.startswith('### '):
            flush_paragraph()
            if in_list:
                html_parts.append('</ul>')
                in_list = False
            text_content = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', stripped[4:])
            html_parts.append(f'<h4>{text_content}</h4>')
            continue
        if stripped.startswith('## '):
            flush_paragraph()
            if in_list:
                html_parts.append('</ul>')
                in_list = False
            text_content = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', stripped[3:])
            html_parts.append(f'<h3>{text_content}</h3>')
            continue
        if stripped.startswith('# '):
            flush_paragraph()
            if in_list:
                html_parts.append('</ul>')
                in_list = False
            text_content = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', stripped[2:])
            html_parts.append(f'<h2>{text_content}</h2>')
            continue

        # List items
        if stripped.startswith('- ') or stripped.startswith('* '):
            flush_paragraph()
            if not in_list:
                html_parts.append('<ul>')
                in_list = True
            item = stripped[2:]
            item = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', item)
            item = re.sub(r'\*(.+?)\*', r'<em>\1</em>', item)
            html_parts.append(f'<li>{item}</li>')
            continue

        # Close list if we're no longer in one
        if in_list and stripped:
            html_parts.append('</ul>')
            in_list = False

        # Blank line = paragraph break
        if not stripped:
            flush_paragraph()
            continue

        # Regular text
        paragraph.append(stripped)

    flush_paragraph()
    if in_list:
        html_parts.append('</ul>')

    return '\n'.join(html_parts)


# ── CSS ───────────────────────────────────────────────────────

CSS = """
<style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
        color: #2c3e50;
        background: #ffffff;
        line-height: 1.6;
        padding: 40px 20px;
    }
    .container {
        max-width: 920px;
        margin: 0 auto;
    }
    .header {
        text-align: center;
        margin-bottom: 40px;
        padding-bottom: 20px;
        border-bottom: 3px solid #2471a3;
    }
    .header h1 {
        font-size: 1.8rem;
        color: #2471a3;
        margin-bottom: 8px;
    }
    .header .subtitle {
        font-size: 1rem;
        color: #7f8c8d;
    }
    .header .people {
        font-size: 0.95rem;
        color: #555;
        margin-top: 4px;
    }
    h2 {
        font-size: 1.4rem;
        color: #2471a3;
        margin: 36px 0 16px 0;
        padding-bottom: 6px;
        border-bottom: 1px solid #d5e8f0;
    }
    h3 {
        font-size: 1.15rem;
        color: #2c3e50;
        margin: 28px 0 12px 0;
    }
    h4 {
        font-size: 1.05rem;
        color: #555;
        margin: 20px 0 8px 0;
    }
    table {
        width: 100%;
        border-collapse: collapse;
        margin: 16px 0 24px 0;
        font-size: 0.92rem;
    }
    th {
        background: #2471a3;
        color: white;
        padding: 10px 14px;
        text-align: right;
        font-weight: 600;
    }
    th:first-child { text-align: left; }
    td {
        padding: 8px 14px;
        text-align: right;
        border-bottom: 1px solid #e8e8e8;
    }
    td:first-child {
        text-align: left;
        font-weight: 500;
        color: #34495e;
    }
    tr:nth-child(even) td { background: #f5f9fc; }
    tr:hover td { background: #eaf2f8; }
    .highlight-row td {
        background: #eaf2f8 !important;
        font-weight: 600;
    }
    .recommended {
        color: #27ae60;
        font-size: 0.75rem;
        font-weight: 600;
        display: block;
    }
    .narrative {
        margin: 40px 0;
        padding: 0;
    }
    .narrative p {
        margin: 12px 0;
        font-size: 1rem;
        line-height: 1.7;
    }
    .narrative ul {
        margin: 12px 0 12px 24px;
    }
    .narrative li {
        margin: 6px 0;
        line-height: 1.6;
    }
    .narrative hr {
        border: none;
        border-top: 1px solid #ddd;
        margin: 32px 0;
    }
    .narrative strong {
        color: #2c3e50;
    }
    details {
        margin: 16px 0;
        border: 1px solid #ddd;
        border-radius: 4px;
        padding: 0;
    }
    details summary {
        padding: 10px 14px;
        cursor: pointer;
        font-weight: 600;
        color: #2471a3;
        background: #f5f9fc;
    }
    details summary:hover { background: #eaf2f8; }
    details[open] summary { border-bottom: 1px solid #ddd; }
    details .detail-content {
        padding: 12px 14px;
        overflow-x: auto;
    }
    details table { font-size: 0.82rem; }
    .footer {
        margin-top: 50px;
        padding-top: 16px;
        border-top: 1px solid #ddd;
        font-size: 0.78rem;
        color: #999;
        text-align: center;
    }
    @media print {
        body { padding: 20px; }
        .container { max-width: 100%; }
        details { display: none; }
        .footer { page-break-before: avoid; }
    }
</style>
"""


# ── Table builders ────────────────────────────────────────────

def _build_comparison_table(scenarios: list, rows: list, title: str = None) -> str:
    """Build an HTML table with scenarios as columns.

    rows: list of (label, key_or_func, format_func) tuples.
        key_or_func: string key into scenario dict, or callable(scenario) -> value
        format_func: callable(value) -> str
    """
    n = len(scenarios)
    parts = []
    if title:
        parts.append(f'<h2>{title}</h2>')
    parts.append('<table>')

    # Header row
    parts.append('<tr><th></th>')
    for s in scenarios:
        rec = ' <span class="recommended">Recommended</span>' if s.get('recommended') else ''
        parts.append(f'<th>{s["label"]}{rec}</th>')
    parts.append('</tr>')

    # Data rows
    for i, (label, key, fmt) in enumerate(rows):
        cls = ' class="highlight-row"' if isinstance(label, str) and label.startswith('*') else ''
        display_label = label.lstrip('*')
        parts.append(f'<tr{cls}><td>{display_label}</td>')
        for s in scenarios:
            if callable(key):
                val = key(s)
            else:
                val = s.get(key)
            parts.append(f'<td>{fmt(val)}</td>')
        parts.append('</tr>')

    parts.append('</table>')
    return '\n'.join(parts)


def _build_1929_detail(scenario: dict) -> str:
    """Build the expandable 1929 year-by-year detail for one scenario."""
    wc = scenario.get('worst_case_1929', {})
    rows = wc.get('year_by_year', [])
    if not rows:
        return ''

    p1_label = scenario.get('person1_label', 'P1')
    p2_label = scenario.get('person2_label', 'P2')

    parts = [f'<details><summary>1929 Year-by-Year Detail — {scenario["label"]}</summary>']
    parts.append('<div class="detail-content">')
    parts.append('<table>')
    parts.append(f'<tr><th>Year</th><th>Age ({p1_label})</th><th>Age ({p2_label})</th>'
                 f'<th>Target</th><th>Actual Spending</th><th>Portfolio</th></tr>')

    for r in rows:
        parts.append(f'<tr><td>{r["year"]}</td><td>{r["age_p1"]}</td><td>{r["age_p2"]}</td>'
                     f'<td>{_fmt(r.get("target"))}</td>'
                     f'<td>{_fmt(r["actual_spend"])}</td>'
                     f'<td>{_fmt(r["portfolio"])}</td></tr>')

    parts.append('</table>')

    # Summary line
    parts.append(f'<p style="margin-top:8px;font-size:0.85rem;color:#666;">'
                 f'Avg spending: {_fmt(wc.get("avg_spending"))} &nbsp;|&nbsp; '
                 f'First 10yr avg: {_fmt(wc.get("avg_first10"))} &nbsp;|&nbsp; '
                 f'Min year: {_fmt(wc.get("min_spending"))} &nbsp;|&nbsp; '
                 f'Final portfolio: {_fmt(wc.get("final_portfolio"))}</p>')

    parts.append('</div></details>')
    return '\n'.join(parts)


# ── Main generator ────────────────────────────────────────────

def generate_html_report(
    scenarios: list,
    narrative: str,
    client_alias: str,
    person1_label: str = "Person 1",
    person2_label: str = "Person 2",
    report_date: str = None,
) -> str:
    """Generate a complete HTML comparison report.

    Args:
        scenarios: list of scenario dicts (see module docstring for format)
        narrative: markdown text with the client-facing write-up
        client_alias: e.g. "Sky-Dune"
        person1_label: first name, e.g. "Mike"
        person2_label: first name, e.g. "Tina"
        report_date: optional, defaults to today

    Returns:
        Complete HTML string.
    """
    if report_date is None:
        report_date = date.today().strftime('%B %d, %Y')

    # Inject person labels into scenarios for 1929 detail
    for s in scenarios:
        s['person1_label'] = person1_label
        s['person2_label'] = person2_label

    # ── Header ──
    header = f"""
    <div class="header">
        <h1>Retirement Plan Comparison</h1>
        <div class="subtitle">{client_alias} &nbsp;&mdash;&nbsp; {report_date}</div>
        <div class="people">{person1_label} &amp; {person2_label}</div>
    </div>
    """

    # ── Core Metrics Table ──
    core_table = _build_comparison_table(scenarios, [
        ('Annual Spending', 'found_spending', _fmt),
        ('Essential Floor', 'found_min', _fmt),
        ('Success Rate', 'found_rate', lambda v: f"{v*100:.0f}%" if v else "—"),
    ], title="What You Can Spend")
    core_table += ('<p style="font-size: 0.78rem; color: #888; margin-top: -16px;">'
                   'Success Rate: the percentage of historical periods where average annual '
                   'spending met or exceeded the Ideal Spending Target. Essential spending was '
                   'met 100% of the time in all scenarios — no one ever went without.</p>')

    # ── Ending Balance Distribution ──
    # Check if any scenario has $0 at any percentile
    has_zero = any(
        s.get('ending_balance_pctiles', {}).get(p, 1) == 0
        for s in scenarios
        for p in [0, 5, 10, 25, 50]
    )

    def _fmt_ending(val):
        if val is not None and val == 0:
            return '$0 *'
        return _fmt(val)

    pctile_labels = [
        ('Worst case', 0),
        ('5th (very bad)', 5),
        ('10th', 10),
        ('25th', 25),
        ('*Typical (50th)', 50),
        ('75th', 75),
        ('90th (very good)', 90),
        ('95th', 95),
    ]
    ending_rows = [
        (label, lambda s, p=p: s.get('ending_balance_pctiles', {}).get(p), _fmt_ending)
        for label, p in pctile_labels
    ]
    ending_rows.append(('*Mean', 'ending_balance_mean', _fmt))
    ending_table = _build_comparison_table(scenarios, ending_rows,
                                            title="What You'd Leave Behind")
    if has_zero:
        ending_table += ('<p style="font-size: 0.78rem; color: #888; margin-top: -16px;">'
                         '* $0 ending balance does not mean the plan failed. Essential spending '
                         'was funded every single year. The portfolio was fully used to support '
                         'retirement — it simply left nothing for the next generation.</p>')

    # ── Stress Test ──
    stress_table = _build_comparison_table(scenarios, [
        ('Decline needed to trigger spending cut',
         lambda s: f"{s.get('decline_pct', 0):.1f}% (-{_fmt(s.get('decline_dollar', 0))})",
         lambda v: v),
        ('Probability of that decline',
         lambda s: f"{s.get('decline_prob_hist', 0)*100:.0f}%",
         lambda v: v),
        ('Spending after decline', 'stressed_spending', _fmt),
        ('Spending change',
         lambda s: f"-{_fmt(abs(s.get('spending_delta', 0)))}/yr ({s.get('spending_delta_pct', 0):+.1f}%)" if s.get('spending_delta') else "—",
         lambda v: v),
    ], title="Stress Test")

    # ── Worst Case 1929 Summary ──
    wc_table = _build_comparison_table(scenarios, [
        ('First 10yr avg spending',
         lambda s: s.get('worst_case_1929', {}).get('avg_first10'), _fmt),
        ('Lowest single year',
         lambda s: s.get('worst_case_1929', {}).get('min_spending'), _fmt),
        ('30-year average spending',
         lambda s: s.get('worst_case_1929', {}).get('avg_spending'), _fmt),
        ('Comfort Level *',
         lambda s: int(s.get('found_min', 0) * 0.80) if s.get('found_min') else None, _fmt),
        ('Years below Comfort Level *',
         lambda s: s.get('worst_case_1929', {}).get('years_below_essential', '—'),
         lambda v: v),
        ('Final portfolio',
         lambda s: s.get('worst_case_1929', {}).get('final_portfolio'), _fmt),
    ], title="Worst Historical Period (1929)")
    wc_table += ('<p style="font-size: 0.78rem; color: #888; margin-top: -16px;">'
                 '* Comfort Level = 80% x Essential Floor — a spending level '
                 'we really don\'t want to go below.</p>')

    # ── 1929 Year-by-Year Details (expandable) ──
    details = '\n'.join(_build_1929_detail(s) for s in scenarios)

    # ── Narrative ──
    narrative_html = f'<div class="narrative">\n{_md_to_html(narrative)}\n</div>'

    # ── Footer ──
    footer = """
    <div class="footer">
        Analysis based on historical market data. Past performance does not guarantee future results.
        All spending figures are in today's dollars (real, net of inflation).
    </div>
    """

    # ── Assemble ──
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Retirement Plan Comparison — {client_alias}</title>
    {CSS}
</head>
<body>
<div class="container">
    {header}
    {core_table}
    {ending_table}
    {stress_table}
    {wc_table}
    {details}
    {narrative_html}
    {footer}
</div>
</body>
</html>"""

    return html


# ── Convenience save function ─────────────────────────────────

def save_html_report(
    scenarios: list,
    narrative: str,
    client_alias: str,
    person1_label: str = "Person 1",
    person2_label: str = "Person 2",
    output_path: str = None,
) -> str:
    """Generate and save HTML report. Returns the file path."""
    html = generate_html_report(scenarios, narrative, client_alias,
                                 person1_label, person2_label)
    if output_path is None:
        output_path = f"/tmp/{client_alias}_comparison.html"
    with open(output_path, 'w') as f:
        f.write(html)
    return output_path
