#!/usr/bin/env python3
"""HTML chart report — year-by-year after-tax spending across three return modes.

Renders four chart panels into a self-contained HTML file:
  1. All-paths spending fan: P10/P50/P90 of annual after-tax spending per year,
     one subplot per mode
  2. Worst-decile cohort spending fan: same fan structure, restricted to the
     worst 10% of paths by first-10-yr drawdown depth
  3. Worst single path: year-by-year spending of the single worst path per mode
     (three lines overlaid)
  4. Worst single path: year-by-year portfolio of the same paths (context for
     spending floors)

Usage:
    python3 charts_html.py <plan.json> [--runs 1000] [--worst-pct 10] [--out charts.html]
"""
import argparse
import base64
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_full_process import build_sim_params
from sim_engine import (
    BOOTSTRAP_MODE_NAME,
    get_all_historical_windows,
    run_historical_parallel,
    run_monte_carlo,
)


MODES = [
    ("Lognormal", "Simulated (lognormal)", "#3b6ea5"),
    ("Historical", "Historical (master_global_factors)", "#2e7d32"),
    ("Bootstrap", BOOTSTRAP_MODE_NAME, "#c1440e"),
]


def _run_one(plan, sim_params, sim_years, inheritor_rate, mode_value, n_runs):
    if mode_value == "Historical (master_global_factors)":
        windows, dates = get_all_historical_windows(sim_years)
        results, ayd = run_historical_parallel(windows, sim_years, inheritor_rate, sim_params)
        return results, ayd, dates
    results, ayd = run_monte_carlo(
        num_runs=n_runs,
        years=sim_years,
        inheritor_rate=inheritor_rate,
        taxable_log_drift=plan.get("taxable_log_drift", 0.09038261),
        taxable_log_volatility=plan.get("taxable_log_volatility", 0.20485277),
        bond_log_drift=plan.get("bond_log_drift", 0.0172918),
        bond_log_volatility=plan.get("bond_log_volatility", 0.04796435),
        return_mode=mode_value,
        **sim_params,
    )
    return results, ayd, None


def _fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _path_drawdown_depth(portfolio, start, dd_years=10):
    window = portfolio[:dd_years]
    baseline = np.concatenate([[start], window])
    running_peak = np.maximum.accumulate(baseline)
    return float((baseline / running_peak - 1.0).min())


def collect_mode_data(plan, sim_params, sim_years, inheritor_rate, n_runs, worst_pct):
    starting = (
        float(plan.get("taxable_start", 0))
        + float(plan.get("tda_start", 0))
        + float(plan.get("tda_spouse_start", 0))
        + float(plan.get("roth_start", 0))
    )
    out = {}
    for label, mode_value, _color in MODES:
        results, ayd, _dates = _run_one(plan, sim_params, sim_years, inheritor_rate, mode_value, n_runs)
        spend_wide = ayd.pivot(index="run", columns="year", values="after_tax_spending").sort_index(axis=1)
        port_wide = ayd.pivot(index="run", columns="year", values="total_portfolio").sort_index(axis=1)
        spending = spend_wide.values  # (n_paths, sim_years)
        portfolios = port_wide.values
        dd_depths = np.array([_path_drawdown_depth(p, starting) for p in portfolios])
        worst_idx = int(np.argmin(dd_depths))
        order = np.argsort(dd_depths)
        cohort_n = max(1, int(round(len(dd_depths) * worst_pct / 100.0)))
        cohort_idx = order[:cohort_n]
        out[label] = {
            "spending": spending,
            "portfolios": portfolios,
            "worst_path_spending": spending[worst_idx],
            "worst_path_portfolio": portfolios[worst_idx],
            "worst_cohort_spending": spending[cohort_idx],
            "worst_cohort_portfolio": portfolios[cohort_idx],
            "worst_dd_depth": float(dd_depths[worst_idx]),
            "starting_portfolio": starting,
            "n_paths": spending.shape[0],
        }
    return out


def _spending_fan_subplot(ax, paths, color, label, ylim=None):
    n_paths, n_yrs = paths.shape
    years = np.arange(1, n_yrs + 1)
    p10 = np.percentile(paths, 10, axis=0)
    p50 = np.percentile(paths, 50, axis=0)
    p90 = np.percentile(paths, 90, axis=0)
    p25 = np.percentile(paths, 25, axis=0)
    p75 = np.percentile(paths, 75, axis=0)
    ax.fill_between(years, p10, p90, color=color, alpha=0.15, label="P10–P90")
    ax.fill_between(years, p25, p75, color=color, alpha=0.30, label="P25–P75")
    ax.plot(years, p50, color=color, linewidth=2.0, label="Median")
    ax.plot(years, paths.T, color=color, alpha=0.04, linewidth=0.5)
    ax.set_title(label)
    ax.set_xlabel("Year")
    ax.grid(True, alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if ylim is not None:
        ax.set_ylim(ylim)


def chart_all_paths_spending(data_by_mode, base_spending):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    # Shared y-axis: union range across modes
    all_vals = np.concatenate([data_by_mode[m]["spending"].flatten() for m, _, _ in MODES])
    ymin = float(max(0, np.percentile(all_vals, 1) - 5000))
    ymax = float(np.percentile(all_vals, 99) + 5000)
    for ax, (label, _mv, color) in zip(axes, MODES):
        _spending_fan_subplot(ax, data_by_mode[label]["spending"], color, label, ylim=(ymin, ymax))
        if base_spending > 0:
            ax.axhline(base_spending, color="#222", linewidth=0.8, linestyle="--", alpha=0.6,
                       label=f"target ${base_spending:,.0f}")
    axes[0].set_ylabel("After-tax spending ($/yr, real)")
    axes[0].legend(loc="lower right", frameon=False, fontsize=9)
    fig.suptitle("After-tax spending across all paths — P10 / P25 / Median / P75 / P90 over time", y=1.02)
    return _fig_to_b64(fig)


def chart_worst_cohort_spending(data_by_mode, worst_pct, base_spending):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    all_vals = np.concatenate([data_by_mode[m]["worst_cohort_spending"].flatten() for m, _, _ in MODES])
    ymin = float(max(0, np.percentile(all_vals, 1) - 5000))
    ymax = float(np.percentile(all_vals, 99) + 5000)
    for ax, (label, _mv, color) in zip(axes, MODES):
        _spending_fan_subplot(ax, data_by_mode[label]["worst_cohort_spending"], color, label, ylim=(ymin, ymax))
        if base_spending > 0:
            ax.axhline(base_spending, color="#222", linewidth=0.8, linestyle="--", alpha=0.6,
                       label=f"target ${base_spending:,.0f}")
    axes[0].set_ylabel("After-tax spending ($/yr, real)")
    axes[0].legend(loc="lower right", frameon=False, fontsize=9)
    fig.suptitle(f"Spending across the worst {worst_pct}% of paths — P10 / P25 / Median / P75 / P90 over time", y=1.02)
    return _fig_to_b64(fig)


def chart_worst_path_spending(data_by_mode, base_spending):
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for label, _mv, color in MODES:
        s = data_by_mode[label]["worst_path_spending"]
        years = np.arange(1, len(s) + 1)
        ax.plot(years, s, color=color, linewidth=2.0, label=label, marker="o", markersize=3)
    if base_spending > 0:
        ax.axhline(base_spending, color="#222", linewidth=0.8, linestyle="--", alpha=0.6,
                   label=f"spending target ${base_spending:,.0f}")
    ax.set_xlabel("Year")
    ax.set_ylabel("After-tax spending ($/yr, real)")
    ax.set_title("Worst single path — year-by-year spending (deepest 10-yr drawdown path per mode)")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(True, alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return _fig_to_b64(fig)


def chart_worst_path_portfolio(data_by_mode):
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for label, _mv, color in MODES:
        p = data_by_mode[label]["worst_path_portfolio"]
        start = data_by_mode[label]["starting_portfolio"]
        years = np.arange(0, len(p) + 1)
        full = np.concatenate([[start], p]) / 1e6
        ax.plot(years, full, color=color, linewidth=2.0, label=label)
    ax.set_xlabel("Year")
    ax.set_ylabel("Portfolio ($M, real)")
    ax.set_title("Worst single path — portfolio trajectory (same paths as spending chart)")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(True, alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return _fig_to_b64(fig)


def _spend_summary_row(label, d, base_spending):
    spend = d["spending"]
    run_avg = spend.mean(axis=1)
    worst_year = spend.min(axis=1)
    success = (
        float((run_avg >= base_spending).mean()) * 100 if base_spending > 0 else float("nan")
    )
    pct = lambda a, q: np.percentile(a, q)
    success_str = f"{success:.1f}%" if not np.isnan(success) else "—"
    return (
        f"<tr><td>{label}</td>"
        f"<td>{d['n_paths']:,}</td>"
        f"<td>{success_str}</td>"
        f"<td>${run_avg.mean():,.0f}</td>"
        f"<td>${pct(run_avg, 10):,.0f}</td>"
        f"<td>${pct(worst_year, 10):,.0f}</td>"
        f"<td>${worst_year.min():,.0f}</td></tr>"
    )


def build_html(plan_path, sim_years, starting, base_spending, target_stock_pct, n_runs, worst_pct, data, charts):
    summary_rows = [_spend_summary_row(label, data[label], base_spending) for label, _mv, _c in MODES]

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Spending — Three-Mode Comparison — {Path(plan_path).name}</title>
<style>
 body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; max-width: 1080px;
        margin: 20px auto; padding: 0 20px; color: #222; }}
 h1, h2 {{ color: #1a3a5e; }}
 h2 {{ border-bottom: 1px solid #ddd; padding-bottom: 6px; margin-top: 32px; }}
 table {{ border-collapse: collapse; margin: 12px 0; }}
 th, td {{ padding: 6px 12px; text-align: right; border-bottom: 1px solid #eee; }}
 th:first-child, td:first-child {{ text-align: left; }}
 th {{ background: #f5f7fa; color: #333; font-weight: 600; }}
 img {{ max-width: 100%; display: block; margin: 8px 0 16px; }}
 .meta {{ background: #f5f7fa; padding: 10px 14px; border-left: 4px solid #1a3a5e; margin-bottom: 16px; font-size: 14px; }}
 .note {{ font-size: 13px; color: #555; margin: 4px 0 16px; }}
</style>
</head>
<body>
<h1>After-Tax Spending — Three-Mode Comparison</h1>
<div class="meta">
  <strong>Plan:</strong> {plan_path}<br>
  <strong>Horizon:</strong> {sim_years} years &nbsp;·&nbsp;
  <strong>Stock allocation:</strong> {target_stock_pct*100:.0f}% &nbsp;·&nbsp;
  <strong>Starting portfolio:</strong> ${starting:,.0f} &nbsp;·&nbsp;
  <strong>Spending target:</strong> ${base_spending:,.0f}<br>
  <strong>Stochastic paths:</strong> {n_runs} &nbsp;·&nbsp;
  <strong>Worst-cohort percentile:</strong> {worst_pct}%
</div>

<h2>Spending summary</h2>
<table>
<tr><th>Mode</th><th>Paths</th>
    <th>Avg ≥ target</th>
    <th>Mean avg spending</th>
    <th>P10 avg spending</th>
    <th>P10 worst-year</th>
    <th>Min single year</th></tr>
{''.join(summary_rows)}
</table>

<h2>All paths — year-by-year spending</h2>
<p class="note">P10, P25, median, P75, P90 of after-tax annual spending across every simulated path, year by year. Faint lines are individual paths. Dashed line = spending target. The bands show how much income variability each return mode produces — narrower = more predictable spending. With guardrails on, you'll typically see the lower bands pull below the target during downside years.</p>
<img src="data:image/png;base64,{charts['all_spend']}">

<h2>Worst-decile cohort — year-by-year spending</h2>
<p class="note">Same fan chart restricted to the worst 10% of paths by first-10-yr portfolio drawdown. This is the cohort that matters for client conversations: the bad twins, side by side. If the spending floor holds across modes, the plan is structurally robust regardless of which model you trust.</p>
<img src="data:image/png;base64,{charts['cohort_spend']}">

<h2>Worst single path — spending</h2>
<p class="note">For the single deepest-drawdown path in each mode, year-by-year after-tax spending. Lognormal and bootstrap can produce different trough timings than the audit's classic 1929 path. Compare to the portfolio trajectory below to see how spending responds to portfolio stress.</p>
<img src="data:image/png;base64,{charts['worst_spend']}">

<h2>Worst single path — portfolio (context)</h2>
<p class="note">Portfolio trajectory for the same single paths as above. Useful for explaining how the guardrail / income floor mechanism kept spending intact (or didn't) given how the portfolio behaved.</p>
<img src="data:image/png;base64,{charts['worst_port']}">

<p class="note" style="margin-top:32px;">Generated by <code>charts_html.py</code>. All values are real (CPI-deflated).</p>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("plan")
    ap.add_argument("--runs", type=int, default=1000)
    ap.add_argument("--worst-pct", type=int, default=10)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    plan = json.loads(Path(args.plan).read_text())
    sim_params, sim_years, target_stock_pct, base_spending, _mg = build_sim_params(plan)
    inheritor_rate = float(plan.get("inheritor_marginal_rate", 0.32))

    print(f"Running 3 modes × {args.runs} paths each ...", flush=True)
    data = collect_mode_data(plan, sim_params, sim_years, inheritor_rate, args.runs, args.worst_pct)

    print("Rendering charts ...", flush=True)
    charts = {
        "all_spend": chart_all_paths_spending(data, base_spending),
        "cohort_spend": chart_worst_cohort_spending(data, args.worst_pct, base_spending),
        "worst_spend": chart_worst_path_spending(data, base_spending),
        "worst_port": chart_worst_path_portfolio(data),
    }

    starting = (
        float(plan.get("taxable_start", 0))
        + float(plan.get("tda_start", 0))
        + float(plan.get("tda_spouse_start", 0))
        + float(plan.get("roth_start", 0))
    )
    html = build_html(args.plan, sim_years, starting, base_spending, target_stock_pct,
                      args.runs, args.worst_pct, data, charts)

    out_path = args.out or f"charts_{Path(args.plan).stem}.html"
    Path(out_path).write_text(html)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    raise SystemExit(main())
