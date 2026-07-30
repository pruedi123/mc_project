"""Spending safety metrics and reserve analysis."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

# The sim's withdrawal solver targets net spending +/- $0.50, so exact >=/<
# comparisons against a spending threshold can count penny-level solver noise
# as a miss.  All pass/fail threshold tests share this tolerance.
SOLVER_EPS = 1.0


def _longest_true_streak(values) -> int:
    longest = 0
    current = 0
    for value in values:
        if bool(value):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _floor_spending(value: float, increment: float = 1000.0) -> float:
    if value < increment:
        return max(0.0, value)
    return float(np.floor(value / increment) * increment)


def spending_safety_metrics(
    all_yearly_df: pd.DataFrame,
    ideal_spending: float,
    acceptable_spending: Optional[float],
    redline_spending: float,
) -> dict:
    """Summarize annual spending against ideal / acceptable / red-line levels."""
    if all_yearly_df.empty:
        return {
            "num_runs": 0,
            "num_years": 0,
            "avg_ideal_success_rate": 0.0,
            "pct_years_at_or_above_ideal": 0.0,
            "pct_years_acceptable_to_ideal": 0.0,
            "pct_years_redline_to_acceptable": 0.0,
            "redline_breach_years": 0,
            "redline_breach_runs": 0,
            "redline_breach_run_rate": 0.0,
            "worst_single_year_spending": 0.0,
            "longest_redline_breach_stretch": 0,
        }

    acceptable = float(acceptable_spending if acceptable_spending is not None else redline_spending)
    ideal = float(ideal_spending)
    redline = float(redline_spending)
    spend = all_yearly_df["after_tax_spending"].astype(float)
    total_years = int(len(all_yearly_df))
    runs = all_yearly_df.groupby("run")
    run_avg = runs["after_tax_spending"].mean()
    # Shift every threshold by the same tolerance so the bands stay complementary
    ideal_t = ideal - SOLVER_EPS
    acceptable_t = acceptable - SOLVER_EPS
    redline_t = redline - SOLVER_EPS
    breach = spend < redline_t
    breach_by_run = all_yearly_df.assign(_breach=breach).groupby("run")["_breach"].any()
    longest = int(
        all_yearly_df.assign(_breach=breach)
        .sort_values(["run", "year"])
        .groupby("run")["_breach"]
        .apply(_longest_true_streak)
        .max()
    )

    return {
        "num_runs": int(run_avg.shape[0]),
        "num_years": total_years,
        "avg_ideal_success_rate": float((run_avg >= ideal_t).mean()) if ideal > 0 else None,
        "pct_years_at_or_above_ideal": float((spend >= ideal_t).mean()) if ideal > 0 else None,
        "pct_years_acceptable_to_ideal": float(((spend >= acceptable_t) & (spend < ideal_t)).mean()) if ideal > 0 else None,
        "pct_years_redline_to_acceptable": float(((spend >= redline_t) & (spend < acceptable_t)).mean()) if acceptable > redline else 0.0,
        "redline_breach_years": int(breach.sum()),
        "redline_breach_runs": int(breach_by_run.sum()),
        "redline_breach_run_rate": float(breach_by_run.mean()),
        "worst_single_year_spending": float(spend.min()),
        "longest_redline_breach_stretch": longest,
    }


def _threshold_stats(all_yearly_df: pd.DataFrame, spending_col: str, threshold: float) -> dict:
    spend = all_yearly_df[spending_col].astype(float)
    below = spend < float(threshold) - SOLVER_EPS
    below_by_run = all_yearly_df.assign(_below=below).groupby("run")["_below"].any()
    return {
        "threshold": float(threshold),
        "years_below": int(below.sum()),
        "pct_years_below": float(below.mean()),
        "runs_below": int(below_by_run.sum()),
        "pct_runs_below": float(below_by_run.mean()),
    }


def essential_reserve_analysis(
    all_yearly_df: pd.DataFrame,
    ideal_spending: float,
    essential_spending: float,
    spending_col: str = "after_tax_spending",
) -> dict:
    """Calculate reserve needed to prevent any year below essential spending.

    The reserve is modeled as a separate real-dollar bucket outside the portfolio.
    It tops up only the shortfall between actual spending and the essential floor.
    """
    if all_yearly_df.empty:
        return {
            "ideal_spending": float(ideal_spending),
            "essential_spending": float(essential_spending),
            "reserve_needed": 0.0,
            "before_reserve": {},
            "after_reserve": {},
            "reserve_percentiles": {},
        }

    df = all_yearly_df[["run", "year", spending_col]].copy()
    if "start_date" in all_yearly_df.columns:
        df["start_date"] = all_yearly_df["start_date"]
    spend = df[spending_col].astype(float)
    ideal = float(ideal_spending)
    essential = float(essential_spending)
    df["essential_reserve_topup"] = (essential - spend).clip(lower=0.0)
    df["spending_with_essential_reserve"] = spend + df["essential_reserve_topup"]

    group_cols = ["run"] + (["start_date"] if "start_date" in df.columns else [])
    by_run = df.groupby(group_cols).agg(
        reserve_needed=("essential_reserve_topup", "sum"),
        essential_breach_years=("essential_reserve_topup", lambda s: int((s > 0).sum())),
        worst_spending=(spending_col, "min"),
    ).reset_index()
    worst_idx = by_run["reserve_needed"].idxmax()
    worst = by_run.loc[worst_idx].to_dict()

    reserve_values = by_run["reserve_needed"].astype(float)
    reserve_percentiles = {
        str(p): float(np.percentile(reserve_values, p))
        for p in [50, 75, 90, 95, 99, 100]
    }

    before = {
        "ideal": _threshold_stats(df, spending_col, ideal),
        "essential": _threshold_stats(df, spending_col, essential),
        "minimum_year": float(spend.min()),
    }
    after = {
        "ideal": _threshold_stats(df, "spending_with_essential_reserve", ideal),
        "essential": _threshold_stats(df, "spending_with_essential_reserve", essential),
        "minimum_year": float(df["spending_with_essential_reserve"].min()),
    }

    return {
        "ideal_spending": ideal,
        "essential_spending": essential,
        "reserve_needed": float(reserve_values.max()),
        "total_reserve_used_all_runs": float(df["essential_reserve_topup"].sum()),
        "runs_needing_reserve": int((reserve_values > 0).sum()),
        "num_runs": int(df["run"].nunique()),
        "num_years": int(len(df)),
        "max_breach_years_in_one_run": int(by_run["essential_breach_years"].max()),
        "worst_reserve_run": worst,
        "reserve_percentiles": reserve_percentiles,
        "before_reserve": before,
        "after_reserve": after,
    }


def solve_essential_floor_for_reserve(
    all_yearly_df: pd.DataFrame,
    reserve_amount: float,
    max_floor: Optional[float] = None,
    spending_col: str = "after_tax_spending",
    increment: float = 100.0,
) -> dict:
    """Find the highest essential floor fully protected by a reserve bucket.

    The reserve is modeled as separate real dollars outside the portfolio. A floor
    is feasible when the largest per-run sum of shortfalls is <= reserve_amount.
    """
    if all_yearly_df.empty:
        return {
            "essential_floor": 0.0,
            "reserve_amount": float(reserve_amount),
            "reserve_needed": 0.0,
            "analysis": essential_reserve_analysis(all_yearly_df, 0.0, 0.0, spending_col),
        }

    reserve = max(0.0, float(reserve_amount))
    spend = all_yearly_df[spending_col].astype(float)
    hi = float(max_floor if max_floor is not None and max_floor > 0 else spend.max())
    lo = 0.0

    def _reserve_needed(floor: float) -> float:
        shortfall = (floor - spend).clip(lower=0.0)
        return float(shortfall.groupby(all_yearly_df["run"]).sum().max())

    for _ in range(45):
        mid = (lo + hi) / 2.0
        if _reserve_needed(mid) <= reserve:
            lo = mid
        else:
            hi = mid

    solved = _floor_spending(lo, increment=increment)
    needed = _reserve_needed(solved)
    while needed > reserve and solved > 0:
        solved = max(0.0, solved - increment)
        needed = _reserve_needed(solved)

    ideal = float(max_floor if max_floor is not None and max_floor > 0 else solved)
    analysis = essential_reserve_analysis(
        all_yearly_df,
        ideal_spending=ideal,
        essential_spending=solved,
        spending_col=spending_col,
    )
    return {
        "essential_floor": float(solved),
        "reserve_amount": reserve,
        "reserve_needed": float(needed),
        "unused_reserve": float(max(0.0, reserve - needed)),
        "analysis": analysis,
    }
