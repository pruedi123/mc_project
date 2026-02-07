import argparse
from pathlib import Path

import pandas as pd


def convert_monthly_to_annual(
    source: Path,
    target: Path,
    factor_column: str,
    months_per_period: int = 12,
):
    """Convert a monthly factor file into non-overlapping annual (or multi-month) periods."""
    df = pd.read_excel(source, parse_dates=["Date"])
    if factor_column not in df.columns:
        raise ValueError(f"Column {factor_column!r} not found in {source}")

    df = df.sort_values("Date").reset_index(drop=True)
    periods = []
    for block_start in range(0, len(df) - months_per_period + 1, months_per_period):
        block = df.iloc[block_start : block_start + months_per_period]
        factor = block[factor_column].prod()
        periods.append(
            {
                "period_start": block["Date"].iloc[0],
                "period_end": block["Date"].iloc[-1],
                "period_factor": factor,
            }
        )

    out = pd.DataFrame(periods)
    out.to_csv(target, index=False)
    print(f"Wrote {len(out)} annual periods to {target}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert a monthly factor spreadsheet into annual factors."
    )
    parser.add_argument("source", type=Path, help="Path to the monthly xlsx file")
    parser.add_argument(
        "--factor-column",
        default="cpi_factors",
        help="Name of the column that stores the monthly multiplier",
    )
    parser.add_argument(
        "--months-per-period",
        type=int,
        default=12,
        help="Number of rows to combine into each annual period",
    )
    parser.add_argument(
        "--target",
        type=Path,
        help="Output file (defaults to annual_{source.stem}.csv)",
    )
    args = parser.parse_args()

    if args.target is None:
        args.target = Path(f"annual_{args.source.stem}.csv")

    convert_monthly_to_annual(
        source=args.source,
        target=args.target,
        factor_column=args.factor_column,
        months_per_period=args.months_per_period,
    )


if __name__ == "__main__":
    main()
