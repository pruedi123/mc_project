import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent
NSIMS = 1_000
DRIFT = 0.07
VOL = 0.127
SEED = 12345

def build_returns(years: int) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    log_returns = rng.normal(DRIFT, VOL, size=(NSIMS, years))
    returns = np.exp(log_returns) - 1.0
    cols = [f"year_{i+1}" for i in range(years)]
    df = pd.DataFrame(returns, columns=cols)
    df.insert(0, "sim_id", np.arange(1, NSIMS + 1))
    return df

def main():
    target = ROOT / "precomputed_returns"
    target.mkdir(exist_ok=True)
    for years in range(1, 41):
        df = build_returns(years)
        path = target / f"lognormal_returns_{years}yrs_1000sims.csv"
        df.to_csv(path, index=False)
    print(f"Generated returns for 1-40 years under {target}")

if __name__ == "__main__":
    main()
