# Monte Carlo Retirement Withdrawal Simulator: A Technical Overview

## 1. Introduction and Purpose

This application is a **Monte Carlo retirement withdrawal simulator** built in Python using the Streamlit framework. Its purpose is to answer a fundamental question in retirement planning: *Given a household's current portfolio, income sources, tax situation, and spending needs, what is the probability that their wealth will sustain them through retirement — and what range of outcomes should they expect?*

The simulator models the year-by-year mechanics of retirement spending with high fidelity, including:

- Multi-account withdrawals (taxable brokerage, tax-deferred IRAs/401(k)s, Roth IRAs)
- Required Minimum Distributions (RMDs) using the IRS Uniform Lifetime Table
- A full federal income tax engine (ordinary brackets, capital gains stacking, NIIT, Social Security provisional income)
- Roth conversions with tax-source flexibility
- Dynamic withdrawal guardrails based on forward-looking success rates
- Two return-generation approaches: parametric (lognormal) Monte Carlo and historical rolling-window analysis

The goal is not merely to produce a single "success rate" number, but to generate rich distributional output — percentile bands for portfolio value, spending, and taxes over time — that supports nuanced financial planning conversations.

---

## 2. Theoretical Foundations

### 2.1 The Retirement Spending Problem

The core problem in decumulation finance is one of **sequential decision-making under uncertainty**. A retiree must decide how much to withdraw each year from a portfolio whose future returns are unknown. Withdraw too much and the portfolio is exhausted prematurely; withdraw too little and utility is needlessly sacrificed.

William Bengen's seminal 1994 work introduced the "4% rule" — the idea that a 50/50 stock/bond portfolio historically survived 30 years at a 4% initial withdrawal rate (inflation-adjusted). While useful as a starting point, this approach has significant limitations:

1. **It assumes a fixed withdrawal rate** regardless of portfolio performance.
2. **It ignores taxes**, which can consume 15-30% of gross withdrawals.
3. **It ignores multiple account types** with different tax characteristics.
4. **It ignores other income sources** (Social Security, pensions) and their interactions with the tax code.

This simulator addresses all four limitations.

### 2.2 Monte Carlo Simulation in Finance

Monte Carlo simulation generates many possible future return paths by random sampling, then analyzes the distribution of outcomes. For retirement planning, the key insight is that **the sequence of returns matters** — two portfolios can earn the same average return but produce very different outcomes depending on when gains and losses occur (sequence-of-returns risk).

By running hundreds or thousands of simulated retirement periods, the simulator produces a probability distribution of outcomes rather than a single point estimate. This allows the planner to assess:

- The probability of portfolio exhaustion (ruin probability)
- The expected range of ending wealth
- The distribution of lifetime spending
- The range of tax burdens across scenarios

### 2.3 Lognormal Return Model

Stock returns are modeled using a **lognormal distribution**, which is the standard assumption in financial economics (following the geometric Brownian motion model underlying Black-Scholes). Under this model:

    ln(1 + R_t) ~ N(mu, sigma^2)

where R_t is the annual return, mu is the log drift (expected log return), and sigma is the log volatility. The lognormal model has desirable properties:

- Returns are always greater than -100% (limited liability)
- The distribution is right-skewed, matching empirical observations
- Compounding is naturally handled through additive log returns
- Two parameters (mu, sigma) fully characterize the distribution

The simulator allows the user to set mu and sigma directly. Typical values for a diversified U.S. equity portfolio might be mu = 0.07, sigma = 0.13, implying an expected arithmetic return of approximately exp(mu + sigma^2/2) - 1, or about 8.0%.

**Bond returns** are not modeled parametrically. Instead, they are sampled with replacement from historical data (the LBM 100 F series from the master dataset). This is a pragmatic choice: bond returns exhibit serial correlation and regime-dependent behavior (interest rate cycles) that a simple parametric model would miss.

---

## 3. Return Generation: Simulated vs. Historical

The simulator offers two fundamentally different approaches to generating return sequences:

### 3.1 Simulated (Lognormal Monte Carlo)

**Method:** For each simulation run, stock returns are drawn independently from a lognormal distribution with user-specified parameters. Bond returns are independently sampled (with replacement) from the historical bond factor series.

**Assumptions:**
- Returns are independently and identically distributed (i.i.d.) across years
- No serial correlation, mean reversion, or regime switching
- The distribution parameters are known and stable

**Strengths:**
- Unlimited number of unique paths — no data recycling
- Easy to test sensitivity to return assumptions
- Clean separation of return magnitude from return sequence
- Produces smooth percentile distributions

**Weaknesses:**
- May understate tail risk (fat tails, crashes)
- Ignores correlation between stock and bond returns
- Assumes stationarity — real markets exhibit time-varying volatility
- No structural relationship between returns and inflation

### 3.2 Historical (All Periods)

**Method:** The simulator extracts every possible starting month from the historical dataset and constructs non-overlapping 12-month return windows for the required number of years. Each such window is a single "run" — the household lives through that exact historical sequence.

For a dataset spanning (say) 1970-2024 and a 30-year retirement, every month from January 1970 through (roughly) 1994 could serve as a starting point, yielding hundreds of overlapping but distinct 30-year paths.

**Assumptions:**
- The future distribution of returns resembles the historical distribution
- The historical record is long enough to capture relevant regimes

**Strengths:**
- Preserves real-world return sequences, including crashes, recoveries, and correlation structures
- No distributional assumptions needed
- Captures fat tails and volatility clustering naturally
- Includes actual stock-bond correlations

**Weaknesses:**
- Limited number of unique paths (determined by data length)
- Overlapping windows are not truly independent
- Implicitly assumes the future will look like the past
- Cannot explore scenarios outside historical experience

### 3.3 Historical (Specific Start Year)

A third mode allows the user to specify a single historical start year and run one deterministic simulation using the actual return sequence from that date forward. This is useful for backtesting ("What would have happened if I retired in 2000?") but provides a single outcome, not a distribution.

### 3.4 When to Use Each

| Criterion | Lognormal MC | Historical All-Periods |
|-----------|-------------|------------------------|
| Testing a specific return assumption | Preferred | Less flexible |
| Stress-testing extreme scenarios | Limited by distribution | Limited by history |
| Preserving real-world correlation structure | No | Yes |
| Number of independent paths | Unlimited | Fixed |
| Sensitivity analysis on risk parameters | Easy | Requires data manipulation |

In practice, running both methods and comparing the results provides a more robust assessment than relying on either alone.

---

## 4. The Simulation Engine: Step by Step

Each simulation run proceeds year by year for the specified retirement horizon. Here is the detailed sequence of operations within each year:

### Step 1: Portfolio Rebalancing

At the start of each year, the entire household portfolio is rebalanced to the target stock/bond allocation using **asset location** principles:

- **Stocks are prioritized in Roth** (tax-free growth for the highest-expected-return asset)
- **Bonds are prioritized in TDAs** (interest is taxed as ordinary income anyway, so there is no tax cost to holding bonds in tax-deferred space)
- Taxable accounts hold whatever allocation is needed to hit the household target after Roth and TDA placements

This reflects the well-established principle of asset location (Daryanani, 2004): place tax-inefficient assets (bonds, high-turnover funds) in tax-sheltered accounts and tax-efficient assets (low-turnover equity) in taxable accounts.

The rebalancing algorithm:
1. Computes total household desired stock and bond dollar amounts
2. Fills Roth with stocks first (up to Roth balance), then taxable, then TDAs
3. Fills TDAs with bonds first (up to TDA balance minus any stocks), then taxable, then Roth
4. Splits TDA allocations between the two spouses proportionally to their balances
5. Adjusts taxable cost basis proportionally to the new stock/bond split

### Step 2: Dynamic Withdrawal Guardrails (if enabled)

If guardrails are active, the simulator evaluates whether the current withdrawal rate remains sustainable:

1. **Forward Success Rate Check:** A fast inner Monte Carlo (default 200 paths) estimates the probability that the current portfolio can sustain the current withdrawal level for the remaining years, using a simplified single-asset lognormal model (no tax engine, for speed).

2. **Dead Band Evaluation:** If the forward success rate falls below the lower guardrail (e.g., 75%) or rises above the upper guardrail (e.g., 90%), the system triggers a recalibration.

3. **Binary Search Recalibration:** A binary search finds the withdrawal amount that produces exactly the target success rate (e.g., 85%). Pre-generating a single random matrix and reusing it across iterations ensures stability. The search runs up to 30 iterations with a $500 tolerance.

4. **Maximum Spending Cap:** The recalibrated withdrawal is capped at a user-specified percentage above the base withdrawal (e.g., 50% means max of 150% of base), preventing unsustainably high spending in strong markets that would leave the retiree vulnerable to subsequent downturns.

This approach draws on the literature of dynamic spending rules (Guyton & Klinger, 2006; Blanchett, Kowara, & Chen, 2012), which demonstrates that flexible withdrawal strategies significantly improve both sustainability and lifetime utility compared to fixed rules.

**Why a simplified inner model:** The guardrail system uses a simplified single-asset model (no tax engine) for the inner MC check. This is a necessary speed compromise — running the full tax engine inside the guardrail check would make the simulation computationally prohibitive (each of 500 outer runs would need 200+ inner simulations per year). The simplified model captures the essential dynamics (portfolio growth and withdrawals) while running orders of magnitude faster.

**Blended parameters for guardrails:** The inner MC uses a single blended return distribution derived from the stock and bond parameters weighted by the target allocation:

    blended_mu = target_stock_pct * stock_mu + (1 - target_stock_pct) * bond_mu
    blended_sigma = target_stock_pct * stock_sigma + (1 - target_stock_pct) * bond_sigma

### Step 3: Roth Conversion (if applicable)

If the user has specified Roth conversions, the system transfers the designated amount from the primary TDA to a pending Roth conversion pool. The actual deposit into Roth occurs after taxes are computed (Step 8), because the conversion generates a tax liability that must be resolved first.

Two tax-payment options are supported:
- **Pay from taxable:** The conversion tax is withdrawn from the taxable account, preserving the full conversion amount in Roth.
- **Pay from TDA (net conversion):** The tax is deducted from the conversion amount itself, reducing what lands in Roth but preserving the taxable account.

The conversion is capped at the current TDA balance (you cannot convert more than you have).

### Step 4: Investment Growth

Returns are applied to all accounts:

- **Stocks** in each account (taxable, TDA primary, TDA spouse, Roth) grow by the period's stock return
- **Bonds** in each account grow by the period's bond return

For the taxable account, stock total return is decomposed into its components:

**Price appreciation:**

    price_return = (1 + stock_total_return) / (1 + dividend_yield) - 1

This formula extracts the price-only return from the total return by removing the dividend component.

**Qualified dividends:** dividend_yield * stock_market_value. Reinvested in the portfolio and added to cost basis. Taxed as qualified dividends (capital gains rates).

**Bond interest:** bond_return * bond_market_value. Reinvested and added to cost basis. Taxed as ordinary income.

**Stock turnover:** A fraction (turnover rate) of stocks is notionally sold and repurchased:

    turnover_sale = stock_mv * turnover_rate
    turnover_basis_sold = stock_basis * turnover_rate
    turnover_realized_gain = max(0, turnover_sale - turnover_basis_sold)

The realized gain generates a tax liability even though no cash leaves the portfolio (it's a round-trip). Basis is stepped up on the repurchase. This models the tax drag of actively managed or indexed funds that experience internal turnover.

### Step 5: Required Minimum Distributions

For each spouse who has reached RMD age (typically 73), the simulator:

1. Looks up the IRS Uniform Lifetime Table divisor for the current age (e.g., age 73 -> divisor 24.7, age 80 -> divisor 18.7)
2. Computes RMD = TDA balance / divisor
3. Withdraws the RMD from the TDA (pro-rata from stocks and bonds within the TDA)

**Handling RMD excess:** If the total RMD exceeds the year's spending need:
- The excess is reinvested in the taxable account (pro-rata to stocks and bonds)
- The excess increases the taxable cost basis (treated as a new deposit at market value)
- The excess is tracked as `rmd_excess_to_taxable` — it is taxable income but not spending cash

If the RMD is less than the spending need, the shortfall is filled from other sources in the next step.

### Step 6: Withdrawal Cascade

The simulator fills the remaining spending need using a priority waterfall:

1. **Taxable bonds** — sold with gross-up to deliver the requested net amount
2. **Taxable stocks** — same gross-up logic
3. **Primary TDA** — additional withdrawals beyond RMD
4. **Spouse TDA** — secondary TDA
5. **Roth IRA** — last resort (preserves tax-free growth)

This order is designed to:
- Satisfy mandatory RMDs first
- Use taxable assets next (where only gains are taxed, not the full withdrawal)
- Defer TDA withdrawals (fully taxable) as long as possible
- Preserve Roth (tax-free growth) as the last resort

**Gross-Up Logic:** When "gross-up withdrawals" is enabled, the simulator solves for the gross sale amount S such that S - tax(S) = net needed:

    S = net / (1 - cap_gains_rate * (1 - basis/market_value))

This ensures the retiree actually receives the intended net spending amount after paying taxes on the sale. Without gross-up, the retiree would receive less than requested because some of the sale proceeds go to taxes.

### Step 7: Tax Computation

The tax engine computes federal income tax in full detail:

**7a. Social Security Taxation**

Using the IRS provisional income formula:

    provisional_income = other_income + capital_gains + 0.5 * SS_income

Taxable portion of Social Security:
- If provisional income <= base threshold ($32,000 MFJ, $25,000 single): 0% taxable
- If provisional income <= upper threshold ($44,000 MFJ, $34,000 single): 50% of excess
- Above upper threshold: up to 85% of Social Security is taxable

This creates a "tax torpedo" where a narrow income range produces very high effective marginal rates as Social Security taxation phases in.

**7b. Ordinary Income Assembly**

Ordinary income before deductions includes:
- Bond interest (from taxable account)
- TDA withdrawals (including RMDs)
- Pension income
- Other income
- Taxable portion of Social Security
- Roth conversion amount (if applicable)

**7c. Deduction**

Either the standard deduction ($29,200 MFJ, $14,600 single for 2024) or an itemized deduction amount specified by the user.

**7d. Ordinary Income Tax**

Applied using 2024 graduated brackets:

| Bracket (MFJ) | Rate |
|---------------|------|
| $0 - $23,200 | 10% |
| $23,200 - $94,300 | 12% |
| $94,300 - $201,050 | 22% |
| $201,050 - $383,900 | 24% |
| $383,900 - $487,450 | 32% |
| $487,450 - $731,200 | 35% |
| Over $731,200 | 37% |

**7e. Capital Gains and Qualified Dividends**

Taxed using the **stacking method** — capital gains brackets are applied on top of ordinary taxable income. This is how the IRS actually computes capital gains tax:

1. Ordinary taxable income fills the income "stack" first
2. Capital gains are placed on top of that stack
3. The gains are taxed at 0%, 15%, or 20% depending on where they fall in the stack

Thresholds for MFJ:
- 0% rate: up to $94,050 of total income
- 15% rate: $94,050 to $583,750
- 20% rate: above $583,750

**7f. Net Investment Income Tax (NIIT)**

A 3.8% surtax on the lesser of:
- Net investment income (capital gains + interest + dividends)
- The amount by which AGI exceeds $250,000 (MFJ) or $200,000 (single)

**7g. Roth Conversion Tax Delta**

The system computes total tax both with and without the Roth conversion to isolate the marginal tax cost:

    roth_conversion_tax_delta = max(0, total_tax_with_conversion - total_tax_without_conversion)

This ensures the conversion tax is properly attributed rather than being spread across all income.

**7h. Marginal Rate Computation**

The simulator tracks the marginal ordinary rate (the bracket the next dollar of ordinary income would fall into) and the marginal capital gains rate (accounting for stacking and NIIT). These marginal rates are used in the non-spending tax computation (Step 9).

### Step 8: Roth Conversion Settlement

After taxes are computed, the Roth conversion is finalized:

If taxes paid from **taxable account**:
1. Withdraw conversion tax from bonds first, then stocks if needed
2. Reduce basis proportionally
3. Full conversion amount deposited into Roth

If taxes paid from **TDA (net conversion)**:
1. Roth deposit = conversion amount - conversion tax
2. Taxable account is unaffected

In either case, the net amount is deposited into Roth as stocks (consistent with the asset location preference for growth assets in Roth).

### Step 9: After-Tax Spending Calculation

The after-tax spending for the year is:

    After-tax spending = Taxable_cash_out + TDA_withdrawals + Roth_withdrawals
                        - RMD_excess_reinvested
                        + SS_income + Pension_income + Other_income
                        - (Total_taxes - Roth_conversion_tax - Non_spending_tax)

The **non-spending tax** adjustment is critical and deserves detailed explanation.

**The Problem:** Total taxes include taxes on income items that were never available as spending cash:

- **RMD excess reinvested in taxable:** This cash went back into the portfolio, not into the retiree's pocket. But it was taxable income, so taxes were computed on it.
- **Bond interest reinvested:** Accrued and reinvested — taxable but not spendable.
- **Qualified dividends reinvested:** Same — reinvested, creating a tax liability but no spending cash.
- **Turnover-realized gains:** A paper round-trip — stocks are sold and immediately repurchased, generating tax but no cash.

These taxes are a **portfolio drag** (they reduce portfolio growth), not a spending reduction. If we subtracted all taxes from spending, we'd be double-counting: the taxes reduce the portfolio AND reduce reported spending, when in reality the retiree never had that money to spend.

**The Solution:** Estimate the tax attributable to non-spending income using marginal rates:

    non_spending_tax = RMD_excess * marginal_ordinary_rate
                     + max(0, interest) * marginal_ordinary_rate
                     + (dividends + turnover_gain) * marginal_cap_gains_rate

This is clamped: `max(0, min(non_spending_tax, total_taxes - roth_conversion_tax))` to ensure:
1. Non-spending tax is never negative (even if bond returns are negative, the tax on that income is zero, not negative)
2. Non-spending tax never exceeds the taxes actually available to allocate (total taxes minus conversion tax)

### Step 10: Mortality Handling

The simulator tracks whether each spouse is alive based on their life expectancy input:
- When a spouse dies, their income sources (SS, pension) stop
- When one spouse dies in an MFJ household, filing status switches to single (higher tax rates, lower bracket thresholds)
- RMDs on a deceased spouse's TDA stop (simplified — in reality, inherited TDA rules vary)

### Step 11: Record Year's Results

All computed values are stored in a row of the output DataFrame: beginning and ending balances for each account (stocks/bonds detail), RMD details, withdrawal sources, all income items, every tax component, marginal rates, and after-tax spending. This produces 40+ columns per year.

---

## 5. Distribution Analysis

After all simulation runs complete, the results are aggregated into several analytical views:

### 5.1 End-of-Period Summary (Percentile Table)

For each run, the simulator computes:

- **After-tax ending value:** Taxable + Roth + TDA * (1 - inheritor_marginal_rate). The TDA haircut reflects the fact that inherited TDA assets will eventually be taxed at the beneficiary's marginal rate under the SECURE Act's 10-year distribution rule.
- **Total lifetime taxes paid**
- **Effective tax rate** (total taxes / total taxable income)
- **Portfolio CAGR** and **Roth CAGR**

These metrics are displayed at the 0th (minimum), 10th, 25th, 50th (median), 75th, and 90th percentiles.

The **"Percent of ending values <= 0"** metric is the classic ruin probability — the fraction of simulations where the portfolio was exhausted before the end of the planning horizon.

### 5.2 Lifetime Spending Distribution

For each run, the simulator computes the sum and average of annual withdrawals and after-tax spending across all years. This distribution answers: *"Over the full retirement period, how much did the household actually spend in each scenario?"*

Displayed metrics (at 0th, 10th, 25th, 50th, 75th, 90th, and 100th percentiles):
- Average annual withdrawal (gross, before tax)
- Average annual after-tax spending
- Total lifetime withdrawal
- Total lifetime after-tax spending

With guardrails enabled, median average annual spending typically exceeds the base withdrawal target — because in favorable markets, the guardrails allow increased spending. The distribution also shows the downside: at the 10th percentile, spending may fall below target.

A summary line compares the median average annual withdrawal to the base target (e.g., "Median avg annual withdrawal: $48,200 (20.5% above target)").

### 5.3 Year-by-Year Median Table

The median (50th percentile) of every output column across all runs, for each year. This provides a "typical" year-by-year retirement path showing how accounts, taxes, income, and spending evolve over time.

**Important distinction:** This is a year-by-year median, not the median run. In each year, the median value may come from a different simulation run. This means the year-by-year medians may not represent any single coherent path — they represent the central tendency at each point in time.

### 5.4 Percentile Band Charts

Five key metrics are displayed as percentile band charts over time:

1. **Total portfolio value** — shows the dispersion of wealth trajectories across all runs. The 0th percentile shows the worst case; the 90th shows favorable outcomes.

2. **Account balances (median)** — shows how the composition shifts over time. Typically: TDAs decline as RMDs draw them down, Roth grows from conversions and tax-free compounding, taxable fluctuates based on market returns and withdrawal needs.

3. **Annual taxes** — shows the range of annual tax burdens. Large dispersions indicate tax sensitivity to return outcomes.

4. **Withdrawal target** — (when guardrails are enabled) shows how the guardrail system adjusts spending in response to portfolio performance. The band width reflects spending volatility.

5. **After-tax spending** — the bottom line: how much the household actually consumed in each year. This is what the retiree cares about most.

### 5.5 Median Representative Run

The simulator identifies the single run whose ending portfolio value is closest to the 50th percentile and displays its full year-by-year detail. This "median run" provides a concrete, inspectable example of a typical retirement path, including the specific withdrawal cascade, tax calculations, and account evolution for each year.

This complements the year-by-year median table: the representative run is a single coherent path, while the year-by-year medians are cross-sectional statistics.

---

## 6. The Tax Engine in Detail

The tax engine is one of the most complex components. Here is a summary of what it captures and what it simplifies:

### What is modeled:
- 2024 federal ordinary income brackets (7 brackets, single and MFJ)
- 2024 federal capital gains brackets (3 tiers: 0%, 15%, 20%) with proper stacking
- Standard and itemized deductions
- Social Security provisional income and taxable benefit calculation (two-threshold formula)
- Net Investment Income Tax (3.8% surtax)
- Roth conversion tax delta (marginal cost of conversion)
- Filing status switch from MFJ to single upon death of a spouse
- Marginal ordinary and capital gains rates for non-spending tax attribution

### What is simplified or omitted:
- No state income taxes
- No AMT (Alternative Minimum Tax)
- No tax bracket inflation adjustment over time
- No itemized deduction phaseouts
- No qualified business income deduction
- No tax credits (child, education, etc.)
- NIIT threshold is not inflation-adjusted
- Social Security thresholds are nominal (not inflation-adjusted), matching actual IRS policy (these thresholds have never been indexed)

---

## 7. Scenario Comparison

The simulator allows saving up to 5 scenarios for side-by-side comparison. Each saved scenario records:

- A user-provided label
- After-tax ending value
- Total lifetime taxes
- Portfolio and Roth CAGRs
- Ruin probability (for distribution runs)
- Full percentile detail (for distribution runs)

This enables direct comparison of strategies — for example, comparing Roth conversion amounts, different withdrawal levels, or different asset allocations.

---

## 8. Key Design Decisions and Their Justification

### 8.1 Asset Location in Rebalancing

The rebalancing algorithm places stocks preferentially in Roth (tax-free compounding on the highest-returning asset) and bonds in TDAs (interest taxed at ordinary rates regardless of account type). This follows the classical asset location framework:

> "Place the most tax-inefficient assets in the most tax-advantaged accounts."

This is a simplification — in practice, optimal asset location depends on marginal tax rates, time horizon, and expected returns (Reichenstein, 2006). But for a simulation tool, this heuristic is well-established and produces realistic results.

### 8.2 Withdrawal Order

The cascade (RMDs first -> taxable -> TDA -> Roth) preserves tax-advantaged growth as long as possible. This is a standard recommended withdrawal order in the financial planning literature, though optimal ordering can depend on specific tax situations.

### 8.3 Guardrail Dead Band

The dead band (lower/upper guardrail around a target) prevents constant adjustment. Without it, the withdrawal would be recalculated every year, creating volatile spending. The dead band ensures spending only changes when the forward success rate moves meaningfully away from target — analogous to the "guardrails and floor" approach advocated by Guyton and Klinger (2006).

### 8.4 Max Spending Cap

Without a cap, guardrails can recommend very high spending in strong markets, essentially spending down the portfolio toward a target ending balance near zero. The max spending cap prevents this by limiting spending to a percentage above the base withdrawal (e.g., 150% of base), ensuring the portfolio retains a buffer against future downturns. This reflects a practical insight: most retirees have a limited capacity to increase spending meaningfully, even when their portfolio allows it.

### 8.5 Bond Returns from Historical Data

Rather than modeling bond returns parametrically, the simulator samples from historical bond factors. This captures the non-normality of bond returns, the relationship between bond returns and the interest rate environment, and avoids the need for additional parameters. The tradeoff is that truly unprecedented interest rate regimes would not be explored.

### 8.6 Non-Spending Tax Separation

The after-tax spending formula required careful treatment of taxes that apply to reinvested income (bond interest, dividends, turnover gains, RMD excess). Without this separation, after-tax spending could appear negative in scenarios where the portfolio generates large taxable income that is reinvested rather than spent. The marginal-rate approach is an approximation (the true incremental tax of each component depends on bracket stacking), but it produces reasonable results and prevents the formula from breaking down at the extremes.

---

## 9. Limitations and Areas for Enhancement

1. **No inflation adjustment.** All values are nominal. The user should input real (inflation-adjusted) return parameters and interpret spending figures in constant dollars. Alternatively, the base withdrawal could be inflation-adjusted each year.

2. **Federal taxes only.** State income taxes are not modeled. For retirees in high-tax states, this understates total tax burden by potentially 5-10%.

3. **2024 tax code.** Brackets and deductions are hardcoded to 2024 values. Future tax law changes (TCJA sunset in 2026, for example) are not modeled.

4. **No estate planning.** The inheritor marginal rate adjustment on TDAs is a rough approximation of the SECURE Act's 10-year distribution requirement. Actual estate tax implications are not modeled.

5. **Independence assumption (lognormal mode).** Returns are drawn independently across years, ignoring mean reversion, momentum, and volatility clustering. Empirical evidence suggests stock returns exhibit weak mean reversion over long horizons, which this model does not capture.

6. **Fixed nominal spending (without guardrails).** Without guardrails enabled, the withdrawal amount is fixed in nominal terms. A more complete model would adjust spending for inflation automatically.

7. **Simplified blended return for guardrails.** The guardrail inner MC uses a linear blend of stock and bond parameters, which is an approximation. A true two-asset model would be more accurate but significantly slower.

8. **No tax-loss harvesting.** The model does not harvest losses in the taxable account to offset gains, which could reduce taxes in practice.

9. **No Social Security claiming optimization.** The model takes Social Security income as an input rather than optimizing the claiming age.

10. **Single rebalancing frequency.** Rebalancing occurs annually. More frequent or opportunistic rebalancing is not modeled.

---

## 10. Summary

This simulator integrates several streams of retirement planning research — Monte Carlo return generation, dynamic withdrawal strategies, multi-account tax optimization, and distributional analysis — into a single interactive tool. Its key contribution is the combination of:

1. **A high-fidelity tax engine** that captures the interaction between income sources, brackets, and capital gains stacking
2. **Dynamic guardrails** that adapt spending to portfolio performance in real time
3. **Two complementary return models** that allow both parametric and historical analysis
4. **Rich distributional output** that goes beyond a single success/failure rate to show the full range of possible retirement experiences

By modeling the actual mechanics of retirement cash flows — RMDs, withdrawal cascades, tax bracket stacking, Social Security taxation, Roth conversions — the simulator produces results that are substantially more realistic than simple "4% rule" analyses. The distributional output (percentile bands, lifetime spending distributions, scenario comparison) supports the kind of nuanced, probabilistic thinking that modern retirement planning demands.

---

## References

- Bengen, W. P. (1994). "Determining Withdrawal Rates Using Historical Data." *Journal of Financial Planning*, 7(4), 171-180.
- Blanchett, D., Kowara, M., & Chen, P. (2012). "Optimal Withdrawal Strategy for Retirement Income Portfolios." *Retirement Management Journal*, 2(3), 7-20.
- Daryanani, G. (2004). "Opportunistic Rebalancing: A New Paradigm for Wealth Managers." *Journal of Financial Planning*, 17(1).
- Guyton, J. T., & Klinger, W. J. (2006). "Decision Rules and Maximum Initial Withdrawal Rates." *Journal of Financial Planning*, 19(3), 49-57.
- Kitces, M. E. (2012). "The Ratcheting Safe Withdrawal Rate: A More Dominant Version of the 4% Rule?" *Nerd's Eye View*.
- Pfau, W. D. (2018). *How Much Can I Spend in Retirement?* Retirement Researcher Media.
- Reichenstein, W. (2006). "Asset Allocation and Asset Location Decisions Revisited." *Journal of Wealth Management*, 9(1), 76-85.
