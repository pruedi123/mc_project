================================================================================
  MONTE CARLO RETIREMENT WITHDRAWAL SIMULATOR
  Complete User Guide
================================================================================

TABLE OF CONTENTS
─────────────────
  1.  What This Tool Does
  2.  How to Run It
  3.  How the Simulation Works
  4.  Accuracy and Realism
  5.  Sidebar Inputs Reference
      5.1  Save / Load Inputs
      5.2  Scenario Comparison
      5.3  Ages & Timeline
      5.4  Account Balances
      5.5  Expected Inheritance
      5.6  Allocation & Roth Conversions
      5.7  Withdrawal Schedule
      5.8  Additional Spending Goals
      5.9  Fund Goal Separately
      5.10 Other Income
      5.11 Pension Buyout
      5.12 Tax Settings
      5.13 Return Assumptions
      5.14 Withdrawal Guardrails
      5.15 Simulation Settings
  6.  Running the Simulation
  7.  Understanding Results
      7.1  Client View
      7.2  Advisor View
      7.3  Scenario Comparison
  8.  Social Security Claiming Age Analysis
  9.  Dollar Growth Distribution
  10. Client PDF Report
  11. Save / Load Plans and Client Management
  12. Technical Architecture


================================================================================
1.  WHAT THIS TOOL DOES
================================================================================

This is a Monte Carlo retirement withdrawal simulator designed for financial
advisors and their clients. It answers the fundamental retirement question:

  "Given my portfolio, income sources, spending goals, and tax situation,
   what is the probability that my money lasts through retirement?"

Unlike simple calculators that use a single average return, this tool runs
your plan through hundreds or thousands of different market scenarios —
either actual historical periods dating back to the 1920s or statistically
simulated market sequences — and shows you the full range of outcomes.

Key capabilities:

  - Multi-account modeling (taxable, tax-deferred IRA/401k, Roth) with
    proper tax treatment for each account type
  - Full federal income tax calculation (ordinary brackets, capital gains
    brackets, Social Security taxation, NIIT, IRMAA)
  - State income tax (flat rate with optional retirement income exemption)
  - Required Minimum Distributions (RMDs) with IRS Uniform Lifetime Table
  - Roth conversion strategies (fixed amount or bracket-fill optimization)
  - Dynamic withdrawal guardrails that adjust spending based on portfolio
    performance
  - Social Security with spousal top-up, survivor benefits, COLA, and
    claiming age optimization
  - Pension income with COLA and survivor benefits
  - Annuity purchase modeling
  - Pension buyout analysis (lump sum vs. annuity)
  - Expected inheritance modeling (taxable + inherited IRA with SECURE Act
    10-year drawdown)
  - TCJA sunset modeling (tax brackets revert to pre-2018 law)
  - Separately funded goals (set aside money today for a future expense)
  - Multi-scenario comparison (test different strategies side by side)
  - Two result views: simplified Client View and detailed Advisor View
  - PDF report generation for client presentations


================================================================================
2.  HOW TO RUN IT
================================================================================

Prerequisites:
  - Python 3.13 (or compatible)
  - Required packages: streamlit, pandas, numpy, altair, openpyxl, fpdf2,
    matplotlib

To start:

    streamlit run main.py

This opens a browser window with the simulator. All inputs are in the left
sidebar. Results appear in the main content area after clicking "Run
simulation."


================================================================================
3.  HOW THE SIMULATION WORKS
================================================================================

The simulator supports two return modes:

HISTORICAL MODE (default, recommended)
───────────────────────────────────────
Uses actual stock and bond returns from a global dataset
(master_global_factors.xlsx) covering nearly 100 years of market history.
The simulator constructs every possible overlapping window of N years
(where N is your plan horizon) from the historical data. Each window uses
the actual stock returns (LBM 100E column) and bond returns (LBM 100 F
column) from that period.

For a 30-year plan, this typically yields 800-900+ distinct historical
scenarios. Each scenario preserves the real sequence of returns — including
the Great Depression (1929), stagflation era (1966), dot-com bust (2000),
and 2008 financial crisis — with their actual year-by-year ordering.

Historical mode also loads actual CPI data (cpi_mo_factors.xlsx) for each
window, so pension purchasing power erosion is computed from the real
inflation experienced in that period rather than an assumed average.

SIMULATED MODE (lognormal)
──────────────────────────
Generates random stock and bond returns from lognormal distributions with
user-specified drift and volatility parameters. Default parameters are
calibrated to match the historical data:

  Stock log drift (mu):      0.09038261
  Stock log volatility (sigma): 0.20485277
  Bond log drift (mu):       0.0172918
  Bond log volatility (sigma):  0.04796435

Each Monte Carlo run draws an independent sequence of returns. The number
of runs is configurable (default: 1,000).

IMPORTANT: ALL RETURNS ARE REAL (NET OF INFLATION)
──────────────────────────────────────────────────
Both historical and simulated returns are expressed in real (inflation-
adjusted) terms. This means every dollar amount in the simulator —
spending targets, portfolio values, tax calculations, Social Security
benefits — is in constant, current-year purchasing power.

Because all values are in real dollars, tax brackets, standard deductions,
IRMAA thresholds, and other nominal tax parameters do NOT need inflation
indexing. They are stated in current-year dollars and remain correct
throughout the simulation.

FOR EACH SCENARIO, THE SIMULATOR DOES THE FOLLOWING EACH YEAR:
──────────────────────────────────────────────────────────────
  1. Rebalance the household portfolio to the target stock/bond allocation
     across all accounts (Roth gets stocks first for tax-free growth,
     TDAs get bonds first since interest is taxed as ordinary income)
  2. Apply investment returns (stock growth, bond growth, dividends,
     interest, capital gains from turnover) to each account
  3. Deduct the advisory fee (if any)
  4. Calculate Required Minimum Distributions for each person
  5. Perform Roth conversions (if configured)
  6. Compute all income: Social Security, pensions, annuities, earned
     income, other income
  7. Use guardrails (if enabled) to check whether spending should be
     adjusted up or down based on forward survival probability
  8. Execute withdrawals using a tax-efficient waterfall:
     Default order: Taxable bonds -> Taxable stocks -> TDA -> Roth
     (or TDA-first if that option is selected)
  9. Calculate full tax liability: ordinary income tax, capital gains tax,
     NIIT, state tax, IRMAA surcharges, Social Security taxation
  10. Record after-tax spending, account balances, and all metrics


================================================================================
4.  ACCURACY AND REALISM
================================================================================

What makes this simulator realistic:

TAX MODELING
  - Uses actual 2024 federal ordinary income brackets (7 brackets from
    10% to 37%)
  - Capital gains brackets (0%/15%/20%) properly stacked on top of
    ordinary income
  - Social Security benefits taxed using the real provisional income
    formula (50% and 85% thresholds)
  - Net Investment Income Tax (3.8% surtax) on high earners
  - Medicare IRMAA surcharges (Part B + Part D) with 2-year lookback
  - State income tax with optional retirement income exemption
  - TCJA sunset modeling (optional): brackets revert to pre-2018 rates
    (10/15/25/28/33/35/39.6%)
  - Standard deduction vs. itemized deduction choice
  - Proper treatment of qualified dividends vs. ordinary dividends
  - Capital gains basis tracking in taxable accounts

ACCOUNT TYPES
  - Taxable accounts: tracks separate stock and bond market values AND
    cost basis. Generates dividends, interest, and turnover-driven
    capital gains. Withdrawals trigger realized gains/losses based on
    actual basis.
  - Tax-deferred accounts (TDA/IRA/401k): Two separate TDAs (one per
    person) for proper RMD calculation. All withdrawals taxed as ordinary
    income.
  - Roth accounts: Tax-free growth and withdrawals. Roth conversion
    amount taxed as ordinary income in the conversion year.

REQUIRED MINIMUM DISTRIBUTIONS
  - Uses the IRS Uniform Lifetime Table (Publication 590-B, effective 2022)
  - Separate RMD calculations for each person based on their own age and
    TDA balance
  - RMDs begin at a configurable age (default 73)
  - RMD amounts are included in ordinary taxable income

SOCIAL SECURITY
  - Early/delayed claiming adjustments using SSA formulas:
    * Before FRA: 5/9 of 1% per month for first 36 months, then 5/12 of
      1% per additional month
    * After FRA: 8% per year delayed retirement credits up to age 70
  - Spousal top-up: 50% of other spouse's PIA minus own PIA, reduced for
    early claiming using the spousal reduction formula (25/36 of 1% per
    month for first 36 months, then 5/12 of 1% beyond)
  - Survivor benefits: surviving spouse gets the higher of own benefit
    (plus spousal) or the deceased's benefit (or 82.5% of PIA, whichever
    is higher), reduced for age if claimed before FRA
  - Filing dependency: spousal benefits require BOTH spouses to have filed
  - Optional COLA adjustment
  - Qualified Charitable Distributions (QCDs) from IRA at age 70+

WITHDRAWAL GUARDRAILS
  - At each year, runs a fast inner Monte Carlo simulation to estimate the
    probability of the plan surviving the remaining years
  - If the survival rate drops below the lower guardrail, spending is
    reduced to restore the target success rate
  - If the survival rate exceeds the upper guardrail, spending is
    increased (up to a configurable cap)
  - Essential goals (needs) are never cut; flexible goals (wants) are cut
    first before base spending is reduced

REBALANCING
  - Annual household-level rebalancing across all accounts
  - Tax-efficient asset location: stocks preferentially in Roth (tax-free
    growth), bonds preferentially in TDAs (interest taxed as ordinary
    income anyway)
  - Rebalancing trades in taxable accounts generate realized capital gains

Known simplifications:
  - Returns are applied annually, not monthly or daily
  - No modeling of FICA/payroll taxes on earned income
  - Bond returns are modeled as a single aggregate (no duration/credit
    quality distinction)
  - Inflation is implicit (real returns) rather than explicit
  - State taxes use a flat rate rather than state-specific brackets
  - No modeling of alternative minimum tax (AMT)
  - Rebalancing happens once per year (no tactical rebalancing)
  - No modeling of sequence-of-returns risk within a single year
  - Annuity pricing is input-based, not actuarially computed


================================================================================
5.  SIDEBAR INPUTS REFERENCE
================================================================================

All inputs are organized in collapsible sections in the left sidebar.

────────────────────────────────────────
5.1  SAVE / LOAD INPUTS
────────────────────────────────────────

Manage client plans. Plans are stored as JSON files in:

    ~/RWM/Current Client Plans/{Last, First}/{plan_name}.json

  Select client        Choose an existing client or create a new one
  Last name / First    Client name (used for folder naming)
  Identifier           Optional disambiguator (e.g., "Portland")
  Plan name            Auto-generated from client name, auto-incremented
  Save Inputs          Save current sidebar settings to JSON
  Load                 Restore all sidebar settings from a saved file
  Delete               Remove a saved plan file
  Rename Client        Change the client folder name

When you save, an automatic timestamped backup is created and changes are
logged to audit.log in the client's folder. Simulation results are also
saved alongside the input file as {plan_name}_results.json.

────────────────────────────────────────
5.2  SCENARIO COMPARISON
────────────────────────────────────────

Test multiple strategies in a single run. Scenario 1 is always the
baseline (using all sidebar settings). Additional scenarios override
specific values:

  Spending override    Same as baseline / Scale by % / Set fixed amount
  Stock % override     Test a different stock allocation
  Roth conversion      Test a different Roth strategy (None, Fixed, Fill)
  Buy annuity          Purchase an annuity from taxable account (scenario
                       adds an income stream and reduces taxable balance)
  Pension buyout       Compare lump sum vs. annuity in this scenario
  Life expectancy      Test early death scenarios

Each scenario runs the full simulation independently. Results are compared
side by side with delta columns showing the impact vs. baseline.

────────────────────────────────────────
5.3  AGES & TIMELINE
────────────────────────────────────────

  Starting age (P1)         Age of Person 1 at simulation start
  Starting age (P2)         Age of Person 2 (spouse) at start
  Primary life expectancy   Last age Person 1 lives through
  Spouse life expectancy    Last age Person 2 lives through

The simulation runs for the longer of the two lifetimes. When one person
dies, the plan switches to single-filer tax brackets, pensions trigger
survivor provisions, and Social Security switches to survivor benefits.

────────────────────────────────────────
5.4  ACCOUNT BALANCES
────────────────────────────────────────

  Taxable account balance        Brokerage/investment account
  Taxable stock basis %          Cost basis as % of market value (stocks)
  Taxable bond basis %           Cost basis as % of market value (bonds)
  Roth account balance           Roth IRA balance
  TDA balance - Person 1         IRA/401k for Person 1
  TDA balance - Person 2         IRA/401k for Person 2

The cost basis percentages determine embedded unrealized gains. A basis of
50% means half the market value is gain. This affects taxes on withdrawals
and rebalancing trades in the taxable account.

────────────────────────────────────────
5.5  EXPECTED INHERITANCE
────────────────────────────────────────

Model a one-time future inheritance:

  Enable                    Toggle inheritance modeling
  Year of inheritance       Simulation year when received (e.g., year 10)
  Inherited taxable assets  Amount received as stocks/bonds (gets
                            stepped-up cost basis = no embedded gains)
  Inherited IRA amount      Amount received as inherited IRA (pre-tax,
                            subject to SECURE Act 10-year drawdown rule:
                            distributed evenly over 10 years as ordinary
                            income)

────────────────────────────────────────
5.6  ALLOCATION & ROTH CONVERSIONS
────────────────────────────────────────

  Target % in stocks           Household-level stock/bond allocation
                               (e.g., 60 = 60% stocks, 40% bonds)

  Prefer TDA before taxable   Changes withdrawal waterfall to deplete TDA
                               faster (useful to reduce future RMDs or
                               take advantage of low tax brackets)

  Roth conversion mode:
    None                       No conversions
    Fixed amount               Convert a set dollar amount per year from
                               TDA to Roth (taxed as ordinary income)
    Fill to bracket            Automatically convert enough to fill up to
                               a chosen tax bracket ceiling (e.g., fill
                               the 22% bracket = convert until taxable
                               income hits $201,050 for MFJ)

  Years to perform conversions   How many years (from year 1) to convert
  Convert from                   Person 1 TDA or Person 2 TDA
  Pay taxes from                 Taxable account or TDA (reduces net
                                 conversion if paid from TDA)

Bracket-fill Roth conversions are powerful tax planning: convert in low-
income years (early retirement before Social Security starts) to fill up
cheap tax brackets, reducing future RMDs and avoiding higher brackets later.

────────────────────────────────────────
5.7  WITHDRAWAL SCHEDULE
────────────────────────────────────────

Define your after-tax spending target in one or more periods:

  Number of periods    Split retirement into phases (e.g., active years
                       vs. later years with lower spending)
  Period N amount      Annual after-tax spending goal for that period
  Period N end year    Last year of the period (auto for final period)

  RMD start age (P1)      Age when RMDs begin for Person 1 (default 73)
  RMD start age (P2)      Age when RMDs begin for Person 2 (default 73)

  Ending balance goal     Portfolio must be >= this amount at the end for
                          the run to count as "successful." Use $1 for
                          "don't run out of money." Use $500,000 for a
                          legacy target. Use $0 for no minimum.

The withdrawal schedule represents your AFTER-TAX spending target. The
simulator calculates the gross (pre-tax) withdrawal needed to deliver
that after-tax amount by solving for taxes iteratively.

────────────────────────────────────────
5.8  ADDITIONAL SPENDING GOALS
────────────────────────────────────────

Layer extra spending on top of base withdrawals:

  Label          Name for the goal (e.g., "Long-term care", "Travel")
  Annual amount  After-tax spending target per year
  Begin year     First year of the goal
  End year       Last year of the goal
  Priority:
    Essential    Always funded at target even in bad markets
    Flexible     Adjusted with base spending when guardrails trigger

  Spending cap   Limits upside in good markets (0 = never exceed target,
                 50 = up to 150% of target, -1 = no cap)

────────────────────────────────────────
5.9  FUND GOAL SEPARATELY
────────────────────────────────────────

For goals like long-term care, you can "fund separately" — meaning:

  1. The simulator calculates how much to set aside TODAY to guarantee
     the goal is fully funded even in the worst historical market scenario
  2. Money is moved from your primary portfolio into shadow accounts
  3. The goal is removed from the regular withdrawal plan
  4. Shadow accounts grow alongside primary accounts and generate real
     tax events (dividends, capital gains, RMD base inflation)
  5. In the goal years, the shadow accounts are liquidated to fund the goal

Inputs when "Fund separately" is checked:

  Stock allocation %     Goal account's own stock/bond mix (lower = more
                         conservative, but higher set-aside cost)
  From Taxable           How much of the set-aside comes from taxable
  From TDA P1            How much from Person 1's TDA
  From TDA P2            How much from Person 2's TDA

Leave all sources at $0 to auto-fill (taxable first, then TDA P1, then
TDA P2).

The set-aside cost uses the 0th percentile (worst case) growth factor
from all historical or simulated paths. This means in most scenarios the
goal account will have surplus money remaining after the goal is funded.

────────────────────────────────────────
5.10  OTHER INCOME
────────────────────────────────────────

SOCIAL SECURITY
  Annual SS - Person 1       Worker benefit at claimed age (per SSA stmt)
  SS start age - P1          When P1 begins collecting
  SS full retirement age     FRA for P1 (66 or 67)
  Annual SS - Person 2       Worker benefit at claimed age
  SS start age - P2          When P2 begins collecting
  SS FRA - P2                FRA for P2
  SS COLA                    Annual cost-of-living adjustment (0 = fixed)

Enter each person's own worker benefit. Spousal top-up and survivor
benefits are computed automatically for married-filing-jointly filers.

PENSION
  Annual pension - P1        P1's pension income
  Pension COLA - P1          Annual adjustment (0 = fixed, erodes with
                             inflation since all values are real)
  Pension survivor % - P1    Fraction paid to survivor after P1 dies
  Annual pension - P2        P2's pension income
  Pension COLA - P2          P2's annual adjustment
  Pension survivor % - P2    Fraction paid to survivor after P2 dies

OTHER
  Other ordinary income      Any other taxable income (rental, etc.)
  Annual earned income       Part-time work income (W-2 or self-employment)
  Years of earned income     How many years earned income continues
  Annual QCD amount          Qualified Charitable Distributions from IRA
                             (age 70+, satisfies RMDs, excluded from
                             taxable income, max $105K/year/person)

────────────────────────────────────────
5.11  PENSION BUYOUT (LUMP SUM VS. ANNUITY)
────────────────────────────────────────

Compare taking a lump sum (rolled into TDA) vs. an annuity stream:

  Baseline choice       Which option you start with
  Buyout for            Person 1 or Person 2
  Lump sum amount       One-time amount rolled into TDA
  Annuity income        Annual income from the annuity alternative
  Annuity COLA          Annual cost-of-living adjustment
  Annuity survivor %    Fraction paid to surviving spouse

When enabled, the simulator automatically creates two scenarios: one with
the lump sum and one with the annuity. If you also set up manual scenarios,
they are cross-multiplied (each override is tested with both the lump sum
and annuity sides).

────────────────────────────────────────
5.12  TAX SETTINGS
────────────────────────────────────────

  Enable taxation          Uncheck to disable all taxes (for testing)
  Filing status            Single or Married Filing Jointly

  Use itemized deductions  Override the standard deduction
  Itemized deduction amt   Your itemized deduction amount

  Inheritor marginal rate  Tax rate applied to TDA balances when computing
                           after-tax ending portfolio (default 35%)
  State income tax rate    Flat state tax rate
  Exempt retirement income Illinois-style exemption: only investment income
                           (interest, dividends, capital gains) is taxed
                           at the state level; SS, pensions, and TDA
                           withdrawals are exempt

  TCJA sunset              Model Tax Cuts and Jobs Act expiration:
                           brackets revert to pre-2018 rates
  TCJA sunset year         Simulation year when TCJA expires

  IRMAA surcharges         Model Medicare Part B + Part D income-related
                           surcharges for persons 65+ (based on MAGI
                           from 2 years prior)

────────────────────────────────────────
5.13  RETURN ASSUMPTIONS
────────────────────────────────────────

  Return mode:
    Historical             Use actual stock/bond returns from the data file.
                           Runs all possible historical windows as a
                           distribution. This is the recommended mode.
    Simulated (lognormal)  Generate random returns from a lognormal
                           distribution with specified drift and volatility.

  Stock log drift / vol    Parameters for lognormal stock returns (only
                           used in simulated mode)
  Bond log drift / vol     Parameters for lognormal bond returns

  Stock dividend yield     Qualified dividend yield on stocks (default 2%)
  Stock turnover rate      Annual portfolio turnover generating short-term
                           capital gains (default 10%)
  Investment fee (bps)     Annual advisory fee in basis points (deducted
                           from returns each year)

────────────────────────────────────────
5.14  WITHDRAWAL GUARDRAILS
────────────────────────────────────────

Dynamic spending adjustment based on portfolio performance:

  Enable guardrails        Toggle on/off

  Lower guardrail          If forward success rate drops below this, cut
                           spending (default 75%)
  Upper guardrail          If forward success rate exceeds this, increase
                           spending (default 90%)
  Target success rate      What to reset to when guardrails trigger
                           (default 85%)
  Inner MC simulations     How many inner simulations per guardrail check
                           (default 200, more = more precise but slower)
  Max spending cap         How much spending can increase above base
                           (0 = never exceed target, 50 = up to 150%,
                           -1 = unlimited)
  Flexible goal minimum    Floor for flexible goal cuts (50% = flex goals
                           can be cut by at most half before base spending
                           is reduced)

How guardrails work:
  At the beginning of each simulation year, the system runs a fast inner
  Monte Carlo to estimate the probability that the remaining portfolio can
  fund the remaining spending schedule. If the probability falls below the
  lower guardrail (e.g., 75%), a binary search finds the spending level
  that restores the target success rate (e.g., 85%). If the probability
  exceeds the upper guardrail (e.g., 90%), spending is increased (again
  via binary search to hit the target).

  This mimics how a good financial advisor adjusts a client's spending
  annually based on how the portfolio has performed.

────────────────────────────────────────
5.15  SIMULATION SETTINGS
────────────────────────────────────────

  Decimal places       Display precision for tables/charts (default 0)
  Monte Carlo runs     Number of random simulations in lognormal mode
                       (default 1,000; higher = more precision)


================================================================================
6.  RUNNING THE SIMULATION
================================================================================

After configuring all inputs, click "Run simulation" (or "Run all
scenarios" if comparing scenarios).

In Historical mode, the simulator runs through all ~800+ historical
windows. A progress bar shows advancement. Typical run time is 30-120
seconds depending on plan complexity and whether guardrails are enabled.

In Simulated mode, the number of runs matches the "Monte Carlo runs"
setting. Each run draws independent random return sequences.


================================================================================
7.  UNDERSTANDING RESULTS
================================================================================

Results are shown in two views, toggled at the top of the results area.

────────────────────────────────────────
7.1  CLIENT VIEW
────────────────────────────────────────

Designed for client presentations. Shows:

THE VERDICT
  A large banner showing "Your plan succeeds in N out of 100 simulations."
  Color-coded: green (>= 90%), yellow (75-89%), red (< 75%).

YOUR SPENDING
  Three cards per spending category (base + each goal):
    If Markets Struggle    Worst-case average annual after-tax spending
    Most Likely            Median (50th percentile) average spending
    If Markets Do Well     90th percentile average spending
  Cards turn gold when at or above target, red when below.

SEPARATELY FUNDED GOALS
  For goals funded separately, three cards showing:
    Set Aside Today        How much was reserved from the portfolio
    Annual Spending        Guaranteed delivery amount
    Median Surplus         Expected leftover in the goal account

WHAT YOU'LL LEAVE BEHIND
  Three legacy cards (0th / 50th / 90th percentile ending portfolio).

PORTFOLIO OVER TIME
  Median portfolio balance chart with pre-tax/after-tax toggle.

WHAT IF YOU HAD THE WORST TIMING?
  Stress tests using actual historical worst periods (1929 Great
  Depression, 1966 Stagflation). Shows year-by-year spending bars with
  target line overlay.

────────────────────────────────────────
7.2  ADVISOR VIEW
────────────────────────────────────────

Full analytical detail. Includes everything in Client View plus:

SPENDING OUTCOMES
  Horizontal bar chart showing 0th/10th/25th/50th/75th/90th/100th
  percentile spending outcomes with spending target line.

SPENDING PERCENTILE TABLE
  Detailed table of average annual and lifetime spending at each
  percentile (gross withdrawal and after-tax).

SEPARATELY FUNDED GOALS (detailed)
  Per-goal cost breakdown table: for each goal year, shows the worst-case
  growth factor, cost per dollar, annual goal, and set-aside amount.
  Surplus distribution table (0th/25th/50th/75th/90th percentiles of
  the goal account balance at each year).
  Combined goal account balance chart across all years.

PORTFOLIO PERCENTILE BANDS
  Year-by-year chart showing portfolio value at 0th/10th/25th/50th/75th/
  90th/100th percentiles, with the ending balance goal line.

ENDING BALANCE TABLE
  Detailed table for each percentile: ending balance (pre-tax and
  after-tax), total taxes paid, effective tax rate.

MEDIAN RUN YEAR-BY-YEAR TABLE
  Full detail for the median simulation run: income sources, withdrawals,
  account balances, taxes, RMDs, Roth conversions, and more.

INCOME SOURCES OVER TIME
  Stacked area chart of median annual income sources (Social Security,
  pension, annuity, withdrawals, Roth conversions, etc.).

────────────────────────────────────────
7.3  SCENARIO COMPARISON
────────────────────────────────────────

When multiple scenarios are run, a comparison section appears above the
detailed results:

  Comparison table at a chosen percentile showing:
    After-tax ending balance (and delta vs. baseline)
    Total taxes paid (and delta)
    Effective tax rate
    Average annual spending (and delta)
    % of runs ending below goal

  Portfolio value comparison chart (median, all scenarios overlaid)
  After-tax spending comparison chart (median, all scenarios overlaid)

  Drill-down selector: pick any scenario to see its full detailed view.


================================================================================
8.  SOCIAL SECURITY CLAIMING AGE ANALYSIS
================================================================================

After running a simulation, an optional "SS Claiming Age Analysis" section
appears. It tests every combination of claiming ages for both spouses
(62-70 for each) and shows:

  - A breakeven table: cumulative SS collected by ages 75, 80, 85, 90, 95
    at each claiming age
  - The PIA (Primary Insurance Amount) back-calculated from the entered
    benefit
  - A grid of plan success rates for every (P1 claim age, P2 claim age)
    combination
  - The optimal claiming age pair that maximizes the plan's success rate

This analysis runs the full simulation for each combination, so it takes
several minutes but provides powerful insight into when to claim.


================================================================================
9.  DOLLAR GROWTH DISTRIBUTION
================================================================================

A standalone analysis tool (also accessible in the results area) that
answers: "If I invest $1 in this allocation, what range of values will I
have after N years?"

Shows percentile bands of cumulative dollar growth across all historical
windows or simulated runs. This is also used internally to price the
"Fund Goal Separately" feature.


================================================================================
10. CLIENT PDF REPORT
================================================================================

After running a simulation, a "Generate Client Report" button appears at
the bottom. It produces a 3-4 page PDF suitable for client presentations:

  Page 1: Big Picture — success rate, assumptions summary
  Page 2: Range of Outcomes — percentile table and portfolio bands chart
  Page 3: Year-by-year median income sources table and stacked chart
  Page 4: Scenario Comparison (only if multiple scenarios were run)

The PDF is generated entirely locally using fpdf2 and matplotlib. No
external services are needed.


================================================================================
11. SAVE / LOAD PLANS AND CLIENT MANAGEMENT
================================================================================

Plans are organized by client in:

    ~/RWM/Current Client Plans/
      Smith, John/
        smith_john.json           <- input settings
        smith_john_results.json   <- simulation results (auto-saved)
        backups/                  <- timestamped backups of prior saves
        audit.log                 <- change log

WHAT IS SAVED:
  - All sidebar widget values (account balances, ages, allocations, etc.)
  - Withdrawal schedule periods
  - Additional spending goals (including fund-separately settings)
  - Scenario override configurations
  - Simulation results (percentile tables, median yearly data)

WHAT IS NOT SAVED:
  - Scenario comparison state (cleared on browser tab close)
  - SS claiming age analysis results (regenerated on demand)

The "Save Scenario" feature at the bottom of the main page is separate —
it saves results to in-memory session state for within-session comparison.
Plans saved with "Save Inputs" persist across browser sessions.

Plans from different clients can be compared by loading them sequentially
and saving scenarios. The "Persistent Plan Comparison" feature allows
loading previously saved results alongside current results.


================================================================================
12. TECHNICAL ARCHITECTURE
================================================================================

The codebase is organized into focused modules with no circular
dependencies:

  tax_engine.py   (~235 lines)  Pure tax math (brackets, deductions,
                                NIIT, IRMAA, state tax). No Streamlit.

  sim_engine.py   (~1,300 lines) Simulation logic: data loading, historical
                                windows, MC runner, withdrawal simulation,
                                Social Security calculations, guardrails,
                                rebalancing. No Streamlit.

  growth_engine.py (~112 lines) Dollar growth distribution (pure investment
                                compounding). No Streamlit.

  ui_inputs.py    (~950 lines)  All sidebar widgets, save/load, client
                                management. Streamlit-dependent.

  main.py         (~1,300 lines) Streamlit orchestrator: builds sim params,
                                runs scenarios, displays results. Ties
                                everything together.

  pdf_report.py   (~600 lines)  PDF generation using fpdf2 + matplotlib.

Import dependency graph (no cycles):

    tax_engine  <-  sim_engine  <-  main.py
                                      ^
                    ui_inputs  --------+
                    pdf_report --------+
                    growth_engine -----+

Data files:
  master_global_factors.xlsx    Historical stock and bond growth factors
  median_cpi_purchasing_power.xlsx  Median purchasing power factors (1-40yr)
  cpi_mo_factors.xlsx           Monthly CPI factors for per-run historical
                                purchasing power calculations

================================================================================
