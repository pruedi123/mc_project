# How Client Aliases Work

## Why This Exists

When you use Claude to build and run retirement plans, Claude sees everything you type and every file it reads. Client names are private information that shouldn't be sent to any AI service. This system keeps real names on your machine and gives Claude a fake codename (alias) instead.

**What Claude sees:** "Teal-Helm, couple, ages 67 and 62, $500k taxable..."
**What Claude never sees:** The real client name

---

## The Two Tools

### 1. Alias Manager (Streamlit App)

**How to open:** Double-click **"Open Alias Manager"** on your Desktop.

This is your private dashboard. It lets you:

- **Add a new client** — type their real name, get an alias like "Teal-Helm"
- **See all clients** — shows aliases and their saved plans
- **Click a plan** to see its inputs (ages, accounts, SS, allocation, etc.)
- **Show real names** — checkbox that reveals which alias belongs to which client (only you see this)

### 2. Claude (this AI)

Claude works entirely with aliases. You talk to Claude using the alias, and Claude never knows or asks for the real name.

---

## Step-by-Step Workflows

### Adding a New Client

1. Open the **Alias Manager** (double-click "Open Alias Manager" on Desktop)
2. Type the client's real name (e.g., "Smith, Bob & Mary")
3. Click **Generate Alias** — you'll see something like "River-Cliff"
4. Come to Claude and say:

   > "New client River-Cliff, couple, ages 63 and 60. Person 1 has $500k in a TDA, person 2 has $200k IRA. Taxable account $300k. Social Security $28k at 67 for person 1, $12k at 62 for person 2. 60% equity, 25% spending cap. Run full process."

5. Claude builds the plan, runs the simulation, and saves everything — all under the alias

### Running a Plan for an Existing Client

1. If you don't remember the alias, open the **Alias Manager** and check
2. Tell Claude:

   > "Run full process for Teal-Helm"

   or

   > "Run a quick sim for Teal-Helm"

### Comparing Scenarios

> "Run full process for Teal-Helm at 60% equity and 80% equity and compare"

### Modifying a Plan

> "Load Teal-Helm's baseline and change person 1 SS to $32,000 at age 70. Run full process."

### Checking What's Saved

Open the **Alias Manager** and click on a plan label (e.g., "Baseline" or "80% Equity") to see all the inputs.

---

## What "Run Full Process" vs "Quick Sim" Means

**"Run full process"** — The comprehensive analysis (takes about 60 seconds):
1. Auto-calculates optimal spending at 90% success rate
2. Shows percentile distribution of ending balances
3. Shows year-by-year detail for the worst historical period (1929)
4. Finds how much the portfolio could decline before hitting 75% success
5. Calculates stressed spending at the declined balance

**"Run a quick sim"** — A basic simulation (takes about 5 seconds):
- Success rate, median ending portfolio, PDF report
- No spending optimization or stress testing

---

## Where Things Are Stored

| What | Where | Who Can See It |
|------|-------|----------------|
| Real name-to-alias mapping | `~/RWM/.client_aliases.json` | Only you (local file) |
| Client plan files | `~/RWM/Current Client Plans/` | Only you (local folders named with real names) |
| Alias Manager app | `mc_project/alias_manager.py` | Only you (local app) |
| Pseudonymization code | `mc_project/pseudonymize.py` | Safe — contains no client data |

---

## Voice Dictation Tips

You can dictate plan details to Claude using macOS dictation (Fn Fn) or any voice input:

- Say numbers naturally: "five hundred thousand" or "500k"
- Say "person one" and "person two" for the spouses
- Use the alias at the beginning: "New client Teal-Helm..."
- Don't worry about perfect formatting — Claude will confirm what it heard before running anything

---

## Quick Reference: What to Say to Claude

| You want to... | Say this |
|----------------|----------|
| Run a full analysis | "Run full process for [alias]" |
| Run a basic sim | "Run a quick sim for [alias]" |
| Compare two scenarios | "Run [alias] at 60% and 80% equity, compare" |
| Change inputs | "Load [alias]'s baseline, change X to Y, run again" |
| Save a plan | "Save this as the baseline for [alias]" |
| See what's saved | Check the Alias Manager app |
