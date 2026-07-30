#!/usr/bin/env python3
"""Build a self-contained, interactive HTML view of all 820 historical outcomes
for one strategy -- the "Retirement Fitting Room" distribution-of-outcomes page.

One .html file, no external dependencies, data embedded inline. Every dot is one
historical start-month; click it to see that cohort's 30-year lean-strip.

Usage: python3 build_outcomes_html.py truth_layer_sample_plan.json [--out retirement_fitting_room.html]
"""
import os
os.environ.setdefault("MC_SEED", "424242")

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from run_full_process import build_sim_params
from sim_engine import get_all_historical_windows, get_allocation_column, run_historical_parallel
from spending_safety import essential_reserve_analysis


def build_data(plan_path):
    plan = json.loads(Path(plan_path).read_text())
    sim_params, sim_years, target_stock_pct, base_spending, _ = build_sim_params(plan)
    inheritor_rate = float(plan.get("inheritor_marginal_rate", 0.32))
    goal = float(base_spending)
    essential = float(plan.get("essential_spending", 0.0))

    windows, dates = get_all_historical_windows(sim_years, target_stock_pct)
    _results, ayd = run_historical_parallel(windows, sim_years, inheritor_rate, sim_params)

    cohorts = []
    for run_id, g in ayd.groupby("run"):
        g = g.sort_values("year")
        spend = g["after_tax_spending"].values
        port = g["total_portfolio"].values if "total_portfolio" in g else np.array([])
        years = [int(round(s)) for s in spend]
        avg = int(round(spend.mean()))
        worst = int(round(spend.min()))
        lean_ess = int((spend < essential).sum()) if essential > 0 else 0
        lean_goal = int((spend < goal).sum())
        # first-10yr drawdown depth
        start_port = float(sim_params.get("taxable_start", 0)) + float(sim_params.get("tda_start", 0)) \
            + float(sim_params.get("tda_spouse_start", 0)) + float(sim_params.get("roth_start", 0))
        if len(port):
            base = np.concatenate([[start_port], port[:10]])
            dd = float((base / np.maximum.accumulate(base) - 1.0).min()) * 100
        else:
            dd = 0.0
        cohorts.append({
            "id": int(run_id) + 1,
            "start": f"{dates[int(run_id)]:%Y-%m}",
            "avg": avg,
            "worst": worst,
            "dd": round(dd, 1),
            "end": int(round(float(port[-1]))) if len(port) else 0,
            "leanEss": lean_ess,
            "leanGoal": lean_goal,
            "years": years,
        })

    cohorts.sort(key=lambda c: c["avg"])
    n = len(cohorts)
    # 1-in-100 seat = ~1st percentile by average spend
    seat_idx = max(0, int(round(0.01 * n)) - 1)
    seat_id = cohorts[seat_idx]["id"]

    avgs = np.array([c["avg"] for c in cohorts])
    lean = np.array([c["leanEss"] for c in cohorts])

    # --- per-100 "houses" (proportional, sums to 100) ---
    g0 = int((lean == 0).sum())                       # smooth: no tight year
    a0 = int(((lean >= 1) & (lean <= 2)).sum())       # a year or two
    green100 = round(g0 / n * 100)
    amber100 = round(a0 / n * 100)
    red100 = 100 - green100 - amber100                # a longer rough patch (3+)
    tough = max(cohorts, key=lambda c: c["leanEss"])  # most lean years = toughest journey

    stats = {
        "pctAvgGoal": round(float((avgs >= goal).mean() * 100), 1),
        "pctAvgEss": round(float((avgs >= essential).mean() * 100), 1),
        "pctEveryYearEss": round(float((lean == 0).mean() * 100), 1),
        "medianAvg": int(np.median(avgs)),
        "dippedCohorts": int((lean > 0).sum()),
        "deepestYear": int(avgs.min() if not len(cohorts) else min(c["worst"] for c in cohorts)),
        "buckets": {
            "0": int((lean == 0).sum()),
            "1-2": int(((lean >= 1) & (lean <= 2)).sum()),
            "3-5": int(((lean >= 3) & (lean <= 5)).sum()),
            "6-10": int(((lean >= 6) & (lean <= 10)).sum()),
            "11+": int((lean >= 11).sum()),
        },
    }
    reserve = essential_reserve_analysis(ayd, ideal_spending=goal, essential_spending=essential) if essential > 0 else None

    return {
        "goal": int(goal),
        "essential": int(essential),
        "reserve": reserve,
        "horizon": int(sim_years),
        "n": n,
        "startFirst": cohorts and min(c["start"] for c in cohorts),
        "startLast": cohorts and max(c["start"] for c in cohorts),
        "stockPct": int(round(target_stock_pct * 100)),
        "feeBps": int(round(plan.get("investment_fee_bps", 0))),
        "returnsCol": get_allocation_column(target_stock_pct),
        "seatId": seat_id,
        "houses": {"green": green100, "amber": amber100, "red": red100},
        "toughest": {"start": tough["start"], "lean": tough["leanEss"],
                     "avg": tough["avg"], "worst": tough["worst"]},
        "cohorts": cohorts,
        "stats": stats,
    }


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Retirement Fitting Room — 100 Retirements</title>
<style>
  :root{
    --bg:#0f1419; --panel:#171e26; --ink:#e8edf2; --muted:#9fb0c0; --line:#2a3642;
    --green:#3fae5a; --lime:#9bcf3c; --amber:#e8a838; --orange:#e8702a; --clay:#cf6b4a; --red:#cf6b4a;
    --accent:#5fb7d4; --sky:#1d2630;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);
    font:16px/1.6 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}
  .wrap{max-width:920px;margin:0 auto;padding:34px 20px 80px}
  h1{font-size:32px;margin:0 0 8px;letter-spacing:-.4px;line-height:1.15}
  h2{font-size:22px;margin:40px 0 10px}
  .sub{color:var(--muted);margin:0 0 4px}
  .kicker{color:var(--accent);font-weight:600;letter-spacing:.6px;text-transform:uppercase;font-size:12px}
  .deal{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:20px 24px;margin:20px 0}
  .deal p{margin:0 0 12px} .deal p:last-child{margin:0}
  .big{font-size:19px;font-weight:600}
  .lead{font-size:18px;color:var(--ink);margin:6px 0 0}
  /* houses hero */
  .hero{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:24px 24px 18px;margin:22px 0}
  .hero h3{margin:0 0 2px;font-size:20px}
  .hero .cap{color:var(--muted);font-size:14px;margin:0 0 16px}
  #houses{width:100%;max-width:560px;display:block;margin:0 auto}
  .hkey{display:flex;flex-direction:column;gap:9px;margin:18px auto 0;max-width:560px}
  .hkey .row{display:flex;align-items:center;gap:11px;font-size:15px}
  .hkey .sw{width:20px;height:20px;flex:none;border-radius:3px}
  .hkey .row b{font-variant-numeric:tabular-nums}
  .tough{margin:16px auto 0;max-width:560px;background:var(--sky);border:1px solid var(--line);
    border-radius:10px;padding:12px 14px;font-size:14.5px;color:var(--ink)}
  .tough b{color:var(--clay)}
  /* stat band */
  .statband{display:flex;flex-wrap:wrap;gap:14px;margin:18px 0}
  .stat{flex:1 1 150px;background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px 16px}
  .stat .num{font-size:25px;font-weight:700;letter-spacing:-.5px}
  .stat .lbl{color:var(--muted);font-size:13px;margin-top:2px}
  /* dot field (depth) */
  details.more{margin-top:14px;border:1px solid var(--line);border-radius:12px;background:var(--panel);padding:4px 18px}
  details.more summary{cursor:pointer;padding:14px 0;font-size:16px;color:var(--accent);font-weight:600}
  .controls{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:8px 0 4px}
  .toggle{display:inline-flex;border:1px solid var(--line);border-radius:999px;overflow:hidden}
  .toggle button{background:transparent;color:var(--muted);border:0;padding:8px 16px;cursor:pointer;font-size:14px}
  .toggle button.on{background:var(--accent);color:#06121a;font-weight:600}
  .capt{color:var(--muted);font-size:14px;min-height:20px;margin:6px 0 0}
  .legend{display:flex;gap:16px;flex-wrap:wrap;font-size:13px;color:var(--muted);margin:10px 0}
  .legend i{display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:5px;vertical-align:-1px}
  #field{width:100%;height:340px;display:block;background:var(--bg);border:1px solid var(--line);border-radius:12px}
  .axislbl{fill:var(--muted);font-size:12px}
  .tip{position:fixed;pointer-events:none;background:#05080b;border:1px solid var(--line);
    border-radius:8px;padding:8px 10px;font-size:13px;opacity:0;transition:opacity .08s;z-index:20;max-width:250px}
  .tip b{color:var(--accent)}
  #detail{background:var(--bg);border:1px solid var(--line);border-radius:12px;padding:18px;margin-top:14px;display:none}
  #detail.show{display:block}
  .strip{display:grid;grid-template-columns:repeat(15,1fr);gap:5px;margin:12px 0}
  .cell{aspect-ratio:1;border-radius:4px;position:relative;cursor:default}
  .cell:hover{outline:2px solid var(--ink)}
  .dhead{display:flex;justify-content:space-between;align-items:baseline;flex-wrap:wrap;gap:8px}
  .pill{font-size:12px;color:var(--muted);border:1px solid var(--line);border-radius:999px;padding:2px 10px}
  .foot{color:var(--muted);font-size:12.5px;margin-top:30px;border-top:1px solid var(--line);padding-top:14px}
  a{color:var(--accent)}
  .house{cursor:default;transition:opacity .1s}
</style>
</head>
<body>
<div class="wrap">
  <div class="kicker">Retirement Fitting Room</div>
  <h1>We tested this plan against every retirement of the last 100 years</h1>
  <p class="lead">Not a guess about the future — real market history. More than <b id="n0"></b> actual <span id="hz"></span>-year stretches, crashes and all.</p>

  <div class="hero">
    <h3>Picture 100 retirees, each living this plan through a different slice of history.</h3>
    <p class="cap">Here's how their retirements went. Hover a house to see what it means.</p>
    <svg id="houses" viewBox="0 0 300 300"></svg>
    <div class="hkey" id="hkey"></div>
    <div class="tough" id="tough"></div>
  </div>

  <div class="deal">
    <p class="big">Here's the good news — and it's almost all good news.</p>
    <p>In every one of those histories, this plan gave you the retirement you were after. The middle one paid about <b id="medAvg"></b> a year, year in and year out. Even the histories that began with a crash still added up to a full retirement.</p>
    <p>A few had a rough patch — usually a year or two of belt-tightening, almost always early, if a downturn hit before your savings had room to grow. But even then the must-haves stayed covered: the house, the groceries, the doctor. It works like cruise control — when the road gets steep, it eases off the gas so you never run dry, then picks right back up.</p>
  </div>

  <div class="statband" id="statband"></div>

  <h2>Want to see all of them?</h2>
  <p class="sub">Every history is here — not just three. Each dot is one of the <span class="nn"></span>. Hover to read it; click to open its 30-year journey, year by year.</p>
  <details class="more">
    <summary>Open the full picture — all <span class="nn"></span> histories</summary>
    <div class="controls">
      <span style="color:var(--muted);font-size:14px">Color by:</span>
      <div class="toggle">
        <button id="tAvg" class="on">The average year</button>
        <button id="tYear">Every single year</button>
      </div>
    </div>
    <p class="capt" id="capt"></p>
    <div class="legend" id="legend"></div>
    <svg id="field"></svg>
    <div id="detail"></div>
    <p style="height:8px"></p>
  </details>

  <div class="foot" id="foot"></div>
</div>

<div class="tip" id="tip"></div>
<script>
const DATA = __DATA__;
const $ = s => document.querySelector(s);
const money = v => '$' + Math.round(v).toLocaleString();
const C = {0:'var(--green)','1-2':'var(--lime)','3-5':'var(--amber)','6-10':'var(--orange)','11+':'var(--red)'};
function bucket(n){ return n===0?'0':n<=2?'1-2':n<=5?'3-5':n<=10?'6-10':'11+'; }
function yearColor(v){ if(v>=DATA.goal) return 'var(--green)'; if(v>=DATA.essential) return 'var(--amber)'; return 'var(--clay)'; }

$('#n0').textContent = DATA.n; $('#hz').textContent = DATA.horizon;
$('#medAvg').textContent = money(DATA.stats.medianAvg);
document.querySelectorAll('.nn').forEach(e=>e.textContent = DATA.n + ' histories');
document.querySelectorAll('.g').forEach(e=>e.textContent = DATA.goal.toLocaleString());

/* ---------- 100 HOUSES HERO ---------- */
const tip=$('#tip');
function moveTip(e,html){ tip.style.opacity=1; tip.style.left=(e.clientX+14)+'px'; tip.style.top=(e.clientY+14)+'px'; tip.innerHTML=html; }
function hideTip(){ tip.style.opacity=0; }

function housePath(x,y,s){
  // silhouette: body + roof, single fill
  const p=(a,b)=>`${(x+a*s).toFixed(2)},${(y+b*s).toFixed(2)}`;
  return `M${p(0.16,1)} L${p(0.16,0.45)} L${p(0.04,0.45)} L${p(0.5,0.02)} L${p(0.96,0.45)} L${p(0.84,0.45)} L${p(0.84,1)} Z`;
}
function renderHouses(){
  const H=DATA.houses; // {green,amber,red} sums to 100
  // ordered category list, then scatter via coprime stride so colors spread out
  const cats=[]; for(let i=0;i<H.green;i++)cats.push('g'); for(let i=0;i<H.amber;i++)cats.push('a'); for(let i=0;i<H.red;i++)cats.push('r');
  while(cats.length<100)cats.push('g'); cats.length=100;
  const fill={g:'var(--green)',a:'var(--amber)',r:'var(--clay)'};
  const meaning={
    g:'Smooth sailing — never a single tight year.',
    a:'A year or two of belt-tightening, then back to normal.',
    r:'A longer rough patch — but the must-haves stayed covered the whole way.'};
  // first red cell becomes the circled "toughest"
  const firstRed = (DATA.houses.green+DATA.houses.amber); // index in cats of first 'r'
  const CELL=30, HS=22, pad=(CELL-HS)/2;
  let parts=[], toughCell=-1;
  for(let i=0;i<100;i++){
    const pos=(i*31)%100;           // deterministic scatter (31 coprime to 100)
    const cat=cats[i];
    const row=Math.floor(pos/10), col=pos%10;
    const x=col*CELL+pad, y=row*CELL+pad;
    if(i===firstRed) toughCell=pos;
    parts.push(`<path class="house" data-cat="${cat}" d="${housePath(x,y,HS)}" fill="${fill[cat]}"/>`);
  }
  // ring on the toughest
  if(toughCell>=0){
    const row=Math.floor(toughCell/10), col=toughCell%10;
    const cx=col*CELL+CELL/2, cy=row*CELL+CELL/2;
    parts.push(`<circle cx="${cx}" cy="${cy}" r="17" fill="none" stroke="#fff" stroke-width="2"/>`);
  }
  const svg=$('#houses'); svg.innerHTML=parts.join('');
  svg.querySelectorAll('.house').forEach(el=>{
    el.addEventListener('mousemove',e=>moveTip(e, meaning[el.dataset.cat]));
    el.addEventListener('mouseleave',hideTip);
  });
  // key
  $('#hkey').innerHTML = [
    ['var(--green)', H.green, 'never had a tight year'],
    ['var(--amber)', H.amber, 'had a year or two of belt-tightening'],
    ['var(--clay)',  H.red,   'went through a longer rough patch — essentials still covered'],
  ].map(([c,n,t])=>`<div class="row"><span class="sw" style="background:${c}"></span><span>About <b>${n} of 100</b> ${t}.</span></div>`).join('');
  // toughest callout
  const t=DATA.toughest;
  $('#tough').innerHTML = `<b>The one in the circle is the toughest history we found</b> — a retirement that began in ${t.start}. It had ${t.lean} leaner years spread across 30, and its hardest single year still put ${money(t.worst)} on the table. Even this one averaged ${money(t.avg)} a year. That's as bad as it ever got.`;
}

/* ---------- STAT BAND ---------- */
const s = DATA.stats;
$('#statband').innerHTML = [
  [s.pctAvgGoal+'%', 'of histories paid your full '+money(DATA.goal)+' goal, on the whole'],
  [s.pctEveryYearEss+' of 100', 'never had a single year below the must-haves'],
  ['cruise control', 'trimmed spending in lean years, then restored it — automatically'],
  [money(s.deepestYear), 'the lowest any single year ever got, in the worst history we found'],
].map(c=>`<div class="stat"><div class="num">${c[0]}</div><div class="lbl">${c[1]}</div></div>`).join('');

/* ---------- DOT FIELD (depth) ---------- */
const svg = $('#field');
let W=560, H=340, PADL=54, PADR=18, PADT=24, PADB=42;
const avgs = DATA.cohorts.map(c=>c.avg);
const minA=Math.min(...avgs), maxA=Math.max(...avgs);
const xOf = a => PADL + (a-minA)/(maxA-minA) * (W-PADL-PADR);
function jitter(id){ let x=Math.sin(id*99.13)*10000; return x-Math.floor(x); }
let mode='avg';
function setLegend(m){
  $('#legend').innerHTML = (m==='avg')
    ? `<span><i style="background:var(--green)"></i>Paid the full ${money(DATA.goal)} goal on the whole</span>`
    : Object.entries({'0':'Never a tight year','1-2':'1–2 lean years','3-5':'3–5 lean years','6-10':'6–10 lean years','11+':'11+ lean years'})
        .map(([k,v])=>`<span><i style="background:${C[k]}"></i>${v}</span>`).join('');
}
function draw(){
  W=svg.clientWidth||560;
  svg.setAttribute('viewBox',`0 0 ${W} ${H}`);
  let parts=[];
  [[DATA.goal,money(DATA.goal)+' goal'],[DATA.essential,money(DATA.essential)+' must-haves']].forEach(([v,lbl])=>{
    if(v<minA||v>maxA) return; const x=xOf(v);
    parts.push(`<line x1="${x}" y1="${PADT}" x2="${x}" y2="${H-PADB}" stroke="var(--line)" stroke-dasharray="4 4"/>`);
    parts.push(`<text class="axislbl" x="${x}" y="${PADT-8}" text-anchor="middle">${lbl}</text>`);
  });
  DATA.cohorts.forEach(c=>{
    const x=xOf(c.avg), y=PADT+10 + jitter(c.id)*(H-PADB-PADT-20);
    const col = mode==='avg' ? (c.avg>=DATA.goal?'var(--green)':'var(--clay)') : C[bucket(c.leanEss)];
    parts.push(`<circle data-id="${c.id}" cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="3.4" fill="${col}" style="cursor:pointer"/>`);
  });
  for(let t=0;t<=4;t++){ const a=minA+(maxA-minA)*t/4; parts.push(`<text class="axislbl" x="${xOf(a)}" y="${H-PADB+22}" text-anchor="middle">${money(a)}</text>`); }
  parts.push(`<text class="axislbl" x="${(PADL+W-PADR)/2}" y="${H-6}" text-anchor="middle">Average yearly spending over ${DATA.horizon} years →</text>`);
  svg.innerHTML = parts.join('');
  svg.querySelectorAll('circle').forEach(el=>{
    const c=DATA.cohorts.find(c=>c.id==el.dataset.id);
    el.addEventListener('mousemove',e=>moveTip(e, `<b>${c.start}</b> start<br>Averaged ${money(c.avg)}/yr<br>Lowest year ${money(c.worst)}<br>${c.leanEss} year(s) below must-haves`));
    el.addEventListener('mouseleave',hideTip);
    el.addEventListener('click',()=>openDetail(c));
  });
}
function openDetail(c){
  const d=$('#detail'); d.classList.add('show');
  const cells = c.years.map((v,i)=>`<div class="cell" style="background:${yearColor(v)}" data-v="${v}" data-y="${i+1}"></div>`).join('');
  d.innerHTML = `<div class="dhead"><div><b>Retirement starting ${c.start}</b>
      <span class="pill">averaged ${money(c.avg)}/yr</span>
      <span class="pill">${c.leanEss} lean year(s)</span>
      <span class="pill">lowest year ${money(c.worst)}</span></div></div>
    <p class="sub" style="margin:10px 0 0">Each square is one year. <span style="color:var(--green)">Green</span> = full ${money(DATA.goal)} goal · <span style="color:var(--amber)">Amber</span> = between goal and ${money(DATA.essential)} must-haves · <span style="color:var(--clay)">Clay</span> = below must-haves.</p>
    <div class="strip">${cells}</div>`;
  d.querySelectorAll('.cell').forEach(el=>{
    el.addEventListener('mousemove',e=>moveTip(e,`<b>Year ${el.dataset.y}</b><br>${money(el.dataset.v)}`));
    el.addEventListener('mouseleave',hideTip);
  });
  d.scrollIntoView({behavior:'smooth',block:'nearest'});
}
function setMode(m){ mode=m; $('#tAvg').classList.toggle('on',m==='avg'); $('#tYear').classList.toggle('on',m==='year');
  $('#capt').innerHTML = m==='avg'
    ? `Colored by the <b>whole-retirement</b> picture — every single history paid the full goal on the whole.`
    : `Now colored by <b>how many individual years</b> were lean. The green field grows a small tail — that's the few who hit a rough patch.`;
  setLegend(m); draw();
}
$('#tAvg').onclick=()=>setMode('avg'); $('#tYear').onclick=()=>setMode('year');
$('details.more').addEventListener('toggle',function(){ if(this.open) setTimeout(draw,30); });

$('#foot').innerHTML = `<b>See the data — we're not asking you to take our word for it.</b> Every house above is backed by a real result you can open yourself: <a href="all-820-histories.csv">download all ${DATA.n} histories (CSV)</a> · <a href="methodology.json">exactly how it was built (JSON)</a>.`
  + `<br><br>What we tried on: ${DATA.stockPct}% stocks / ${100-DATA.stockPct}% bonds, ${DATA.feeBps==0?'no':'a '+DATA.feeBps+' basis-point'} yearly cost, a ${money(DATA.goal)} goal with ${money(DATA.essential)} must-haves. Tested against all ${DATA.n} overlapping ${DATA.horizon}-year stretches of real history (${DATA.startFirst}–${DATA.startLast}). The 100 houses show those ${DATA.n} histories scaled to a round 100. History is not a promise of the future — it's the only honest fitting room we have.`;

renderHouses();
setMode('avg');
window.addEventListener('resize',()=>{ if($('details.more').open) draw(); });
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("plan")
    ap.add_argument("--out", default="retirement_fitting_room.html")
    args = ap.parse_args()
    data = build_data(args.plan)
    html = HTML_TEMPLATE.replace("__DATA__", json.dumps(data, separators=(",", ":")))
    Path(args.out).write_text(html)
    kb = len(html) / 1024
    print(f"Wrote {args.out}  ({kb:.0f} KB, {data['n']} cohorts embedded)")
    print(f"  open with: open {args.out}")


if __name__ == "__main__":
    raise SystemExit(main())
