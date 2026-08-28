#!/usr/bin/env python
"""Build a concise, animated HTML technical memo from frozen Phase 2 results.

The memo is generated from the same matrices and figures used by
``phase2_report.py``.  It never reads training data or refits/rescores a model,
so it is safe to regenerate after the once-only 2023 test.  The page keeps the
result values inline and refers to the publication figures by repository-
relative paths, making it portable with the accompanying ``figures/`` folder.
"""
from __future__ import annotations

import argparse
import html
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.common.config import PATHS, ROOT, Config, load_config
from src.common.logio import get_logger

log = get_logger("phase2_techmemo")

OUTPUT_PATH = ROOT / "docs" / "phase2_technical_memo.html"
MATRIX_PATH = PATHS.processed / "phase2_results_matrix.csv"
CASE_PATH = PATHS.processed / "phase2_gefs_case_matrix.csv"
COST_LOSS_PATH = PATHS.processed / "phase2_cost_loss.csv"
DECISION_PATH = PATHS.processed / "phase2_decision_summary.json"
TEST_PREDICTION_PATH = PATHS.processed / "phase2_test_predictions.parquet"
FORECAST_PROB_PATH = PATHS.processed / "phase2_forecast_county_probs.parquet"

AUTHOR = "Afahad Abdullah"
INSTITUTION = "NASA Goddard Space Flight Center"

FIGURES = [
    ("phase2_skill_summary.png", "Model skill, calibration, and cross-validation"),
    ("phase2_county_diagnostics.png", "County-level diagnostic maps"),
    ("phase2_gefs_case_studies.png", "GEFS case-study forecast distributions"),
    ("phase2_case_hazards.png", "ERA5 hazard fields on the two case-study days"),
    ("phase2_cost_loss_value.png", "Relative economic value by cost-loss ratio"),
]


def _frame(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _records(frame: pd.DataFrame) -> list[dict]:
    """JSON records with missing values represented as null, never NaN."""
    return json.loads(frame.to_json(orient="records")) if not frame.empty else []


def _number(value) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _metric(matrix: pd.DataFrame, key: str, split: str) -> float | None:
    if matrix.empty or split not in matrix or "key" not in matrix:
        return None
    match = matrix.loc[matrix.key.eq(key), split]
    return _number(match.iloc[0]) if len(match) else None


def _metric_text(value: float | None, digits: int = 3, signed: bool = False) -> str:
    if value is None:
        return "not available"
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def interruption_cost_rate(cfg: Config) -> float:
    """Configured interruption-cost proxy, not infrastructure-damage cost."""
    ice = cfg["ice"]
    return round(
        float(ice["mix_residential"]) * float(ice["residential_usd_per_cust_hr"])
        + float(ice["mix_commercial"]) * float(ice["commercial_usd_per_cust_hr"]), 2)


def economic_scenarios(cfg: Config, cases: pd.DataFrame) -> dict:
    """Counterfactual case impacts and forecast-triggered mitigation scenarios.

    These calculations make the causal assumption visible: an action that is
    taken in a forecast-triggered county reduces its realized interruption
    consequence by the configured delta. They are potential avoided impacts,
    not observed savings or direct physical-damage estimates.
    """
    rate = interruption_cost_rate(cfg)
    deltas = [float(delta) for delta in cfg["hazard_reduction_deltas"]]
    delta = deltas[len(deltas) // 2]
    result = {
        "usd_per_customer_hour": rate,
        "hazard_reduction_delta": delta,
        "annual_observed_customer_hours": None,
        "annual_interruption_cost_proxy_usd": None,
        "case_impacts": [],
        "triggered_actions": [],
    }
    if TEST_PREDICTION_PATH.exists():
        test = pd.read_parquet(TEST_PREDICTION_PATH)
        if "customer_hours" in test:
            annual = float(test.customer_hours.sum())
            result["annual_observed_customer_hours"] = annual
            result["annual_interruption_cost_proxy_usd"] = annual * rate

    if not cases.empty and "observed_customer_hours" in cases:
        for case, group in cases.groupby("case", sort=True):
            observed = _number(group.observed_customer_hours.iloc[0])
            if observed is None:
                continue
            result["case_impacts"].append({
                "case": case,
                "observed_customer_hours": observed,
                "interruption_cost_proxy_usd": observed * rate,
                **{f"avoided_customer_hours_{int(delta_value * 100)}pct": observed * delta_value
                   for delta_value in deltas},
                **{f"avoided_cost_proxy_usd_{int(delta_value * 100)}pct": observed * delta_value * rate
                   for delta_value in deltas},
            })

    if not (FORECAST_PROB_PATH.exists() and TEST_PREDICTION_PATH.exists()):
        return result
    forecast = pd.read_parquet(FORECAST_PROB_PATH)
    test = pd.read_parquet(TEST_PREDICTION_PATH, columns=["fips", "date", "customer_hours"])
    forecast["date"] = pd.to_datetime(forecast.date).dt.normalize()
    test["date"] = pd.to_datetime(test.date).dt.normalize()
    joined = forecast.merge(test, on=["fips", "date"], how="inner")
    if joined.empty:
        return result
    # C/L=0.10 is the middle of the configured operational decision grid.
    threshold = min(cfg["cost_loss_ratios"], key=lambda value: abs(float(value) - 0.10))
    for (case, lead), group in joined.groupby(["case", "lead_days"], sort=True):
        trigger = group.probability.ge(float(threshold))
        total = float(group.customer_hours.sum())
        exposed = float(group.loc[trigger, "customer_hours"].sum())
        result["triggered_actions"].append({
            "case": case,
            "lead_days": int(lead),
            "cost_loss_threshold": float(threshold),
            "triggered_counties": int(trigger.sum()),
            "reporting_counties": int(len(group)),
            "observed_customer_hours": total,
            "triggered_customer_hours": exposed,
            "covered_observed_loss_share": exposed / total if total > 0 else None,
            "potential_avoided_customer_hours": exposed * delta,
            "potential_avoided_cost_proxy_usd": exposed * delta * rate,
        })
    return result


def build_payload(cfg: Config) -> dict:
    """Return the small, serialisable results payload consumed by the memo."""
    matrix = _frame(MATRIX_PATH)
    has_test = _metric(matrix, "occurrence_brier", "test") is not None
    split = "test" if has_test else "validation"
    scope = "the full held-out 2023 test" if has_test else "Jul–Dec 2022 validation"
    brier = _metric(matrix, "occurrence_brier", split)
    ap = _metric(matrix, "occurrence_average_precision", split)
    skill = _metric(matrix, "occurrence_brier_skill_vs_climatology", split)
    abstract_parts = [
        f"This short memo reports the frozen Michigan county-day outage-risk model on {scope}.",
        f"Occurrence Brier score was {_metric_text(brier, 4)} and average precision was {_metric_text(ap)}.",
    ]
    if skill is not None:
        abstract_parts.append(
            f"Brier skill relative to county climatology was {_metric_text(skill, signed=True)}.")
    if has_test:
        abstract_parts.append(
            "All 2023 values were produced after the model and calibrator were frozen.")
    else:
        abstract_parts.append(
            "The 2023 test remains sealed; these are validation results, not final evidence.")

    cases = _frame(CASE_PATH)
    if not cases.empty:
        keep = [column for column in [
            "case", "lead_days", "median_customer_hours", "p10_customer_hours",
            "p90_customer_hours", "observed_customer_hours",
            "meteorological_variance_share", "input",
        ] if column in cases]
        cases = cases[keep].sort_values(["case", "lead_days"], ascending=[True, False])
    cost_loss = _frame(COST_LOSS_PATH)
    if not cost_loss.empty:
        keep = [column for column in [
            "lead_days", "source", "cost_loss_ratio", "relative_economic_value",
            "n_county_days",
        ] if column in cost_loss]
        cost_loss = cost_loss[keep]
    decision = json.loads(DECISION_PATH.read_text()) if DECISION_PATH.exists() else {}
    economics = economic_scenarios(cfg, cases)
    figures = [
        {"src": f"../figures/{name}", "caption": caption}
        for name, caption in FIGURES if (PATHS.figures / name).exists()
    ]
    return {
        "region": str(cfg["region_name"]),
        "split": {
            "train": f"{cfg['train_start']} to {cfg['train_end']}",
            "validation": f"{cfg['val_start']} to {cfg['val_end']}",
            "test": f"{cfg['test_start']} to {cfg['test_end']}",
            "has_test": has_test,
            "scope": scope,
        },
        "abstract": " ".join(abstract_parts),
        "primary_split": split,
        "metrics": _records(matrix),
        "cases": _records(cases),
        "cost_loss": _records(cost_loss),
        "decision": decision,
        "economics": economics,
        "figures": figures,
        "author": AUTHOR,
        "institution": INSTITUTION,
    }


def render_html(payload: dict) -> str:
    """Render an intentionally short, self-contained technical memo page."""
    status = ("Final 2023 test scored once" if payload["split"]["has_test"]
              else "2023 test sealed — validation-only")
    template = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Phase 2 technical memo — Michigan storm outage risk</title>
<style>
:root { --ink:#17232d; --muted:#5d6b76; --line:#dce4e8; --teal:#12626f; --blue:#31699f; --orange:#c05e17; --red:#a43a25; --paper:#fff; }
* { box-sizing:border-box; }
body { margin:0; background:#f3f6f7; color:var(--ink); font:16px/1.58 Arial, Helvetica, sans-serif; }
main { width:min(1040px, calc(100% - 32px)); margin:0 auto 48px; }
header { padding:58px 0 32px; border-bottom:4px solid var(--teal); }
h1 { font-size:clamp(2rem, 5vw, 3.65rem); letter-spacing:-.035em; line-height:1.04; margin:0 0 15px; max-width:880px; }
h2 { font-size:1.42rem; margin:0 0 12px; letter-spacing:-.015em; }
h3 { font-size:1rem; margin:0; }
p { margin:0 0 13px; }
.eyebrow { color:var(--teal); text-transform:uppercase; letter-spacing:.12em; font-size:.75rem; font-weight:700; }
.byline { color:var(--muted); font-size:.96rem; }
.status { display:inline-block; margin-top:10px; border:1px solid var(--line); padding:5px 9px; color:var(--muted); font-size:.82rem; }
section { margin-top:24px; padding:25px 28px; background:var(--paper); border:1px solid var(--line); opacity:0; transform:translateY(18px); transition:opacity .55s ease, transform .55s ease; }
section.in { opacity:1; transform:none; }
.split { display:grid; grid-template-columns:repeat(3,1fr); gap:1px; background:var(--line); border:1px solid var(--line); margin-top:17px; }
.split div { background:var(--paper); padding:13px; }
.split strong { display:block; font-size:.76rem; letter-spacing:.08em; text-transform:uppercase; color:var(--muted); }
.split span { display:block; margin-top:4px; font-size:.91rem; }
.impact-grid,.operation-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin-top:16px; }
.impact,.operation { border:1px solid var(--line); padding:14px; background:#fff; }
.impact strong { display:block; font-size:1.48rem; line-height:1.15; color:var(--teal); margin:4px 0; }
.impact span,.operation span { color:var(--muted); font-size:.83rem; }
.operation h3 { margin-bottom:5px; }
.chart { overflow-x:auto; }
svg { display:block; width:100%; min-width:620px; height:auto; }
.figure-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:18px; }
figure { margin:0; border:1px solid var(--line); background:#fff; }
figure img { width:100%; display:block; }
figcaption { padding:9px 11px; font-size:.84rem; color:var(--muted); }
table { border-collapse:collapse; width:100%; font-size:.87rem; }
th, td { padding:8px; border-bottom:1px solid var(--line); text-align:right; white-space:nowrap; }
th:first-child, td:first-child { text-align:left; white-space:normal; }
th { color:var(--muted); font-size:.76rem; text-transform:uppercase; letter-spacing:.04em; }
.scroll { overflow-x:auto; }
.note { color:var(--muted); font-size:.88rem; }
.conclusion { border-left:5px solid var(--teal); }
footer { color:var(--muted); font-size:.78rem; padding:22px 4px; }
@keyframes grow { from { transform:scaleY(.03); } to { transform:scaleY(1); } }
.bar { transform-box:fill-box; transform-origin:bottom; animation:grow .8s cubic-bezier(.2,.8,.2,1) both; }
@media (max-width:680px) { main { width:min(100% - 20px, 1040px); } header { padding-top:36px; } section { padding:20px 16px; } .split,.figure-grid,.impact-grid,.operation-grid { grid-template-columns:1fr; } }
</style>
</head>
<body>
<main>
  <header>
    <div class="eyebrow">Technical memo · frozen Phase 2 results</div>
    <h1>Storm-driven outage risk and forecast value in Michigan</h1>
    <div id="byline" class="byline"></div>
    <div id="status" class="status"></div>
  </header>
  <section class="in">
    <div class="eyebrow">Abstract</div>
    <p id="abstract"></p>
  </section>
  <section>
    <div class="eyebrow">Design</div>
    <h2>Temporal separation keeps the final test honest</h2>
    <div class="split" id="split"></div>
    <p class="note" style="margin-top:15px">The occurrence, conditional magnitude, and restoration models are composed probabilistically. GEFS drives two case studies after the model is frozen.</p>
  </section>
  <section>
    <div class="eyebrow">Results</div>
    <h2>Occurrence skill against required reference models</h2>
    <div class="chart" id="brier-chart" aria-label="Brier score comparison chart"></div>
    <p class="note">Lower Brier score is better. Bar heights animate on load; labels are the stored frozen-run values.</p>
  </section>
  <section id="case-section">
    <div class="eyebrow">Forecast case studies</div>
    <h2>GEFS distributions remain separate from the annual test</h2>
    <div id="case-table" class="scroll"></div>
    <p class="note">The case-study forecasts use GEFS ensemble members and conditional model draws. Observations appear only after the held-out 2023 test is opened.</p>
  </section>
  <section id="value-section">
    <div class="eyebrow">Decision value</div>
    <h2>Value varies with the cost-loss setting</h2>
    <div class="chart" id="value-chart" aria-label="Relative economic value chart"></div>
    <p id="decision-note" class="note"></p>
  </section>
  <section id="economic-section">
    <div class="eyebrow">2023 interruption-cost scenarios</div>
    <h2>Translate customer-hours into action-relevant impact</h2>
    <div id="impact-cards" class="impact-grid"></div>
    <div class="chart" id="action-chart" aria-label="Forecast-triggered potential avoided interruption cost chart"></div>
    <div id="action-table" class="scroll"></div>
    <p class="note">This is an interruption-cost proxy, not measured infrastructure damage. Potential avoided cost assumes a forecast-triggered action reduces realized interruption consequence by the configured effectiveness; it does not claim those savings were observed.</p>
  </section>
  <section>
    <div class="eyebrow">Industry-style risk operations</div>
    <h2>Use probabilities to target preparation before the event</h2>
    <div class="operation-grid">
      <div class="operation"><h3>1. Forecast risk</h3><span>GEFS members feed calibrated county-day outage probabilities and customer-hour distributions at each lead.</span></div>
      <div class="operation"><h3>2. Trigger actions</h3><span>Act where probability exceeds the selected cost-loss threshold; stage crews, materials, inspections, or mutual assistance according to local authority.</span></div>
      <div class="operation"><h3>3. Audit value</h3><span>Compare triggered coverage, realized customer-hours, action cost, and assumed effectiveness after each event to revise the decision rule.</span></div>
    </div>
  </section>
  <section id="figure-section">
    <div class="eyebrow">Figures</div>
    <h2>Publication diagnostics</h2>
    <div id="figures" class="figure-grid"></div>
  </section>
  <section class="conclusion">
    <div class="eyebrow">Conclusion</div>
    <h2>Use the model as calibrated decision support, not a county ranking</h2>
    <p id="conclusion"></p>
  </section>
  <footer>Generated from frozen Phase 2 artifacts. The full metric matrix and methods record are in <code>docs/phase2_results.md</code> and <code>docs/PHASE2_RUNBOOK.md</code>.</footer>
</main>
<script>
const data = __PAYLOAD__;
const finite = value => value !== null && value !== undefined && value !== "" && Number.isFinite(Number(value));
const fmt = (value, digits=3) => finite(value) ? Number(value).toFixed(digits) : "—";
const comma = value => finite(value) ? Math.round(Number(value)).toLocaleString() : "—";
document.querySelector("#status").textContent = "__STATUS__";
document.querySelector("#byline").textContent = `${data.author} · ${data.institution}`;
document.querySelector("#abstract").textContent = data.abstract;
document.querySelector("#split").innerHTML = [["Training", data.split.train], ["Calibration / validation", data.split.validation], ["Held-out test", data.split.test]].map(([name, range]) => `<div><strong>${name}</strong><span>${range}</span></div>`).join("");

const metric = key => data.metrics.find(row => row.key === key) || {};
const split = data.primary_split;
const brier = [
  ["Frozen model", metric("occurrence_brier")[split], "#12626f"],
  ["Logistic GLM", metric("reference_logistic_glm_brier")[split], "#91a0aa"],
  ["Gust > 20 m/s", metric("threshold_gust20_brier")[split], "#91a0aa"],
  ["County climatology", metric("occurrence_brier_ref_county_climatology")[split], "#91a0aa"],
].filter(row => finite(row[1]));
function brierChart() {
  const host = document.querySelector("#brier-chart");
  if (!brier.length) { host.textContent = "Metric matrix unavailable."; return; }
  const width=720, height=280, left=166, right=58, top=28, bottom=38, max=Math.max(...brier.map(d => +d[1])) * 1.15;
  const scale = value => left + (+value / max) * (width-left-right);
  const rows = brier.map((d, i) => { const y=top+i*52; return `<text x="${left-12}" y="${y+19}" text-anchor="end" fill="#5d6b76" font-size="13">${d[0]}</text><rect class="bar" x="${left}" y="${y}" width="${Math.max(1, scale(d[1])-left)}" height="28" fill="${d[2]}" style="animation-delay:${i*.12}s"/><text x="${Math.min(width-right,scale(d[1])+8)}" y="${y+19}" fill="#17232d" font-size="13">${fmt(d[1],4)}</text>`; }).join("");
  host.innerHTML = `<svg viewBox="0 0 ${width} ${height}" role="img"><text x="${left}" y="${height-8}" fill="#5d6b76" font-size="12">Occurrence Brier score — lower is better</text><line x1="${left}" x2="${width-right}" y1="${height-bottom}" y2="${height-bottom}" stroke="#dce4e8"/>${rows}</svg>`;
}
brierChart();

function caseTable() {
  const host = document.querySelector("#case-table");
  if (!data.cases.length) { document.querySelector("#case-section").style.display="none"; return; }
  const rows = data.cases.map(row => `<tr><td>${row.case}</td><td>day −${row.lead_days}</td><td>${comma(row.median_customer_hours)}</td><td>${comma(row.p10_customer_hours)}–${comma(row.p90_customer_hours)}</td><td>${comma(row.observed_customer_hours)}</td><td>${finite(row.meteorological_variance_share) ? (100*row.meteorological_variance_share).toFixed(0)+"%" : "—"}</td></tr>`).join("");
  host.innerHTML = `<table><thead><tr><th>Case</th><th>Lead</th><th>Median customer-hours</th><th>10–90% interval</th><th>Observed</th><th>Meteorological variance</th></tr></thead><tbody>${rows}</tbody></table>`;
}
caseTable();

function valueChart() {
  const host=document.querySelector("#value-chart");
  const values=data.cost_loss.filter(row => row.lead_days === 0 && finite(row.relative_economic_value)).sort((a,b) => a.cost_loss_ratio-b.cost_loss_ratio);
  if (!values.length) { document.querySelector("#value-section").style.display="none"; return; }
  const width=720, height=285, left=66, right=28, top=24, bottom=48;
  const xs=values.map(d=>+d.cost_loss_ratio), ys=values.map(d=>+d.relative_economic_value);
  const x=v=>left+(v-Math.min(...xs))/(Math.max(...xs)-Math.min(...xs)||1)*(width-left-right);
  const low=Math.min(0,...ys), high=Math.max(0,...ys), span=high-low||1;
  const y=v=>top+(high-v)/span*(height-top-bottom);
  const path=values.map((d,i)=>`${i?"L":"M"}${x(+d.cost_loss_ratio).toFixed(1)},${y(+d.relative_economic_value).toFixed(1)}`).join(" ");
  const dots=values.map(d=>`<circle cx="${x(+d.cost_loss_ratio)}" cy="${y(+d.relative_economic_value)}" r="4" fill="#12626f"><title>C/L ${d.cost_loss_ratio}: ${fmt(d.relative_economic_value)}</title></circle>`).join("");
  host.innerHTML=`<svg viewBox="0 0 ${width} ${height}" role="img"><line x1="${left}" x2="${width-right}" y1="${y(0)}" y2="${y(0)}" stroke="#9aa6ae" stroke-dasharray="4 4"/><path d="${path}" fill="none" stroke="#12626f" stroke-width="3" stroke-dasharray="1000" stroke-dashoffset="1000"><animate attributeName="stroke-dashoffset" from="1000" to="0" dur="1.2s" fill="freeze"/></path>${dots}<text x="${left}" y="${height-8}" fill="#5d6b76" font-size="12">Cost-loss ratio C/L</text><text x="10" y="18" fill="#5d6b76" font-size="12">Relative economic value</text></svg>`;
}
valueChart();
const decision=data.decision;
document.querySelector("#decision-note").textContent = decision.ice_values_are_region_yaml_placeholders ? "Dollar outputs retain the documented placeholder ICE assumptions and should not be used externally until replaced with region-specific ICE Calculator inputs." : "Decision values use the configured ICE inputs; see the runbook for the assumptions chain.";
const dollars = value => finite(value) ? "$" + Math.round(Number(value)).toLocaleString() : "—";
const economics = data.economics;
function economicImpact() {
  const section = document.querySelector("#economic-section");
  const host = document.querySelector("#impact-cards");
  if (!economics.case_impacts.length && !finite(economics.annual_observed_customer_hours)) { section.style.display="none"; return; }
  const shortLead = economics.triggered_actions.filter(row => row.lead_days === 1);
  const potential = shortLead.reduce((total, row) => total + (+row.potential_avoided_cost_proxy_usd || 0), 0);
  const cards = [
    [comma(economics.annual_observed_customer_hours), "observed 2023 customer-hours"],
    [dollars(economics.annual_interruption_cost_proxy_usd), "2023 interruption-cost proxy"],
    [dollars(potential), `potential avoided proxy at day-1, ${(100*economics.hazard_reduction_delta).toFixed(0)}% effectiveness`],
  ];
  host.innerHTML = cards.map(([value, label]) => `<div class="impact"><span>${label}</span><strong>${value}</strong></div>`).join("");
  const actions = economics.triggered_actions.filter(row => row.triggered_counties > 0 && finite(row.potential_avoided_cost_proxy_usd) && row.potential_avoided_cost_proxy_usd > 0);
  const table = document.querySelector("#action-table");
  if (!actions.length) { document.querySelector("#action-chart").style.display="none"; table.innerHTML="<p class='note'>Forecast-triggered coverage needs the stored county-level GEFS probabilities and test predictions.</p>"; return; }
  const delta = (100 * economics.hazard_reduction_delta).toFixed(0);
  const rows = actions.map(row => `<tr><td>${row.case}</td><td>day −${row.lead_days}</td><td>${row.triggered_counties}/${row.reporting_counties}</td><td>${finite(row.covered_observed_loss_share) ? (100*row.covered_observed_loss_share).toFixed(0)+"%" : "—"}</td><td>${comma(row.potential_avoided_customer_hours)}</td><td>${dollars(row.potential_avoided_cost_proxy_usd)}</td></tr>`).join("");
  table.innerHTML=`<table><thead><tr><th>Case</th><th>Lead</th><th>Counties triggered</th><th>Observed loss covered</th><th>Potential avoided customer-hours</th><th>Potential avoided cost proxy</th></tr></thead><tbody>${rows}</tbody></table>`;
  const hostChart = document.querySelector("#action-chart");
  const width=720, height=310, left=165, right=72, top=28, bottom=46, max=Math.max(...actions.map(row => +row.potential_avoided_cost_proxy_usd), 1) * 1.15;
  const plotted = actions.map((row, index) => { const y=top+index*31; const amount=+row.potential_avoided_cost_proxy_usd; const bar=Math.max(1, amount/max*(width-left-right)); return `<text x="${left-10}" y="${y+16}" text-anchor="end" fill="#5d6b76" font-size="11">${row.case.replace(" 2023 ", " ")} · −${row.lead_days}</text><rect class="bar" x="${left}" y="${y}" width="${bar}" height="20" fill="#31699f" style="animation-delay:${index*.08}s"/><text x="${Math.min(width-right, left+bar+6)}" y="${y+15}" fill="#17232d" font-size="11">${dollars(amount)}</text>`; }).join("");
  hostChart.innerHTML=`<svg viewBox="0 0 ${width} ${height}" role="img"><text x="${left}" y="${height-10}" fill="#5d6b76" font-size="12">Potential avoided interruption-cost proxy at ${delta}% effectiveness</text>${plotted}</svg>`;
}
economicImpact();
const figures=document.querySelector("#figures");
if (!data.figures.length) document.querySelector("#figure-section").style.display="none";
figures.innerHTML=data.figures.map(item => `<figure><img src="${item.src}" alt="${item.caption}"><figcaption>${item.caption}</figcaption></figure>`).join("");
const conclusion = data.split.has_test ? "The final test evaluates the already-frozen model over all of 2023. Interpret the county panels as spatial diagnostics, retain the predictive distributions for operational choices, and keep GEFS case studies distinct from the annual test." : "Review calibration, reference-model comparisons, cross-validation, and the two GEFS case studies before opening 2023. The validation figures are not substitutes for the planned annual held-out test.";
document.querySelector("#conclusion").textContent=conclusion;
const observer=new IntersectionObserver(entries => entries.forEach(entry => { if (entry.isIntersecting) entry.target.classList.add("in"); }), {threshold:.12});
document.querySelectorAll("section").forEach(section => observer.observe(section));
</script>
</body>
</html>'''
    return template.replace("__PAYLOAD__", json.dumps(payload, allow_nan=False)).replace(
        "__STATUS__", html.escape(status))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config" / "region.yaml"))
    parser.add_argument("--phase2", default=str(ROOT / "config" / "phase2.yaml"))
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    args = parser.parse_args()
    if not MATRIX_PATH.exists():
        raise SystemExit("results matrix missing — run `make phase2-report` first")
    cfg = load_config(args.config, args.phase2)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_html(build_payload(cfg)), encoding="utf-8")
    log.info("technical memo -> %s", output)


if __name__ == "__main__":
    main()
