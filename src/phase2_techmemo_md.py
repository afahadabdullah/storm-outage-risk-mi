#!/usr/bin/env python
"""Build the GitHub-readable Markdown version of the Phase 2 technical memo."""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.config import ROOT, Config, load_config
from src.common.logio import get_logger
from src.phase2_techmemo import build_payload

log = get_logger("phase2_techmemo_md")
OUTPUT_MD = ROOT / "docs" / "phase2_technical_memo.md"


def _value(row: dict, key: str):
    value = row.get(key)
    try:
        return float(value) if value is not None and math.isfinite(float(value)) else None
    except (TypeError, ValueError):
        return None


def _fmt(value, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _money(value) -> str:
    return "—" if value is None else f"${value:,.0f}"


def render_markdown(payload: dict) -> str:
    split = payload["split"]
    status = ("Final 2023 test scored once" if split["has_test"]
              else "2023 test sealed — validation-only")
    lines = [
        "# Storm-driven outage risk and forecast value in Michigan", "",
        f"**Abdullah Al Fahad · NASA Goddard Space Flight Center**", "",
        f"*{status}*", "",
        "## Abstract", "", payload["abstract"], "",
        "## Study design", "",
        "| Period | Dates | Role |", "|---|---|---|",
        f"| Training | {split['train']} | Fit the frozen model |",
        f"| Calibration / validation | {split['validation']} | Select and verify calibration |",
        f"| Held-out test | {split['test']} | One-time final evaluation |", "",
        "The occurrence, conditional magnitude, and restoration distributions are "
        "composed probabilistically. GEFS drives two separate 2023 case studies "
        "and does not replace the annual test.", "",
        "## Final results", "",
        "| Metric | Validation | 2023 test |", "|---|---:|---:|",
    ]
    for row in payload["metrics"]:
        validation = row.get("validation_formatted") or _fmt(_value(row, "validation"))
        test = row.get("test_formatted") or _fmt(_value(row, "test"))
        lines.append(f"| {row.get('metric', row.get('key', 'metric'))} | {validation} | {test} |")
    lines += [
        "", "## GEFS case studies", "",
        "The August wind event was contained by all four 10–90% intervals; the "
        "February ice-storm observation exceeded every lead-time interval. These "
        "are operational case diagnostics, not replacements for annual verification.", "",
        "| Case | Lead | Median customer-hours | 10–90% interval | Observed | Met. variance |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["cases"]:
        met = _value(row, "meteorological_variance_share")
        met_text = "—" if met is None else f"{100 * met:.1f}%"
        observed = _value(row, "observed_customer_hours")
        lines.append(
            f"| {row.get('case', '—')} | day −{row.get('lead_days', '—')} | "
            f"{_money(_value(row, 'median_customer_hours'))} | "
            f"{_money(_value(row, 'p10_customer_hours'))}–{_money(_value(row, 'p90_customer_hours'))} | "
            f"{_money(observed)} | {met_text} |")

    economics = payload["economics"]
    lines += ["", "## Economic impact and action scenarios", "",
              f"The configured interruption-cost proxy is **${economics['usd_per_customer_hour']:.2f} per customer-hour**. "
              "It is not a direct estimate of physical damage, repair expense, or realized savings. "
              f"Potential avoided impact assumes a forecast-triggered action reduces consequence by "
              f"{100 * economics['hazard_reduction_delta']:.0f}%.", ""]
    if economics["annual_observed_customer_hours"] is not None:
        lines.append(
            f"Across the 2023 test, observed interruption was **{economics['annual_observed_customer_hours']:,.0f} "
            f"customer-hours**, corresponding to **{_money(economics['annual_interruption_cost_proxy_usd'])}** "
            "under that configured proxy.")
        lines.append("")
    if economics["case_impacts"]:
        lines += ["| Case | Observed customer-hours | Interruption-cost proxy | Avoided proxy at 10% / 20% / 30% |",
                  "|---|---:|---:|---:|"]
        for row in economics["case_impacts"]:
            avoided = " / ".join(_money(row.get(f"avoided_cost_proxy_usd_{d}pct"))
                                 for d in (10, 20, 30))
            lines.append(f"| {row['case']} | {_money(row['observed_customer_hours'])} | "
                         f"{_money(row['interruption_cost_proxy_usd'])} | {avoided} |")
        lines.append("")
    actions = [row for row in economics["triggered_actions"]
               if row.get("triggered_counties", 0) > 0
               and (_value(row, "potential_avoided_cost_proxy_usd") or 0) > 0]
    if actions:
        lines += ["Forecast-triggered actions use the configured C/L threshold closest to 0.10.", "",
                  "| Case | Lead | Counties triggered | Observed loss covered | Potential avoided proxy |",
                  "|---|---:|---:|---:|---:|"]
        for row in actions:
            covered = _value(row, "covered_observed_loss_share")
            lines.append(f"| {row['case']} | day −{row['lead_days']} | "
                         f"{row['triggered_counties']}/{row['reporting_counties']} | "
                         f"{'—' if covered is None else f'{100 * covered:.0f}%'} | "
                         f"{_money(row['potential_avoided_cost_proxy_usd'])} |")
        lines.append("")
    lines += [
        "## Publication figures", "",
        "These static PNGs render directly in GitHub; the companion HTML keeps the "
        "animated charts for local/browser review.", "",
    ]
    for item in payload["figures"]:
        name = Path(item["src"]).name
        lines += [f"### {item['caption']}", "", f"![{item['caption']}](../figures/{name})", ""]
    lines += [
        "## Conclusion", "",
        ("The final test evaluates the frozen model over all of 2023. Use the "
         "calibrated probabilities and predictive distributions to prioritize "
         "preparation, and treat county maps and action savings as diagnostics "
         "until utility-specific costs and intervention outcomes are available."
         if split["has_test"] else
         "Review calibration, reference-model comparisons, and GEFS diagnostics "
         "before opening 2023; validation is not a substitute for the final test."),
        "",
        "Generated from the frozen Phase 2 artifacts by `make phase2-techmemo`.", "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config" / "region.yaml"))
    parser.add_argument("--phase2", default=str(ROOT / "config" / "phase2.yaml"))
    parser.add_argument("--output", default=str(OUTPUT_MD))
    args = parser.parse_args()
    cfg = load_config(args.config, args.phase2)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_markdown(build_payload(cfg)), encoding="utf-8")
    log.info("GitHub-readable technical memo -> %s", output)


if __name__ == "__main__":
    main()
