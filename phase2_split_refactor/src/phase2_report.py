#!/usr/bin/env python
"""Phase 2 results: the headline metric matrix and the publication figures.

One command turns whatever run artifacts exist into the things a write-up
actually needs:

  docs/phase2_results.md                     the headline table, in prose order
  data/processed/phase2_results_matrix.csv   the same numbers, machine-readable
  data/processed/phase2_county_skill.csv     per-county diagnostics behind the maps
  figures/fig1_skill_summary.png             reliability, PR, PIT, stratified skill
  figures/fig2_county_maps.png               where the model works, geographically
  figures/fig3_hazard_rasters.png            the ERA5 fields behind a case study
  figures/fig4_forecast_leads.png            statewide distribution by lead time
  figures/fig5_decision_value.png            cost-loss value and break-even cost

Every figure is optional: the script reports what it could and could not build
rather than failing because the test year is still sealed or the forecast stage
has not been run. That matters because the natural time to look at figures is
BEFORE the final test, when half of these inputs do not exist yet.

Nothing here refits, rescores or reads a held-out outcome that has not already
been opened by `phase2_train --evaluate-test`. It is a reporting layer over
artifacts that already exist.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.common import plotstyle as ps
from src.common.config import PATHS, ROOT, Config, load_config
from src.common.logio import get_logger

log = get_logger("phase2_report")
ps.apply_style()

RESULTS_MD = ROOT / "docs" / "phase2_results.md"
MATRIX_CSV = PATHS.processed / "phase2_results_matrix.csv"
COUNTY_CSV = PATHS.processed / "phase2_county_skill.csv"

# Rows of the headline table, in the order a reader should meet them: what the
# model does, then what it beats, then the two conditional stages.
METRIC_ROWS = [
    ("occurrence_brier", "Occurrence Brier", "lower", "{:.4f}"),
    ("occurrence_brier_skill_vs_climatology",
     "  ...skill vs COUNTY climatology", "higher", "{:+.3f}"),
    ("occurrence_brier_skill_vs_global_climatology",
     "  ...skill vs state-wide climatology", "higher", "{:+.3f}"),
    ("occurrence_average_precision", "Average precision", "higher", "{:.3f}"),
    ("occurrence_roc_auc", "ROC AUC", "higher", "{:.3f}"),
    ("occurrence_log_loss", "Log loss", "lower", "{:.4f}"),
    ("reference_logistic_glm_brier", "Reference: logistic GLM Brier", "lower", "{:.4f}"),
    ("threshold_gust20_brier", "Reference: gust > 20 m/s rule Brier", "lower", "{:.4f}"),
    ("occurrence_brier_ref_county_climatology",
     "Reference: county climatology Brier", "lower", "{:.4f}"),
    ("magnitude_crps", "Magnitude CRPS (customer-hours)", "lower", "{:,.0f}"),
    ("magnitude_crps_skill_vs_climatology",
     "  ...skill vs climatology", "higher", "{:+.3f}"),
    ("magnitude_log_score", "Magnitude log score", "lower", "{:.3f}"),
    ("magnitude_median_rmse", "Magnitude median RMSE", "lower", "{:,.0f}"),
    ("magnitude_spread_skill_ratio", "Magnitude spread-skill ratio", "near 1", "{:.2f}"),
    ("duration_concordance", "Duration concordance index", "higher", "{:.3f}"),
    ("duration_median_mae", "Duration median MAE (h)", "lower", "{:.2f}"),
    ("duration_persistence_mae",
     "Reference: county persistence MAE (h)", "lower", "{:.2f}"),
]


# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #
def read_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def read_frame(path: Path) -> pd.DataFrame | None:
    return pd.read_parquet(path) if path.exists() else None


def load_artifacts(cfg: Config) -> dict:
    """Everything the report can use. Missing pieces are None, never an error."""
    art = {
        "validation_metrics": read_json(PATHS.processed / "phase2_validation_metrics.json"),
        "test_metrics": read_json(PATHS.processed / "phase2_test_metrics.json"),
        "composed_metrics": read_json(PATHS.processed / "phase2_composed_metrics.json"),
        "decision_summary": read_json(PATHS.processed / "phase2_decision_summary.json"),
        "predictions": read_frame(PATHS.processed / "phase2_predictions.parquet"),
        "test_predictions": read_frame(PATHS.processed / "phase2_test_predictions.parquet"),
        "cv": (pd.read_csv(PATHS.processed / "phase2_cv_metrics.csv")
               if (PATHS.processed / "phase2_cv_metrics.csv").exists() else None),
        "uncertainty": (pd.read_csv(PATHS.processed / "phase2_uncertainty_by_lead.csv")
                        if (PATHS.processed / "phase2_uncertainty_by_lead.csv").exists() else None),
        "cost_loss": (pd.read_csv(PATHS.processed / "phase2_cost_loss.csv")
                      if (PATHS.processed / "phase2_cost_loss.csv").exists() else None),
        "decision_value": (pd.read_csv(PATHS.processed / "phase2_decision_value.csv",
                                       dtype={"fips": str})
                           if (PATHS.processed / "phase2_decision_value.csv").exists() else None),
        "realizations": (np.load(PATHS.processed / "phase2_forecast_realizations.npz")
                         if (PATHS.processed / "phase2_forecast_realizations.npz").exists() else None),
        "coverage": read_json(PATHS.processed / "phase2_coverage_exclusions.json"),
    }
    have = [k for k, v in art.items() if v is not None]
    missing = [k for k, v in art.items() if v is None]
    log.info("artifacts present: %s", ", ".join(have) or "none")
    if missing:
        log.warning("artifacts absent (their figures will be skipped): %s",
                    ", ".join(missing))
    return art


def split_label(cfg: Config) -> dict[str, str]:
    return {
        "train": f"{cfg['train_start']} to {cfg['train_end']}",
        "validation": f"{cfg['val_start']} to {cfg['val_end']}",
        "test": f"{cfg['test_start']} to {cfg['test_end']}",
    }


# --------------------------------------------------------------------------- #
# the metric matrix
# --------------------------------------------------------------------------- #
def results_matrix(art: dict) -> pd.DataFrame:
    """Validation and (if opened) test, side by side, with the direction of good."""
    columns = {}
    if art["validation_metrics"]:
        columns["validation"] = art["validation_metrics"]
    if art["test_metrics"]:
        columns["test"] = art["test_metrics"]
    if not columns:
        raise SystemExit(
            "no metrics to report -- run `make phase2-train` first")

    rows = []
    for key, label, better, fmt in METRIC_ROWS:
        row = {"metric": label, "key": key, "better": better}
        for name, blob in columns.items():
            value = blob.get(key)
            row[name] = float(value) if isinstance(value, (int, float)) else np.nan
            row[f"{name}_formatted"] = (fmt.format(value)
                                        if isinstance(value, (int, float)) else "--")
        rows.append(row)
    return pd.DataFrame(rows)


def stratified_table(metrics: dict, key: str, index_name: str) -> pd.DataFrame:
    """Flatten one of the by_regime / by_rurality_tercile blocks."""
    block = (metrics or {}).get(key) or {}
    rows = []
    for name, values in block.items():
        row = {index_name: name}
        row.update({k: v for k, v in values.items()} if isinstance(values, dict)
                   else {"value": values})
        rows.append(row)
    return pd.DataFrame(rows)


def write_results_markdown(cfg: Config, art: dict, matrix: pd.DataFrame,
                           figures: dict[str, Path | None]) -> Path:
    splits = split_label(cfg)
    val = art["validation_metrics"] or {}
    test = art["test_metrics"]
    has_test = test is not None

    lines = [
        "# Phase 2 results",
        "",
        "Generated by `make phase2-report`. Every number here is read from a run",
        "artifact; this file computes no scores of its own.",
        "",
        "## Frozen design",
        "",
        "| Split | Window |",
        "|---|---|",
        f"| Train | {splits['train']} |",
        f"| Calibration / validation | {splits['validation']} |",
        f"| Held-out test | {splits['test']}"
        + (" — **opened**" if has_test else " — *sealed*") + " |",
        "",
    ]

    if art["coverage"]:
        cov = art["coverage"]
        lines += [
            f"Reporting cohort: **{cov.get('n_reporting', '?')} counties**, fixed by the "
            f"train and validation years. "
            f"{len(cov.get('excluded_unstable_counties') or [])} excluded for unstable "
            "annual EAGLE-I coverage"
            + (f"; {len(cov.get('test_year_coverage_gaps') or [])} have test-year "
               "coverage gaps that are reported, not acted on" if has_test else "")
            + ".",
            "",
        ]

    lines += ["## Headline metrics", "", "| Metric | Better |"
              + (" Validation | Test |" if has_test else " Validation |"),
              "|---|---|" + ("---:|---:|" if has_test else "---:|")]
    for _, row in matrix.iterrows():
        cells = f"| {row['metric']} | {row['better']} | {row['validation_formatted']} |"
        if has_test:
            cells += f" {row['test_formatted']} |"
        lines.append(cells)
    lines.append("")

    model_brier = val.get("occurrence_brier")
    refs = {
        "county climatology": val.get("occurrence_brier_ref_county_climatology"),
        "logistic GLM": val.get("reference_logistic_glm_brier"),
        "gust > 20 m/s rule": val.get("threshold_gust20_brier"),
    }
    beaten = [name for name, value in refs.items()
              if isinstance(value, (int, float)) and isinstance(model_brier, (int, float))
              and model_brier < value]
    lost = [name for name, value in refs.items()
            if isinstance(value, (int, float)) and isinstance(model_brier, (int, float))
            and model_brier >= value]
    lines += ["### Against the required baselines (spec 7.3)", ""]
    if beaten:
        lines.append(f"On validation Brier the model beats: **{', '.join(beaten)}**.")
    if lost:
        lines.append(
            f"It does **not** beat: **{', '.join(lost)}**. That is a finding and belongs "
            "in the write-up as one — a model that barely beats a threshold rule is "
            "informative about the problem, and reporting it makes every other claim "
            "more credible.")
    if not beaten and not lost:
        lines.append("_Baseline comparisons unavailable in this run._")
    lines.append("")

    if val.get("magnitude_model_actually_used"):
        used = val["magnitude_model_actually_used"]
        configured = cfg.get("magnitude_model", "ngboost")
        lines += [
            f"Magnitude model actually used: **{used}**"
            + ("" if used == configured
               else f" — note this differs from the configured `{configured}`, "
                    "so NGBoost failed to import or fit and the write-up must say so"),
            "",
        ]

    if val.get("calibration_note"):
        lines += ["### Calibration", "",
                  val["calibration_note"] + ".",
                  ""]
        if "occurrence_brier_insample_calibration" in val:
            lines.append(
                f"In-sample calibrated Brier is "
                f"{val['occurrence_brier_insample_calibration']:.4f} against the "
                f"out-of-fold {val['occurrence_brier']:.4f}; the difference is the "
                "optimism that scoring a calibrator on its own fitting rows would have "
                "bought.")
            lines.append("")

    if art["cv"] is not None and len(art["cv"]):
        cv = art["cv"]
        lines += ["## Cross-validation (spec 7.1)", "",
                  "| Scheme | Folds | Mean Brier | Mean AUC |", "|---|---:|---:|---:|"]
        for scheme, group in cv.groupby("scheme"):
            lines.append(f"| {scheme} | {len(group)} | {group.brier.mean():.4f} "
                         f"| {group.roc_auc.mean():.3f} |")
        lines.append("")
        schemes = set(cv.scheme)
        expected = {"storm_blocked", "leave_one_county_out", "forward_year"}
        if not expected <= schemes:
            lines.append(
                f"**Missing CV scheme(s): {sorted(expected - schemes)}.** Spec 7.1 asks "
                "for all three; a missing one is a finding, not a formatting issue.")
            lines.append("")
        if {"storm_blocked", "leave_one_county_out"} <= schemes:
            sb = cv[cv.scheme.eq("storm_blocked")].brier.mean()
            lo = cv[cv.scheme.eq("leave_one_county_out")].brier.mean()
            lines.append(
                f"Leave-one-county-out Brier is {lo:.4f} against storm-blocked "
                f"{sb:.4f}. "
                + ("Spatial generalisation is the weaker of the two, which is the "
                   "signature of a model leaning on county-specific baselines — say so."
                   if lo > sb * 1.10 else
                   "The two are close, so there is no strong evidence the model is "
                   "memorising county-specific baselines."))
            lines.append("")

    for key, name, heading in (("by_regime", "regime", "By hazard regime (spec 7.5)"),
                               ("by_rurality_tercile", "tercile",
                                "By county rurality tercile (spec 7.5)")):
        table = stratified_table(val, key, name)
        if table.empty:
            continue
        lines += [f"## {heading}", "", "| " + " | ".join(table.columns) + " |",
                  "|" + "---|" * len(table.columns)]
        for _, row in table.iterrows():
            lines.append("| " + " | ".join(
                f"{v:.4f}" if isinstance(v, float) else str(v) for v in row) + " |")
        lines.append("")

    if art["composed_metrics"]:
        comp = art["composed_metrics"]
        lines += ["## Composed forecast (spec 6.4, 7.2)", "",
                  f"Composed CRPS {comp['composed_crps_customer_minutes']:,.0f} "
                  f"customer-minutes, skill {comp['composed_crps_skill_vs_climatology']:+.3f} "
                  "against climatology.",
                  "",
                  f"Rank histogram: `{comp['composed_rank_histogram']}` — "
                  + comp.get("composed_rank_histogram_note", ""), ""]

    if art["uncertainty"] is not None and len(art["uncertainty"]):
        unc = art["uncertainty"]
        lines += ["## Uncertainty decomposition by lead (spec 8.3)", "",
                  "| Case | Lead (days) | Realizations | Meteorological share |",
                  "|---|---:|---:|---:|"]
        for _, row in unc.sort_values(["case", "lead_days"], ascending=[True, False]).iterrows():
            lines.append(f"| {row['case']} | {int(row['lead_days'])} | "
                         f"{int(row['n_realizations']):,} | "
                         f"{row['meteorological_share']:.1%} |")
        lines.append("")
        lines.append(
            "Meteorological uncertainty should dominate at day-5 and shrink toward "
            "day-1, where the damage model becomes the binding constraint. The "
            "crossover is the operationally useful number: it says whether better "
            "forecasts or a better damage model would help more at each horizon.")
        lines.append("")

    if art["decision_summary"]:
        dec = art["decision_summary"]
        lines += ["## Decision value (spec 9)", ""]
        if dec.get("ice_values_are_region_yaml_placeholders"):
            lines.append(
                "> **The ICE dollar values are still the `region.yaml` placeholders.** "
                "Every figure in this section is therefore arithmetic, not a result. "
                "Replace them with ICE Calculator output for this region and customer "
                "mix before any dollar amount leaves the repository.")
            lines.append("")
        lines += [
            f"- Forecast has positive economic value for cost-loss ratios in "
            f"**{dec.get('cost_loss_positive_value_range', 'n/a')}**.",
            f"- Break-even inspection cost at delta = {dec.get('headline_delta', 0):.0%}: "
            f"up to **${dec.get('max_cost_per_asset_usd', float('nan')):,.0f} per asset**.",
            f"- The top decile of counties by risk-spend efficiency captures "
            f"**{dec.get('top_decile_share_of_avoidable_customer_hours', float('nan')):.0%}** "
            "of avoidable customer-hours.",
            f"- EVPI ceiling: **${dec.get('evpi_usd', float('nan')):,.0f}**.",
            "",
            "The weakest assumption in the chain is the asset-count proxy "
            "(customers / customers_per_asset). With real network data it would be a "
            "pole and span count. Name it first when asked, not last.",
            "",
        ]

    built = {k: v for k, v in figures.items() if v is not None}
    skipped = [k for k, v in figures.items() if v is None]
    lines += ["## Figures", ""]
    for name, path in built.items():
        lines.append(f"- `{path.relative_to(ROOT)}` — {name}")
    if skipped:
        lines += ["", f"Not built (inputs absent): {', '.join(skipped)}."]
    lines.append("")

    if not has_test:
        lines += ["---", "",
                  "**The test year is still sealed.** Everything above is validation "
                  "evidence. Freeze every model and hyperparameter decision in Git "
                  "before running `make phase2-test`, and remember that spec 7.4 gives "
                  "you exactly one look.", ""]

    RESULTS_MD.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_MD.write_text("\n".join(lines))
    return RESULTS_MD


# --------------------------------------------------------------------------- #
# county diagnostics behind the maps
# --------------------------------------------------------------------------- #
def county_skill(prediction: pd.DataFrame, quantiles: list[float]) -> pd.DataFrame:
    """Per-county occurrence and magnitude diagnostics."""
    from sklearn.metrics import brier_score_loss

    from src.phase2_train import crps_from_quantiles

    qcols = [f"magnitude_q{round(q * 100):02d}" for q in quantiles]
    qcols = [c for c in qcols if c in prediction.columns]
    rows = []
    for fips, group in prediction.groupby("fips"):
        y = group.event.astype(int).to_numpy()
        p = group.probability.to_numpy()
        clim = group.reference_climatology_county.to_numpy()
        brier = float(brier_score_loss(y, p))
        reference = float(np.mean((y - clim) ** 2))
        events = group[group.event.eq(1)]
        crps = np.nan
        if len(events) >= 3 and qcols:
            crps = crps_from_quantiles(events.customer_hours.to_numpy(),
                                       events[qcols].to_numpy(),
                                       [float(q) for q in quantiles][:len(qcols)])
        rows.append({
            "fips": str(fips).zfill(5),
            "n_county_days": int(len(group)),
            "n_events": int(y.sum()),
            "observed_event_rate": float(np.mean(y)),
            "mean_probability": float(np.mean(p)),
            "probability_bias": float(np.mean(p) - np.mean(y)),
            "brier": brier,
            "brier_skill": 1 - brier / reference if reference > 0 else np.nan,
            "magnitude_crps": crps,
            "observed_customer_hours": float(group.customer_hours.sum()),
        })
    return pd.DataFrame(rows).sort_values("fips").reset_index(drop=True)


def load_geometry(cfg: Config):
    import geopandas as gpd

    path = PATHS.raw / f"tiger_counties_{cfg['sources']['tiger_year']}.parquet"
    if not path.exists():
        return None
    gdf = gpd.read_parquet(path)
    key = next((c for c in ("GEOID", "fips", "FIPS") if c in gdf.columns), None)
    if key is None:
        return None
    gdf["fips"] = gdf[key].astype(str).str.zfill(5)
    return gdf.to_crs(cfg["crs_analysis"])


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def fig_skill_summary(art: dict, cfg: Config, label: str) -> Path | None:
    """Reliability, discrimination, calibration of the magnitude density, and
    the two stratifications the spec asks for."""
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import precision_recall_curve

    pred = art["test_predictions"] if art["test_metrics"] else art["predictions"]
    metrics = art["test_metrics"] or art["validation_metrics"]
    if pred is None or metrics is None:
        return None
    if art["test_metrics"] is None and art["predictions"] is not None:
        from src.phase2_train import masks
        pred = art["predictions"][masks(art["predictions"], cfg)["val"]]

    y = pred.event.astype(int).to_numpy()
    p = pred.probability.to_numpy()

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.2))

    # (a) reliability -- the plot that decides whether a probability is a probability
    ax = axes[0, 0]
    frac, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
    ax.plot([0, 1], [0, 1], color=ps.FAINT, lw=1, ls="--", zorder=1)
    ax.plot(mean_pred, frac, "o-", color=ps.ACCENT, zorder=3,
            markeredgecolor="white", markeredgewidth=1.2)
    ax.set_xlabel("forecast probability")
    ax.set_ylabel("observed frequency")
    lim = max(mean_pred.max(), frac.max()) * 1.08
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ps.panel_label(ax, "a", "Reliability")

    # (b) precision-recall, with the base rate as the no-skill floor
    ax = axes[0, 1]
    precision, recall, _ = precision_recall_curve(y, p)
    ax.plot(recall, precision, color=ps.ACCENT, zorder=3)
    base = float(y.mean())
    ax.axhline(base, color=ps.FAINT, lw=1, ls="--", zorder=1)
    ax.annotate(f"no skill ({base:.3f})", (0.02, base), xytext=(0.02, base + 0.03),
                fontsize=8, color=ps.MUTED)
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ps.panel_label(ax, "b", f"Precision-recall (AP {metrics.get('occurrence_average_precision', float('nan')):.3f})")

    # (c) PIT -- flat means the magnitude density is calibrated
    ax = axes[0, 2]
    pit = metrics.get("magnitude_pit_histogram_10bin")
    if pit:
        counts = np.asarray(pit, dtype=float)
        centres = np.linspace(0.05, 0.95, len(counts))
        ax.bar(centres, counts / counts.sum(), width=0.085, color=ps.ACCENT,
               edgecolor="white", linewidth=1.2, zorder=3)
        ax.axhline(1 / len(counts), color=ps.FAINT, lw=1, ls="--", zorder=1)
        ax.set_xlabel("probability integral transform")
        ax.set_ylabel("share of events")
        ax.set_xlim(0, 1)
    else:
        ax.text(0.5, 0.5, "no PIT histogram", ha="center", va="center", color=ps.MUTED)
        ax.set_axis_off()
    ps.panel_label(ax, "c", "Magnitude PIT (flat = calibrated)")

    # (d) skill against every required baseline, as a Brier comparison
    ax = axes[1, 0]
    entries = [
        ("Model", metrics.get("occurrence_brier"), ps.ACCENT),
        ("Logistic GLM", metrics.get("reference_logistic_glm_brier"), ps.REFERENCE),
        ("Gust > 20 m/s", metrics.get("threshold_gust20_brier"), ps.REFERENCE),
        ("County climatology",
         metrics.get("occurrence_brier_ref_county_climatology"), ps.REFERENCE),
    ]
    entries = [(n, v, c) for n, v, c in entries if isinstance(v, (int, float))]
    names = [e[0] for e in entries][::-1]
    values = [e[1] for e in entries][::-1]
    colors = [e[2] for e in entries][::-1]
    bars = ax.barh(names, values, color=colors, height=0.62, zorder=3)
    for bar, value in zip(bars, values):
        ax.text(bar.get_width() * 1.02, bar.get_y() + bar.get_height() / 2,
                f"{value:.4f}", va="center", fontsize=8, color=ps.MUTED)
    ax.set_xlabel("Brier score (lower is better)")
    ax.set_xlim(0, max(values) * 1.25)
    ax.set_ylim(-0.6, max(len(values) - 0.4, 2.6))
    ax.grid(axis="y", visible=False)
    ps.panel_label(ax, "d", "Against the required baselines")

    # (e) by hazard regime -- categorical hues, fixed order, never cycled
    ax = axes[1, 1]
    regimes = stratified_table(metrics, "by_regime", "regime")
    if not regimes.empty and "brier" in regimes:
        regimes = regimes.sort_values("n", ascending=True)
        colors = [ps.REGIME_COLORS.get(r, ps.REFERENCE) for r in regimes.regime]
        bars = ax.barh(regimes.regime, regimes.brier, color=colors, height=0.62, zorder=3)
        for bar, n in zip(bars, regimes.n):
            ax.text(bar.get_width() * 1.02, bar.get_y() + bar.get_height() / 2,
                    f"n={int(n):,}", va="center", fontsize=7.5, color=ps.MUTED)
        ax.set_xlabel("Brier score")
        ax.set_xlim(0, float(regimes.brier.max()) * 1.3)
        # A single surviving regime would otherwise be drawn as one bar filling
        # the panel, which reads as a rendering fault rather than as "n=1".
        ax.set_ylim(-0.6, max(len(regimes) - 0.4, 2.6))
        ax.grid(axis="y", visible=False)
    else:
        ax.text(0.5, 0.5, "no regime breakdown", ha="center", va="center", color=ps.MUTED)
        ax.set_axis_off()
    ps.panel_label(ax, "e", "By hazard regime")

    # (f) cross-validation schemes side by side
    ax = axes[1, 2]
    cv = art["cv"]
    if cv is not None and len(cv):
        order = ["storm_blocked", "leave_one_county_out", "forward_year"]
        present = [s for s in order if s in set(cv.scheme)]
        data = [cv[cv.scheme.eq(s)].brier.to_numpy() for s in present]
        parts = ax.boxplot(data, vert=True, patch_artist=True, widths=0.55,
                           medianprops={"color": ps.INK, "linewidth": 1.6},
                           flierprops={"marker": "o", "markersize": 3,
                                       "markerfacecolor": ps.MUTED,
                                       "markeredgecolor": "none"})
        for patch in parts["boxes"]:
            patch.set_facecolor(ps.SEQ_HEX[2])
            patch.set_edgecolor(ps.ACCENT)
            patch.set_linewidth(1.2)
        for whisker in parts["whiskers"] + parts["caps"]:
            whisker.set_color(ps.ACCENT)
        ax.set_xticks(range(1, len(present) + 1))
        ax.set_xticklabels([s.replace("_", "\n") for s in present], fontsize=8)
        ax.set_ylabel("fold Brier score")
    else:
        ax.text(0.5, 0.5, "no CV metrics", ha="center", va="center", color=ps.MUTED)
        ax.set_axis_off()
    ps.panel_label(ax, "f", "Cross-validation by scheme")

    fig.suptitle(f"Phase 2 occurrence and magnitude skill — {label}",
                 fontsize=13, fontweight="semibold", x=0.005, ha="left", y=0.995)
    fig.tight_layout(rect=(0, 0.015, 1, 0.975))
    return ps.save(fig, PATHS.figures / "fig1_skill_summary.png",
                   f"{label}; n = {len(pred):,} county-days, {int(y.sum()):,} events. "
                   "Probabilities are out-of-fold calibrated on the validation year.")


def fig_county_maps(art: dict, cfg: Config, counties: pd.DataFrame,
                    label: str) -> Path | None:
    """Where the model works, geographically. Diverging fields are pinned at zero."""
    import matplotlib.pyplot as plt

    geometry = load_geometry(cfg)
    if geometry is None:
        log.warning("county geometry missing; skipping maps")
        return None
    mapped = geometry.merge(counties, on="fips", how="left")

    panels = [
        ("observed_event_rate", "Observed event rate", "seq", "share of days"),
        ("mean_probability", "Mean forecast probability", "seq", "probability"),
        ("probability_bias", "Forecast bias (mean p − observed)", "div", "probability"),
        ("brier_skill", "Brier skill vs county climatology", "div", "skill"),
        ("n_events", "Events observed", "seq", "count"),
        ("magnitude_crps", "Magnitude CRPS", "seq", "customer-hours"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 9.4))
    for ax, (field, title, kind, unit), letter in zip(axes.flat, panels, "abcdef"):
        if field not in mapped:
            ax.set_axis_off()
            continue
        values = pd.to_numeric(mapped[field], errors="coerce")
        norm = ps.diverging_norm(values) if kind == "div" else None
        cmap = ps.diverging() if kind == "div" else ps.sequential()
        mapped.assign(_v=values).plot(
            column="_v", ax=ax, cmap=cmap, norm=norm,
            linewidth=0.3, edgecolor="white",
            legend=True,
            legend_kwds={"shrink": 0.62, "label": unit, "pad": 0.01},
            missing_kwds={"color": ps.MISSING, "edgecolor": "white",
                          "linewidth": 0.3, "label": "no data"})
        ps.map_axes(ax)
        ps.panel_label(ax, letter, title)

    fig.suptitle(f"Phase 2 county diagnostics — {label}",
                 fontsize=13, fontweight="semibold", x=0.005, ha="left", y=0.995)
    fig.tight_layout(rect=(0, 0.02, 1, 0.975))
    n_days = int(counties.n_county_days.median()) if len(counties) else 0
    return ps.save(
        fig, PATHS.figures / "fig2_county_maps.png",
        f"{label}; {len(counties)} reporting counties, median {n_days:,} evaluated "
        "days each. Diverging panels are centred on zero; grey counties have no data. "
        "Per-county event counts are small, so read county differences as indicative.")


def fig_hazard_rasters(art: dict, cfg: Config) -> Path | None:
    """The ERA5 fields themselves, on the day of a case study, with counties over.

    Choropleths show what the model concluded; this shows what it was given. A
    reader who wants to believe a county-level result needs to see that the
    underlying grid resolved the storm at all -- which for a convective event on
    a 0.25 degree reanalysis is exactly the thing in question.
    """
    import matplotlib.pyplot as plt
    import xarray as xr

    geometry = load_geometry(cfg)
    cases = list(cfg.get("case_studies") or [])
    if geometry is None or not cases:
        return None

    found = []
    for case in cases:
        target = pd.Timestamp(case["date"])
        path = PATHS.raw / "era5_monthly" / f"era5_{target.year}{target.month:02d}.nc"
        if path.exists():
            found.append((case, target, path))
    if not found:
        log.warning("no ERA5 month covering a case study; skipping rasters")
        return None

    fields = [("i10fg", "10 m wind gust", "m s$^{-1}$", "max"),
              ("tp", "Total precipitation", "mm", "sum"),
              ("t2m", "2 m temperature", "$^\\circ$C", "min")]
    fig, axes = plt.subplots(len(found), 3, figsize=(13.5, 4.4 * len(found)),
                             squeeze=False)
    bounds = geometry.to_crs("EPSG:4326").total_bounds

    for row, (case, target, path) in enumerate(found):
        with xr.open_dataset(path) as ds:
            ren = {o: n for o, n in {"valid_time": "time", "latitude": "lat",
                                     "longitude": "lon"}.items() if o in ds}
            ds = ds.rename(ren).sortby("time")
            day = ds.sel(time=str(target.date()))
            for col, (var, title, unit, how) in enumerate(fields):
                ax = axes[row][col]
                if var not in day:
                    ax.set_axis_off()
                    continue
                arr = day[var]
                field = (arr.max("time") if how == "max"
                         else arr.sum("time") if how == "sum" else arr.min("time"))
                values = field.values
                if var == "tp":
                    values = values * 1000.0
                if var == "t2m":
                    values = values - 273.15
                lat, lon = field.lat.values, field.lon.values
                # pcolormesh on cell EDGES, so a 0.25 deg cell is drawn as the
                # 0.25 deg box it actually is rather than smoothed to a point.
                half = 0.125
                lon_e = np.append(lon - half, lon[-1] + half)
                lat_e = np.append(lat - half, lat[-1] + half)
                mesh = ax.pcolormesh(lon_e, lat_e, values, cmap=ps.sequential(),
                                     shading="auto", zorder=1)
                geometry.to_crs("EPSG:4326").boundary.plot(
                    ax=ax, color=ps.INK, linewidth=0.45, alpha=0.55, zorder=2)
                bar = fig.colorbar(mesh, ax=ax, shrink=0.78, pad=0.02)
                bar.set_label(unit, fontsize=8)
                bar.ax.tick_params(labelsize=7.5)
                ax.set_xlim(bounds[0], bounds[2])
                ax.set_ylim(bounds[1], bounds[3])
                ps.map_axes(ax)
                ax.set_title(f"{title} — daily {how}", loc="left", fontsize=9.5)
    # Row headings are placed once per row in FIGURE coordinates, after the
    # layout is settled. Putting them at transAxes y>1 on the first panel put
    # them on top of that panel's own title.
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    for row, (case, target, _) in enumerate(found):
        top = axes[row][0].get_position().y1
        fig.text(0.005, min(top + 0.035, 0.985),
                 f"{case['name']}  ·  {target.date()}", fontsize=11,
                 fontweight="semibold", color=ps.INK, va="bottom")

    fig.suptitle("ERA5 hazard fields on the case-study days, with county boundaries",
                 fontsize=13, fontweight="semibold", x=0.005, ha="left", y=0.999)
    return ps.save(
        fig, PATHS.figures / "fig3_hazard_rasters.png",
        "ERA5 single-levels, native 0.25 deg cells drawn on their true footprint "
        "(no interpolation). Gust and temperature are cell maxima/minima over the "
        "day; precipitation is the daily accumulation. County aggregation uses the "
        "max over intersecting cells for gust and the area-weighted mean for "
        "precipitation — a county-mean gust would erase the damage signal.")


def observed_statewide_by_case(cfg: Config) -> dict[str, float]:
    """Observed statewide customer-hours on each case-study day.

    Test-year discipline applies: the case studies fall inside the held-out
    year, so the truth line only appears once that year has been formally
    opened. Before then the money plot is drawn without it rather than quietly
    peeking at an outcome the model has not earned the right to see.
    """
    marker = PATHS.models / "TEST_YEAR_OPENED.txt"
    source = PATHS.processed / "phase2_merged_test.parquet"
    if not marker.exists() or not source.exists():
        return {}
    frame = pd.read_parquet(source, columns=["date", "customer_hours"])
    frame["date"] = pd.to_datetime(frame.date).dt.normalize()
    out = {}
    for case in cfg.get("case_studies") or []:
        day = pd.Timestamp(case["date"]).normalize()
        block = frame[frame.date.eq(day)]
        if len(block):
            out[case["name"]] = float(block.customer_hours.sum())
    return out


def fig_forecast_leads(art: dict, cfg: Config) -> Path | None:
    """Statewide predicted distribution sharpening as the lead shortens."""
    import matplotlib.pyplot as plt

    real = art["realizations"]
    unc = art["uncertainty"]
    if real is None:
        return None
    keys = list(real.files)
    cases = sorted({k.split("|")[0] for k in keys})
    leads = sorted({int(k.split("|")[1]) for k in keys}, reverse=True)
    if not cases or not leads:
        return None

    n_rows = len(cases) + (1 if unc is not None and len(unc) else 0)
    fig, axes = plt.subplots(n_rows, len(leads), figsize=(3.35 * len(leads), 3.5 * n_rows),
                             squeeze=False)

    truth = observed_statewide_by_case(cfg)
    for row, case in enumerate(cases):
        series = {lead: real[f"{case}|{lead}"] for lead in leads
                  if f"{case}|{lead}" in keys}
        if not series:
            continue
        observed = truth.get(case)
        pooled = np.concatenate(list(series.values()))
        hi = float(np.percentile(pooled, 99.5))
        bins = np.linspace(0, max(hi, 1.0), 46)
        for col, lead in enumerate(leads):
            ax = axes[row][col]
            values = series.get(lead)
            if values is None:
                ax.set_axis_off()
                continue
            ax.hist(values, bins=bins, color=ps.SEQ_HEX[3 + min(col, 3)],
                    edgecolor="white", linewidth=0.4, zorder=3)
            median = float(np.median(values))
            ax.axvline(median, color=ps.ACCENT, lw=1.6, zorder=4)
            ax.set_xlabel("statewide customer-hours" if row == len(cases) - 1 else "")
            if observed is not None:
                ax.axvline(observed, color=ps.BAD, lw=2.0, zorder=5)
                if col == 0:
                    ax.annotate("observed", xy=(observed, ax.get_ylim()[1] * 0.92),
                                xytext=(4, 0), textcoords="offset points",
                                fontsize=8, color=ps.BAD, fontweight="semibold")
            if col == 0:
                ax.set_ylabel("realizations")
            ax.set_title(f"day-{lead}  (n={values.size:,})", loc="left", fontsize=9.5)
            ax.tick_params(labelsize=7.5)

    if unc is not None and len(unc):
        row = len(cases)
        for col, lead in enumerate(leads):
            ax = axes[row][col]
            block = unc[unc.lead_days.eq(lead)]
            if block.empty:
                ax.set_axis_off()
                continue
            met = float(block.meteorological_share.mean())
            ax.bar(["meteorological", "model"], [met, 1 - met],
                   color=[ps.REGIME_COLORS["synoptic_wind"], ps.REFERENCE],
                   width=0.62, edgecolor="white", linewidth=1.2, zorder=3)
            ax.set_ylim(0, 1)
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.tick_params(labelsize=7.5)
            for x, v in enumerate([met, 1 - met]):
                ax.text(x, v + 0.02, f"{v:.0%}", ha="center", fontsize=8, color=ps.MUTED)
            if col == 0:
                ax.set_ylabel("share of predictive variance")
            ax.set_title(f"day-{lead}", loc="left", fontsize=9.5)
            ax.grid(axis="x", visible=False)

    fig.suptitle("Forecast distribution of statewide customers-out by lead time",
                 fontsize=13, fontweight="semibold", x=0.005, ha="left", y=0.999)
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    headings = list(cases) + (["Uncertainty decomposition"]
                              if unc is not None and len(unc) else [])
    for row, heading in enumerate(headings):
        top = axes[row][0].get_position().y1
        fig.text(0.005, min(top + 0.028, 0.985), heading, fontsize=11,
                 fontweight="semibold", color=ps.INK, va="bottom")
    synthetic = (art.get("forecast_summary") or {}).get("synthetic_gefs")
    note = ("GEFS members quantile-mapped onto the ERA5 climatology per grid cell "
            "and season; the transfer function is fitted once across all leads and "
            "applied unchanged, so lead-dependent spread survives. Vertical line is "
            "the ensemble median.")
    if synthetic:
        note = "SYNTHETIC GEFS STAND-INS — NO FORECAST SKILL IS IMPLIED. " + note
    return ps.save(fig, PATHS.figures / "fig4_forecast_leads.png", note)


def fig_decision_value(art: dict, cfg: Config) -> Path | None:
    """Cost-loss value, and where a hazard-reduction programme pays for itself."""
    import matplotlib.pyplot as plt

    curves = art["cost_loss"]
    value = art["decision_value"]
    if curves is None and value is None:
        return None

    geometry = load_geometry(cfg)
    n_panels = 2 + (1 if (value is not None and geometry is not None) else 0)
    fig = plt.figure(figsize=(4.9 * n_panels, 4.6))
    axes = [fig.add_subplot(1, n_panels, i + 1) for i in range(n_panels)]

    ax = axes[0]
    if curves is not None and len(curves):
        hind = curves[curves.lead_days.eq(0)].sort_values("cost_loss_ratio")
        if len(hind):
            ax.plot(hind.cost_loss_ratio, hind.relative_economic_value,
                    "o-", color=ps.ACCENT, zorder=4, label="hindcast (analysis)",
                    markeredgecolor="white", markeredgewidth=1.1)
        leads = sorted(set(curves[curves.lead_days.gt(0)].lead_days), reverse=True)
        for i, lead in enumerate(leads):
            block = curves[curves.lead_days.eq(lead)].groupby(
                "cost_loss_ratio", as_index=False).relative_economic_value.mean()
            ax.plot(block.cost_loss_ratio, block.relative_economic_value,
                    ls="--", lw=1.3, color=ps.SEQ_HEX[2 + min(i, 4)],
                    zorder=3, label=f"day-{int(lead)}")
        ax.axhline(0, color=ps.FAINT, lw=1, zorder=1)
        ax.set_xlabel("cost–loss ratio  C/L")
        ax.set_ylabel("relative economic value")
        ax.legend(loc="best")
    else:
        ax.text(0.5, 0.5, "no cost-loss curves", ha="center", va="center", color=ps.MUTED)
        ax.set_axis_off()
    ps.panel_label(ax, "a", "Value of acting on the forecast")

    ax = axes[1]
    if value is not None and len(value):
        # The concentration curve is INVARIANT to delta: a hazard reduction
        # scales every county's avoided customer-hours by the same factor, so it
        # cancels in both the ranking and the cumulative share. Drawing one curve
        # per delta produced three exactly coincident lines and a legend that
        # implied a difference that cannot exist. One curve, and say why.
        deltas = sorted(value.hazard_reduction_delta.unique())
        block = value[value.hazard_reduction_delta.eq(deltas[0])]
        ordered = block.sort_values("avoided_cust_hours", ascending=False)
        share = np.arange(1, len(ordered) + 1) / len(ordered)
        cumulative = (ordered.avoided_cust_hours.cumsum()
                      / max(ordered.avoided_cust_hours.sum(), 1e-9))
        ax.plot(np.concatenate([[0], share]), np.concatenate([[0], cumulative]),
                color=ps.ACCENT, zorder=3)
        ax.fill_between(np.concatenate([[0], share]),
                        np.concatenate([[0], cumulative]),
                        np.concatenate([[0], share]),
                        color=ps.SEQ_HEX[1], alpha=0.55, zorder=2, lw=0)
        ax.plot([0, 1], [0, 1], color=ps.FAINT, lw=1, ls="--", zorder=1)
        # Label the true top decile only when a decile actually exists. With a
        # handful of counties `n // 10` collapses to one county and calling that
        # "the top decile" would be a wrong number on a published figure.
        n = len(ordered)
        k = max(n // 10, 1)
        top = float(cumulative[k - 1])
        caption = (f"top decile: {top:.0%}" if n >= 10
                   else f"top {k}/{n} counties: {top:.0%}")
        ax.annotate(caption, xy=(k / n, top),
                    xytext=(0.34, min(top + 0.14, 0.93)), fontsize=8.5,
                    color=ps.INK,
                    arrowprops={"arrowstyle": "-", "color": ps.MUTED, "lw": 0.9})
        ax.set_xlabel("share of counties, ranked by avoidable customer-hours")
        ax.set_ylabel("cumulative share of avoidable customer-hours")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
    else:
        ax.text(0.5, 0.5, "no decision-value table", ha="center", va="center",
                color=ps.MUTED)
        ax.set_axis_off()
    ps.panel_label(ax, "b", "Concentration of avoidable risk (invariant to δ)")

    if n_panels == 3:
        ax = axes[2]
        deltas = sorted(value.hazard_reduction_delta.unique())
        middle = deltas[len(deltas) // 2]
        block = value[value.hazard_reduction_delta.eq(middle)]
        mapped = geometry.merge(block, on="fips", how="left")
        mapped.plot(column="max_cost_per_asset_usd", ax=ax, cmap=ps.sequential(),
                    linewidth=0.3, edgecolor="white", legend=True,
                    legend_kwds={"shrink": 0.7, "label": "USD per asset", "pad": 0.01},
                    missing_kwds={"color": ps.MISSING, "edgecolor": "white",
                                  "linewidth": 0.3})
        ps.map_axes(ax)
        ps.panel_label(ax, "c", f"Break-even inspection cost (δ = {middle:.0%})")

    fig.suptitle("Decision value", fontsize=13, fontweight="semibold",
                 x=0.005, ha="left", y=0.995)
    fig.tight_layout(rect=(0, 0.03, 1, 0.955))
    dec = art["decision_summary"] or {}
    note = ("Cost–loss value is the classical forecast-to-decision bridge: act when "
            "p > C/L, valued against climatology as zero and perfect information as "
            "one. Per-lead curves rest on the case-study days only — read the "
            "n_county_days column in phase2_cost_loss.csv before quoting them.")
    if dec.get("ice_values_are_region_yaml_placeholders"):
        note = ("DOLLAR VALUES USE PLACEHOLDER ICE INPUTS FROM region.yaml AND ARE NOT "
                "RESULTS. ") + note
    return ps.save(fig, PATHS.figures / "fig5_decision_value.png", note)


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "config" / "region.yaml"))
    ap.add_argument("--phase2", default=str(ROOT / "config" / "phase2.yaml"))
    ap.add_argument("--only", choices=["all", "table", "figures"], default="all")
    args = ap.parse_args()
    cfg = load_config(args.config, args.phase2)

    art = load_artifacts(cfg)
    art["forecast_summary"] = read_json(
        PATHS.processed / "phase2_forecast_summary.json")
    opened = art["test_metrics"] is not None
    label = (f"test {pd.Timestamp(cfg['test_start']).year}" if opened
             else f"validation {pd.Timestamp(cfg['val_start']).year}")
    log.info("reporting on the %s split", label)

    matrix = results_matrix(art)
    MATRIX_CSV.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(MATRIX_CSV, index=False)

    counties = pd.DataFrame()
    quantiles = [float(q) for q in cfg["quantiles"]]
    source = art["test_predictions"] if opened else art["predictions"]
    if source is not None:
        if not opened:
            from src.phase2_train import masks
            source = source[masks(source, cfg)["val"]]
        counties = county_skill(source, quantiles)
        counties.to_csv(COUNTY_CSV, index=False)
        log.info("county diagnostics for %d counties -> %s",
                 len(counties), COUNTY_CSV.name)

    figures: dict[str, Path | None] = {}
    if args.only in ("all", "figures"):
        figures["occurrence and magnitude skill"] = fig_skill_summary(art, cfg, label)
        figures["county diagnostics"] = (
            fig_county_maps(art, cfg, counties, label) if len(counties) else None)
        figures["ERA5 hazard rasters"] = fig_hazard_rasters(art, cfg)
        figures["forecast by lead time"] = fig_forecast_leads(art, cfg)
        figures["decision value"] = fig_decision_value(art, cfg)

    path = write_results_markdown(cfg, art, matrix, figures)

    log.info("-" * 66)
    log.info("results table  -> %s", path.relative_to(ROOT))
    log.info("metric matrix  -> %s", MATRIX_CSV.relative_to(ROOT))
    for name, figure in figures.items():
        log.info("%-14s -> %s", "figure", figure.relative_to(ROOT) if figure
                 else f"SKIPPED ({name}: inputs absent)")
    if not opened:
        log.info("test year is sealed; this reports the validation split only")
    log.info("-" * 66)


if __name__ == "__main__":
    main()
