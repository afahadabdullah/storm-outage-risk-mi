#!/usr/bin/env python
"""Build result matrices and publication figures from frozen Phase 2 artifacts.

This reporting layer never fits or scores a model. It uses the honest OOF
validation predictions, or the once-only 2023 predictions after the test marker
exists, and summarizes the two configured GEFS case studies separately.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.common import plotstyle
from src.common.config import PATHS, ROOT, Config, load_config
from src.common.logio import get_logger
from src.phase2_train import config_digest, masks

log = get_logger("phase2_report")
plotstyle.apply()

MATRIX_PATH = PATHS.processed / "phase2_results_matrix.csv"
CASE_MATRIX_PATH = PATHS.processed / "phase2_gefs_case_matrix.csv"
RESULTS_PATH = ROOT / "docs" / "phase2_results.md"

METRICS = [
    ("occurrence_brier", "Occurrence Brier score", "lower", ".4f"),
    ("occurrence_brier_skill_vs_climatology",
     "Brier skill vs county climatology", "higher", "+.3f"),
    ("occurrence_average_precision", "Average precision", "higher", ".3f"),
    ("occurrence_roc_auc", "ROC AUC", "higher", ".3f"),
    ("occurrence_log_loss", "Occurrence log loss", "lower", ".4f"),
    ("reference_logistic_glm_brier", "Logistic GLM Brier", "lower", ".4f"),
    ("threshold_gust20_brier", "Gust > 20 m/s rule Brier", "lower", ".4f"),
    ("occurrence_brier_ref_county_climatology",
     "County climatology Brier", "lower", ".4f"),
    ("magnitude_crps", "Magnitude CRPS (customer-hours)", "lower", ",.0f"),
    ("magnitude_crps_skill_vs_climatology",
     "Magnitude CRPS skill vs climatology", "higher", "+.3f"),
    ("magnitude_log_score", "Magnitude log score", "lower", ".3f"),
    ("magnitude_median_rmse", "Magnitude median RMSE", "lower", ",.0f"),
    ("duration_concordance", "Restoration concordance", "higher", ".3f"),
    ("duration_median_mae", "Restoration median MAE (hours)", "lower", ".2f"),
    ("duration_persistence_mae", "County persistence MAE (hours)", "lower", ".2f"),
]


def _json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def _frame(path: Path) -> pd.DataFrame | None:
    return pd.read_parquet(path) if path.exists() else None


def _check_digest(blob: dict | None, expected: str, name: str) -> None:
    if blob is None:
        return
    found = blob.get("config_sha256")
    if found != expected:
        reason = "has no digest" if found is None else "belongs to another configuration"
        raise SystemExit(
            f"{name} {reason}; rerun the pipeline before publishing results")


def load_artifacts(cfg: Config) -> dict:
    expected = config_digest(cfg)
    validation = _json(PATHS.processed / "phase2_validation_metrics.json")
    test = _json(PATHS.processed / "phase2_test_metrics.json")
    _check_digest(validation, expected, "validation metrics")
    _check_digest(test, expected, "test metrics")

    forecast_summary = _json(PATHS.processed / "phase2_forecast_summary.json")
    if forecast_summary is not None:
        found = forecast_summary.get("config_sha256")
        if found != expected:
            raise SystemExit(
                "GEFS forecast artifacts do not match the frozen model/configuration; "
                "rerun `make phase2-forecast`")

    validation_predictions = _frame(PATHS.processed / "phase2_predictions.parquet")
    test_predictions = _frame(PATHS.processed / "phase2_test_predictions.parquet")
    opened = (PATHS.models / "TEST_YEAR_OPENED.txt").exists()
    if test is not None and not opened:
        raise SystemExit("test metrics exist without TEST_YEAR_OPENED.txt; refusing report")
    if opened and test is None:
        raise SystemExit("TEST_YEAR_OPENED.txt exists but test metrics are missing")
    if test is not None and test_predictions is None:
        raise SystemExit("test metrics exist but test predictions are missing")
    return {
        "validation_metrics": validation,
        "test_metrics": test,
        "predictions": validation_predictions,
        "test_predictions": test_predictions,
        "cv": (pd.read_csv(PATHS.processed / "phase2_cv_metrics.csv")
               if (PATHS.processed / "phase2_cv_metrics.csv").exists() else None),
        "uncertainty": (pd.read_csv(PATHS.processed / "phase2_uncertainty_by_lead.csv")
                        if (PATHS.processed / "phase2_uncertainty_by_lead.csv").exists()
                        else None),
        "realizations": (np.load(PATHS.processed / "phase2_forecast_realizations.npz")
                         if (PATHS.processed / "phase2_forecast_realizations.npz").exists()
                         else None),
        "forecast_summary": forecast_summary,
        "opened": opened,
    }


def results_matrix(artifacts: dict) -> pd.DataFrame:
    """Headline validation and final-test metrics in publication order."""
    columns = {
        "validation": artifacts.get("validation_metrics"),
        "test": artifacts.get("test_metrics"),
    }
    if all(value is None for value in columns.values()):
        raise SystemExit("no model metrics found; run `make phase2-train` first")
    rows = []
    for key, label, better, fmt in METRICS:
        row = {"key": key, "metric": label, "better": better}
        for split, values in columns.items():
            value = (values or {}).get(key)
            row[split] = float(value) if isinstance(value, (int, float)) else np.nan
            row[f"{split}_formatted"] = (
                format(value, fmt) if isinstance(value, (int, float)) else "--")
        rows.append(row)
    return pd.DataFrame(rows)


def _observed_cases(cfg: Config, artifacts: dict) -> dict[str, float]:
    if not artifacts["opened"] or artifacts["test_predictions"] is None:
        return {}
    pred = artifacts["test_predictions"].copy()
    pred["date"] = pd.to_datetime(pred.date).dt.normalize()
    observed = {}
    for case in cfg.get("case_studies", []):
        day = pd.Timestamp(case["date"]).normalize()
        block = pred[pred.date.eq(day)]
        if len(block):
            observed[case["name"]] = float(block.customer_hours.sum())
    return observed


def gefs_case_matrix(cfg: Config, artifacts: dict) -> pd.DataFrame:
    """Case/lead verification matrix for the frozen model driven by GEFS."""
    archive = artifacts.get("realizations")
    if archive is None:
        return pd.DataFrame()
    observed = _observed_cases(cfg, artifacts)
    uncertainty = artifacts.get("uncertainty")
    summary = artifacts.get("forecast_summary") or {}
    rows = []
    for key in archive.files:
        case, lead_text = key.rsplit("|", 1)
        lead = int(lead_text)
        values = np.asarray(archive[key], dtype=float)
        truth = observed.get(case, np.nan)
        p10, median, p90 = np.quantile(values, [0.10, 0.50, 0.90])
        row = {
            "case": case,
            "lead_days": lead,
            "n_realizations": int(values.size),
            "p10_customer_hours": float(p10),
            "median_customer_hours": float(median),
            "p90_customer_hours": float(p90),
            "observed_customer_hours": truth,
            "median_absolute_error": abs(float(median) - truth)
            if np.isfinite(truth) else np.nan,
            "observed_inside_80pct_interval": bool(p10 <= truth <= p90)
            if np.isfinite(truth) else np.nan,
            "input": "synthetic" if summary.get("synthetic_gefs") else "GEFS",
        }
        if uncertainty is not None:
            hit = uncertainty[
                uncertainty["case"].eq(case) & uncertainty["lead_days"].eq(lead)]
            if len(hit):
                row["meteorological_variance_share"] = float(
                    hit.iloc[0]["meteorological_share"])
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["case", "lead_days"], ascending=[True, False]).reset_index(drop=True)


def _scored_predictions(cfg: Config, artifacts: dict) -> tuple[pd.DataFrame | None, str]:
    if artifacts["test_metrics"] is not None and artifacts["test_predictions"] is not None:
        return artifacts["test_predictions"].copy(), "held-out test · 2023"
    pred = artifacts.get("predictions")
    if pred is None:
        return None, "validation"
    return pred[masks(pred, cfg)["val"]].copy(), "validation · Jul–Dec 2022"


def plot_skill_summary(cfg: Config, artifacts: dict) -> tuple[Path, Path] | None:
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import precision_recall_curve

    pred, label = _scored_predictions(cfg, artifacts)
    metrics = artifacts["test_metrics"] or artifacts["validation_metrics"]
    if pred is None or metrics is None or pred.empty:
        return None
    y = pred.event.astype(int).to_numpy()
    probability = pred.probability.to_numpy()
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.0))

    ax = axes[0, 0]
    frac, mean = calibration_curve(y, probability, n_bins=10, strategy="quantile")
    limit = max(float(np.max(frac)), float(np.max(mean)), 0.05) * 1.08
    ax.plot([0, limit], [0, limit], "--", color=plotstyle.FAINT, lw=1)
    ax.plot(mean, frac, "o-", color=plotstyle.ACCENT,
            markeredgecolor="white", markeredgewidth=1)
    ax.set(xlim=(0, limit), ylim=(0, limit), xlabel="forecast probability",
           ylabel="observed event frequency")
    plotstyle.panel(ax, "a", "Reliability")

    ax = axes[0, 1]
    precision, recall, _ = precision_recall_curve(y, probability)
    ax.plot(recall, precision, color=plotstyle.ACCENT)
    ax.axhline(y.mean(), color=plotstyle.FAINT, ls="--", lw=1,
               label=f"event rate {y.mean():.3f}")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="recall", ylabel="precision")
    ax.legend(loc="upper right")
    plotstyle.panel(
        ax, "b", f"Precision–recall · AP {metrics['occurrence_average_precision']:.3f}")

    ax = axes[1, 0]
    entries = [
        ("Frozen model", metrics.get("occurrence_brier"), plotstyle.ACCENT),
        ("Logistic GLM", metrics.get("reference_logistic_glm_brier"),
         plotstyle.REFERENCE),
        ("Gust > 20 m/s", metrics.get("threshold_gust20_brier"),
         plotstyle.REFERENCE),
        ("County climatology", metrics.get("occurrence_brier_ref_county_climatology"),
         plotstyle.REFERENCE),
    ]
    entries = [entry for entry in entries if isinstance(entry[1], (int, float))]
    names = [entry[0] for entry in entries][::-1]
    values = [entry[1] for entry in entries][::-1]
    colors = [entry[2] for entry in entries][::-1]
    bars = ax.barh(names, values, color=colors, height=0.6)
    for bar, value in zip(bars, values):
        ax.text(value, bar.get_y() + bar.get_height() / 2, f"  {value:.4f}",
                va="center", fontsize=8, color=plotstyle.MUTED)
    ax.set_xlabel("Brier score · lower is better")
    ax.set_xlim(0, max(values, default=1) * 1.25)
    ax.grid(axis="y", visible=False)
    plotstyle.panel(ax, "c", "Required reference models")

    ax = axes[1, 1]
    cv = artifacts.get("cv")
    if cv is not None and len(cv):
        order = ["storm_blocked", "leave_one_county_out", "forward_year"]
        present = [name for name in order if name in set(cv.scheme)]
        data = [cv.loc[cv.scheme.eq(name), "brier"].dropna().to_numpy()
                for name in present]
        boxes = ax.boxplot(data, patch_artist=True, widths=0.55,
                           medianprops={"color": plotstyle.INK, "linewidth": 1.5})
        for box in boxes["boxes"]:
            box.set(facecolor=plotstyle.SEQUENCE[1], edgecolor=plotstyle.ACCENT)
        ax.set_xticks(range(1, len(present) + 1),
                      [name.replace("_", "\n") for name in present])
        ax.set_ylabel("fold Brier score")
    else:
        ax.text(0.5, 0.5, "cross-validation artifact unavailable",
                ha="center", va="center", color=plotstyle.MUTED)
    plotstyle.panel(ax, "d", "Temporal, spatial, and storm-blocked CV")

    fig.suptitle(f"Frozen outage-risk model performance — {label}",
                 x=0.01, ha="left", fontsize=14, fontweight="semibold")
    fig.tight_layout(rect=(0, 0.035, 1, 0.965))
    note = (f"n={len(pred):,} county-days; events={int(y.sum()):,}. "
            + ("Validation reliability uses out-of-fold isotonic probabilities."
               if artifacts["test_metrics"] is None else
               "The 2023 test was scored once with the frozen model and calibrator."))
    return plotstyle.save(fig, PATHS.figures / "phase2_skill_summary.png", note)


def plot_gefs_cases(cfg: Config, artifacts: dict,
                    matrix: pd.DataFrame) -> tuple[Path, Path] | None:
    import matplotlib.pyplot as plt

    archive = artifacts.get("realizations")
    if archive is None or matrix.empty:
        return None
    cases = [case["name"] for case in cfg.get("case_studies", [])
             if case["name"] in set(matrix.case)]
    leads = sorted(matrix.lead_days.unique(), reverse=True)
    if not cases or not leads:
        return None
    fig, axes = plt.subplots(len(cases), len(leads),
                             figsize=(3.15 * len(leads), 3.25 * len(cases)),
                             squeeze=False)
    for row, case in enumerate(cases):
        blocks = [np.asarray(archive[f"{case}|{lead}"], dtype=float)
                  for lead in leads if f"{case}|{lead}" in archive.files]
        upper = max(float(np.quantile(np.concatenate(blocks), 0.995)), 1.0)
        bins = np.linspace(0, upper, 42)
        for col, lead in enumerate(leads):
            ax = axes[row, col]
            key = f"{case}|{lead}"
            if key not in archive.files:
                ax.set_axis_off()
                continue
            values = np.asarray(archive[key], dtype=float)
            ax.hist(values, bins=bins, color=plotstyle.SEQUENCE[min(col + 1, 4)],
                    edgecolor="white", linewidth=0.35)
            median = float(np.median(values))
            ax.axvline(median, color=plotstyle.ACCENT, lw=1.8, label="median")
            truth = matrix.loc[
                matrix.case.eq(case) & matrix.lead_days.eq(lead),
                "observed_customer_hours"].iloc[0]
            if np.isfinite(truth):
                ax.axvline(truth, color=plotstyle.OBSERVED, lw=2, label="observed")
            if row == len(cases) - 1:
                ax.set_xlabel("statewide customer-hours")
            if col == 0:
                ax.set_ylabel(f"{case}\nrealizations")
            ax.set_title(f"day −{lead}", loc="left")
            if row == 0 and col == len(leads) - 1:
                ax.legend(loc="upper right")
    summary = artifacts.get("forecast_summary") or {}
    synthetic = bool(summary.get("synthetic_gefs"))
    title = "Frozen-model GEFS case-study forecasts"
    if synthetic:
        title += " · SYNTHETIC INPUTS (NOT A RESULT)"
    fig.suptitle(title, x=0.01, ha="left", fontsize=14, fontweight="semibold")
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    note = ("31 GEFS members × 100 conditional model draws per lead. Gust and "
            "precipitation are quantile-mapped to the pre-test ERA5 climatology. "
            "Observed lines appear only after the once-only 2023 test is opened.")
    return plotstyle.save(fig, PATHS.figures / "phase2_gefs_case_studies.png", note)


def _markdown_table(matrix: pd.DataFrame, include_test: bool) -> list[str]:
    header = "| Metric | Better | Validation |" + (" Test |" if include_test else "")
    divider = "|---|---|---:|" + ("---:|" if include_test else "")
    lines = [header, divider]
    for row in matrix.itertuples():
        line = f"| {row.metric} | {row.better} | {row.validation_formatted} |"
        if include_test:
            line += f" {row.test_formatted} |"
        lines.append(line)
    return lines


def write_results(cfg: Config, artifacts: dict, matrix: pd.DataFrame,
                  cases: pd.DataFrame, figures: list[tuple[Path, Path] | None]) -> Path:
    include_test = artifacts["test_metrics"] is not None
    lines = [
        "# Phase 2 results",
        "",
        "Generated from frozen run artifacts by `make phase2-report`.",
        "",
        "## Study split",
        "",
        f"- Training: {cfg['train_start']} through {cfg['train_end']}",
        f"- Calibration/validation: {cfg['val_start']} through {cfg['val_end']}",
        f"- Held-out test: {cfg['test_start']} through {cfg['test_end']}"
        + (" (opened once)" if include_test else " (still sealed)"),
        "",
        "The full 2023 table tests the frozen model on ERA5/outage observations. "
        "The two GEFS storm cases are a separate operational forecast evaluation "
        "of that same frozen model; they do not replace the year-long test.",
        "",
        "## Headline metric matrix",
        "",
        *_markdown_table(matrix, include_test),
        "",
    ]
    if len(cases):
        lines += ["## GEFS case-study matrix", "",
                  "| Case | Lead | Median | 10–90% interval | Observed | Met. variance |",
                  "|---|---:|---:|---:|---:|---:|"]
        for row in cases.itertuples():
            truth = (f"{row.observed_customer_hours:,.0f}"
                     if np.isfinite(row.observed_customer_hours) else "--")
            met = getattr(row, "meteorological_variance_share", np.nan)
            met_text = f"{met:.1%}" if np.isfinite(met) else "--"
            lines.append(
                f"| {row.case} | day −{row.lead_days} | "
                f"{row.median_customer_hours:,.0f} | "
                f"{row.p10_customer_hours:,.0f}–{row.p90_customer_hours:,.0f} | "
                f"{truth} | {met_text} |")
        lines.append("")
    made = [pair for pair in figures if pair is not None]
    if made:
        lines += ["## Publication figures", ""]
        for png, pdf in made:
            lines.append(f"- `{png.relative_to(ROOT)}`; vector: `{pdf.relative_to(ROOT)}`")
        lines.append("")
    if not include_test:
        lines += ["The held-out 2023 results are intentionally absent. Rerun this report "
                  "after the one-time final-test job to add them.", ""]
    RESULTS_PATH.write_text("\n".join(lines))
    return RESULTS_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config" / "region.yaml"))
    parser.add_argument("--phase2", default=str(ROOT / "config" / "phase2.yaml"))
    args = parser.parse_args()
    cfg = load_config(args.config, args.phase2)
    artifacts = load_artifacts(cfg)
    matrix = results_matrix(artifacts)
    cases = gefs_case_matrix(cfg, artifacts)
    matrix.to_csv(MATRIX_PATH, index=False)
    cases.to_csv(CASE_MATRIX_PATH, index=False)
    figures = [plot_skill_summary(cfg, artifacts),
               plot_gefs_cases(cfg, artifacts, cases)]
    results = write_results(cfg, artifacts, matrix, cases, figures)
    log.info("metric matrix -> %s", MATRIX_PATH)
    log.info("GEFS case matrix -> %s", CASE_MATRIX_PATH)
    log.info("results summary -> %s", results)


if __name__ == "__main__":
    main()
