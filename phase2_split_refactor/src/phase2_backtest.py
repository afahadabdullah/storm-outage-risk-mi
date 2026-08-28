#!/usr/bin/env python
"""Score the frozen Phase 2 bundle on a retrospective, pre-test year.

This is intentionally separate from ``phase2_train --evaluate-test``.  It
does not touch the final-test marker or its artifacts, so 2021 can be used to
inspect temporal generalisation while the configured 2023 test remains sealed.
The companion builder chooses the last contiguous, locally available ERA5
month, rather than silently treating absent months as zero-information days.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.common.config import PATHS, ROOT, load_config
from src.common.gates import set_phase
from src.common.logio import get_logger, timed
from src.phase2_train import (
    MODEL_PATH,
    config_digest,
    crps_from_quantiles,
    evaluate,
    masks,
    predict,
)

log = get_logger("phase2_backtest")


def _safe_score(y: pd.Series, probability: pd.Series,
                climatology: pd.Series) -> dict[str, float]:
    """Occurrence scores for a small stratum, returning NaN when undefined."""
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    yv = y.astype(int).to_numpy()
    pv = probability.to_numpy()
    cv = climatology.to_numpy()
    if len(yv) == 0:
        return {"brier_skill": np.nan, "average_precision": np.nan, "roc_auc": np.nan}
    brier = float(brier_score_loss(yv, pv))
    reference = float(np.mean((yv - cv) ** 2))
    result = {"brier_skill": 1 - brier / reference if reference > 0 else np.nan,
              "average_precision": np.nan, "roc_auc": np.nan}
    if np.unique(yv).size > 1:
        result["average_precision"] = float(average_precision_score(yv, pv))
        result["roc_auc"] = float(roc_auc_score(yv, pv))
    return result


def skill_matrix(prediction: pd.DataFrame, train: pd.DataFrame,
                 quantiles: list[float]) -> pd.DataFrame:
    """Monthly occurrence/magnitude skill matrix used by the diagnostics plot."""
    train_events = train.loc[train.event.eq(1), "customer_hours"].to_numpy()
    rows: list[dict] = []
    for month, group in prediction.groupby(pd.to_datetime(prediction.date).dt.month):
        scores = _safe_score(group.event, group.probability,
                             group.reference_climatology_county)
        row = {"month": int(month), "n_county_days": int(len(group)),
               "n_events": int(group.event.sum()), **scores}
        events = group[group.event.eq(1)]
        qcols = [f"magnitude_q{int(round(q * 100)):02d}" for q in quantiles]
        if len(events) >= 3 and len(train_events):
            observed = events.customer_hours.to_numpy()
            score = crps_from_quantiles(observed, events[qcols].to_numpy(), quantiles)
            climate = np.tile(np.quantile(train_events, quantiles), (len(events), 1))
            reference = crps_from_quantiles(observed, climate, quantiles)
            row["magnitude_crps_skill"] = 1 - score / reference if reference > 0 else np.nan
        else:
            row["magnitude_crps_skill"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values("month")


def county_skill(prediction: pd.DataFrame) -> pd.DataFrame:
    """County-level occurrence diagnostics, with the same reference as BSS."""
    from sklearn.metrics import brier_score_loss

    rows: list[dict] = []
    for fips, group in prediction.groupby("fips"):
        y = group.event.astype(int).to_numpy()
        p = group.probability.to_numpy()
        climate = group.reference_climatology_county.to_numpy()
        brier = float(brier_score_loss(y, p))
        reference = float(np.mean((y - climate) ** 2))
        rows.append({
            "fips": str(fips).zfill(5), "n_county_days": int(len(group)),
            "n_events": int(y.sum()), "brier": brier,
            "brier_skill": 1 - brier / reference if reference > 0 else np.nan,
            "observed_event_rate": float(np.mean(y)),
            "mean_probability": float(np.mean(p)),
            "probability_bias": float(np.mean(p) - np.mean(y)),
        })
    return pd.DataFrame(rows).sort_values("fips")


def plot_diagnostics(prediction: pd.DataFrame, matrix: pd.DataFrame,
                     metrics: dict, label: str) -> Path:
    """Write reliability, discrimination, and month-by-metric skill figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.calibration import CalibrationDisplay
    from sklearn.metrics import PrecisionRecallDisplay

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    CalibrationDisplay.from_predictions(prediction.event, prediction.probability,
                                        n_bins=10, strategy="quantile", ax=axes[0, 0])
    axes[0, 0].set_title("Occurrence reliability")
    PrecisionRecallDisplay.from_predictions(prediction.event, prediction.probability,
                                             ax=axes[0, 1])
    axes[0, 1].set_title("Occurrence precision-recall")

    score_columns = ["brier_skill", "average_precision", "roc_auc", "magnitude_crps_skill"]
    values = matrix.reindex(columns=score_columns).to_numpy(dtype=float).T
    image = axes[1, 0].imshow(values, aspect="auto", cmap="RdYlGn", vmin=-0.25, vmax=1)
    axes[1, 0].set_xticks(np.arange(len(matrix)))
    axes[1, 0].set_xticklabels([pd.Timestamp(2000, int(m), 1).strftime("%b")
                                for m in matrix.month])
    axes[1, 0].set_yticks(np.arange(len(score_columns)))
    axes[1, 0].set_yticklabels(["BSS", "AP", "AUC", "magnitude CRPSS"])
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            if np.isfinite(value):
                axes[1, 0].text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=8)
    axes[1, 0].set_title("Skill matrix by available month")
    fig.colorbar(image, ax=axes[1, 0], shrink=0.85, label="score")

    labels = ["Model BSS", "AP", "AUC", "Magnitude CRPSS"]
    values = [metrics.get("occurrence_brier_skill_vs_climatology", np.nan),
              metrics.get("occurrence_average_precision", np.nan),
              metrics.get("occurrence_roc_auc", np.nan),
              metrics.get("magnitude_crps_skill_vs_climatology", np.nan)]
    colors = ["#2979b8" if np.isfinite(v) and v >= 0 else "#bd3d3d" for v in values]
    axes[1, 1].barh(labels, values, color=colors)
    axes[1, 1].axvline(0, color="0.3", linewidth=0.8)
    finite = [v for v in values if np.isfinite(v)]
    axes[1, 1].set_xlim(min(-0.1, min(finite, default=0.0) - 0.05), 1.0)
    axes[1, 1].set_title("Whole-window summary")
    axes[1, 1].grid(axis="x", alpha=0.25)

    fig.suptitle(f"Phase 2 retrospective backtest — {label}", y=1.01)
    fig.tight_layout()
    path = PATHS.figures / f"phase2_backtest_{label}_diagnostics.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_maps(counties: pd.DataFrame, label: str) -> Path:
    """Map county BSS, observed frequency, forecast mean, and probability bias."""
    import geopandas as gpd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    tiger = PATHS.raw / "tiger_counties_2023.parquet"
    if not tiger.exists():
        raise SystemExit(f"{tiger} missing; static county geometry is required for maps")
    geography = gpd.read_parquet(tiger)
    key = next((name for name in ("GEOID", "fips", "FIPS") if name in geography.columns), None)
    if key is None:
        raise ValueError(f"No FIPS column in {tiger}; found {list(geography.columns)}")
    geography["fips"] = geography[key].astype(str).str.zfill(5)
    mapped = geography.merge(counties, on="fips", how="inner", validate="one_to_one")
    if mapped.empty:
        raise ValueError("No county skill rows joined to the TIGER geometry")

    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    fields = [
        ("brier_skill", "Brier skill vs county climatology", "RdYlGn", True),
        ("observed_event_rate", "Observed event rate", "magma", None),
        ("mean_probability", "Mean forecast probability", "viridis", None),
        ("probability_bias", "Mean probability − observed rate", "RdBu_r", True),
    ]
    for axis, (field, title, cmap, centred) in zip(axes.flat, fields):
        norm = None
        if centred:
            finite = pd.to_numeric(mapped[field], errors="coerce").dropna()
            extent = max(float(finite.abs().max()) if len(finite) else 0.0, 1e-6)
            norm = TwoSlopeNorm(vmin=-extent, vcenter=0.0, vmax=extent)
        mapped.plot(column=field, ax=axis, cmap=cmap, norm=norm,
                    legend=True, legend_kwds={"shrink": 0.7}, missing_kwds={"color": "lightgrey"})
        axis.set_title(title)
        axis.set_axis_off()
    fig.suptitle(f"Phase 2 county diagnostics — {label}", y=0.94)
    fig.tight_layout()
    path = PATHS.figures / f"phase2_backtest_{label}_maps.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "config" / "region.yaml"))
    ap.add_argument("--phase2", default=str(ROOT / "config" / "phase2.yaml"))
    ap.add_argument("--year", type=int, required=True)
    args = ap.parse_args()
    cfg = load_config(args.config, args.phase2)
    set_phase(2)
    year = int(args.year)
    val_year = pd.Timestamp(cfg["val_end"]).year
    test_year = pd.Timestamp(cfg["test_start"]).year
    if not (val_year < year < test_year):
        gap = list(range(val_year + 1, test_year))
        raise SystemExit(
            f"Backtest year must lie strictly between validation ({val_year}) and "
            f"the sealed final test ({test_year}). "
            + (f"Available gap years: {gap}."
               if gap else
               "There is no gap year under the current frozen split -- validation "
               "runs to the end of the year before the test year, which is the "
               "intended design. Every pre-test year is now training or validation "
               "data, so a retrospective backtest would not be out of sample. Use "
               "the cross-validation in `make phase2-train` for temporal "
               "generalisation instead."))

    import joblib

    if not MODEL_PATH.exists():
        raise SystemExit("Frozen Phase 2 model missing; run `make phase2-train` first")
    bundle = joblib.load(MODEL_PATH)
    if bundle["config_sha256"] != config_digest(cfg):
        raise SystemExit("configuration changed after model freeze; refusing backtest "
                         "evaluation")
    source = PATHS.processed / f"phase2_merged_backtest_{year}.parquet"
    if not source.exists():
        raise SystemExit(f"{source} missing; run `make phase2-backtest-{year}` "
                         "first")
    frame = pd.read_parquet(source)
    dates = pd.to_datetime(frame.date)
    scored = frame.loc[dates.dt.year.eq(year)].copy()
    if scored.empty:
        raise SystemExit(f"No {year} rows in {source}")
    frozen = set(bundle.get("reporting_counties") or [])
    rebuilt = set(frame.fips.astype(str).unique())
    if frozen and frozen != rebuilt:
        raise SystemExit("Reporting-county cohort changed between the frozen model and "
                         "backtest build")

    train = frame[masks(frame, cfg)["train"]].copy()
    backtest_dates = dates[dates.dt.year.eq(year)]
    label = f"{year}_{backtest_dates.min():%Y%m%d}_{backtest_dates.max():%Y%m%d}"
    with timed("phase2_backtest_score", log):
        prediction = predict(bundle, scored)
        metrics = evaluate(prediction, train, f"backtest_{label}",
                           bundle["magnitude"]["quantiles"])
        matrix = skill_matrix(prediction, train, bundle["magnitude"]["quantiles"])
        counties = county_skill(prediction)
        metrics.update({"backtest_kind": "retrospective_pre_final_test",
                        "window_start": str(pd.to_datetime(prediction.date).min().date()),
                        "window_end": str(pd.to_datetime(prediction.date).max().date()),
                        "available_months": [int(v) for v in matrix.month],
                        "n_reporting_counties": int(len(counties))})

    prediction.to_parquet(PATHS.processed / f"phase2_backtest_{year}_predictions.parquet",
                          index=False)
    matrix.to_csv(PATHS.processed / f"phase2_backtest_{year}_skill_matrix.csv", index=False)
    counties.to_csv(PATHS.processed / f"phase2_backtest_{year}_county_skill.csv", index=False)
    (PATHS.processed / f"phase2_backtest_{year}_metrics.json").write_text(
        json.dumps(metrics, indent=2))
    diagnostics = plot_diagnostics(prediction, matrix, metrics, label)
    maps = plot_maps(counties, label)
    log.info("BACKTEST %s: BSS %.3f, AP %.3f, AUC %.3f, magnitude CRPSS %.3f",
             label, metrics["occurrence_brier_skill_vs_climatology"],
             metrics["occurrence_average_precision"], metrics["occurrence_roc_auc"],
             metrics["magnitude_crps_skill_vs_climatology"])
    log.info("wrote diagnostics %s and maps %s", diagnostics.name, maps.name)


if __name__ == "__main__":
    main()
