#!/usr/bin/env python
"""Fit, validate, and once-only test the full Phase 2 hurdle model.

Default mode fits on 2018--2021 and calibrates/evaluates on 2022. The held-out
2023 test is scored only with ``--evaluate-test``; that mode loads the frozen
bundle and never refits it. Cross-validation summaries are computed on pre-test
data only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.common.config import PATHS, ROOT, Config, load_config
from src.common.logio import get_logger, timed

log = get_logger("phase2_train")
MODEL_PATH = PATHS.models / "phase2_models.joblib"
PRED_PATH = PATHS.processed / "phase2_predictions.parquet"
METRIC_PATH = PATHS.processed / "phase2_validation_metrics.json"
CV_PATH = PATHS.processed / "phase2_cv_metrics.csv"
TEST_MARKER = PATHS.models / "TEST_YEAR_OPENED.txt"

TARGETS = {
    "event", "customer_hours", "restoration_hours", "peak_frac_out",
    "peak_customers_out", "n_events", "censored", "concurrent_state_load",
    "mcc", "customer_hours_per_customer",
}
IDS = {"fips", "date", "regime_label", "month"}


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return sorted(c for c in frame.columns if c not in TARGETS | IDS
                  and pd.api.types.is_numeric_dtype(frame[c]))


def masks(frame: pd.DataFrame, cfg: Config) -> dict[str, pd.Series]:
    dates = pd.to_datetime(frame.date)
    return {
        "train": dates.between(cfg["train_start"], cfg["train_end"]),
        "val": dates.between(cfg["val_start"], cfg["val_end"]),
        "test": dates.between(cfg["test_start"], cfg["test_end"]),
    }


def lgb_classifier(cfg: Config, n_estimators: int | None = None):
    import lightgbm as lgb

    return lgb.LGBMClassifier(
        objective="binary", n_estimators=n_estimators or int(cfg["n_boost_rounds"]),
        learning_rate=0.05, num_leaves=15, min_child_samples=30,
        subsample=0.9, colsample_bytree=0.9,
        random_state=int(cfg.get("random_seed", 0)), n_jobs=-1, verbose=-1)


def fit_occurrence(frame: pd.DataFrame, feats: list[str], split, cfg: Config):
    from sklearn.isotonic import IsotonicRegression

    train, val = frame[split["train"]], frame[split["val"]]
    y = train.event.astype(int)
    weight = float((1 - y).sum() / max(y.sum(), 1))
    model = lgb_classifier(cfg)
    model.set_params(scale_pos_weight=weight)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(train[feats], y)
    raw_val = model.predict_proba(val[feats])[:, 1]
    if val.event.nunique() < 2:
        raise ValueError("2022 validation year has only one occurrence class")
    calibrator = IsotonicRegression(out_of_bounds="clip").fit(
        raw_val, val.event.astype(int))
    return model, calibrator


def fit_magnitude(frame: pd.DataFrame, feats: list[str], split, cfg: Config):
    train = frame[split["train"] & frame.event.eq(1)]
    if len(train) < 50:
        raise ValueError(f"only {len(train)} training events for magnitude")
    X = train[feats]
    y = np.log1p(train.customer_hours.to_numpy())
    quantiles = [float(q) for q in cfg["quantiles"]]
    if cfg.get("magnitude_model", "ngboost") == "ngboost":
        try:
            from ngboost import NGBRegressor
            from ngboost.distns import Normal
            model = NGBRegressor(
                Dist=Normal, n_estimators=int(cfg["n_boost_rounds"]),
                learning_rate=0.03, verbose=False,
                random_state=int(cfg.get("random_seed", 0)))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X, y)
            return {"kind": "ngboost", "model": model, "quantiles": quantiles}
        except Exception as err:  # noqa: BLE001
            log.warning("NGBoost failed (%s); using LightGBM quantiles", err)

    import lightgbm as lgb
    models = {}
    for q in quantiles:
        model = lgb.LGBMRegressor(
            objective="quantile", alpha=q, n_estimators=int(cfg["n_boost_rounds"]),
            learning_rate=0.04, num_leaves=15, min_child_samples=20,
            random_state=int(cfg.get("random_seed", 0)), n_jobs=-1, verbose=-1)
        model.fit(X, y)
        models[q] = model
    return {"kind": "lgbm_quantile", "models": models, "quantiles": quantiles}


def magnitude_quantiles(bundle: dict, X: pd.DataFrame) -> np.ndarray:
    qs = bundle["quantiles"]
    if bundle["kind"] == "ngboost":
        dist = bundle["model"].pred_dist(X)
        log_q = np.column_stack([dist.ppf(q) for q in qs])
    else:
        log_q = np.column_stack([bundle["models"][q].predict(X) for q in qs])
    return np.expm1(np.sort(log_q, axis=1)).clip(min=0)


def duration_frame(frame: pd.DataFrame, numeric: list[str],
                   fill: dict[str, float] | None = None) -> tuple[pd.DataFrame, dict[str, float]]:
    values = frame.reindex(columns=numeric).replace([np.inf, -np.inf], np.nan)
    if fill is None:
        fill = {col: float(values[col].median()) if values[col].notna().any() else 0.0
                for col in values}
    values = values.fillna(fill)
    regimes = pd.get_dummies(frame.regime_label, prefix="regime", dtype=float)
    values = pd.concat([values.reset_index(drop=True), regimes.reset_index(drop=True)], axis=1)
    return values.astype(float), fill


def fit_duration(frame: pd.DataFrame, feats: list[str], split, cfg: Config):
    from lifelines import CoxPHFitter, WeibullAFTFitter
    from lifelines.statistics import proportional_hazard_test

    events = frame[split["train"] & frame.event.eq(1)].copy().reset_index(drop=True)
    numeric = feats + ["peak_frac_out", "concurrent_state_load"]
    X, fill = duration_frame(events, numeric)
    nunique = X.nunique(dropna=False)
    X = X.loc[:, nunique.gt(1)]
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), 1).astype(bool))
    X = X.drop(columns=[c for c in upper if upper[c].gt(0.999).any()])
    duration = events.restoration_hours.fillna(0.5).clip(lower=0.5).to_numpy()
    observed = (~events.censored.fillna(False)).astype(int).to_numpy()
    fit = X.copy()
    fit["duration_"] = duration
    fit["observed_"] = observed

    aft = WeibullAFTFitter(penalizer=0.1)
    aft.fit(fit, duration_col="duration_", event_col="observed_")
    cox = CoxPHFitter(penalizer=0.1)
    cox.fit(fit, duration_col="duration_", event_col="observed_")
    ph = proportional_hazard_test(cox, fit, time_transform="rank")
    ph.summary.to_csv(PATHS.processed / "phase2_cox_ph_test.csv")
    return {"model": aft, "cox": cox, "columns": list(X.columns),
            "numeric": numeric, "fill": fill}


def predict_duration(bundle: dict, frame: pd.DataFrame) -> np.ndarray:
    X, _ = duration_frame(frame, bundle["numeric"], bundle["fill"])
    X = X.reindex(columns=bundle["columns"], fill_value=0.0)
    return np.asarray(bundle["model"].predict_median(X)).ravel()


def predict(bundle: dict, frame: pd.DataFrame) -> pd.DataFrame:
    X = frame[bundle["features"]]
    raw = bundle["occurrence"].predict_proba(X)[:, 1]
    probability = bundle["calibrator"].predict(raw)
    qpred = magnitude_quantiles(bundle["magnitude"], X)
    duration = predict_duration(bundle["duration"], frame)
    out = frame[["fips", "date", "event", "customer_hours", "restoration_hours",
                 "censored", "regime_label", "customer_density", "gust_max"]].copy()
    out["probability_raw"] = raw
    out["probability"] = probability
    for idx, q in enumerate(bundle["magnitude"]["quantiles"]):
        out[f"magnitude_q{int(round(q * 100)):02d}"] = qpred[:, idx]
    out["duration_median"] = duration
    return out


def crps_from_quantiles(observed: np.ndarray, qpred: np.ndarray,
                        qs: list[float]) -> float:
    from properscoring import crps_ensemble

    grid = np.linspace(max(min(qs), 0.01), min(max(qs), 0.99), 199)
    ensemble = np.vstack([np.interp(grid, qs, row) for row in qpred])
    return float(np.mean(crps_ensemble(observed, ensemble)))


def evaluate(pred: pd.DataFrame, train: pd.DataFrame, label: str,
             quantiles: list[float]) -> dict:
    from lifelines.utils import concordance_index
    from sklearn.metrics import (average_precision_score, brier_score_loss,
                                 log_loss, roc_auc_score)

    y = pred.event.astype(int).to_numpy()
    p = pred.probability.to_numpy()
    climatology = float(train.event.mean())
    brier = float(brier_score_loss(y, p))
    brier_ref = float(np.mean((y - climatology) ** 2))
    metrics = {
        "split": label, "n_rows": len(pred), "n_events": int(y.sum()),
        "occurrence_brier": brier,
        "occurrence_brier_skill_vs_climatology": 1 - brier / brier_ref,
        "occurrence_average_precision": float(average_precision_score(y, p)),
        "occurrence_roc_auc": float(roc_auc_score(y, p)),
        "occurrence_log_loss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))),
        "threshold_gust20_brier": float(brier_score_loss(
            y, pred.gust_max.gt(20).astype(int))),
    }
    event_pred = pred[pred.event.eq(1)]
    qcols = [f"magnitude_q{int(round(q * 100)):02d}" for q in quantiles]
    qpred = event_pred[qcols].to_numpy()
    observed = event_pred.customer_hours.to_numpy()
    metrics["magnitude_crps"] = crps_from_quantiles(observed, qpred, quantiles)
    train_mag = train.loc[train.event.eq(1), "customer_hours"].to_numpy()
    climate_q = np.quantile(train_mag, quantiles)
    climate_pred = np.tile(climate_q, (len(observed), 1))
    ref = crps_from_quantiles(observed, climate_pred, quantiles)
    metrics["magnitude_crps_skill_vs_climatology"] = 1 - metrics["magnitude_crps"] / ref
    median_col = f"magnitude_q{int(round(quantiles[len(quantiles) // 2] * 100)):02d}"
    mag_error = event_pred[median_col].to_numpy() - observed
    central_width = qpred[:, -2] - qpred[:, 1] if qpred.shape[1] >= 4 else qpred[:, -1] - qpred[:, 0]
    metrics["magnitude_median_rmse"] = float(np.sqrt(np.mean(mag_error ** 2)))
    metrics["magnitude_spread_skill_ratio"] = float(
        np.mean(central_width) / max(metrics["magnitude_median_rmse"], 1e-9))
    pit = np.mean(qpred <= observed[:, None], axis=1)
    metrics["magnitude_pit_histogram_10bin"] = np.histogram(
        pit, bins=np.linspace(0, 1, 11))[0].astype(int).tolist()
    uncensored = event_pred[~event_pred.censored]
    if len(uncensored) > 1:
        metrics["duration_concordance"] = float(concordance_index(
            uncensored.restoration_hours, uncensored.duration_median))
        metrics["duration_median_mae"] = float(np.mean(np.abs(
            uncensored.restoration_hours - uncensored.duration_median)))
        persistence = float(train.loc[train.event.eq(1) & ~train.censored,
                                      "restoration_hours"].median())
        metrics["duration_persistence_mae"] = float(np.mean(np.abs(
            uncensored.restoration_hours - persistence)))
    by_regime = {}
    for regime, group in pred.groupby("regime_label"):
        if len(group) and group.event.nunique() > 1:
            by_regime[str(regime)] = {
                "n": len(group), "events": int(group.event.sum()),
                "brier": float(brier_score_loss(group.event, group.probability)),
                "average_precision": float(average_precision_score(
                    group.event, group.probability)),
            }
    metrics["by_regime"] = by_regime
    density_edges = train.customer_density.quantile([0, 1 / 3, 2 / 3, 1]).to_numpy()
    density_edges[0], density_edges[-1] = -np.inf, np.inf
    rurality = pd.cut(pred.customer_density, bins=np.unique(density_edges),
                      labels=False, include_lowest=True)
    metrics["by_rurality_tercile"] = {
        str(int(group)): {
            "n": int(mask.sum()),
            "brier": float(brier_score_loss(pred.loc[mask, "event"],
                                              pred.loc[mask, "probability"])),
        }
        for group in sorted(rurality.dropna().unique())
        for mask in [rurality.eq(group)]
    }
    if len(event_pred) >= 3:
        severity = pd.qcut(event_pred.customer_hours, 3,
                           labels=["low", "medium", "high"], duplicates="drop")
        metrics["magnitude_crps_by_severity_tercile"] = {
            str(level): crps_from_quantiles(
                event_pred.loc[severity.eq(level), "customer_hours"].to_numpy(),
                event_pred.loc[severity.eq(level), qcols].to_numpy(), quantiles)
            for level in severity.dropna().unique()
        }
    return metrics


def storm_groups(frame: pd.DataFrame, gap_days: int) -> pd.Series:
    daily = frame.groupby("date").event.max().sort_index()
    event_dates = pd.to_datetime(daily[daily.eq(1)].index)
    episodes = []
    episode = 0
    previous = None
    episode_start = None
    for date in event_dates:
        if previous is None or (date - previous).days > gap_days:
            if previous is not None:
                episodes.append((episode_start, previous, f"storm_{episode:04d}"))
            episode += 1
            episode_start = date
        previous = date
    if previous is not None:
        episodes.append((episode_start, previous, f"storm_{episode:04d}"))
    dates = pd.to_datetime(frame.date)
    labels = []
    for date in dates:
        label = f"quiet_{date:%Y%m%d}"
        for start, finish, storm in episodes:
            if start - pd.Timedelta(days=gap_days) <= date <= finish + pd.Timedelta(days=gap_days):
                label = storm
                break
        labels.append(label)
    return pd.Series(labels, index=frame.index)


def occurrence_cv(frame: pd.DataFrame, feats: list[str], cfg: Config) -> pd.DataFrame:
    from sklearn.metrics import brier_score_loss, roc_auc_score
    from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

    rows = []

    def run_scheme(name, groups, splitter):
        for fold, (tr_idx, va_idx) in enumerate(splitter.split(frame, frame.event, groups)):
            train, val = frame.iloc[tr_idx], frame.iloc[va_idx]
            if train.event.nunique() < 2 or val.event.nunique() < 2:
                continue
            model = lgb_classifier(cfg, int(cfg.get("cv_n_estimators", 200)))
            y = train.event.astype(int)
            model.set_params(scale_pos_weight=float((1 - y).sum() / max(y.sum(), 1)))
            model.fit(train[feats], y)
            p = model.predict_proba(val[feats])[:, 1]
            rows.append({"scheme": name, "fold": fold, "n": len(val),
                         "events": int(val.event.sum()),
                         "brier": float(brier_score_loss(val.event, p)),
                         "roc_auc": float(roc_auc_score(val.event, p))})

    storm = storm_groups(frame, int(cfg.get("storm_gap_days", 2)))
    run_scheme("storm_blocked", storm,
               GroupKFold(n_splits=min(5, storm.nunique())))
    if bool(cfg.get("cv_leave_one_county_out", True)):
        run_scheme("leave_one_county_out", frame.fips, LeaveOneGroupOut())

    years = pd.to_datetime(frame.date).dt.year
    for year in sorted(years.unique())[1:]:
        train, val = frame[years < year], frame[years == year]
        if train.event.nunique() < 2 or val.event.nunique() < 2:
            continue
        model = lgb_classifier(cfg, int(cfg.get("cv_n_estimators", 200)))
        y = train.event.astype(int)
        model.set_params(scale_pos_weight=float((1 - y).sum() / max(y.sum(), 1)))
        model.fit(train[feats], y)
        p = model.predict_proba(val[feats])[:, 1]
        rows.append({"scheme": "forward_year", "fold": int(year), "n": len(val),
                     "events": int(val.event.sum()),
                     "brier": float(brier_score_loss(val.event, p)),
                     "roc_auc": float(roc_auc_score(val.event, p))})
    return pd.DataFrame(rows)


def plot_validation(pred: pd.DataFrame, label: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.calibration import CalibrationDisplay
    from sklearn.metrics import PrecisionRecallDisplay

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    CalibrationDisplay.from_predictions(pred.event, pred.probability,
                                        n_bins=10, strategy="quantile", ax=axes[0])
    PrecisionRecallDisplay.from_predictions(pred.event, pred.probability, ax=axes[1])
    axes[0].set_title(f"Occurrence reliability — {label}")
    axes[1].set_title(f"Occurrence precision–recall — {label}")
    fig.tight_layout()
    fig.savefig(PATHS.figures / f"phase2_occurrence_{label}.png", dpi=160)
    plt.close(fig)


def config_digest(cfg: Config) -> str:
    content = {k: v for k, v in cfg.items() if not k.startswith("_")}
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def fit_and_validate(frame: pd.DataFrame, cfg: Config) -> None:
    import joblib

    split = masks(frame, cfg)
    if split["test"].any():
        raise SystemExit("Training input contains 2023 rows; rebuild through validation")
    feats = feature_columns(frame)
    with timed("phase2_model_fit", log):
        occurrence, calibrator = fit_occurrence(frame, feats, split, cfg)
        magnitude = fit_magnitude(frame, feats, split, cfg)
        duration = fit_duration(frame, feats, split, cfg)
    bundle = {"occurrence": occurrence, "calibrator": calibrator,
              "magnitude": magnitude, "duration": duration, "features": feats,
              "config_sha256": config_digest(cfg), "trained_through": cfg["val_end"]}
    joblib.dump(bundle, MODEL_PATH)
    prediction = predict(bundle, frame)
    prediction.to_parquet(PRED_PATH, index=False)
    train = frame[split["train"]]
    val_pred = prediction[split["val"].to_numpy()]
    metrics = evaluate(val_pred, train, "validation_2022", magnitude["quantiles"])
    METRIC_PATH.write_text(json.dumps(metrics, indent=2))
    plot_validation(val_pred, "validation_2022")
    with timed("phase2_cross_validation", log):
        cv = occurrence_cv(frame[split["train"] | split["val"]].reset_index(drop=True),
                           feats, cfg)
    cv.to_csv(CV_PATH, index=False)
    log.info("validation BSS %.3f, AP %.3f, AUC %.3f, magnitude CRPSS %.3f",
             metrics["occurrence_brier_skill_vs_climatology"],
             metrics["occurrence_average_precision"], metrics["occurrence_roc_auc"],
             metrics["magnitude_crps_skill_vs_climatology"])


def evaluate_test(frame: pd.DataFrame, cfg: Config, force: bool = False) -> None:
    import joblib

    if not MODEL_PATH.exists():
        raise SystemExit("Frozen Phase 2 model missing; run `make phase2-train` first")
    if TEST_MARKER.exists() and not force:
        raise SystemExit(f"{TEST_MARKER} exists: test year was already opened. "
                         "Refusing a second look; use --force-test only for rerun recovery.")
    bundle = joblib.load(MODEL_PATH)
    if bundle["config_sha256"] != config_digest(cfg):
        raise SystemExit("configuration changed after model freeze; refusing test evaluation")
    split = masks(frame, cfg)
    test = frame[split["test"]].copy()
    if test.empty:
        raise SystemExit("No 2023 rows; first run `make phase2-build-test`")
    TEST_MARKER.write_text(
        f"Test year opened once at {pd.Timestamp.utcnow()} UTC\n"
        f"config_sha256={bundle['config_sha256']}\n")
    prediction = predict(bundle, test)
    prediction.to_parquet(PATHS.processed / "phase2_test_predictions.parquet", index=False)
    train = frame[split["train"]]
    metrics = evaluate(prediction, train, "test_2023", bundle["magnitude"]["quantiles"])
    (PATHS.processed / "phase2_test_metrics.json").write_text(json.dumps(metrics, indent=2))
    plot_validation(prediction, "test_2023")
    log.info("FINAL TEST: BSS %.3f, AP %.3f, AUC %.3f, magnitude CRPSS %.3f",
             metrics["occurrence_brier_skill_vs_climatology"],
             metrics["occurrence_average_precision"], metrics["occurrence_roc_auc"],
             metrics["magnitude_crps_skill_vs_climatology"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "config" / "region.yaml"))
    ap.add_argument("--phase2", default=str(ROOT / "config" / "phase2.yaml"))
    ap.add_argument("--evaluate-test", action="store_true")
    ap.add_argument("--force-test", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config, args.phase2)
    if args.evaluate_test and TEST_MARKER.exists() and not args.force_test:
        raise SystemExit(f"{TEST_MARKER} exists: test year was already opened. "
                         "Refusing to read the full table again.")
    frame = pd.read_parquet(PATHS.processed / "phase2_merged.parquet")
    if args.evaluate_test:
        evaluate_test(frame, cfg, force=args.force_test)
    else:
        fit_and_validate(frame, cfg)


if __name__ == "__main__":
    main()
