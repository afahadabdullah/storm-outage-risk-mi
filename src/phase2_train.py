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
from src.common.gates import set_phase
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
    """Fit the occurrence model and its isotonic calibrator (spec 6.1).

    Returns the model, the calibrator fitted on the whole validation year (the
    one that goes into the frozen bundle and scores the test year), and an
    OUT-OF-FOLD calibrated probability for each validation row.

    That third return value is the point. Isotonic is a flexible non-parametric
    fit; scoring it on the same rows it was fitted on flatters every occurrence
    metric and makes the reliability diagram look better calibrated than the
    model is. The honest validation number comes from K-fold-within-2022; the
    full-year calibrator is still what ships, because for the test year 2022 is
    genuinely out of sample.
    """
    from sklearn.isotonic import IsotonicRegression
    from sklearn.model_selection import StratifiedKFold

    train, val = frame[split["train"]], frame[split["val"]]
    y = train.event.astype(int)
    weight = float((1 - y).sum() / max(y.sum(), 1))
    model = lgb_classifier(cfg)
    model.set_params(scale_pos_weight=weight)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(train[feats], y)
    raw_val = model.predict_proba(val[feats])[:, 1]
    y_val = val.event.astype(int).to_numpy()
    if val.event.nunique() < 2:
        raise ValueError(
            f"{pd.Timestamp(cfg['val_start']).year} validation year has only "
            "one occurrence class")

    calibrator = IsotonicRegression(out_of_bounds="clip").fit(raw_val, y_val)

    n_folds = int(cfg.get("calibration_cv_folds", 5))
    oof = np.empty_like(raw_val)
    folds = StratifiedKFold(n_splits=n_folds, shuffle=True,
                            random_state=int(cfg.get("random_seed", 0)))
    for fit_idx, score_idx in folds.split(raw_val.reshape(-1, 1), y_val):
        inner = IsotonicRegression(out_of_bounds="clip").fit(
            raw_val[fit_idx], y_val[fit_idx])
        oof[score_idx] = inner.predict(raw_val[score_idx])
    log.info("isotonic calibration: %d-fold out-of-fold probabilities computed "
             "for the validation year (in-sample fit is what ships in the bundle)",
             n_folds)
    return model, calibrator, pd.Series(oof, index=val.index)


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


def magnitude_quantiles(bundle: dict, X: pd.DataFrame,
                        quantiles: list[float] | None = None) -> np.ndarray:
    """Predicted customer-hours at the requested quantiles (natural units).

    `quantiles` defaults to the reporting set. CRPS asks for a finer, wider grid
    -- see `crps_from_quantiles` -- and NGBoost can supply it directly from the
    fitted density rather than interpolating between seven reported points.
    """
    qs = list(quantiles) if quantiles is not None else bundle["quantiles"]
    if bundle["kind"] == "ngboost":
        dist = bundle["model"].pred_dist(X)
        log_q = np.column_stack([dist.ppf(q) for q in qs])
    else:
        # LightGBM quantile models exist only at the trained quantiles, so
        # anything outside them is extrapolated log-linearly from the outer
        # pair rather than clamped -- clamping is what truncated the tails.
        trained = list(bundle["quantiles"])
        base = np.column_stack([bundle["models"][q].predict(X) for q in trained])
        base = np.sort(base, axis=1)
        log_q = _interp_log_quantiles(base, trained, qs)
    return np.expm1(np.sort(log_q, axis=1)).clip(min=0)


def _interp_log_quantiles(base: np.ndarray, trained: list[float],
                          wanted: list[float]) -> np.ndarray:
    """Interpolate in log space, extrapolating the tails on a normal scale.

    Beyond the trained quantile range the quantile function is extended using
    the local slope in probit space, which keeps the extrapolated tail finite
    and monotone instead of flat. A flat tail is what made CRPS optimistic:
    the outer 10% of the predictive mass was being treated as if it sat exactly
    on q05 and q95.
    """
    from scipy.stats import norm

    z_trained = norm.ppf(np.asarray(trained, dtype=float))
    z_wanted = norm.ppf(np.asarray(wanted, dtype=float))
    lo_slope = (base[:, 1] - base[:, 0]) / max(z_trained[1] - z_trained[0], 1e-9)
    hi_slope = (base[:, -1] - base[:, -2]) / max(z_trained[-1] - z_trained[-2], 1e-9)

    # Vectorised over rows: every row shares z_trained, so one searchsorted on
    # the wanted grid gives the bracketing pair and weight for all rows at once.
    # A per-row np.interp inside the grid loop would be ~1e6 python-level calls
    # on a real validation year.
    k = np.clip(np.searchsorted(z_trained, z_wanted, side="right") - 1,
                0, len(z_trained) - 2)
    span = z_trained[k + 1] - z_trained[k]
    w = np.where(span > 0, (z_wanted - z_trained[k]) / np.where(span > 0, span, 1.0), 0.0)
    out = base[:, k] * (1.0 - w)[None, :] + base[:, k + 1] * w[None, :]

    below = z_wanted < z_trained[0]
    above = z_wanted > z_trained[-1]
    if below.any():
        out[:, below] = (base[:, [0]]
                         + lo_slope[:, None] * (z_wanted[below] - z_trained[0])[None, :])
    if above.any():
        out[:, above] = (base[:, [-1]]
                         + hi_slope[:, None] * (z_wanted[above] - z_trained[-1])[None, :])
    return out


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


DURATION_QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)


def predict_duration(bundle: dict, frame: pd.DataFrame) -> np.ndarray:
    X, _ = duration_frame(frame, bundle["numeric"], bundle["fill"])
    X = X.reindex(columns=bundle["columns"], fill_value=0.0)
    return np.asarray(bundle["model"].predict_median(X)).ravel()


def predict_duration_quantiles(bundle: dict, frame: pd.DataFrame) -> np.ndarray:
    """Predicted restoration-time quantiles, for the spec 7.2 calibration check."""
    X, _ = duration_frame(frame, bundle["numeric"], bundle["fill"])
    X = X.reindex(columns=bundle["columns"], fill_value=0.0)
    cols = []
    for q in DURATION_QUANTILES:
        # lifelines' predict_percentile takes the SURVIVAL probability p, so the
        # q-th quantile of the duration is p = 1 - q.
        cols.append(np.asarray(bundle["model"].predict_percentile(X, p=1.0 - q)).ravel())
    out = np.column_stack(cols)
    return np.sort(np.nan_to_num(out, nan=0.0, posinf=np.nanmax(out[np.isfinite(out)])
                                 if np.isfinite(out).any() else 0.0), axis=1)


def _log_score_from_quantiles(observed: np.ndarray, qpred: np.ndarray,
                              qs: list[float]) -> float:
    """Mean negative log predictive density, read off the quantile function.

    The density at the observed value is the reciprocal of the quantile
    function's local slope: f(y) = d tau / d q. Approximated by a central
    difference on a fine probability grid, which is enough for a comparative
    score and avoids assuming a parametric head that the LightGBM-quantile
    route does not have.
    """
    observed = np.asarray(observed, dtype=float)
    if observed.size == 0:
        return float("nan")
    grid = np.linspace(0.002, 0.998, 501)
    q_full = np.sort(np.clip(_interp_log_quantiles(
        np.sort(np.asarray(qpred, dtype=float), axis=1), [float(v) for v in qs],
        list(grid)), 0.0, None), axis=1)
    dens = np.gradient(grid)[None, :] / np.maximum(np.gradient(q_full, axis=1), 1e-9)
    idx = np.clip(np.array([np.searchsorted(row, val)
                            for row, val in zip(q_full, observed)]),
                  0, q_full.shape[1] - 1)
    at_obs = dens[np.arange(len(observed)), idx]
    return float(-np.mean(np.log(np.clip(at_obs, 1e-300, None))))


def _duration_calibration(frame: pd.DataFrame) -> dict:
    """Empirical coverage of the predicted restoration-time quantiles.

    A well-calibrated Weibull AFT puts ~10% of observed restorations below its
    own 10th percentile, ~50% below its median, and so on. Reported as observed
    share per nominal level -- the quantile calibration spec 7.2 asks for.
    """
    out = {}
    for q in DURATION_QUANTILES:
        col = f"duration_q{int(round(q * 100)):02d}"
        if col not in frame:
            continue
        out[f"{int(round(q * 100)):02d}"] = round(float(
            (frame.restoration_hours <= frame[col]).mean()), 4)
    return out


def fit_reference_models(frame: pd.DataFrame, feats: list[str], split, cfg: Config):
    """Spec 6.1 / 6.2 skill floors: a logistic GLM and a negative-binomial GLM.

    Absolute scores mean nothing without these. A gradient-boosted model that
    only just beats a logistic regression on a reduced feature set is a finding,
    and reporting it makes everything else in the write-up more believable.
    """
    import statsmodels.api as sm

    # A deliberately reduced, interpretable feature set -- not the full 26.
    reduced = [c for c in ("gust_max", "hours_gust_gt_18", "precip_total",
                           "freezing_rain_proxy", "snow_water_equiv_wet",
                           "soil_moisture_mean", "canopy_pct", "customer_density",
                           "leaf_on") if c in feats]
    train = frame[split["train"]]
    Xtr = sm.add_constant(train[reduced].astype(float), has_constant="add")
    out: dict = {"columns": reduced}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            out["logistic"] = sm.GLM(train.event.astype(float), Xtr,
                                     family=sm.families.Binomial()).fit(maxiter=200)
        except Exception as err:                              # noqa: BLE001
            log.warning("logistic GLM reference failed (%s); skipping", err)
        events = train[train.event.eq(1)]
        if len(events) >= 50:
            Xev = sm.add_constant(events[reduced].astype(float), has_constant="add")
            try:
                # Negative binomial on rounded customer-hours (a count-like,
                # over-dispersed target) -- spec 6.2's stated reference.
                y = np.rint(events.customer_hours.to_numpy()).clip(min=0)
                out["negbin"] = sm.GLM(
                    y, Xev, family=sm.families.NegativeBinomial(alpha=1.0)
                ).fit(maxiter=200)
            except Exception as err:                          # noqa: BLE001
                log.warning("negative-binomial GLM reference failed (%s); skipping", err)
    log.info("reference models fitted on %d reduced features: %s",
             len(reduced), sorted(k for k in out if k != "columns"))
    return out


def predict(bundle: dict, frame: pd.DataFrame) -> pd.DataFrame:
    X = frame[bundle["features"]]
    raw = bundle["occurrence"].predict_proba(X)[:, 1]
    probability = bundle["calibrator"].predict(raw)
    qpred = magnitude_quantiles(bundle["magnitude"], X)
    duration = predict_duration(bundle["duration"], frame)
    dur_q = predict_duration_quantiles(bundle["duration"], frame)
    out = frame[["fips", "date", "event", "customer_hours", "restoration_hours",
                 "censored", "regime_label", "customer_density", "gust_max"]].copy()
    out["probability_raw"] = raw
    out["probability"] = probability
    for idx, q in enumerate(bundle["magnitude"]["quantiles"]):
        out[f"magnitude_q{int(round(q * 100)):02d}"] = qpred[:, idx]
    out["duration_median"] = duration
    for idx, q in enumerate(DURATION_QUANTILES):
        out[f"duration_q{int(round(q * 100)):02d}"] = dur_q[:, idx]

    # Spec 7.3 references, carried alongside the model's own predictions so
    # every metric can be reported relative to them.
    out["reference_climatology_county"] = frame.fips.map(
        bundle.get("county_climatology", {})).fillna(
        bundle.get("global_climatology", 0.0)).to_numpy()
    refs = bundle.get("references") or {}
    if refs.get("logistic") is not None:
        import statsmodels.api as sm
        Xr = sm.add_constant(frame[refs["columns"]].astype(float), has_constant="add")
        out["reference_logistic_glm"] = np.asarray(
            refs["logistic"].predict(Xr)).ravel()
    if refs.get("negbin") is not None:
        import statsmodels.api as sm
        Xr = sm.add_constant(frame[refs["columns"]].astype(float), has_constant="add")
        out["reference_magnitude_glm"] = np.asarray(
            refs["negbin"].predict(Xr)).ravel()
    return out


def crps_from_quantiles(observed: np.ndarray, qpred: np.ndarray,
                        qs: list[float], n_grid: int = 999) -> float:
    """CRPS via the pinball-loss integral, with the tails extrapolated.

        CRPS(F, y) = 2 * integral_0^1 pinball_tau(y - F^-1(tau)) d tau

    This is an identity, not an ensemble approximation, so it needs no
    truncation. The previous implementation built an equally weighted ensemble
    on linspace(min(qs), max(qs)) -- 0.05 to 0.95 -- which discards the outer
    10% of the predictive mass and treats the distribution as if it ended
    there. For a target spanning four to five orders of magnitude that is where
    most of the customer-hours live, so absolute CRPS came out understated and
    the high-severity tercile worst of all.

    Quantiles outside the supplied set are extrapolated with the local slope in
    probit space (see `_interp_log_quantiles`), keeping the tail monotone and
    finite rather than flat.
    """
    observed = np.asarray(observed, dtype=float)
    if observed.size == 0:
        return float("nan")
    qs = [float(q) for q in qs]
    grid = np.linspace(0.001, 0.999, n_grid)
    q_full = _interp_log_quantiles(np.sort(np.asarray(qpred, dtype=float), axis=1),
                                   qs, list(grid))
    # customer-hours are non-negative; a probit-slope lower tail can undershoot
    q_full = np.sort(np.clip(q_full, 0.0, None), axis=1)
    diff = observed[:, None] - q_full
    pinball = np.where(diff >= 0, diff * grid[None, :], diff * (grid[None, :] - 1.0))
    return float(np.mean(2.0 * np.trapezoid(pinball, grid, axis=1)
                         if hasattr(np, "trapezoid")
                         else 2.0 * np.trapz(pinball, grid, axis=1)))


def evaluate(pred: pd.DataFrame, train: pd.DataFrame, label: str,
             quantiles: list[float]) -> dict:
    from lifelines.utils import concordance_index
    from sklearn.metrics import (average_precision_score, brier_score_loss,
                                 log_loss, roc_auc_score)

    y = pred.event.astype(int).to_numpy()
    p = pred.probability.to_numpy()

    # Spec 7.3 baseline 1: climatology is the COUNTY's historical event rate,
    # not one number for the whole state. A global rate is the weaker reference
    # by a wide margin in a state whose county event rates vary by an order of
    # magnitude, so skill measured against it flatters the model.
    global_rate = float(train.event.mean())
    county_rate = train.groupby("fips").event.mean()
    clim_county = pred.fips.map(county_rate).fillna(global_rate).to_numpy()

    brier = float(brier_score_loss(y, p))
    brier_ref_global = float(np.mean((y - global_rate) ** 2))
    brier_ref_county = float(np.mean((y - clim_county) ** 2))
    metrics = {
        "split": label, "n_rows": len(pred), "n_events": int(y.sum()),
        "occurrence_brier": brier,
        "occurrence_brier_skill_vs_climatology": 1 - brier / brier_ref_county,
        "occurrence_brier_skill_vs_global_climatology": 1 - brier / brier_ref_global,
        "occurrence_brier_ref_county_climatology": brier_ref_county,
        "occurrence_brier_ref_global_climatology": brier_ref_global,
        "occurrence_average_precision": float(average_precision_score(y, p)),
        "occurrence_roc_auc": float(roc_auc_score(y, p)),
        "occurrence_log_loss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))),
        "threshold_gust20_brier": float(brier_score_loss(
            y, pred.gust_max.gt(20).astype(int))),
    }
    if "probability_insample_calibrated" in pred:
        # C1: kept alongside the honest number so the optimism is visible
        # rather than reported as the result.
        metrics["occurrence_brier_insample_calibration"] = float(brier_score_loss(
            y, pred.probability_insample_calibrated.to_numpy()))
    for name in ("reference_logistic_glm", "reference_climatology_county"):
        if name in pred:
            q = np.clip(pred[name].to_numpy(), 1e-6, 1 - 1e-6)
            metrics[f"{name}_brier"] = float(brier_score_loss(y, q))
            metrics[f"{name}_log_loss"] = float(log_loss(y, q))
            if len(np.unique(y)) > 1:
                metrics[f"{name}_roc_auc"] = float(roc_auc_score(y, q))
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

    # Spec 7.2: log score for the magnitude density. Computed from the quantile
    # function by finite difference -- f(y) = dtau/dq at the observed value.
    metrics["magnitude_log_score"] = _log_score_from_quantiles(
        observed, qpred, quantiles)
    if "reference_magnitude_glm" in event_pred:
        ref_mag = event_pred.reference_magnitude_glm.to_numpy()
        metrics["reference_magnitude_glm_mae"] = float(
            np.mean(np.abs(ref_mag - observed)))

    uncensored = event_pred[~event_pred.censored]
    if len(uncensored) > 1:
        metrics["duration_concordance"] = float(concordance_index(
            uncensored.restoration_hours, uncensored.duration_median))
        metrics["duration_median_mae"] = float(np.mean(np.abs(
            uncensored.restoration_hours - uncensored.duration_median)))
        # Spec 7.3 baseline 3: persistence is THE COUNTY's historical median
        # restoration time, with the global median only as a fallback for a
        # county with no observed training restorations.
        observed_train = train[train.event.eq(1) & ~train.censored]
        global_persist = float(observed_train.restoration_hours.median())
        county_persist = observed_train.groupby("fips").restoration_hours.median()
        persist = uncensored.fips.map(county_persist).fillna(global_persist).to_numpy()
        metrics["duration_persistence_mae"] = float(np.mean(np.abs(
            uncensored.restoration_hours.to_numpy() - persist)))
        metrics["duration_persistence_global_mae"] = float(np.mean(np.abs(
            uncensored.restoration_hours.to_numpy() - global_persist)))
        metrics["duration_quantile_calibration"] = _duration_calibration(
            uncensored)
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


def storm_groups(frame: pd.DataFrame, gap_days: int,
                 min_county_frac: float = 0.10) -> pd.Series:
    """Label each county-day with the storm episode it belongs to (spec 7.1).

    A storm day is one where at least `min_county_frac` of the reporting
    counties are in event -- NOT one where any single county is, which is what
    an unqualified `groupby("date").event.max()` gives you. That distinction is
    the whole fix: Michigan has ~83 reporting counties and a per-county event
    rate of a few percent, so "any county anywhere" is true on almost every
    calendar day. Consecutive days then never exceed the merge gap, the entire
    multi-year record fuses into ONE episode, and `GroupKFold(n_splits=1)`
    raises. Where it does survive, the "storms" span weeks, which blocks nothing
    and hands back exactly the meaningless score section 7.1 exists to prevent.

    Days outside any episode are their own singleton `quiet_` group, so a fold
    never splits a storm across train and test while quiet days stay poolable.
    """
    frame = frame.copy()
    dates = pd.to_datetime(frame.date)
    n_counties = max(frame.fips.nunique(), 1)
    share = frame.groupby(dates).event.mean().sort_index()
    stormy = share[share.ge(min_county_frac)].index
    log.info("storm blocking: %d/%d days have >=%.0f%% of %d counties in event",
             len(stormy), len(share), 100 * min_county_frac, n_counties)
    if len(stormy) == 0:
        log.warning("no day reaches the %.0f%% threshold, so every county-day "
                    "becomes its own quiet group and storm blocking is inert. "
                    "Lower storm_min_county_frac in phase2.yaml to the share a "
                    "real regional storm actually reaches.", 100 * min_county_frac)
    elif len(stormy) > 0.8 * len(share):
        log.warning("%.0f%% of all days clear the storm threshold -- blocking "
                    "will fuse most of the record into a few groups. Raise "
                    "storm_min_county_frac in phase2.yaml.",
                    100 * len(stormy) / len(share))

    # Contiguous runs of storm days, merged across gaps of <= gap_days.
    episodes: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for date in pd.to_datetime(stormy):
        if episodes and (date - episodes[-1][1]).days <= gap_days:
            episodes[-1] = (episodes[-1][0], date)
        else:
            episodes.append((date, date))

    # Assign each day to the NEAREST episode, if it is within gap_days of one.
    # Nearest rather than first-match, because padded windows of two episodes
    # that are just far enough apart to stay separate can still overlap in the
    # middle; merging them there would silently fuse distinct storms, and
    # taking whichever happens to come first is arbitrary.
    #
    # Done on the UNIQUE dates (a few thousand) and mapped back, not per row --
    # the real table is ~151k rows and the python loop this replaces was
    # O(rows x episodes).
    values = np.array([f"quiet_{d:%Y%m%d}" for d in dates], dtype=object)
    n_episodes = len(episodes)
    if episodes:
        pad = np.timedelta64(gap_days, "D")
        starts = np.array([e[0] for e in episodes], dtype="datetime64[ns]")
        ends = np.array([e[1] for e in episodes], dtype="datetime64[ns]")
        uniq = np.unique(dates.to_numpy().astype("datetime64[ns]"))
        # candidate = the episode starting at or before each date, and the next
        after = np.searchsorted(starts, uniq, side="right")
        assigned = np.full(len(uniq), -1, dtype=int)
        best = np.full(len(uniq), np.timedelta64(np.iinfo(np.int64).max, "ns"))
        for cand in (after - 1, after):
            ok = (cand >= 0) & (cand < len(starts))
            idx = np.clip(cand, 0, len(starts) - 1)
            dist = np.maximum(starts[idx] - uniq, uniq - ends[idx])
            dist = np.maximum(dist, np.timedelta64(0, "ns"))
            take = ok & (dist <= pad) & (dist < best)
            assigned[take] = idx[take]
            best[take] = dist[take]
        lookup = {pd.Timestamp(d): i for d, i in zip(uniq, assigned) if i >= 0}
        hit = np.array([lookup.get(d, -1) for d in dates])
        inside = hit >= 0
        values[inside] = [f"storm_{i:04d}" for i in hit[inside]]
    labels = pd.Series(values, index=frame.index, dtype=object)
    log.info("storm blocking: %d episodes covering %d/%d county-days",
             n_episodes, int(labels.str.startswith("storm_").sum()), len(labels))
    return labels


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

    storm = storm_groups(frame, int(cfg.get("storm_gap_days", 2)),
                         float(cfg.get("storm_min_county_frac", 0.10)))
    n_groups = storm.nunique()
    if n_groups < 2:
        # Never construct GroupKFold(n_splits=1) -- sklearn raises, and it would
        # raise here, at the end of a multi-hour job. Skip loudly instead.
        log.error("storm-blocked CV skipped: only %d storm group(s). Lower "
                  "storm_min_county_frac in phase2.yaml -- at the current "
                  "threshold every day is a storm day and blocking is inert.",
                  n_groups)
    else:
        run_scheme("storm_blocked", storm, GroupKFold(n_splits=min(5, n_groups)))
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


def require_lazy_imports(cfg: Config) -> None:
    """Fail in the first second, not after the model fits.

    Every package below is imported lazily somewhere downstream of the fitting
    -- `properscoring` inside evaluate(), `statsmodels` inside the reference
    models, `ngboost` inside fit_magnitude. A node missing one used to surface
    it hours in, with a model already dumped and no metrics beside it.
    """
    import importlib.util as iu

    need = {"properscoring": "CRPS reporting",
            "lifelines": "Weibull AFT and Cox duration models"}
    if bool(cfg.get("fit_reference_models", True)):
        need["statsmodels"] = "logistic and negative-binomial GLM references (spec 6.1, 6.2)"
    if cfg.get("magnitude_model", "ngboost") == "ngboost":
        need["ngboost"] = "the configured magnitude model"
    missing = {m: why for m, why in need.items() if iu.find_spec(m) is None}
    if missing:
        raise SystemExit(
            "Missing packages this run imports lazily, after the expensive "
            "fitting:\n" + "\n".join(f"  {m:<14} {why}" for m, why in missing.items())
            + "\nInstall them (env/requirements.txt) and re-run `make doctor-phase2`.")


def config_digest(cfg: Config) -> str:
    content = {k: v for k, v in cfg.items() if not k.startswith("_")}
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def fit_and_validate(frame: pd.DataFrame, cfg: Config) -> None:
    import joblib

    require_lazy_imports(cfg)
    test_year = pd.Timestamp(cfg["test_start"]).year
    split = masks(frame, cfg)
    if split["test"].any():
        raise SystemExit(
            f"Training input contains {test_year} rows. `make phase2-build` "
            "writes the validation-scope table; the test-scope table lives in "
            "phase2_merged_test.parquet and is only read by --evaluate-test.")
    feats = feature_columns(frame)
    label = f"validation_{pd.Timestamp(cfg['val_start']).year}"

    with timed("phase2_model_fit", log):
        occurrence, calibrator, oof_val = fit_occurrence(frame, feats, split, cfg)
        magnitude = fit_magnitude(frame, feats, split, cfg)
        duration = fit_duration(frame, feats, split, cfg)
        references = (fit_reference_models(frame, feats, split, cfg)
                      if bool(cfg.get("fit_reference_models", True)) else {})

    train = frame[split["train"]]
    bundle = {
        "occurrence": occurrence, "calibrator": calibrator,
        "magnitude": magnitude, "duration": duration, "features": feats,
        "references": references,
        "config_sha256": config_digest(cfg),
        "trained_through": cfg["val_end"],
        # C3: the cohort is part of the frozen artifact. If the test-year build
        # produces a different reporting county set, the frozen model is being
        # scored against a different study population -- and config_sha256
        # cannot see that, because it hashes config and not data.
        "reporting_counties": sorted(frame.fips.astype(str).unique()),
        "county_climatology": train.groupby("fips").event.mean().to_dict(),
        "global_climatology": float(train.event.mean()),
        "magnitude_model_actually_used": magnitude["kind"],
    }
    if magnitude["kind"] != cfg.get("magnitude_model", "ngboost"):
        log.warning("magnitude model requested %r but %r was used -- say so in "
                    "the write-up", cfg.get("magnitude_model"), magnitude["kind"])

    # Cross-validation BEFORE the artifact writes. It used to run last, so a CV
    # failure left a model, predictions, metrics and a figure on disk with no
    # phase2_cv_metrics.csv -- the one file the runbook says to review before
    # opening the test year.
    with timed("phase2_cross_validation", log):
        cv = occurrence_cv(frame[split["train"] | split["val"]].reset_index(drop=True),
                           feats, cfg)

    prediction = predict(bundle, frame)
    val_pred = prediction[split["val"].to_numpy()].copy()
    # C1: report out-of-fold calibrated probabilities for the validation year,
    # keeping the in-sample number beside them so the optimism stays visible.
    val_pred["probability_insample_calibrated"] = val_pred.probability.to_numpy()
    val_pred["probability"] = oof_val.to_numpy()

    metrics = evaluate(val_pred, train, label, magnitude["quantiles"])
    metrics["calibration_note"] = (
        "occurrence probabilities are out-of-fold within the validation year; "
        "occurrence_brier_insample_calibration is the same calibrator scored on "
        "the rows it was fitted on, kept only to show the optimism")
    metrics["magnitude_model_actually_used"] = magnitude["kind"]
    metrics["reporting_counties"] = len(bundle["reporting_counties"])

    joblib.dump(bundle, MODEL_PATH)
    prediction.to_parquet(PRED_PATH, index=False)
    METRIC_PATH.write_text(json.dumps(metrics, indent=2))
    cv.to_csv(CV_PATH, index=False)
    plot_validation(val_pred, label)

    log.info("validation BSS %.3f (vs county climatology), AP %.3f, AUC %.3f, "
             "magnitude CRPSS %.3f",
             metrics["occurrence_brier_skill_vs_climatology"],
             metrics["occurrence_average_precision"], metrics["occurrence_roc_auc"],
             metrics["magnitude_crps_skill_vs_climatology"])
    log.info("CV schemes recorded: %s",
             cv.scheme.value_counts().to_dict() if len(cv) else "NONE")


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
    test_year = pd.Timestamp(cfg["test_start"]).year
    split = masks(frame, cfg)
    test = frame[split["test"]].copy()
    if test.empty:
        raise SystemExit(f"No {test_year} rows; first run `make phase2-build-test`")

    # C3: the study population must be the one the model was fitted on.
    frozen = set(bundle.get("reporting_counties") or [])
    rebuilt = set(frame.fips.astype(str).unique())
    if frozen and frozen != rebuilt:
        raise SystemExit(
            "Reporting-county cohort changed between the frozen model and the "
            f"test build.\n  only in frozen model: {sorted(frozen - rebuilt)}\n"
            f"  only in test build:   {sorted(rebuilt - frozen)}\n"
            "Scoring a frozen model against a different study population is not "
            "a held-out test. Rebuild the test table with the same cohort, or "
            "refit and re-freeze -- and say in the write-up which you did.")

    prediction = predict(bundle, test)
    train = frame[split["train"]]
    metrics = evaluate(prediction, train, f"test_{test_year}",
                       bundle["magnitude"]["quantiles"])
    metrics["magnitude_model_actually_used"] = bundle.get(
        "magnitude_model_actually_used", bundle["magnitude"]["kind"])

    prediction.to_parquet(PATHS.processed / "phase2_test_predictions.parquet", index=False)
    (PATHS.processed / "phase2_test_metrics.json").write_text(json.dumps(metrics, indent=2))
    plot_validation(prediction, f"test_{test_year}")
    # O2: the marker is written only once the scoring has actually succeeded.
    # Writing it first meant a crash in predict() burned the single attempt and
    # left --force-test as the only way forward.
    TEST_MARKER.write_text(
        f"Test year {test_year} opened once at {pd.Timestamp.now(tz='UTC')} UTC\n"
        f"config_sha256={bundle['config_sha256']}\n"
        f"reporting_counties={len(rebuilt)}\n")
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
    set_phase(2)
    if args.evaluate_test and TEST_MARKER.exists() and not args.force_test:
        raise SystemExit(f"{TEST_MARKER} exists: test year was already opened. "
                         "Refusing to read the full table again.")
    # O4: the two scopes are separate files, so the final-test build no longer
    # destroys the validation-scope table and force a 72-file ERA5 rebuild.
    name = "phase2_merged_test.parquet" if args.evaluate_test else "phase2_merged.parquet"
    path = PATHS.processed / name
    if not path.exists():
        raise SystemExit(
            f"{path} missing -- run "
            f"`{'make phase2-build-test' if args.evaluate_test else 'make phase2-build'}` first")
    frame = pd.read_parquet(path)
    if args.evaluate_test:
        evaluate_test(frame, cfg, force=args.force_test)
    else:
        fit_and_validate(frame, cfg)


if __name__ == "__main__":
    main()
