#!/usr/bin/env python
"""phase2ML: a like-for-like machine-learning bake-off against the Phase 2 model.

    python src/phase2ML_train.py                     # all three heads
    python src/phase2ML_train.py --heads occurrence  # one head
    python src/phase2ML_train.py --quick             # tiny budgets, for tests

What this does and does not do
------------------------------
It fits every available learner in ``phase2ML_models`` on the SAME merged
table, the SAME frozen train/validation split, the SAME feature block and the
SAME storm-blocked grouping the incumbent Phase 2 hurdle model uses, scores
them with the incumbent's own metric primitives, and writes a leaderboard.

It never reads the held-out test year, never refits or overwrites the frozen
incumbent bundle, and never selects a champion on the validation period. The
champion is chosen inside the training window by storm-blocked cross-
validation; validation-period numbers are reported for every learner so the
selection can be audited, not so it can be made.

Two numbers on the leaderboard deserve to be read before any other:

``incumbent_reproduced``    the harness refits the incumbent's exact
                            configuration as an ordinary roster row. If that
                            row does not land on the published incumbent score,
                            phase2ML differs from the Phase 2 pipeline
                            somewhere and no other row can be trusted yet.
``selection_optimism``      the gap between the champion's train-CV score and
                            its validation score. With a roster this size the
                            best validation score is optimistic by construction;
                            quoting it as the ML result is the standard way a
                            bake-off overstates itself.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.common.config import ROOT
from src.common.logio import get_logger, timed
from src.phase2ML_common import (
    DURATION_QUANTILES,
    LEADERBOARD_PATH,
    METRICS_PATH,
    ML_MODEL_DIR,
    PRED_DIR,
    TUNING_PATH,
    cohort_matches_incumbent,
    config_digest,
    crps_from_quantiles,
    design_matrix,
    duration_frame,
    ensure_dirs,
    evaluate,
    feature_columns,
    incumbent_metrics,
    isotonic_calibrate,
    load_ml_config,
    load_training_frame,
    masks,
    period_slug,
    tuning_folds,
    write_json,
)
from src.phase2ML_models import Learner, roster
from src.phase2_train import _duration_calibration, _log_score_from_quantiles

log = get_logger("phase2ML.train")

# Higher is better for these, lower is better for everything else.
HIGHER_IS_BETTER = {"occurrence": True, "magnitude": False, "duration": True}
SELECTION_METRIC = {"occurrence": "average_precision",
                    "magnitude": "crps",
                    "duration": "concordance"}


# =============================================================================
# Feature blocks
# =============================================================================

def head_design(frame: pd.DataFrame, feats: list[str], head: str,
                fill: dict[str, float] | None):
    """The feature matrix each head is entitled to see.

    Occurrence and magnitude get the standard feature block. Duration gets the
    incumbent's extended block -- the same features plus ``peak_frac_out`` and
    ``concurrent_state_load`` -- because at restoration time those are known:
    the outage has already happened and its peak is observed. Giving the
    occurrence model those columns would be leakage; withholding them from the
    duration model would handicap it relative to the incumbent it is being
    compared against. Both mistakes are easy and both invalidate the table.
    """
    if head == "duration":
        numeric = [*feats, "peak_frac_out", "concurrent_state_load"]
        return duration_frame(frame, numeric, fill)
    return design_matrix(frame, feats, fill)


# =============================================================================
# Per-head scorers -- the incumbent's primitives, nothing re-derived
# =============================================================================

def score_occurrence(y: np.ndarray, p: np.ndarray, clim: np.ndarray) -> dict:
    from sklearn.metrics import (
        average_precision_score, brier_score_loss, log_loss, roc_auc_score)

    brier = float(brier_score_loss(y, p))
    ref = float(np.mean((y - clim) ** 2))
    return {
        "brier": brier,
        "brier_skill_vs_county_climatology": 1 - brier / ref if ref > 0 else np.nan,
        "average_precision": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan,
        "log_loss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))),
    }


def score_magnitude(observed: np.ndarray, qpred: np.ndarray,
                    quantiles: list[float], train_mag: np.ndarray) -> dict:
    crps = crps_from_quantiles(observed, qpred, quantiles)
    climate = np.tile(np.quantile(train_mag, quantiles), (len(observed), 1))
    ref = crps_from_quantiles(observed, climate, quantiles)
    median = qpred[:, len(quantiles) // 2]
    rmse = float(np.sqrt(np.mean((median - observed) ** 2)))
    width = (qpred[:, -2] - qpred[:, 1]) if qpred.shape[1] >= 4 else (
        qpred[:, -1] - qpred[:, 0])
    pit = np.mean(qpred <= observed[:, None], axis=1)
    return {
        "crps": crps,
        "crps_skill_vs_climatology": 1 - crps / ref if ref > 0 else np.nan,
        "median_rmse": rmse,
        "spread_skill_ratio": float(np.mean(width) / max(rmse, 1e-9)),
        "log_score": _log_score_from_quantiles(observed, qpred, quantiles),
        "pit_histogram_10bin": np.histogram(pit, bins=np.linspace(0, 1, 11))[0]
                                 .astype(int).tolist(),
    }


def score_duration(frame: pd.DataFrame, qpred: np.ndarray,
                   train_events: pd.DataFrame) -> dict:
    from lifelines.utils import concordance_index

    median = qpred[:, list(DURATION_QUANTILES).index(0.50)]
    scored = frame.assign(duration_median=median)
    for idx, q in enumerate(DURATION_QUANTILES):
        scored[f"duration_q{int(round(q * 100)):02d}"] = qpred[:, idx]
    uncensored = scored[~scored.censored.astype(bool)]
    if len(uncensored) < 2:
        return {"concordance": np.nan, "median_mae": np.nan,
                "n_uncensored": int(len(uncensored))}
    observed_train = train_events[~train_events.censored.astype(bool)]
    county = observed_train.groupby("fips").restoration_hours.median()
    fallback = float(observed_train.restoration_hours.median())
    persist = uncensored.fips.map(county).fillna(fallback).to_numpy()
    truth = uncensored.restoration_hours.to_numpy()
    return {
        "concordance": float(concordance_index(truth,
                                               uncensored.duration_median)),
        "median_mae": float(np.mean(np.abs(truth - uncensored.duration_median))),
        "persistence_mae": float(np.mean(np.abs(truth - persist))),
        "quantile_calibration": _duration_calibration(uncensored),
        "n_uncensored": int(len(uncensored)),
        "censored_fraction": float(scored.censored.astype(bool).mean()),
    }


# =============================================================================
# Fitting one learner, once
# =============================================================================

def positive_weight(y: np.ndarray) -> np.ndarray:
    """The incumbent's scale_pos_weight, expressed as a sample weight.

    LightGBM's ``scale_pos_weight = neg/pos`` and a per-row sample weight of
    neg/pos on the positive class are the same reweighting; expressing it as a
    sample weight is the only form every other learner in the roster accepts,
    so the imbalance correction is identical across the leaderboard instead of
    being whatever each library's own knob happens to mean.
    """
    y = np.asarray(y).astype(int)
    weight = float((1 - y).sum() / max(y.sum(), 1))
    return np.where(y == 1, weight, 1.0)


def fit_occurrence_model(model, supports_weight: bool, Xtr, ytr, Xva):
    """Fit one occurrence estimator and return it with raw validation probabilities."""
    ytr = np.asarray(ytr).astype(int)
    if supports_weight:
        try:
            model.fit(Xtr, ytr, **_weight_kwargs(model, positive_weight(ytr)))
        except TypeError:
            log.warning("estimator rejected sample_weight; fitting unweighted")
            model.fit(Xtr, ytr)
    else:
        model.fit(Xtr, ytr)
    return model, np.asarray(model.predict_proba(Xva))[:, 1]


def _weight_kwargs(model, weights) -> dict:
    """sklearn Pipelines need the weight addressed to the final step."""
    from sklearn.pipeline import Pipeline
    if isinstance(model, Pipeline):
        return {f"{model.steps[-1][0]}__sample_weight": weights}
    return {"sample_weight": weights}


# =============================================================================
# Cross-validation and tuning -- inside the training window only
# =============================================================================

def candidates(grid: dict, limit: int, tune_enabled: bool) -> list[dict]:
    """Hyperparameter candidates, always including the default recipe."""
    if not grid or not tune_enabled:
        return [{}]
    keys = list(grid)
    combos = [dict(zip(keys, values))
              for values in itertools.product(*(grid[k] for k in keys))]
    if len(combos) > limit:
        # Deterministic thinning, not a random sample: the same budget has to
        # select the same candidates on a rerun, or the leaderboard is not
        # reproducible from the config alone.
        step = len(combos) / limit
        combos = [combos[int(i * step)] for i in range(limit)]
    return [{}] + [c for c in combos if c]


def cross_validate(head: str, learner: Learner, cfg, seed, train, folds, feats,
                   fill, quantiles) -> tuple[dict, float, list[dict]]:
    """Score every candidate on storm-blocked TRAINING folds.

    This runs even when a learner has no grid, because the returned score is
    what selects the champion. Selecting on the validation period instead would
    make the reported validation metrics a best-of-N statistic rather than an
    out-of-sample one -- with a roster this size that is a difference of real
    size, not a technicality.
    """
    combos = candidates(learner.grid, int(cfg.get("tuning_max_candidates", 6)),
                        bool(cfg.get("tune", True)))
    rows = []
    for params in combos:
        scores = [s for s in (
            _fold_score(head, learner, cfg, seed, train, fit_idx, hold_idx,
                        feats, fill, quantiles, params)
            for fit_idx, hold_idx in folds) if s is not None and np.isfinite(s)]
        if not scores:
            continue
        rows.append({"head": head, "learner": learner.name,
                     "params": json.dumps(params, default=str),
                     "cv_score": float(np.mean(scores)),
                     "cv_score_std": float(np.std(scores)),
                     "n_folds": len(scores)})
    if not rows:
        log.warning("%s/%s: every training fold failed; no CV score", head,
                    learner.name)
        return {}, float("nan"), []
    better = max if HIGHER_IS_BETTER[head] else min
    best = better(rows, key=lambda r: r["cv_score"])
    log.info("%s/%s: %d candidate(s), best train-CV %s=%.5f%s", head,
             learner.name, len(rows), SELECTION_METRIC[head], best["cv_score"],
             f" with {best['params']}" if best["params"] != "{}" else "")
    return json.loads(best["params"]), best["cv_score"], rows


def _set_params(model, params: dict):
    """Apply a candidate's hyperparameters to whatever kind of model this is."""
    if not params:
        return model
    if hasattr(model, "set_params"):
        try:
            return model.set_params(**params)
        except (ValueError, TypeError):
            pass
    for key, value in params.items():
        if hasattr(model, key):
            setattr(model, key, value)
    return model


def _fold_score(head, learner, cfg, seed, train, fit_idx, hold_idx, feats,
                fill, quantiles, params):
    fit_rows = train.iloc[fit_idx]
    hold_rows = train.iloc[hold_idx]
    try:
        if head == "occurrence":
            y = fit_rows.event.astype(int).to_numpy()
            if len(np.unique(y)) < 2 or hold_rows.event.nunique() < 2:
                return None
            Xf, _ = head_design(fit_rows, feats, head, fill)
            Xh, _ = head_design(hold_rows, feats, head, fill)
            model = _set_params(learner.build(cfg, seed), params)
            _, p = fit_occurrence_model(model, learner.supports_sample_weight,
                                        Xf, y, Xh)
            from sklearn.metrics import average_precision_score
            # Ranking, not calibration: isotonic calibration is fitted later on
            # the validation window and is the same for every learner, so
            # tuning on a calibration-sensitive score would rank the learners
            # by how raw their probabilities happen to be.
            return float(average_precision_score(
                hold_rows.event.astype(int).to_numpy(), p))

        if head == "magnitude":
            fit_ev = fit_rows[fit_rows.event.eq(1)]
            hold_ev = hold_rows[hold_rows.event.eq(1)]
            if len(fit_ev) < 50 or len(hold_ev) < 10:
                return None
            Xf, _ = head_design(fit_ev, feats, head, fill)
            Xh, _ = head_design(hold_ev, feats, head, fill)
            model = _set_params(learner.build(cfg, seed), params)
            model.fit(Xf, np.log1p(fit_ev.customer_hours.to_numpy()))
            qpred = np.expm1(np.sort(
                model.predict_log_quantiles(Xh, quantiles), axis=1)).clip(min=0)
            return crps_from_quantiles(hold_ev.customer_hours.to_numpy(),
                                       qpred, quantiles)

        fit_ev = fit_rows[fit_rows.event.eq(1)]
        hold_ev = hold_rows[hold_rows.event.eq(1) & ~hold_rows.censored.astype(bool)]
        if len(fit_ev) < 50 or len(hold_ev) < 10:
            return None
        Xf, fold_fill = head_design(fit_ev, feats, head, fill)
        Xh, _ = head_design(hold_ev, feats, head, fold_fill)
        Xh = Xh.reindex(columns=Xf.columns, fill_value=0.0)
        model = _set_params(learner.build(cfg, seed), params)
        model.fit(Xf,
                  np.clip(fit_ev.restoration_hours.fillna(0.5).to_numpy(), 0.5, None),
                  ~fit_ev.censored.astype(bool).to_numpy())
        qpred = model.predict_quantiles(Xh, list(DURATION_QUANTILES))
        from lifelines.utils import concordance_index
        return float(concordance_index(
            hold_ev.restoration_hours.to_numpy(),
            qpred[:, list(DURATION_QUANTILES).index(0.50)]))
    except Exception as err:
        log.warning("%s/%s: training fold failed (%s)", head, learner.name, err)
        return None


# =============================================================================
# One head, every learner
# =============================================================================

def _align(reference: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
    """Give `other` exactly the columns of `reference`, in the same order.

    duration_frame one-hot encodes the regime label, and a regime that occurs
    in the training window but not in the validation window (or the reverse)
    changes the column set. Fitting on one set and predicting on another is a
    silent shape mismatch in the best case and a silently permuted design
    matrix in the worst.
    """
    return other.reindex(columns=reference.columns, fill_value=0.0)


def run_head(head: str, frame: pd.DataFrame, cfg, feats: list[str], split,
             folds, train_cv: pd.DataFrame, seed: int,
             only: list[str] | None) -> tuple[list[dict], dict, list[dict]]:
    quantiles = [float(q) for q in cfg["quantiles"]]
    train = frame[split["train"]]
    val = frame[split["val"]]
    if head in ("magnitude", "duration"):
        train_fit = train[train.event.eq(1)]
        val_fit = val[val.event.eq(1)]
    else:
        train_fit, val_fit = train, val

    if head == "magnitude" and len(train_fit) < 50:
        raise SystemExit(f"only {len(train_fit)} training events for magnitude")
    if head == "duration" and len(train_fit) < 50:
        raise SystemExit(f"only {len(train_fit)} training events for duration")

    Xtr, fill = head_design(train_fit, feats, head, None)
    Xva = _align(Xtr, head_design(val_fit, feats, head, fill)[0])
    log.info("%s: %d training rows, %d validation rows, %d columns",
             head, len(Xtr), len(Xva), Xtr.shape[1])

    county_rate = train.groupby("fips").event.mean()
    global_rate = float(train.event.mean())
    clim = val.fips.map(county_rate).fillna(global_rate).to_numpy()
    y_val = val.event.astype(int).to_numpy()
    train_mag = train.loc[train.event.eq(1), "customer_hours"].to_numpy()

    rows: list[dict] = []
    tuning_rows: list[dict] = []
    fitted: dict[str, object] = {}
    predictions: dict[str, np.ndarray] = {}

    for learner in roster(head, only):
        from src.phase2ML_models import missing as missing_packages
        gaps = missing_packages(learner.requires)
        if gaps:
            log.warning("%s/%s: SKIPPED, missing %s", head, learner.name, gaps)
            rows.append({"head": head, "learner": learner.name,
                         "status": "SKIPPED", "missing_packages": ",".join(gaps),
                         "family": learner.family, "note": learner.note})
            continue

        started = time.monotonic()
        try:
            params, cv_score, candidate_rows = cross_validate(
                head, learner, cfg, seed, train_cv, folds, feats, fill, quantiles)
            tuning_rows.extend(candidate_rows)
            model = _set_params(learner.build(cfg, seed), params)
            row = {"head": head, "learner": learner.name, "status": "OK",
                   "family": learner.family, "is_incumbent": learner.is_incumbent,
                   "params": json.dumps(params, default=str),
                   "train_cv_score": cv_score,
                   "selection_metric": SELECTION_METRIC[head],
                   "note": learner.note}

            if head == "occurrence":
                model, raw = fit_occurrence_model(
                    model, learner.supports_sample_weight, Xtr,
                    train_fit.event.astype(int).to_numpy(), Xva)
                calibrator, oof = isotonic_calibrate(
                    raw, y_val, int(cfg.get("calibration_cv_folds", 5)), seed)
                row.update(score_occurrence(y_val, oof, clim))
                row["brier_insample_calibration"] = score_occurrence(
                    y_val, calibrator.predict(raw), clim)["brier"]
                row["class_weighting"] = ("sample_weight"
                                          if learner.supports_sample_weight
                                          else "NONE (unsupported)")
                fitted[learner.name] = {"model": model, "calibrator": calibrator,
                                        "columns": list(Xtr.columns), "fill": fill}
                predictions[learner.name] = oof

            elif head == "magnitude":
                model.fit(Xtr, np.log1p(train_fit.customer_hours.to_numpy()))
                qpred = np.expm1(np.sort(
                    model.predict_log_quantiles(Xva, quantiles), axis=1)).clip(min=0)
                row.update(score_magnitude(val_fit.customer_hours.to_numpy(),
                                           qpred, quantiles, train_mag))
                row["distribution_route"] = getattr(model, "route", learner.route)
                fitted[learner.name] = {"model": model, "columns": list(Xtr.columns),
                                        "fill": fill, "quantiles": quantiles}
                predictions[learner.name] = qpred

            else:
                model.fit(
                    Xtr,
                    np.clip(train_fit.restoration_hours.fillna(0.5).to_numpy(), 0.5, None),
                    ~train_fit.censored.astype(bool).to_numpy())
                qpred = model.predict_quantiles(Xva, list(DURATION_QUANTILES))
                row.update(score_duration(val_fit, qpred, train_fit))
                row["censoring_handling"] = learner.censoring
                row["survival_curve_truncated_fraction"] = float(
                    getattr(model, "truncated_fraction", 0.0))
                fitted[learner.name] = {"model": model, "columns": list(Xtr.columns),
                                        "fill": fill}
                predictions[learner.name] = qpred

            row["fit_seconds"] = round(time.monotonic() - started, 2)
            rows.append(row)
            log.info("%s/%s: %s=%.5f (validation) in %.1fs", head, learner.name,
                     SELECTION_METRIC[head],
                     row.get(SELECTION_METRIC[head], float("nan")),
                     row["fit_seconds"])
        except Exception as err:
            log.warning("%s/%s: FAILED (%s: %s)", head, learner.name,
                        type(err).__name__, err)
            rows.append({"head": head, "learner": learner.name, "status": "FAILED",
                         "family": learner.family, "error": f"{type(err).__name__}: {err}",
                         "fit_seconds": round(time.monotonic() - started, 2),
                         "note": learner.note})

    for name, prediction in predictions.items():
        frame_out = val_fit[["fips", "date", "event", "customer_hours",
                             "restoration_hours", "censored"]].reset_index(drop=True)
        block = np.asarray(prediction)
        if block.ndim == 1:
            frame_out["prediction"] = block
        else:
            labels = (quantiles if head == "magnitude" else list(DURATION_QUANTILES))
            for idx, q in enumerate(labels):
                frame_out[f"q{int(round(q * 100)):02d}"] = block[:, idx]
        frame_out.to_parquet(PRED_DIR / f"{head}_{name}.parquet", index=False)

    return rows, fitted, tuning_rows


def pick_champion(rows: list[dict], head: str) -> tuple[str | None, str]:
    """Champion by TRAIN-CV score; validation is only a fallback, and says so."""
    usable = [r for r in rows if r.get("status") == "OK"
              and np.isfinite(r.get("train_cv_score", np.nan))]
    better = max if HIGHER_IS_BETTER[head] else min
    if usable:
        best = better(usable, key=lambda r: r["train_cv_score"])
        return best["learner"], "train_cv"
    usable = [r for r in rows if r.get("status") == "OK"
              and np.isfinite(r.get(SELECTION_METRIC[head], np.nan))]
    if not usable:
        return None, "none"
    best = better(usable, key=lambda r: r[SELECTION_METRIC[head]])
    log.warning("%s: no training-CV score available, so the champion was picked "
                "on the VALIDATION period. Its validation metrics are a "
                "best-of-%d statistic, not an out-of-sample one -- say so.",
                head, len(usable))
    return best["learner"], "validation_FALLBACK"


# =============================================================================
# The champion trio, scored with the incumbent's own evaluate()
# =============================================================================

def predict_full(head: str, entry: dict, rows: pd.DataFrame, feats: list[str],
                 quantiles: list[float]) -> np.ndarray:
    """Apply a fitted head to EVERY row of a frame (not just the event rows)."""
    design = _align(pd.DataFrame(columns=entry["columns"]),
                    head_design(rows, feats, head, entry["fill"])[0])
    if head == "occurrence":
        return np.asarray(entry["model"].predict_proba(design))[:, 1]
    if head == "magnitude":
        return np.expm1(np.sort(
            entry["model"].predict_log_quantiles(design, quantiles), axis=1)).clip(min=0)
    return entry["model"].predict_quantiles(design, list(DURATION_QUANTILES))


def composite_metrics(frame: pd.DataFrame, cfg, feats: list[str], split,
                      champions: dict, fitted: dict) -> dict:
    """Score the three champions together, with the incumbent's evaluate().

    Per-head scores answer "which learner is best at this head". They do not
    answer "is the ML trio better than the shipped model", because the
    incumbent's published metrics come from one joint evaluation over one
    prediction frame. Running that same function over the champions' joint
    predictions is the only row in the leaderboard that is directly comparable
    to phase2_validation_metrics.json, number for number.
    """
    needed = ("occurrence", "magnitude", "duration")
    if any(champions.get(h) is None for h in needed):
        return {"skipped": "composite needs a champion for all three heads; "
                           f"have {[h for h in needed if champions.get(h)]}"}

    quantiles = [float(q) for q in cfg["quantiles"]]
    train = frame[split["train"]]
    val = frame[split["val"]].reset_index(drop=True)
    label = f"phase2ml_validation_{period_slug(cfg['val_start'], cfg['val_end'])}"

    occ_entry = fitted["occurrence"][champions["occurrence"]]
    out = val[["fips", "date", "event", "customer_hours", "restoration_hours",
               "censored", "regime_label", "customer_density", "gust_max"]].copy()
    out["probability_raw"] = predict_full("occurrence", occ_entry, val, feats, quantiles)
    # The honest number, exactly as the incumbent reports it: out-of-fold
    # calibrated probabilities, with the in-sample value kept beside them.
    out["probability"] = occ_entry["oof"]
    out["probability_insample_calibrated"] = occ_entry["calibrator"].predict(
        out.probability_raw.to_numpy())

    qpred = predict_full("magnitude", fitted["magnitude"][champions["magnitude"]],
                         val, feats, quantiles)
    for idx, q in enumerate(quantiles):
        out[f"magnitude_q{int(round(q * 100)):02d}"] = qpred[:, idx]

    dur = predict_full("duration", fitted["duration"][champions["duration"]],
                       val, feats, quantiles)
    out["duration_median"] = dur[:, list(DURATION_QUANTILES).index(0.50)]
    for idx, q in enumerate(DURATION_QUANTILES):
        out[f"duration_q{int(round(q * 100)):02d}"] = dur[:, idx]

    county_rate = train.groupby("fips").event.mean()
    out["reference_climatology_county"] = val.fips.map(county_rate).fillna(
        float(train.event.mean())).to_numpy()

    out.to_parquet(PRED_DIR / "champion_trio_validation.parquet", index=False)
    metrics = evaluate(out, train, label, quantiles)
    metrics["champions"] = dict(champions)
    metrics["calibration_note"] = (
        "occurrence probabilities are out-of-fold within the validation year, "
        "the same discipline the incumbent reports")
    return metrics


def integrity(rows: list[dict], champions: dict, cohort_note: str,
              cohort_ok: bool, composite: dict) -> dict:
    """The checks that decide whether the leaderboard may be quoted at all."""
    incumbent = incumbent_metrics()
    checks: dict = {"cohort_vs_incumbent": {"passed": cohort_ok, "detail": cohort_note}}

    harness = next((r for r in rows if r["head"] == "occurrence"
                    and r.get("is_incumbent") and r.get("status") == "OK"), None)
    published = incumbent.get("occurrence_brier")
    if harness is None or published is None:
        checks["incumbent_reproduced"] = {
            "passed": None,
            "detail": ("no published incumbent Brier on this machine (run "
                       "`make phase2-train` first)" if published is None else
                       "the incumbent configuration did not fit in this harness"),
        }
    else:
        delta = float(harness["brier"] - published)
        tolerance = max(0.05 * published, 1e-4)
        checks["incumbent_reproduced"] = {
            "passed": bool(abs(delta) <= tolerance),
            "harness_brier": harness["brier"], "published_brier": published,
            "delta": delta, "tolerance": tolerance,
            "detail": ("the harness refit of the incumbent configuration lands "
                       "within tolerance of the published number, so the other "
                       "rows are measured on the same footing"
                       if abs(delta) <= tolerance else
                       "the harness refit of the incumbent configuration does "
                       "NOT reproduce the published number. Something differs "
                       "between phase2ML and phase2_train -- resolve that "
                       "before quoting any row of this leaderboard."),
        }

    optimism = {}
    for head, champion in champions.items():
        row = next((r for r in rows if r["head"] == head
                    and r["learner"] == champion), None)
        if not row or not np.isfinite(row.get("train_cv_score", np.nan)):
            continue
        metric = SELECTION_METRIC[head]
        if metric not in row or not np.isfinite(row.get(metric, np.nan)):
            continue
        gap = float(row[metric] - row["train_cv_score"])
        if not HIGHER_IS_BETTER[head]:
            gap = -gap
        optimism[head] = {
            "metric": metric, "train_cv": row["train_cv_score"],
            "validation": row[metric], "validation_better_by": gap,
        }
    checks["selection_optimism"] = {
        "per_head": optimism,
        "detail": ("the champion was chosen on training-window folds, so its "
                   "validation score is out-of-sample; a large positive gap "
                   "means the training folds were the harder problem, not that "
                   "the model improved"),
    }
    n_ok = sum(1 for r in rows if r.get("status") == "OK")
    checks["multiple_comparisons"] = {
        "learners_fitted": n_ok,
        "detail": (f"{n_ok} learner-head fits were scored on the same "
                   "validation period. The BEST validation score in that set is "
                   "optimistically biased even though the champion was not "
                   "selected on it; report the champion's number, not the "
                   "table's minimum, as the ML result."),
    }
    if composite and "skipped" not in composite and incumbent:
        pairs = {}
        for key in ("occurrence_brier", "occurrence_average_precision",
                    "occurrence_roc_auc", "magnitude_crps",
                    "magnitude_crps_skill_vs_climatology",
                    "occurrence_brier_skill_vs_climatology",
                    "duration_concordance", "duration_median_mae"):
            if key in composite and key in incumbent:
                pairs[key] = {"phase2ml": composite[key],
                              "incumbent": incumbent[key],
                              "difference": float(composite[key] - incumbent[key])}
        checks["head_to_head"] = pairs
    return checks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "config" / "region.yaml"))
    ap.add_argument("--phase2", default=str(ROOT / "config" / "phase2.yaml"))
    ap.add_argument("--phase2ml", default=str(ROOT / "config" / "phase2ml.yaml"))
    ap.add_argument("--heads", nargs="+", default=["occurrence", "magnitude", "duration"],
                    choices=["occurrence", "magnitude", "duration"])
    ap.add_argument("--learners", nargs="+", default=None,
                    help="restrict the roster to these learner names")
    ap.add_argument("--no-tune", action="store_true",
                    help="score only each learner's default recipe (CV still runs, "
                         "because the champion must be selected on training folds)")
    ap.add_argument("--quick", action="store_true",
                    help="tiny budgets for smoke tests; NEVER for a reported run")
    args = ap.parse_args()

    ensure_dirs()
    cfg = load_ml_config(args.config, args.phase2, args.phase2ml)
    if args.no_tune:
        cfg["tune"] = False
    if args.quick:
        cfg.update({"n_boost_rounds": 40, "sk_gbm_rounds": 40, "aft_rounds": 40,
                    "forest_trees": 40, "tune": False, "cv_folds": 2,
                    "calibration_cv_folds": 3, "qrf_draws_per_tree": 4})
        log.warning("--quick: budgets reduced to smoke-test size. These numbers "
                    "are plumbing evidence, not results.")

    seed = int(cfg.get("random_seed", 0))
    frame = load_training_frame(cfg)
    feats = feature_columns(frame)
    split = masks(frame, cfg)
    cohort_ok, cohort_note = cohort_matches_incumbent(frame)
    log.info("phase2ML: %d rows, %d features, train=%d val=%d | %s", len(frame),
             len(feats), int(split["train"].sum()), int(split["val"].sum()),
             cohort_note)
    if not cohort_ok:
        log.warning(cohort_note)

    train_cv, _, folds = tuning_folds(frame, cfg, int(cfg.get("cv_folds", 3)))

    all_rows: list[dict] = []
    all_tuning: list[dict] = []
    fitted: dict[str, dict] = {}
    champions: dict[str, str | None] = {}
    selection_basis: dict[str, str] = {}

    with timed("phase2ml_bakeoff", log):
        for head in args.heads:
            log.info("=== %s ===", head)
            rows, models, tuning = run_head(head, frame, cfg, feats, split, folds,
                                            train_cv, seed, args.learners)
            all_rows.extend(rows)
            all_tuning.extend(tuning)
            fitted[head] = models
            champion, basis = pick_champion(rows, head)
            champions[head] = champion
            selection_basis[head] = basis
            log.info("%s champion: %s (selected on %s)", head, champion, basis)

    if "occurrence" in fitted:
        for name, entry in fitted["occurrence"].items():
            val = frame[split["val"]]
            raw = predict_full("occurrence", entry, val, feats,
                               [float(q) for q in cfg["quantiles"]])
            _, entry["oof"] = isotonic_calibrate(
                raw, val.event.astype(int).to_numpy(),
                int(cfg.get("calibration_cv_folds", 5)), seed)
            del name

    composite = (composite_metrics(frame, cfg, feats, split, champions, fitted)
                 if set(args.heads) == {"occurrence", "magnitude", "duration"}
                 else {"skipped": f"heads run: {sorted(args.heads)}"})

    board = pd.DataFrame(all_rows)
    board.to_csv(LEADERBOARD_PATH, index=False)
    if all_tuning:
        pd.DataFrame(all_tuning).to_csv(TUNING_PATH, index=False)

    import joblib
    for head, models in fitted.items():
        champion = champions.get(head)
        if champion and champion in models:
            joblib.dump({"head": head, "learner": champion, **models[champion],
                         "config_sha256": config_digest(cfg),
                         "split_windows": {"train": [cfg["train_start"], cfg["train_end"]],
                                           "validation": [cfg["val_start"], cfg["val_end"]]},
                         "reporting_counties": sorted(frame.fips.astype(str).unique())},
                        ML_MODEL_DIR / f"phase2ml_{head}_champion.joblib")

    write_json(METRICS_PATH, {
        "generated": str(pd.Timestamp.now(tz="UTC")),
        "config_sha256": config_digest(cfg),
        "quick_mode": bool(args.quick),
        "heads": args.heads,
        "champions": champions,
        "selection_basis": selection_basis,
        "leaderboard": all_rows,
        "composite_champion_trio": composite,
        "integrity": integrity(all_rows, champions, cohort_note, cohort_ok,
                               composite),
    })
    log.info("wrote %s and %s", LEADERBOARD_PATH.name, METRICS_PATH.name)
    ok = [r for r in all_rows if r.get("status") == "OK"]
    skipped = [r for r in all_rows if r.get("status") == "SKIPPED"]
    failed = [r for r in all_rows if r.get("status") == "FAILED"]
    log.info("leaderboard: %d fitted, %d skipped (missing packages), %d failed",
             len(ok), len(skipped), len(failed))
    for row in failed:
        log.warning("FAILED %s/%s: %s", row["head"], row["learner"], row.get("error"))


if __name__ == "__main__":
    main()
