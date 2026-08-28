#!/usr/bin/env python
"""Step 5 -- the three-part hurdle model (spec section 6; phase 1 step 6).

  occurrence   P(event | county, day)          LightGBM binary + isotonic
  magnitude    customer-hours | event          NGBoost lognormal, or LGBM quantiles
  duration     restoration time | event        Weibull AFT, right-censored

EVERY PART OUTPUTS A DISTRIBUTION, NOT A POINT ESTIMATE. That is the whole
reason this artifact is credible for a UQ role, and it is what step 6 of the
Monte Carlo composition consumes.

PHASE 1: the goal is only that each fit() and predict() returns valid output of
the right shape.

  Expect and IGNORE: convergence warnings, singular matrices from constant
  columns, AUC near 0.5, absurd confidence intervals. All are consequences of a
  ~60-row sample.

  Do NOT ignore: exceptions, NaN predictions, negative durations, quantile
  crossing, or a shape that does not match your row count. Those are code bugs
  and more data will not fix them.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.common.config import PATHS, base_parser, config_from_args
from src.common.gates import book
from src.common.logio import get_logger, record, timed

log = get_logger("05_models")
MERGED_PATH = PATHS.processed / "phase1_merged.parquet"

TARGETS = {"event", "customer_hours", "restoration_hours", "peak_frac_out",
           "peak_customers_out", "n_events", "censored", "concurrent_state_load",
           "mcc", "customer_hours_per_customer"}
IDS = {"fips", "date", "regime_label", "month"}


# --------------------------------------------------------------------------- #
def feature_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns
            if c not in TARGETS | IDS and pd.api.types.is_numeric_dtype(df[c])]
    return sorted(cols)


def drop_degenerate(X: pd.DataFrame, gb, tag: str) -> pd.DataFrame:
    """lifelines and GLMs fail hard on constant or collinear covariates, which is
    far more likely in a five-day slice than in six years. Drop, and NAME them:
    in Phase 2 they will have variance and should come back."""
    nunique = X.nunique(dropna=False)
    dead = list(nunique[nunique <= 1].index)
    if dead:
        log.warning("[%s] dropping %d zero-variance columns: %s", tag, len(dead), dead)
        gb.note(f"{tag}_zero_variance_dropped", ", ".join(dead))
    keep = X.drop(columns=dead)
    if keep.shape[1] > 1:
        corr = keep.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        collinear = [c for c in upper.columns if (upper[c] > 0.999).any()]
        if collinear:
            log.warning("[%s] dropping %d near-collinear columns: %s",
                        tag, len(collinear), collinear)
            gb.note(f"{tag}_collinear_dropped", ", ".join(collinear))
            keep = keep.drop(columns=collinear)
    return keep


def chronological_split(df: pd.DataFrame, cfg) -> dict[str, pd.DataFrame]:
    """Phase 1 splits are MEANINGLESS. They exist only to prove the split code
    runs. Phase 2 uses the frozen 2018-Jun 2022 / Jul-Dec 2022 / 2023 boundaries."""
    days = sorted(df.date.unique())
    n_tr, n_va = int(cfg.get("train_days", 3)), int(cfg.get("val_days", 1))
    tr, va, te = days[:n_tr], days[n_tr:n_tr + n_va], days[n_tr + n_va:]
    log.warning("PHASE 1 SPLIT (meaningless): train=%s val=%s test=%s",
                [str(pd.Timestamp(d).date()) for d in tr],
                [str(pd.Timestamp(d).date()) for d in va],
                [str(pd.Timestamp(d).date()) for d in te])
    return {"train": df[df.date.isin(tr)], "val": df[df.date.isin(va)],
            "test": df[df.date.isin(te)], "all": df}


# --------------------------------------------------------------------------- #
def fit_occurrence(splits, feats, cfg, gb):
    import lightgbm as lgb

    tr = splits["train"]
    X, y = tr[feats], tr.event.astype(int)
    pos, neg = int(y.sum()), int((1 - y).sum())
    spw = (neg / max(pos, 1))
    log.info("occurrence: %d rows, %d positive, scale_pos_weight=%.1f "
             "(weights, NOT resampling -- resampling destroys calibration and "
             "calibration is the point)", len(tr), pos, spw)

    model = lgb.LGBMClassifier(
        n_estimators=int(cfg["n_boost_rounds"]), learning_rate=0.05,
        num_leaves=7, min_child_samples=1, min_split_gain=0.0,
        scale_pos_weight=spw, random_state=int(cfg.get("random_seed", 0)),
        verbose=-1,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y)

    proba = model.predict_proba(splits["all"][feats])[:, 1]
    gb.require("occurrence_shape", proba.shape == (len(splits["all"]),),
               f"proba shape {proba.shape} vs {len(splits['all'])} rows", criterion=7)
    gb.require("occurrence_in_unit_interval",
               bool(((proba >= 0) & (proba <= 1)).all()),
               f"proba in [{proba.min():.4f}, {proba.max():.4f}]", criterion=7)
    gb.require("occurrence_no_nan", bool(np.isfinite(proba).all()),
               "no NaN predictions", criterion=7)

    calibrator = None
    if not bool(cfg.get("skip_isotonic", False)):
        from sklearn.isotonic import IsotonicRegression
        va = splits["val"]
        if va.event.nunique() < 2:
            log.warning("validation fold has one class -- skipping isotonic")
        else:
            calibrator = IsotonicRegression(out_of_bounds="clip").fit(
                model.predict_proba(va[feats])[:, 1], va.event.astype(int))
    else:
        log.warning("skip_isotonic=true: cannot calibrate on one day. "
                    "Phase 2 fits isotonic on the VALIDATION YEAR ONLY -- never "
                    "on train (leaks), never on test (cheating).")
    return model, calibrator, proba


def fit_magnitude(splits, feats, cfg, gb):
    """Model in log space: customer-hours spans four to five orders of magnitude."""
    ev = splits["all"][splits["all"].event == 1]
    qs = [float(q) for q in cfg.get("quantiles", [.05, .1, .25, .5, .75, .9, .95])]
    y = np.log1p(ev.customer_hours.to_numpy())
    X = ev[feats]
    kind = cfg.get("magnitude_model", "lgbm_quantile")

    if kind == "ngboost":
        try:
            from ngboost import NGBRegressor
            from ngboost.distns import LogNormal
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = NGBRegressor(Dist=LogNormal, n_estimators=int(cfg["n_boost_rounds"]),
                                 verbose=False,
                                 random_state=int(cfg.get("random_seed", 0))).fit(X, y)
            params = m.pred_dist(splits["all"][feats]).params
            qpred = np.column_stack([
                m.pred_dist(splits["all"][feats]).ppf(q) for q in qs])
            qpred = _assert_quantiles(qpred, qs, gb, "ngboost")
            return {"kind": "ngboost", "model": m, "quantiles": qs}, qpred, params
        except Exception as err:
            log.warning("NGBoost unavailable/failed (%s) -- falling back to "
                        "LightGBM quantile regression", err)

    import lightgbm as lgb
    models = {}
    preds = []
    for q in qs:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = lgb.LGBMRegressor(objective="quantile", alpha=q,
                                  n_estimators=int(cfg["n_boost_rounds"]),
                                  learning_rate=0.08, num_leaves=7,
                                  min_child_samples=1, verbose=-1,
                                  random_state=int(cfg.get("random_seed", 0))).fit(X, y)
        models[q] = m
        preds.append(m.predict(splits["all"][feats]))
    qpred = _assert_quantiles(np.column_stack(preds), qs, gb, "lgbm_quantile")
    return {"kind": "lgbm_quantile", "models": models, "quantiles": qs}, qpred, None


def _assert_quantiles(qpred, qs, gb, kind):
    """Independently fitted quantile regressions can cross, and a non-monotone
    quantile function is not a distribution. Rearranging (sorting each row) is
    the standard fix and never increases estimation error -- but the RAW
    crossing count is recorded, because a large one says the quantile route is
    straining and NGBoost is the better primary model."""
    gb.require("magnitude_is_a_distribution", qpred.shape[1] == len(qs),
               f"{kind}: {qpred.shape[1]} quantile columns for {len(qs)} quantiles",
               criterion=7)
    gb.require("magnitude_no_nan", bool(np.isfinite(qpred).all()),
               f"{kind}: finite predictions", criterion=7)
    raw_crossings = int((np.diff(qpred, axis=1) < -1e-9).sum())
    rearranged = np.sort(qpred, axis=1)
    gb.note(f"{kind}_raw_quantile_crossings",
            f"{raw_crossings} crossing pairs before rearrangement "
            f"({raw_crossings / max(qpred.shape[0] * (len(qs) - 1), 1):.1%} of pairs)")
    gb.require("no_quantile_crossing",
               bool((np.diff(rearranged, axis=1) >= -1e-9).all()),
               f"{kind}: quantile function monotone after rearrangement "
               f"({raw_crossings} raw crossings repaired)", criterion=7)
    return rearranged


def fit_duration(splits, feats, cfg, gb):
    """The survival stage. Right-censored at end-of-record and coverage gaps."""
    from lifelines import WeibullAFTFitter

    ev = splits["all"][splits["all"].event == 1].copy()
    dur = ev.restoration_hours.fillna(1.0).clip(lower=0.5)
    observed = (~ev.censored.fillna(False).astype(bool)).astype(int)
    if observed.sum() == 0:
        log.warning("every event is censored -- AFT needs at least one observed "
                    "restoration; treating the longest as observed for phase 1")
        observed.iloc[int(dur.argmax())] = 1

    cov = drop_degenerate(ev[[*feats, "concurrent_state_load"]].fillna(0.0),
                          gb, "duration")
    # the covariate that signals domain understanding: restoration is slower when
    # crews are stretched across many simultaneously-damaged counties
    if "concurrent_state_load" not in cov.columns:
        log.warning("concurrent_state_load dropped as degenerate -- in Phase 2 it "
                    "will vary and must come back")

    frame = cov.copy()
    frame["duration_"] = dur.to_numpy()
    frame["observed_"] = observed.to_numpy()

    aft = WeibullAFTFitter(penalizer=0.1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        aft.fit(frame, duration_col="duration_", event_col="observed_")

    med = aft.predict_median(frame)
    gb.require("duration_summary_finite", bool(aft.summary.notna().all().all()),
               "AFT summary has no NaNs", criterion=7)
    gb.require("duration_positive", bool((med > 0).all()),
               f"median predicted restoration in "
               f"[{float(med.min()):.2f}, {float(med.max()):.2f}] h", criterion=7)
    return {"model": aft, "columns": list(cov.columns)}, med


# --------------------------------------------------------------------------- #
def main() -> None:
    args = base_parser(__doc__).parse_args()
    cfg = config_from_args(args)
    gb = book("05_models")

    df = pd.read_parquet(MERGED_PATH)
    feats = feature_columns(df)
    log.info("%d rows, %d features", len(df), len(feats))
    splits = chronological_split(df, cfg)

    import joblib
    with timed("model_fit", log):
        occ, calib, proba = fit_occurrence(splits, feats, cfg, gb)
        mag, qpred, mag_params = fit_magnitude(splits, feats, cfg, gb)
        dur, med = fit_duration(splits, feats, cfg, gb)

    joblib.dump({"occurrence": occ, "calibrator": calib, "magnitude": mag,
                 "duration": dur, "features": feats},
                PATHS.models / "phase1_models.joblib")
    np.savez(PATHS.processed / "phase1_predictions.npz",
             proba=proba, qpred=qpred,
             quantiles=np.array(mag["quantiles"]),
             duration_median=med.to_numpy())
    record("model_fit", n_rows=len(df), n_features=len(feats),
           magnitude_kind=mag["kind"], positives=int(df.event.sum()))
    gb.flush()

    log.info("PHASE 1 MODELS ARE DISPOSABLE. `make clean-phase1` deletes them. "
             "A stale model file loaded in Phase 2 is a nasty bug.")
    log.info("occurrence proba: mean %.4f  magnitude median col: %.2f (log1p) "
             "duration median: %.1f h", proba.mean(),
             float(np.median(qpred[:, len(mag["quantiles"]) // 2])), float(med.median()))


if __name__ == "__main__":
    main()
