#!/usr/bin/env python
"""Shared plumbing for the Phase 2 ML bake-off (phase2ML).

The whole point of this track is a FAIR comparison against the incumbent
Phase 2 hurdle model, and the only way to get one is to reuse the incumbent's
own definitions rather than re-implement them. Every function below either
delegates to ``src.phase2_train`` or exists because the bake-off needs
something the single-model pipeline never did.

Three things this module refuses to let the bake-off do, because each one
silently converts a comparison into a flattering number:

1. **Read the test year.** phase2ML reads ``phase2_merged.parquet`` only. The
   held-out year belongs to the frozen incumbent bundle and to whichever ML
   champion is frozen afterwards -- never to a model search.
2. **Tune on the validation period.** Hyperparameters and the champion are
   selected inside the TRAINING window using storm-blocked folds. The
   validation period is scored once per learner, at the end, and is reported
   for transparency, not used to choose. With ~10 learners, picking the best
   validation score and then quoting it is a selection-bias number, not a
   result.
3. **Score an uncalibrated or in-sample-calibrated probability.** Every
   occurrence learner gets the same isotonic treatment the incumbent gets,
   including the out-of-fold calibration that keeps the reported Brier honest
   (the C1 defect in the Phase 2 audit). A learner scored with an in-sample
   calibrator beats one scored out-of-fold for reasons that have nothing to do
   with the learner.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.common.config import PATHS, ROOT, Config, load_config
from src.common.logio import get_logger

# Delegation, not duplication. If any of these change in the incumbent, the
# bake-off changes with them and the comparison stays apples-to-apples.
from src.phase2_train import (  # noqa: F401  (re-exported on purpose)
    DURATION_QUANTILES,
    config_digest,
    crps_from_quantiles,
    duration_frame,
    evaluate,
    feature_columns,
    masks,
    period_slug,
    storm_groups,
    validate_temporal_split,
)

log = get_logger("phase2ML")

MERGED_PATH = PATHS.processed / "phase2_merged.parquet"
TEST_MERGED_PATH = PATHS.processed / "phase2_merged_test.parquet"

# phase2ML artifacts are namespaced so nothing here can be mistaken for, or
# overwrite, an incumbent Phase 2 artifact. `make phase2-report` globs
# data/processed/phase2_*.json; without the distinct prefix the ML leaderboard
# would be read as an incumbent metrics file.
ML_DIR = PATHS.processed / "phase2ml"
ML_MODEL_DIR = PATHS.models / "phase2ml"
LEADERBOARD_PATH = ML_DIR / "phase2ml_leaderboard.csv"
METRICS_PATH = ML_DIR / "phase2ml_metrics.json"
TUNING_PATH = ML_DIR / "phase2ml_tuning.csv"
PRED_DIR = ML_DIR / "predictions"
INCUMBENT_METRICS = PATHS.processed / "phase2_validation_metrics.json"

DEFAULT_ML_CONFIG = ROOT / "config" / "phase2ml.yaml"


def ensure_dirs() -> None:
    for directory in (ML_DIR, ML_MODEL_DIR, PRED_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def load_ml_config(region: str | Path, phase2: str | Path,
                   phase2ml: str | Path) -> Config:
    """region.yaml <- phase2.yaml <- phase2ml.yaml, in that precedence order.

    phase2ml.yaml may only add ML-track keys and must never move a split
    boundary or a quantile set: doing so would make the ML numbers
    incomparable with the incumbent's while looking like a like-for-like table.
    That is checked, not trusted.
    """
    import yaml

    cfg = load_config(region, phase2)
    frozen_before = {k: cfg.get(k) for k in FROZEN_KEYS}
    overrides = yaml.safe_load(Path(phase2ml).read_text()) or {}
    cfg["_phase2ml_file"] = str(phase2ml)
    cfg["_phase2ml_overrides"] = sorted(overrides)
    cfg.update(overrides)
    changed = [k for k in FROZEN_KEYS if cfg.get(k) != frozen_before[k]]
    if changed:
        raise SystemExit(
            "config/phase2ml.yaml overrides keys that must stay identical to "
            f"the incumbent Phase 2 run: {changed}. The bake-off is only "
            "meaningful on the same split, the same cohort rule and the same "
            "reporting quantiles. Move the change to region.yaml/phase2.yaml "
            "and refit BOTH tracks, or drop it.")
    return cfg


# Keys that define the comparison itself rather than how a learner is fitted.
FROZEN_KEYS = (
    "train_start", "train_end", "val_start", "val_end", "test_start",
    "test_end", "quantiles", "event_frac_threshold", "event_min_duration_hr",
    "event_merge_gap_hr", "restoration_end_frac", "baseline_method",
    "baseline_window_days", "baseline_quantile", "storm_gap_days",
    "storm_min_county_frac",
)


def load_training_frame(cfg: Config) -> pd.DataFrame:
    """The validation-scope merged table, with the test year proven absent."""
    if not MERGED_PATH.exists():
        raise SystemExit(
            f"{MERGED_PATH} missing -- run `make phase2-build` first. phase2ML "
            "deliberately builds nothing itself: it must consume the exact "
            "table the incumbent was fitted on, or the comparison is between "
            "two different datasets.")
    frame = pd.read_parquet(MERGED_PATH)
    split = masks(frame, cfg)
    if split["test"].any():
        raise SystemExit(
            f"{MERGED_PATH} contains rows inside the held-out test window "
            f"({cfg['test_start']}..{cfg['test_end']}). Refusing to run a model "
            "search over the test year.")
    if not split["train"].any() or not split["val"].any():
        raise SystemExit("training or validation window is empty in the merged table")
    return frame


def assert_no_test_rows(frame: pd.DataFrame, cfg: Config, where: str) -> None:
    if masks(frame, cfg)["test"].any():
        raise SystemExit(f"{where}: test-window rows present; refusing to continue")


def cohort_matches_incumbent(frame: pd.DataFrame) -> tuple[bool, str]:
    """Compare the ML track's county cohort with the frozen incumbent bundle.

    Not fatal -- the incumbent may simply not be fitted yet on this machine --
    but a mismatch means the two tracks are being scored on different study
    populations, and that has to appear in the leaderboard rather than be
    discovered by a reviewer.
    """
    bundle_path = PATHS.models / "phase2_models.joblib"
    if not bundle_path.exists():
        return True, "incumbent bundle not present; cohort not cross-checked"
    try:
        import joblib
        frozen = set(joblib.load(bundle_path).get("reporting_counties") or [])
    except Exception as err:  # pragma: no cover - defensive
        return True, f"incumbent bundle unreadable ({err}); cohort not cross-checked"
    if not frozen:
        return True, "incumbent bundle carries no cohort; not cross-checked"
    here = set(frame.fips.astype(str).unique())
    if frozen == here:
        return True, f"cohort matches incumbent ({len(here)} counties)"
    return False, (
        f"COHORT MISMATCH vs incumbent: only in incumbent "
        f"{sorted(frozen - here)}; only here {sorted(here - frozen)}")


def tuning_folds(frame: pd.DataFrame, cfg: Config, n_splits: int = 4):
    """Storm-blocked folds INSIDE the training window only.

    Model selection has to happen somewhere, and the validation period is not
    available for it: that period is what calibrates the occurrence models and
    produces every reported number. Selecting there and reporting there is the
    classic bake-off failure -- with a dozen learners the winner's validation
    score is optimistic by roughly the spread of the roster.

    Folds are blocked by storm episode for the same reason the incumbent's CV
    is (spec 7.1): adjacent county-days of one storm are not independent, and
    random folds put the same storm on both sides of the split, which rewards
    memorising storms rather than learning hazard response.
    """
    from sklearn.model_selection import GroupKFold

    train = frame[masks(frame, cfg)["train"]].reset_index(drop=True)
    groups = storm_groups(train, int(cfg.get("storm_gap_days", 2)),
                          float(cfg.get("storm_min_county_frac", 0.10)))
    n_groups = groups.nunique()
    folds = int(min(n_splits, n_groups))
    if folds < 2:
        raise SystemExit(
            f"storm blocking produced {n_groups} group(s) in the training "
            "window, so no grouped tuning split exists. Lower "
            "storm_min_county_frac in phase2.yaml.")
    if folds < n_splits:
        log.warning("tuning: only %d storm groups, using %d folds instead of %d",
                    n_groups, folds, n_splits)
    splitter = GroupKFold(n_splits=folds)
    return train, groups, list(splitter.split(train, train.event, groups))


def isotonic_calibrate(raw_val: np.ndarray, y_val: np.ndarray, folds: int,
                       seed: int) -> tuple[object, np.ndarray]:
    """The incumbent's calibration discipline, applied identically per learner.

    Returns (calibrator fitted on the whole validation period, out-of-fold
    calibrated probabilities). The first is what a frozen ML bundle would ship;
    the second is what gets scored. Reporting the first would make every
    learner in the roster look better than the incumbent for a reason that is
    purely an artefact of how it was measured.
    """
    from sklearn.isotonic import IsotonicRegression
    from sklearn.model_selection import StratifiedKFold

    calibrator = IsotonicRegression(out_of_bounds="clip").fit(raw_val, y_val)
    oof = np.empty_like(raw_val, dtype=float)
    n = int(min(folds, np.bincount(y_val.astype(int)).min())) if y_val.sum() else 0
    if n < 2:
        log.warning("calibration: only %d minority-class rows in the validation "
                    "window; falling back to the in-sample calibrator and "
                    "flagging it", int(y_val.sum()))
        return calibrator, calibrator.predict(raw_val)
    splitter = StratifiedKFold(n_splits=n, shuffle=True, random_state=seed)
    for fit_idx, score_idx in splitter.split(raw_val.reshape(-1, 1), y_val):
        fold = IsotonicRegression(out_of_bounds="clip").fit(
            raw_val[fit_idx], y_val[fit_idx])
        oof[score_idx] = fold.predict(raw_val[score_idx])
    return calibrator, oof


def design_matrix(frame: pd.DataFrame, feats: list[str],
                  fill: dict[str, float] | None = None
                  ) -> tuple[pd.DataFrame, dict[str, float]]:
    """Numeric, finite, NaN-free features.

    LightGBM and the other boosters take NaNs natively; scikit-learn's forests,
    the MLP and most of the survival learners do not. Imputing per learner
    would mean each one silently sees a different matrix, so it is done once
    here with medians taken from the TRAINING rows and reused unchanged
    everywhere else -- a median recomputed on the validation window is a small,
    real leak that is very hard to see afterwards.
    """
    values = frame.reindex(columns=feats).astype(float)
    values = values.replace([np.inf, -np.inf], np.nan)
    if fill is None:
        fill = {c: (float(values[c].median()) if values[c].notna().any() else 0.0)
                for c in values.columns}
    return values.fillna(fill).fillna(0.0), fill


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_jsonable))


def _jsonable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp,)):
        return str(obj)
    return str(obj)


def incumbent_metrics() -> dict:
    """The published incumbent validation metrics, if this machine has them."""
    if not INCUMBENT_METRICS.exists():
        return {}
    try:
        return json.loads(INCUMBENT_METRICS.read_text())
    except Exception as err:  # pragma: no cover - defensive
        log.warning("could not read incumbent metrics (%s)", err)
        return {}
