#!/usr/bin/env python
"""Phase 2 stage 3b -- Monte Carlo composition (spec section 6.4).

    total customer-minutes = P(event) x E[customer_hours | event] x duration adj.

propagated by SAMPLING, not multiplication:

    for each county-day, for each of n_mc_draws:
        occurrence ~ Bernoulli(p_calibrated)
        magnitude  ~ inverse-CDF of the predicted magnitude distribution
        duration   ~ inverse-survival of the fitted Weibull AFT

Composing MEDIANS instead of sampling is the single most damaging silent bug
available here: the output looks entirely reasonable and carries no uncertainty
at all. The spread assertions below exist to catch exactly that.

This is Phase 1's 06_compose_mc.py rebuilt against the Phase 2 bundle and the
frozen validation-scope table. Two things changed beyond the paths:

  * magnitude is drawn from the NGBoost density directly where one was fitted,
    instead of interpolating between seven reported quantiles and clamping the
    tails at q05/q95;
  * the composed ensemble is scored against the observed outcome (CRPS and a
    rank histogram, spec section 7.2 "Composed"), which Phase 1 never did.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.common.config import PATHS, ROOT, Config, load_config
from src.common.gates import book, set_phase
from src.common.logio import get_logger, record, timed

log = get_logger("phase2_compose")

SAMPLES_PATH = PATHS.processed / "phase2_mc_samples.npy"
SUMMARY_PATH = PATHS.processed / "phase2_composed_summary.parquet"
METRIC_PATH = PATHS.processed / "phase2_composed_metrics.json"

# One MC draw per county-day is 8 bytes; 83 counties x 1826 days x 1000 draws
# is ~1.2 GB. Chunking keeps peak memory flat and is why this runs in 128 GB
# with room to spare.
CHUNK_ROWS = 20_000


def sample_magnitude(bundle: dict, X: pd.DataFrame, rng, n_draws: int) -> np.ndarray:
    """Draw customer-hours from the fitted magnitude distribution.

    NGBoost gives a real density, so draws come from its own ppf on uniform
    probabilities and the tails are whatever the fitted Normal says they are.
    The LightGBM-quantile fallback has no density, so it interpolates the
    trained quantile function and extrapolates beyond it on a probit slope --
    the same treatment `phase2_train.crps_from_quantiles` uses, and for the same
    reason: clamping at q05/q95 truncates the distribution.
    """
    from src.phase2_train import _interp_log_quantiles

    u = rng.uniform(0.001, 0.999, size=(len(X), n_draws))
    if bundle["kind"] == "ngboost":
        dist = bundle["model"].pred_dist(X)
        loc = np.asarray(dist.params["loc"], dtype=float)
        scale = np.asarray(dist.params["scale"], dtype=float)
        from scipy.stats import norm
        log_draws = loc[:, None] + scale[:, None] * norm.ppf(u)
    else:
        trained = [float(q) for q in bundle["quantiles"]]
        base = np.sort(np.column_stack(
            [bundle["models"][q].predict(X) for q in trained]), axis=1)
        grid = np.linspace(0.001, 0.999, 501)
        q_full = np.sort(_interp_log_quantiles(base, trained, list(grid)), axis=1)
        log_draws = np.empty_like(u)
        for i in range(len(X)):
            log_draws[i] = np.interp(u[i], grid, q_full[i])
    return np.expm1(log_draws).clip(min=0.0)


def sample_duration(bundle: dict, frame: pd.DataFrame, rng, n_draws: int) -> np.ndarray:
    """Inverse-survival sampling from the Weibull AFT: S(t) = u => t = S^-1(u)."""
    from src.phase2_train import duration_frame

    X, _ = duration_frame(frame, bundle["numeric"], bundle["fill"])
    X = X.reindex(columns=bundle["columns"], fill_value=0.0)
    aft = bundle["model"]
    # predict_percentile is a per-p call, so draw a shared ladder of survival
    # probabilities once and assign each row a random rung. Calling it n_draws
    # times per row would be O(rows x draws) model evaluations.
    ladder = np.linspace(0.02, 0.98, min(n_draws, 99))
    curves = np.column_stack(
        [np.asarray(aft.predict_percentile(X, p=float(p))).ravel() for p in ladder])
    curves = np.nan_to_num(curves, nan=np.nan)
    median = np.nanmedian(curves)
    curves = np.nan_to_num(curves, nan=median, posinf=np.nanmax(curves[np.isfinite(curves)])
                           if np.isfinite(curves).any() else median)
    pick = rng.integers(0, curves.shape[1], size=(len(X), n_draws))
    return np.take_along_axis(curves, pick, axis=1).clip(min=0.1)


def compose(bundle: dict, frame: pd.DataFrame, cfg: Config, gb) -> np.ndarray:
    """(rows, n_draws) composed customer-MINUTES -- the decision-value unit."""
    from src.phase2_train import predict_duration

    rng = np.random.default_rng(int(cfg.get("random_seed", 0)))
    n_draws = int(cfg["n_mc_draws"])
    feats = bundle["features"]
    out = np.empty((len(frame), n_draws), dtype=np.float32)

    raw = bundle["occurrence"].predict_proba(frame[feats])[:, 1]
    proba = bundle["calibrator"].predict(raw)
    dur_ref = float(np.median(predict_duration(bundle["duration"], frame)))
    log.info("composing %d county-days x %d draws (reference duration %.1f h)",
             len(frame), n_draws, dur_ref)

    mag_spread_min, dur_spread_min = np.inf, np.inf
    for start in range(0, len(frame), CHUNK_ROWS):
        stop = min(start + CHUNK_ROWS, len(frame))
        chunk = frame.iloc[start:stop]
        X = chunk[feats]
        occ = (rng.uniform(size=(len(chunk), n_draws)) < proba[start:stop, None])
        mag = sample_magnitude(bundle["magnitude"], X, rng, n_draws)
        dur = sample_duration(bundle["duration"], chunk, rng, n_draws)
        mag_spread_min = min(mag_spread_min, float(mag.std(axis=1).min()))
        dur_spread_min = min(dur_spread_min, float(dur.std(axis=1).min()))
        # customer-MINUTES, scaled by how long restoration runs relative to the
        # model's own median restoration time
        out[start:stop] = (occ * mag * (dur / max(dur_ref, 1e-6)) * 60.0).astype(np.float32)
        log.info("  rows %d-%d done", start, stop)

    gb.require("mc_shape", out.shape == (len(frame), n_draws),
               f"{out.shape} == ({len(frame)}, {n_draws})")
    gb.require("mc_non_negative", bool(np.isfinite(out).all() and (out >= 0).all()),
               f"min sample {out.min():.3f}, all finite")
    # The conditional stages must have spread on EVERY row -- that is the direct
    # test for "medians were composed instead of sampled". The composed rows
    # cannot be tested that way: a low-probability county-day can legitimately
    # draw zero events in all N draws and be all-zero.
    gb.require("magnitude_draws_have_spread", mag_spread_min > 0,
               f"min per-row magnitude std {mag_spread_min:.4g}")
    gb.require("duration_draws_have_spread", dur_spread_min > 0,
               f"min per-row duration std {dur_spread_min:.4g}")
    fired = out.max(axis=1) > 0
    row_std = out[fired].std(axis=1) if fired.any() else np.array([1.0])
    gb.require(
        "composed_rows_have_spread", bool((row_std > 0).all()),
        f"{int(fired.sum())}/{len(frame)} rows drew at least one event and all "
        f"have non-zero spread ({int((~fired).sum())} never fired at "
        f"n_mc_draws={n_draws})",
        on_fail="A fired row with zero spread means point estimates were "
                "composed instead of sampled -- the output looks entirely "
                "reasonable and carries no uncertainty at all.")
    return out


def score_composed(samples: np.ndarray, frame: pd.DataFrame) -> dict:
    """Spec 7.2 'Composed': CRPS on total customer-minutes and a rank histogram."""
    from src.phase2_train import crps_from_quantiles

    observed = (frame.customer_hours.to_numpy() * 60.0)
    qs = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    qpred = np.quantile(samples, qs, axis=1).T
    crps = crps_from_quantiles(observed, qpred, qs)

    # Climatology reference: the county's own composed distribution, pooled.
    clim = np.quantile(observed, qs)
    ref = crps_from_quantiles(observed, np.tile(clim, (len(observed), 1)), qs)

    # Rank histogram: where the observation falls among the MC members. A flat
    # histogram means the ensemble is reliable; U-shaped means under-dispersed.
    n_bins = 11
    thinned = samples[:, :: max(samples.shape[1] // (n_bins - 1), 1)][:, : n_bins - 1]
    rank = (thinned < observed[:, None]).sum(axis=1)
    hist = np.bincount(rank, minlength=n_bins)[:n_bins]
    return {
        "n_rows": int(len(frame)),
        "composed_crps_customer_minutes": crps,
        "composed_crps_skill_vs_climatology": 1 - crps / ref if ref else float("nan"),
        "composed_rank_histogram": hist.astype(int).tolist(),
        "composed_rank_histogram_note":
            "flat = reliable ensemble; U-shaped = under-dispersed; "
            "dome = over-dispersed",
        "statewide_customer_minutes_p10_p50_p90": [
            float(np.percentile(samples.sum(axis=0), p)) for p in (10, 50, 90)],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "config" / "region.yaml"))
    ap.add_argument("--phase2", default=str(ROOT / "config" / "phase2.yaml"))
    ap.add_argument("--split", choices=["validation", "train", "all"],
                    default="validation",
                    help="which rows to compose; the test year is never included")
    args = ap.parse_args()
    cfg = load_config(args.config, args.phase2)
    set_phase(2)
    gb = book("phase2_compose")

    import joblib
    model_path = PATHS.models / "phase2_models.joblib"
    if not model_path.exists():
        raise SystemExit("frozen Phase 2 model missing -- run `make phase2-train` first")
    merged = PATHS.processed / "phase2_merged.parquet"
    if not merged.exists():
        raise SystemExit(f"{merged} missing -- run `make phase2-build` first")

    bundle = joblib.load(model_path)
    frame = pd.read_parquet(merged)
    from src.phase2_train import masks
    split = masks(frame, cfg)
    if split["test"].any():
        raise SystemExit("composition input contains test-year rows; it must run "
                         "on the validation-scope table only")
    if args.split == "validation":
        frame = frame[split["val"]].reset_index(drop=True)
    elif args.split == "train":
        frame = frame[split["train"]].reset_index(drop=True)
    else:
        frame = frame.reset_index(drop=True)

    with timed("phase2_monte_carlo", log):
        samples = compose(bundle, frame, cfg, gb)

    metrics = score_composed(samples, frame)
    np.save(SAMPLES_PATH, samples)
    summary = frame[["fips", "date", "event", "customer_hours"]].copy()
    summary["composed_mean_customer_minutes"] = samples.mean(axis=1)
    for p in (10, 50, 90):
        summary[f"composed_p{p}"] = np.percentile(samples, p, axis=1)
    summary.to_parquet(SUMMARY_PATH, index=False)
    METRIC_PATH.write_text(json.dumps(metrics, indent=2))

    record("phase2_monte_carlo", n_rows=len(frame), n_draws=int(cfg["n_mc_draws"]),
           composed_crpss=metrics["composed_crps_skill_vs_climatology"])
    gb.note("statewide_customer_minutes_p10_p50_p90",
            " / ".join(f"{v:,.0f}" for v in
                       metrics["statewide_customer_minutes_p10_p50_p90"]))
    gb.flush()
    log.info("composed CRPS %.1f customer-minutes, CRPSS %.3f vs climatology",
             metrics["composed_crps_customer_minutes"],
             metrics["composed_crps_skill_vs_climatology"])
    log.info("rank histogram %s", metrics["composed_rank_histogram"])
    log.info("MC samples %s -> %s", samples.shape, SAMPLES_PATH.name)


if __name__ == "__main__":
    main()
