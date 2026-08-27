#!/usr/bin/env python
"""Step 6 -- Monte Carlo composition (spec section 6.4; phase 1 step 7).

    total customer-minutes = P(event) x E[customer_hours | event] x duration adj.

propagated by SAMPLING, not multiplication:

    for each county-day, for each draw:
        occurrence ~ Bernoulli(p)
        magnitude  ~ inverse-CDF of the predicted quantiles (log space)
        duration   ~ inverse-survival of the fitted Weibull AFT

The assertion that per-row std > 0 catches the single most damaging silent bug
in the whole design: composing MEDIANS instead of sampling. The output looks
entirely reasonable and carries no uncertainty at all.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.common.config import PATHS, base_parser, config_from_args
from src.common.gates import book
from src.common.logio import get_logger, record, timed

log = get_logger("06_compose")
SAMPLES_PATH = PATHS.processed / "phase1_mc_samples.npy"


def sample_magnitude(qpred: np.ndarray, quantiles, rng, n_draws: int) -> np.ndarray:
    """Inverse-transform sampling from the predicted quantile function.

    Rows are sorted first: independently fitted quantile regressions can cross,
    and an unsorted quantile function is not a distribution. Draws outside
    [q_min, q_max] clamp to the end quantiles, so the tails are truncated -- an
    honest limitation of the quantile route, and one reason NGBoost is the
    primary magnitude model in Phase 2.
    """
    qs = np.asarray(quantiles, dtype=float)
    q_sorted = np.sort(qpred, axis=1)
    u = rng.uniform(qs[0], qs[-1], size=(qpred.shape[0], n_draws))
    out = np.empty_like(u)
    for i in range(qpred.shape[0]):
        out[i] = np.interp(u[i], qs, q_sorted[i])
    return np.expm1(out).clip(min=0)


def sample_duration(aft, frame: pd.DataFrame, rng, n_draws: int) -> np.ndarray:
    """Inverse-survival sampling: S(t) = u  =>  t = S^-1(u), u ~ U(0,1)."""
    draws = np.empty((len(frame), n_draws))
    for d in range(n_draws):
        p = float(rng.uniform(0.02, 0.98))
        draws[:, d] = np.asarray(aft.predict_percentile(frame, p=p)).ravel()
    return np.nan_to_num(draws, nan=np.nanmedian(draws), posinf=np.nanmax(draws)).clip(min=0.1)


def main() -> None:
    args = base_parser(__doc__).parse_args()
    cfg = config_from_args(args)
    gb = book("06_compose")
    rng = np.random.default_rng(int(cfg.get("random_seed", 0)))

    import joblib
    df = pd.read_parquet(PATHS.processed / "phase1_merged.parquet")
    pred = np.load(PATHS.processed / "phase1_predictions.npz")
    bundle = joblib.load(PATHS.models / "phase1_models.joblib")

    proba = pred["proba"]
    if bundle.get("calibrator") is not None:
        proba = bundle["calibrator"].predict(proba)
    n_draws = int(cfg["n_mc_draws"])
    n_rows = len(df)

    with timed("monte_carlo", log):
        occ = (rng.uniform(size=(n_rows, n_draws)) < proba[:, None]).astype(float)
        mag = sample_magnitude(pred["qpred"], pred["quantiles"], rng, n_draws)

        aft = bundle["duration"]["model"]
        cols = bundle["duration"]["columns"]
        frame = df.reindex(columns=cols).fillna(0.0)
        dur = sample_duration(aft, frame, rng, n_draws)
        dur_ref = float(np.median(pred["duration_median"]))
        adj = dur / max(dur_ref, 1e-6)

        # customer-MINUTES: the unit the decision-value stage speaks in
        samples = occ * mag * adj * 60.0

    gb.require("mc_shape", samples.shape == (n_rows, n_draws),
               f"{samples.shape} == ({n_rows}, {n_draws})", criterion=8)
    gb.require("mc_non_negative", bool((samples >= 0).all()),
               f"min sample {samples.min():.3f}", criterion=8)

    # The spec's assertion is `samples.std(axis=1) > 0` for every row. Taken
    # literally that is wrong, and Phase 1 is where you find out: a county-day
    # with a low occurrence probability can draw zero events in all N draws, so
    # its composed samples are legitimately all-zero. With 50 draws that is
    # common; with 1000 it is rare but still possible.
    #
    # The assertion's INTENT is to catch composing medians instead of sampling,
    # so test that directly on the conditional stages -- which must have spread
    # on every row -- and on composed rows that actually fired.
    fired = samples.max(axis=1) > 0
    row_std = samples.std(axis=1)
    gb.require("magnitude_draws_have_spread", bool((mag.std(axis=1) > 0).all()),
               f"min per-row magnitude std {mag.std(axis=1).min():.4g}", criterion=8)
    gb.require("duration_draws_have_spread", bool((dur.std(axis=1) > 0).all()),
               f"min per-row duration std {dur.std(axis=1).min():.4g}", criterion=8)
    gb.require(
        "mc_has_spread", bool((row_std[fired] > 0).all()),
        f"{int(fired.sum())}/{n_rows} rows drew at least one event; all of them "
        f"have non-zero spread ({int((~fired).sum())} rows never fired at "
        f"n_mc_draws={n_draws})",
        criterion=8,
        on_fail="A fired row with zero spread means point estimates were "
                "composed instead of sampled -- the output looks entirely "
                "reasonable and carries no uncertainty at all.")

    np.save(SAMPLES_PATH, samples)
    statewide = samples.sum(axis=0)
    record("monte_carlo", n_rows=n_rows, n_draws=n_draws,
           statewide_median_cust_min=float(np.median(statewide)))
    gb.note("statewide_customer_minutes_p10_p50_p90",
            " / ".join(f"{np.percentile(statewide, p):,.0f}" for p in (10, 50, 90)))
    gb.flush()
    log.info("MC samples %s -> %s", samples.shape, SAMPLES_PATH.name)
    log.info("statewide customer-minutes  p10 %.0f  p50 %.0f  p90 %.0f",
             np.percentile(statewide, 10), np.percentile(statewide, 50),
             np.percentile(statewide, 90))


if __name__ == "__main__":
    main()
