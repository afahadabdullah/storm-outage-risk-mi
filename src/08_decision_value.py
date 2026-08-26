#!/usr/bin/env python
"""Step 8 -- decision value (spec section 9; phase 1 step 8, second half).

Two short calculations. This is the section that answers the value-of-data
question directly, and with section 7 validation it is the reason the artifact
works at all.

  9.1  cost-loss relative economic value    V vs C/L, per lead time
  9.2  break-even inspection cost           max defensible $ per asset, by county

PHASE 1: THE NUMBERS WILL BE NONSENSE. Confirm only that they are finite,
positive, and change monotonically with C/L. Do not read the dollar figure. Do
not tune anything to make it look sensible.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.common.config import PATHS, base_parser, config_from_args
from src.common.gates import book
from src.common.logio import get_logger, record

log = get_logger("08_value")


# --------------------------------------------------------------------------- #
def cost_loss_value(prob: np.ndarray, obs: np.ndarray, alpha: float) -> float:
    """Classical relative economic value at cost-loss ratio alpha = C/L.

        V = (E_climatology - E_forecast) / (E_climatology - E_perfect)

    Expenses are in units of L. The decision rule is act when p > alpha, which
    is the optimal threshold for a cost-loss decision maker -- that is why the
    forecast has to be CALIBRATED to be worth anything, and why isotonic
    regression on the validation year is not a cosmetic step.
    """
    s = float(obs.mean())
    if s in (0.0, 1.0):
        return float("nan")
    act = prob > alpha
    hits = float(((act) & (obs == 1)).mean())
    false_alarms = float(((act) & (obs == 0)).mean())
    misses = float(((~act) & (obs == 1)).mean())

    e_forecast = alpha * (hits + false_alarms) + misses
    e_climate = min(alpha, s)
    e_perfect = s * alpha
    denom = e_climate - e_perfect
    return float((e_climate - e_forecast) / denom) if denom else float("nan")


def ice_dollars_per_customer_hour(cfg) -> float:
    """Blended ICE Calculator value. REPLACE the config numbers with real ICE
    output for this region and customer mix before any figure leaves the repo."""
    ice = cfg["ice"]
    return (ice["mix_residential"] * ice["residential_usd_per_cust_hr"]
            + ice["mix_commercial"] * ice["commercial_usd_per_cust_hr"])


def break_even(samples: np.ndarray, df: pd.DataFrame, cfg, delta: float) -> pd.DataFrame:
    """Max defensible cost per asset inspected, by county, for hazard reduction delta.

    Assumption chain, all of it visible on purpose: an inspection programme
    reduces the failure hazard on treated assets by delta -> avoided
    customer-minutes scale by delta -> ICE converts them to dollars -> divide by
    the asset count. The weakest link is the asset count proxy, which here is
    customers / customers_per_asset. With real network data it would be pole and
    span counts.
    """
    usd_per_cust_hr = ice_dollars_per_customer_hour(cfg)
    per_asset = float(cfg["ice"].get("customers_per_asset", 12))

    exp_cust_min = samples.mean(axis=1)
    out = df[["fips", "date"]].copy()
    out["expected_customer_minutes"] = exp_cust_min
    by = out.groupby("fips").expected_customer_minutes.sum().rename("cust_min")
    by = by.to_frame()
    by["avoided_cust_hours"] = by.cust_min * delta / 60.0
    by["avoided_usd"] = by.avoided_cust_hours * usd_per_cust_hr

    mcc = df.groupby("fips").mcc.max() if "mcc" in df else None
    if mcc is None:
        mcc = pd.Series(1.0, index=by.index)
    by["assets"] = (mcc / per_asset).reindex(by.index).clip(lower=1)
    by["max_cost_per_asset_usd"] = by.avoided_usd / by.assets
    by["risk_reduction_per_dollar"] = by.avoided_cust_hours / by.avoided_usd.replace(0, np.nan)
    return by.sort_values("max_cost_per_asset_usd", ascending=False)


# --------------------------------------------------------------------------- #
def main() -> None:
    args = base_parser(__doc__).parse_args()
    cfg = config_from_args(args)
    gb = book("08_value")

    df = pd.read_parquet(PATHS.processed / "phase1_merged.parquet")
    samples = np.load(PATHS.processed / "phase1_mc_samples.npy")
    pred = np.load(PATHS.processed / "phase1_predictions.npz")
    proba = pred["proba"]
    obs = df.event.to_numpy()

    ratios = [float(r) for r in cfg.get("cost_loss_ratios", [0.05, 0.1, 0.2])]
    if cfg.is_phase1 and len(ratios) > 3:
        # three SPREAD values, not the first three: a cluster of tiny C/L
        # ratios all sit on the same side of every predicted probability and
        # produce a flat, uninformative curve
        ratios = [ratios[0], ratios[len(ratios) // 2], ratios[-1]]
    values = {a: cost_loss_value(proba, obs, a) for a in ratios}
    for a, v in values.items():
        log.info("C/L = %.2f   relative economic value = %+.3f", a, v)

    finite = [v for v in values.values() if np.isfinite(v)]
    gb.require("cost_loss_finite", len(finite) == len(values) and len(finite) > 0,
               f"values: {values}")
    gb.check("cost_loss_varies_with_ratio",
             len(set(np.round(finite, 6))) > 1,
             "value changes with C/L (a flat curve means the threshold rule is "
             "not binding, or every probability is on one side of every ratio)")

    deltas = [float(d) for d in cfg.get("hazard_reduction_deltas", [0.20])]
    delta = deltas[len(deltas) // 2]          # the middle scenario
    be = break_even(samples, df, cfg, delta)
    costs = be.max_cost_per_asset_usd
    invalid = ~np.isfinite(costs) | costs.lt(0)
    gb.require("break_even_finite_positive",
               bool(not invalid.any()),
               f"max cost per asset in "
               f"[{costs.min():,.2f}, {costs.max():,.2f}] USD at delta={delta}; "
               f"{int(invalid.sum())} invalid of {len(costs)} counties")

    out = PATHS.processed / "phase1_decision_value.csv"
    be.to_csv(out)
    headline = float(be.max_cost_per_asset_usd.max())
    top_decile_share = float(
        be.avoided_cust_hours.head(max(len(be) // 10, 1)).sum()
        / max(be.avoided_cust_hours.sum(), 1e-9))
    record("decision_value", delta=delta, cost_loss=values,
           headline_max_cost_per_asset_usd=headline,
           top_decile_share=top_decile_share)
    gb.note("phase1_decision_number",
            f"${headline:,.0f} per asset at delta={delta} "
            "(MEANINGLESS -- a plumbing output, not a result)")
    gb.flush()

    log.info("--------------------------------------------------------------")
    log.info("PHASE 1 DECISION NUMBER: $%s per asset inspected at delta=%.0f%%",
             f"{headline:,.0f}", 100 * delta)
    log.info("top decile of counties captures %.0f%% of avoidable customer-hours",
             100 * top_decile_share)
    log.info("THIS NUMBER IS NONSENSE. It exists to prove the arithmetic runs "
             "end to end. Do not interpret it, quote it, or tune anything to "
             "improve it.")
    log.info("--------------------------------------------------------------")
    log.info("county table -> %s", out.name)


if __name__ == "__main__":
    main()
