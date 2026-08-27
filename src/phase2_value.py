#!/usr/bin/env python
"""Phase 2 stage 6 -- decision value (spec section 9).

Three calculations, all short. This is the section that answers the
value-of-data question directly.

  9.1  cost-loss relative economic value   V vs C/L, one curve per lead time
  9.2  break-even inspection cost          max defensible $ per asset, by county
  9.3  risk-spend efficiency               counties ranked by risk reduction / $

Plus the EVPI ceiling: what perfect knowledge of which assets will fail would
be worth.

Unlike the Phase 1 version of this arithmetic, these numbers are meant to be
read -- which is exactly why the assumption chain is kept visible rather than
buried. The weakest link is the asset-count proxy (customers / customers_per_
asset); with real network data it is a pole and span count. Name it first when
asked, not last.

The ICE dollar values come from config/region.yaml and are placeholders until
they are replaced with ICE Calculator output for this region and customer mix.
A run whose ICE block is still at its defaults says so in the output.
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
from src.common.logio import get_logger, record

log = get_logger("phase2_value")

VALUE_PATH = PATHS.processed / "phase2_decision_value.csv"
COSTLOSS_PATH = PATHS.processed / "phase2_cost_loss.csv"
SUMMARY_PATH = PATHS.processed / "phase2_decision_summary.json"

# The region.yaml defaults, so a run can say out loud that they were never
# replaced with real ICE Calculator output.
PLACEHOLDER_ICE = {"residential_usd_per_cust_hr": 4.0,
                   "commercial_usd_per_cust_hr": 180.0}


# --------------------------------------------------------------------------- #
# 9.1 cost-loss relative economic value
# --------------------------------------------------------------------------- #
def cost_loss_value(prob: np.ndarray, obs: np.ndarray, alpha: float) -> float:
    """Relative economic value at cost-loss ratio alpha = C/L.

        V = (E_climatology - E_forecast) / (E_climatology - E_perfect)

    Expenses are in units of L. The decision rule is act when p > alpha, the
    optimal threshold for a cost-loss decision maker -- which is why the
    forecast has to be CALIBRATED to be worth anything, and why isotonic
    regression on the validation year is not a cosmetic step.
    """
    obs = np.asarray(obs, dtype=float)
    prob = np.asarray(prob, dtype=float)
    s = float(obs.mean())
    if s in (0.0, 1.0) or len(obs) == 0:
        return float("nan")
    act = prob > alpha
    hits = float((act & (obs == 1)).mean())
    false_alarms = float((act & (obs == 0)).mean())
    misses = float(((~act) & (obs == 1)).mean())
    e_forecast = alpha * (hits + false_alarms) + misses
    e_climate = min(alpha, s)
    e_perfect = s * alpha
    denom = e_climate - e_perfect
    return float((e_climate - e_forecast) / denom) if denom else float("nan")


def cost_loss_curves(cfg: Config, validation: pd.DataFrame,
                     forecast: pd.DataFrame | None) -> pd.DataFrame:
    """One curve per lead time, plus the analysis-time (lead 0) hindcast curve.

    The hindcast curve is the statistically meaningful one: it has a whole
    validation year behind it. The per-lead curves come from the case-study
    forecasts and rest on a handful of days, so they are reported with their
    sample size attached and should be read as illustration, not as an estimate.
    """
    ratios = [float(r) for r in cfg["cost_loss_ratios"]]
    rows = []
    for alpha in ratios:
        rows.append({
            "lead_days": 0, "source": "hindcast_validation_year",
            "cost_loss_ratio": alpha, "n_county_days": len(validation),
            "base_rate": float(validation.event.mean()),
            "relative_economic_value": cost_loss_value(
                validation.probability.to_numpy(), validation.event.to_numpy(), alpha),
        })
    if forecast is not None and len(forecast):
        for (case, lead), group in forecast.groupby(["case", "lead_days"]):
            for alpha in ratios:
                rows.append({
                    "lead_days": int(lead), "source": f"forecast:{case}",
                    "cost_loss_ratio": alpha, "n_county_days": len(group),
                    "base_rate": float(group.event.mean()),
                    "relative_economic_value": cost_loss_value(
                        group.probability.to_numpy(), group.event.to_numpy(), alpha),
                })
    return pd.DataFrame(rows)


def plot_cost_loss(curves: pd.DataFrame) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for (lead, source), group in curves.groupby(["lead_days", "source"]):
        group = group.sort_values("cost_loss_ratio")
        label = ("hindcast (analysis)" if lead == 0 else f"day-{lead}")
        ax.plot(group.cost_loss_ratio, group.relative_economic_value,
                marker="o", ms=4, lw=1.8 if lead == 0 else 1.2,
                ls="-" if lead == 0 else "--", label=label)
    ax.axhline(0, color="#6B7A88", lw=1)
    ax.set_xlabel("cost-loss ratio  C/L")
    ax.set_ylabel("relative economic value")
    ax.set_title("Value of acting on the forecast, by lead time")
    ax.legend(fontsize=8)
    ax.grid(alpha=.25)
    fig.tight_layout()
    out = PATHS.figures / "phase2_cost_loss_value.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# 9.2 break-even inspection cost, 9.3 risk-spend efficiency
# --------------------------------------------------------------------------- #
def ice_dollars_per_customer_hour(cfg: Config) -> float:
    ice = cfg["ice"]
    return (ice["mix_residential"] * ice["residential_usd_per_cust_hr"]
            + ice["mix_commercial"] * ice["commercial_usd_per_cust_hr"])


def using_placeholder_ice(cfg: Config) -> bool:
    ice = cfg["ice"]
    return all(float(ice.get(k, 0)) == v for k, v in PLACEHOLDER_ICE.items())


def break_even(samples: np.ndarray, frame: pd.DataFrame, cfg: Config,
               delta: float) -> pd.DataFrame:
    """Max defensible cost per asset inspected, by county, for hazard reduction delta.

    Assumption chain, visible on purpose:
      an inspection programme reduces failure hazard on treated assets by delta
      -> avoided customer-minutes scale by delta
      -> ICE converts them to dollars
      -> divide by the asset count.
    """
    usd_per_cust_hr = ice_dollars_per_customer_hour(cfg)
    per_asset = float(cfg["ice"].get("customers_per_asset", 12))

    out = frame[["fips"]].copy()
    out["expected_customer_minutes"] = samples.mean(axis=1)
    by = out.groupby("fips").expected_customer_minutes.sum().to_frame("cust_min")
    by["avoided_cust_hours"] = by.cust_min * delta / 60.0
    by["avoided_usd"] = by.avoided_cust_hours * usd_per_cust_hr

    mcc = (frame.groupby("fips").mcc.max() if "mcc" in frame
           else pd.Series(1.0, index=by.index))
    by["customers"] = mcc.reindex(by.index)
    by["assets"] = (by.customers / per_asset).clip(lower=1)
    by["max_cost_per_asset_usd"] = by.avoided_usd / by.assets
    # 9.3: the metric utilities actually use in regulatory filings -- avoided
    # customer-hours per dollar of programme spend at the break-even price.
    by["risk_reduction_per_dollar"] = (
        by.avoided_cust_hours / by.avoided_usd.replace(0, np.nan))
    by["hazard_reduction_delta"] = delta
    return by.sort_values("max_cost_per_asset_usd", ascending=False)


def evpi(samples: np.ndarray, frame: pd.DataFrame, cfg: Config) -> float:
    """Expected value of perfect information, in dollars.

    The ceiling on any programme: the difference between acting with perfect
    foreknowledge of which county-days fail and acting on the expectation alone,
    valued with the same ICE conversion. Nothing an inspection programme can buy
    is worth more than this.
    """
    usd_per_cust_hr = ice_dollars_per_customer_hour(cfg)
    per_draw_total = samples.sum(axis=0) / 60.0        # customer-hours per draw
    # Under perfect information you act only where it matters, so the residual
    # loss is the mean outcome; under uncertainty you carry the whole spread.
    return float((per_draw_total.mean() - np.percentile(per_draw_total, 10))
                 * usd_per_cust_hr)


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "config" / "region.yaml"))
    ap.add_argument("--phase2", default=str(ROOT / "config" / "phase2.yaml"))
    args = ap.parse_args()
    cfg = load_config(args.config, args.phase2)
    set_phase(2)
    gb = book("phase2_value")

    samples_path = PATHS.processed / "phase2_mc_samples.npy"
    summary_path = PATHS.processed / "phase2_composed_summary.parquet"
    if not samples_path.exists() or not summary_path.exists():
        raise SystemExit("composed Monte Carlo output missing -- run "
                         "`make phase2-compose` first")
    samples = np.load(samples_path)
    composed = pd.read_parquet(summary_path)

    predictions = pd.read_parquet(PATHS.processed / "phase2_predictions.parquet")
    from src.phase2_train import masks
    split = masks(predictions, cfg)
    validation = predictions[split["val"]].copy()

    # Per-lead forecast probabilities, joined to the observed outcome. The
    # observed side lives in the test-scope table, so this only contributes
    # once the test year has formally been opened -- otherwise the cost-loss
    # figure carries the hindcast curve alone and says so.
    forecast = None
    prob_path = PATHS.processed / "phase2_forecast_county_probs.parquet"
    truth_path = PATHS.processed / "phase2_merged_test.parquet"
    marker = PATHS.models / "TEST_YEAR_OPENED.txt"
    if prob_path.exists() and truth_path.exists() and marker.exists():
        probs = pd.read_parquet(prob_path)
        truth = pd.read_parquet(truth_path, columns=["fips", "date", "event"])
        truth["date"] = pd.to_datetime(truth.date).dt.normalize()
        probs["date"] = pd.to_datetime(probs.date).dt.normalize()
        forecast = probs.merge(truth, on=["fips", "date"], how="inner")
        log.info("cost-loss: %d case-study county-days with observed outcomes",
                 len(forecast))
    elif prob_path.exists():
        log.warning("forecast probabilities exist but the test year is not "
                    "open, so per-lead cost-loss curves are skipped. The "
                    "hindcast curve is unaffected.")

    curves = cost_loss_curves(cfg, validation, forecast)
    curves.to_csv(COSTLOSS_PATH, index=False)
    figure = plot_cost_loss(curves)

    hind = curves[curves.lead_days.eq(0)]
    finite = hind.relative_economic_value.dropna()
    gb.require("cost_loss_finite", len(finite) == len(hind) and len(finite) > 0,
               f"{len(finite)}/{len(hind)} hindcast cost-loss values are finite")
    gb.check("cost_loss_varies_with_ratio", finite.round(6).nunique() > 1,
             "value changes with C/L (a flat curve means the threshold rule is "
             "not binding, or every probability sits on one side of every ratio)")
    positive = hind[hind.relative_economic_value.gt(0)].cost_loss_ratio
    span = (f"{positive.min():.2f}-{positive.max():.2f}" if len(positive) else "none")
    gb.note("cost_loss_positive_value_range",
            f"forecast has positive economic value for C/L in {span}")

    # 9.2 / 9.3 across all three hazard-reduction scenarios
    deltas = [float(d) for d in cfg["hazard_reduction_deltas"]]
    merged_frame = pd.read_parquet(PATHS.processed / "phase2_merged.parquet",
                                   columns=["fips", "mcc"])
    mcc = merged_frame.groupby("fips").mcc.max()
    tables = []
    for delta in deltas:
        table = break_even(samples, composed.assign(mcc=composed.fips.map(mcc)),
                           cfg, delta)
        tables.append(table.reset_index())
    value = pd.concat(tables, ignore_index=True)
    value.to_csv(VALUE_PATH, index=False)

    invalid = ~np.isfinite(value.max_cost_per_asset_usd) | value.max_cost_per_asset_usd.lt(0)
    gb.require("break_even_finite_positive", bool(not invalid.any()),
               f"{int(invalid.sum())} invalid of {len(value)} county x delta rows")

    middle = deltas[len(deltas) // 2]
    scenario = value[value.hazard_reduction_delta.eq(middle)].sort_values(
        "max_cost_per_asset_usd", ascending=False)
    headline = float(scenario.max_cost_per_asset_usd.max())
    top_decile = max(len(scenario) // 10, 1)
    top_share = float(scenario.avoided_cust_hours.head(top_decile).sum()
                      / max(scenario.avoided_cust_hours.sum(), 1e-9))
    ceiling = evpi(samples, composed, cfg)
    placeholder = using_placeholder_ice(cfg)

    summary = {
        "ice_usd_per_customer_hour": ice_dollars_per_customer_hour(cfg),
        "ice_values_are_region_yaml_placeholders": placeholder,
        "hazard_reduction_deltas": deltas,
        "headline_delta": middle,
        "max_cost_per_asset_usd": headline,
        "top_decile_share_of_avoidable_customer_hours": top_share,
        "evpi_usd": ceiling,
        "cost_loss_positive_value_range": span,
        "n_counties": int(scenario.fips.nunique()),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    record("phase2_decision_value", **{k: v for k, v in summary.items()
                                       if isinstance(v, (int, float, bool))})
    gb.flush()

    log.info("-" * 66)
    if placeholder:
        log.warning("ICE values are still the region.yaml placeholders. Replace "
                    "them with ICE Calculator output for this region and "
                    "customer mix before any dollar figure leaves the repo.")
    log.info("cost-loss: positive economic value for C/L in %s", span)
    log.info("break-even at delta=%.0f%%: up to $%s per asset inspected",
             100 * middle, f"{headline:,.0f}")
    log.info("top decile of counties captures %.0f%% of avoidable customer-hours",
             100 * top_share)
    log.info("EVPI ceiling: $%s", f"{ceiling:,.0f}")
    log.info("-" * 66)
    log.info("county table -> %s ; cost-loss -> %s ; figure -> %s",
             VALUE_PATH.name, COSTLOSS_PATH.name, figure.name)


if __name__ == "__main__":
    main()
