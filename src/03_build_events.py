#!/usr/bin/env python
"""Step 3 -- the outage event table (spec section 4; phase 1 spec steps 1-3).

    raw 15-min county series
      -> hourly max per county
      -> normalise by covered customers      frac_out = customers_out / mcc
      -> remove baseline                     rolling 30-day 10th pct, subtracted
      -> flag coverage discontinuities
      -> threshold + merge                   -> event table

Baseline removal matters. Every county carries a persistent floor of 20-200
customers out from routine faults. Left in, it inflates small-county event rates
and contaminates the occurrence model.

PHASE 1 APPROXIMATION: five days cannot support a 30-day rolling baseline, so
`baseline_method: window_percentile` substitutes the window's own 10th
percentile per county. This is logged loudly and reverted in Phase 2 (spec
section 9.2).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.common.config import PATHS, base_parser, config_from_args
from src.common.gates import book
from src.common.geo import load_counties
from src.common.io_outage import normalize_outage_frame
from src.common.logio import get_logger, record, timed

log = get_logger("03_events")
EVENTS_PATH = PATHS.processed / "phase1_events.parquet"
SERIES_PATH = PATHS.processed / "phase1_county_hourly.parquet"


# --------------------------------------------------------------------------- #
def load_window_outages(cfg, gb) -> pd.DataFrame:
    start = pd.Timestamp(cfg.required("window_start"), tz="UTC")
    end = start + pd.Timedelta(days=int(cfg["window_days"]))
    year = start.year
    csv = PATHS.raw / f"eaglei_outages_{year}.csv"
    if not csv.exists():
        raise SystemExit(f"{csv} missing -- run `make fetch`")

    df = normalize_outage_frame(pd.read_csv(csv, dtype={"fips_code": "string"}), cfg)

    # ---- step 1 assertions (phase 1 spec section 6) -------------------------
    gb.require("fips_5char", df.fips_code.str.len().eq(5).all(),
               "every fips_code is a 5-character string")
    gb.require("timestamps_tz_aware", df.run_start_time.dt.tz is not None,
               f"tz={df.run_start_time.dt.tz} -- one reference, UTC, set at ingest",
               criterion=2)
    gb.require("customers_out_nonneg", df["sum"].min() >= 0,
               f"min customers out = {df['sum'].min()}")

    win = df[(df.run_start_time >= start) & (df.run_start_time < end)].copy()
    gb.require("window_non_empty", len(win) > 0,
               f"{len(win):,} records in {start.date()} .. {end.date()}")
    log.info("window %s .. %s: %s records, %d counties",
             start.date(), end.date(), f"{len(win):,}", win.fips_code.nunique())
    return win


def check_fips_join(df, counties, gb) -> None:
    """Criterion 1. Print the actual unmatched lists, not a percentage.

    A partial join fails silently by dropping rows, which is why this is the
    highest-probability failure in the whole project.
    """
    outage_fips = set(df.fips_code.unique())
    tiger_fips = set(counties.index)
    only_outage = sorted(outage_fips - tiger_fips)
    only_tiger = sorted(tiger_fips - outage_fips)
    if only_outage:
        log.error("FIPS in EAGLE-I but not TIGER (%d): %s", len(only_outage), only_outage)
    if only_tiger:
        log.warning("FIPS in TIGER but not EAGLE-I (%d): %s", len(only_tiger), only_tiger)
    gb.require("fips_join_outage_side", not only_outage,
               f"unmatched EAGLE-I FIPS: {only_outage or 'none'}", criterion=1)
    gb.check("fips_join_tiger_side", not only_tiger,
             f"counties with no outage record: {only_tiger or 'none'} "
             "(acceptable only if the utility genuinely does not report there)",
             criterion=1)


def load_mcc(cfg, counties, gb) -> pd.Series:
    path = PATHS.raw / "MCC.csv"
    if not path.exists():
        raise SystemExit(f"{path} missing -- raw customer counts are meaningless "
                         "without the coverage denominator.")
    mcc = pd.read_csv(path, dtype=str)
    mcc.columns = [c.strip().lower() for c in mcc.columns]
    fips_col = next(c for c in mcc.columns if "fips" in c)
    cust_col = next(c for c in mcc.columns if "customer" in c)
    year_col = next((c for c in mcc.columns if c in ("year", "mcc_year")), None)
    if year_col:                      # some releases are per-year; take the latest
        mcc = mcc.sort_values(year_col).groupby(fips_col, as_index=False).last()
    s = (mcc.assign(**{fips_col: mcc[fips_col].str.strip().str.zfill(5)})
            .set_index(fips_col)[cust_col].astype(float))
    s = s[s.index.isin(counties.index)]
    gb.require("mcc_covers_all_counties",
               set(counties.index) <= set(s.index),
               f"MCC missing {sorted(set(counties.index) - set(s.index))}")
    gb.require("mcc_positive", (s > 0).all(), f"min MCC = {s.min():,.0f}")
    return s


# --------------------------------------------------------------------------- #
def hourly_frac(df, mcc, cfg, gb) -> pd.DataFrame:
    """15-min -> hourly max per county, normalised, baseline removed."""
    hourly = (df.set_index("run_start_time")
                .groupby(["fips_code", pd.Grouper(freq="1h")])["sum"]
                .max().rename("customers_out").reset_index()
                .rename(columns={"run_start_time": "time", "fips_code": "fips"}))
    # a county with no record in an hour has no reported outage, not a gap
    full = pd.MultiIndex.from_product(
        [sorted(hourly.fips.unique()), sorted(hourly.time.unique())],
        names=["fips", "time"])
    hourly = (hourly.set_index(["fips", "time"]).reindex(full, fill_value=0.0)
                    .reset_index())

    hourly["mcc"] = hourly.fips.map(mcc)
    hourly["frac_out"] = hourly.customers_out / hourly.mcc

    gb.require("frac_out_le_1", (hourly.frac_out <= 1.0).all(),
               f"max frac_out = {hourly.frac_out.max():.4f} "
               "(>1 means the MCC denominator is wrong)")
    gb.require("mcc_not_null", hourly.mcc.notna().all(), "MCC joined for every row")

    method = cfg.get("baseline_method", "rolling")
    q = float(cfg.get("baseline_quantile", 0.10))
    if method == "window_percentile":
        log.warning("PHASE 1 APPROXIMATION: baseline = window %g-th percentile "
                    "per county, not a 30-day rolling window. Revert in Phase 2.", q)
        base = hourly.groupby("fips").frac_out.transform(lambda s: s.quantile(q))
    else:
        w = int(cfg.get("baseline_window_days", 30)) * 24
        base = (hourly.groupby("fips").frac_out
                      .transform(lambda s: s.rolling(w, min_periods=w // 4)
                                            .quantile(q).bfill()))
    hourly["baseline"] = base.fillna(0.0)
    hourly["frac_excess"] = (hourly.frac_out - hourly.baseline).clip(lower=0)

    # A county's outage fraction is not comparable with the statewide row
    # median: the latter is dominated by zeroes from other counties.  The
    # window-percentile baseline is a floor only relative to the same county.
    county_median = hourly.groupby("fips").frac_out.transform("median")
    gb.check("baseline_is_a_floor", bool((hourly.baseline <= county_median).all()),
             "baseline <= its own county median for every county "
             f"(max baseline {hourly.baseline.max():.5f})")
    return hourly


# --------------------------------------------------------------------------- #
def detect_events(hourly, cfg, gb) -> pd.DataFrame:
    thr = float(cfg["event_frac_threshold"])
    min_hr = int(cfg["event_min_duration_hr"])
    gap_hr = int(cfg.get("event_merge_gap_hr", 6))
    end_frac = float(cfg.get("restoration_end_frac", 0.10))
    if cfg.is_phase1:
        log.warning("RELAXED THRESHOLDS IN USE: frac>%g for >=%dh "
                    "(region.yaml: 0.005 / 2h). Temporary, phase 1 only.", thr, min_hr)

    statewide = hourly.groupby("time").customers_out.sum()
    t_end = hourly.time.max()
    rows = []
    for fips, g in hourly.groupby("fips", sort=True):
        g = g.sort_values("time").reset_index(drop=True)
        above = (g.frac_excess > thr).to_numpy()
        if not above.any():
            continue
        # contiguous runs, then merge runs separated by <= gap_hr
        idx = np.flatnonzero(above)
        splits = np.flatnonzero(np.diff(idx) > gap_hr + 1)
        runs = np.split(idx, splits + 1)
        for run in runs:
            i0, i1 = run[0], run[-1]
            if (i1 - i0 + 1) < min_hr:
                continue
            seg = g.iloc[i0:i1 + 1]
            peak_i = int(seg.frac_excess.idxmax())
            peak_val = float(g.frac_excess.iloc[peak_i])

            # restoration: peak -> first hour below end_frac * peak
            tail = g.iloc[peak_i:]
            below = np.flatnonzero(
                (tail.frac_excess <= end_frac * peak_val).to_numpy())
            censored = len(below) == 0
            end_i = int(tail.index[below[0]]) if not censored else int(g.index[-1])
            end_time = g.time.iloc[end_i]
            censored = censored or end_time >= t_end

            rows.append({
                "event_id": f"{fips}_{g.time.iloc[i0]:%Y%m%dT%H}",
                "fips": fips,
                "start_time": g.time.iloc[i0],
                "peak_time": g.time.iloc[peak_i],
                "end_time": end_time,
                "date": g.time.iloc[peak_i].normalize(),
                "peak_frac_out": peak_val,
                "peak_customers_out": float(g.customers_out.iloc[peak_i]),
                # magnitude target: integral of customers-out over the event
                "customer_hours": float(
                    (g.frac_excess.iloc[i0:end_i + 1] * g.mcc.iloc[i0:end_i + 1]).sum()),
                # duration target
                "restoration_hours": float(end_i - peak_i),
                "censored": bool(censored),
                "concurrent_state_load": float(statewide.loc[g.time.iloc[peak_i]]),
                "mcc": float(g.mcc.iloc[0]),
            })
    ev = pd.DataFrame(rows)

    # ---- step 3 assertions --------------------------------------------------
    lo, hi = cfg.get("expected_event_count", [20, 60])
    gb.require("event_table_non_empty", len(ev) > 0, f"{len(ev)} events detected",
               criterion=3,
               on_fail="EMPTY EVENT TABLE -- the window missed the storm or the "
                       "threshold is still too high. Restart at spec section 3.")
    gb.require("customer_hours_positive", bool(ev.customer_hours.gt(0).all()),
               f"min customer_hours = {ev.customer_hours.min():,.1f}", criterion=3)
    # A right-censored event that begins in the last observed hour has no
    # observed restoration interval.  Its end is the window boundary, which
    # can equal its start; only observed restorations must be strictly ordered.
    observed = ~ev.censored
    gb.require("end_after_start", bool((ev.loc[observed, "end_time"]
                                         > ev.loc[observed, "start_time"]).all()),
               f"every observed restoration ends after it starts; "
               f"{int(ev.censored.sum())} right-censored events may end at the "
               "window boundary", criterion=3)
    n_observed = int((~ev.censored).sum())
    gb.require("some_restorations_observed", n_observed > 0,
               f"{n_observed} of {len(ev)} events uncensored",
               criterion=3)
    gb.check("event_count_in_expected_range", lo <= len(ev) <= hi,
             f"{len(ev)} events (expected {lo}-{hi}; thousands means baseline "
             "removal is not working, zero means the window missed the storm)",
             criterion=3)
    return ev


def plot_three_events(hourly, ev, cfg) -> Path:
    """The single most valuable ten minutes in Phase 1 (spec step 3).

    Eyeballing three events catches definition bugs no assertion will. Look for:
    a start that begins mid-rise, an end that fires on a noise dip, a peak on the
    wrong local maximum.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    picks = ev.sort_values("customer_hours", ascending=False).head(3)
    fig, axes = plt.subplots(len(picks), 1, figsize=(11, 3 * len(picks)), squeeze=False)
    for ax, (_, e) in zip(axes[:, 0], picks.iterrows()):
        g = hourly[hourly.fips == e.fips].sort_values("time")
        ax.plot(g.time, g.customers_out, lw=1.1, label="customers out (raw)")
        ax.plot(g.time, g.baseline * g.mcc, lw=0.9, ls=":", label="baseline")
        for t, c, lab in ((e.start_time, "tab:green", "start"),
                          (e.peak_time, "tab:red", "peak"),
                          (e.end_time, "tab:purple", "end")):
            ax.axvline(t, color=c, lw=1, ls="--", label=lab)
        ax.set_title(f"{e.fips}  {e.event_id}   {e.customer_hours:,.0f} customer-hours"
                     f"   restoration {e.restoration_hours:.0f} h"
                     + ("  [CENSORED]" if e.censored else ""))
        ax.legend(fontsize=7, ncol=5)
    fig.tight_layout()
    out = PATHS.figures / "phase1_three_events.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main() -> None:
    args = base_parser(__doc__).parse_args()
    cfg = config_from_args(args)
    gb = book("03_events")

    counties = load_counties(cfg)
    with timed("event_detection", log):
        df = load_window_outages(cfg, gb)
        check_fips_join(df, counties, gb)
        mcc = load_mcc(cfg, counties, gb)
        hourly = hourly_frac(df, mcc, cfg, gb)
        ev = detect_events(hourly, cfg, gb)

    hourly.to_parquet(SERIES_PATH)
    ev.to_parquet(EVENTS_PATH)
    fig = plot_three_events(hourly, ev, cfg)
    record("event_detection", n_events=len(ev), n_county_hours=len(hourly))
    gb.note("manual_inspection_figure", str(fig))
    gb.flush()

    log.info("%d events, %d county-hours -> %s", len(ev), len(hourly), EVENTS_PATH.name)
    log.info("INSPECT %s BY EYE before continuing (spec step 3).", fig)
    log.info("event customer_hours: min %.0f  median %.0f  max %.0f",
             ev.customer_hours.min(), ev.customer_hours.median(),
             ev.customer_hours.max())


if __name__ == "__main__":
    main()
