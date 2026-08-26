#!/usr/bin/env python
"""Build the multi-year Phase 2 event and county-day feature tables.

The default stops at the end of validation (2022). Building 2023 requires an
explicit acknowledgement so the held-out outcomes are not opened accidentally.
"""
from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.common.config import PATHS, Config, ROOT, load_config
from src.common.gates import book
from src.common.io_outage import normalize_outage_frame
from src.common.logio import get_logger, timed

log = get_logger("phase2_build")


def output_paths() -> dict[str, Path]:
    return {
        "hourly": PATHS.processed / "phase2_county_hourly.parquet",
        "events": PATHS.processed / "phase2_events.parquet",
        "weather": PATHS.processed / "phase2_county_day.parquet",
        "merged": PATHS.processed / "phase2_merged.parquet",
        "coverage": PATHS.processed / "phase2_coverage_exclusions.json",
    }


def load_mcc() -> pd.Series:
    path = PATHS.raw / "MCC.csv"
    if not path.exists():
        raise SystemExit(f"{path} missing -- run `make phase2-download-outages`")
    frame = pd.read_csv(path, dtype=str)
    frame.columns = [c.strip().lower() for c in frame.columns]
    fips = next(c for c in frame.columns if "fips" in c)
    customers = next(c for c in frame.columns if "customer" in c)
    year = next((c for c in frame.columns if c in ("year", "mcc_year")), None)
    if year:
        frame = frame.sort_values(year).groupby(fips, as_index=False).last()
    frame[fips] = frame[fips].str.strip().str.zfill(5)
    values = pd.to_numeric(frame[customers], errors="coerce")
    return pd.Series(values.to_numpy(), index=frame[fips], dtype=float)


def read_outage_year(cfg: Config, year: int) -> pd.DataFrame:
    path = PATHS.raw / f"eaglei_outages_{year}.csv"
    if not path.exists():
        raise SystemExit(f"{path} missing -- run `make phase2-download-outages`")
    pieces = []
    for chunk in pd.read_csv(path, dtype={"fips_code": "string"},
                             chunksize=1_000_000):
        chunk = normalize_outage_frame(chunk, cfg)
        if chunk.empty:
            continue
        hourly = (chunk.set_index("run_start_time")
                  .groupby(["fips_code", pd.Grouper(freq="1h")])["sum"]
                  .max().rename("customers_out").reset_index()
                  .rename(columns={"fips_code": "fips",
                                   "run_start_time": "time"}))
        pieces.append(hourly)
    if not pieces:
        return pd.DataFrame(columns=["fips", "time", "customers_out"])
    # A CSV chunk can split an hour; consolidate the partial hourly maxima.
    return (pd.concat(pieces, ignore_index=True)
            .groupby(["fips", "time"], as_index=False).customers_out.max())


def build_hourly(cfg: Config, end: pd.Timestamp) -> tuple[pd.DataFrame, list[str]]:
    model_start = pd.Timestamp(cfg["train_start"], tz="UTC")
    baseline_days = int(cfg.get("baseline_window_days", 30))
    start = model_start - pd.Timedelta(days=baseline_days)
    end_utc = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    years = range(start.year, pd.Timestamp(end).year + 1)
    annual = {year: read_outage_year(cfg, year) for year in years
              if year <= pd.Timestamp(end).year}
    # The prior-year slice is baseline context only. Coverage stability is
    # assessed from the frozen study period, which intentionally starts in 2018.
    annual_sets = {year: set(frame.fips.astype(str)) for year, frame in annual.items()
                   if year >= model_start.year}
    seen = set().union(*annual_sets.values()) if annual_sets else set()
    stable = set.intersection(*annual_sets.values()) if annual_sets else set()
    unstable = sorted(seen - stable)
    if unstable:
        log.warning("excluding %d counties with changing annual reporting coverage: %s",
                    len(unstable), unstable)
    reporting = sorted(stable)
    raw = pd.concat(annual.values(), ignore_index=True)
    raw = raw[(raw.time >= start) & (raw.time < end_utc)
              & raw.fips.astype(str).isin(reporting)]
    raw = raw.groupby(["fips", "time"], as_index=False).customers_out.max()

    times = pd.date_range(start, end_utc, freq="1h", inclusive="left")
    index = pd.MultiIndex.from_product([reporting, times], names=["fips", "time"])
    hourly = (raw.set_index(["fips", "time"]).reindex(index, fill_value=0.0)
              .reset_index())
    mcc = load_mcc()
    hourly["mcc"] = hourly.fips.map(mcc)
    if hourly.mcc.isna().any() or hourly.mcc.le(0).any():
        missing = sorted(hourly.loc[hourly.mcc.isna() | hourly.mcc.le(0), "fips"].unique())
        raise ValueError(f"MCC missing/non-positive for reporting counties: {missing}")
    hourly["frac_out"] = hourly.customers_out / hourly.mcc
    if hourly.frac_out.gt(1).any():
        raise ValueError(f"frac_out > 1 (max={hourly.frac_out.max():.4f})")

    hourly = hourly.sort_values(["fips", "time"])
    window = baseline_days * 24
    min_periods = window
    quantile = float(cfg.get("baseline_quantile", 0.10))
    hourly["baseline"] = (hourly.groupby("fips", sort=False).frac_out
                          .transform(lambda values: values.rolling(
                              window, min_periods=min_periods).quantile(quantile)))
    if hourly.loc[hourly.time >= model_start, "baseline"].isna().any():
        raise ValueError("rolling baseline is incomplete at the model boundary")
    hourly["frac_excess"] = (hourly.frac_out - hourly.baseline).clip(lower=0)
    return hourly[hourly.time >= model_start].copy(), unstable


def build_events(hourly: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    mod = runpy.run_path(str(ROOT / "src" / "03_build_events.py"))
    gb = book("phase2_events")
    events = mod["detect_events"](hourly, cfg, gb)
    if not 500 <= len(events) <= 20_000:
        log.warning("Phase 2 detected %d events; inspect county/event distributions", len(events))
    return events


def open_month(path: Path):
    import xarray as xr

    ds = xr.open_dataset(path)
    rename = {old: new for old, new in {
        "valid_time": "time", "latitude": "lat", "longitude": "lon"
    }.items() if old in ds}
    ds = ds.rename(rename)
    for dim in ("expver", "number"):
        if dim in ds.dims and ds.sizes[dim] == 1:
            ds = ds.squeeze(dim, drop=True)
    return ds.sortby("time")


def build_weather(cfg: Config, end: pd.Timestamp) -> pd.DataFrame:
    mod = runpy.run_path(str(ROOT / "src" / "04_build_features.py"))
    county_day_features = mod["county_day_features"]
    derived_features = mod["derived_features"]
    label_regimes = mod["label_regimes"]
    months = pd.period_range(pd.Timestamp(cfg["train_start"]), end, freq="M")
    frames = []
    for period in months:
        path = PATHS.raw / "era5_monthly" / f"era5_{period.year}{period.month:02d}.nc"
        if not path.exists():
            raise SystemExit(f"{path} missing -- run the monthly ERA5 download job")
        log.info("aggregating %s", path.name)
        with open_month(path) as ds:
            local = Config(cfg.copy())
            local["window_days"] = int(ds.sizes["time"] // 24)
            frame = county_day_features(ds, local, book(f"era5_{period}"))
            frames.append(frame)
    weather = pd.concat(frames, ignore_index=True)
    weather = weather[pd.to_datetime(weather.date) <= pd.Timestamp(end)].copy()
    weather = derived_features(weather, cfg)
    canopy_path = PATHS.raw / "canopy_county.csv"
    if canopy_path.exists():
        canopy = pd.read_csv(canopy_path, dtype={"fips": str}).set_index("fips").canopy_pct
        weather["canopy_pct"] = weather.fips.astype(str).map(canopy)
        weather["gust_x_canopy"] = weather.gust_max * weather.canopy_pct
        weather["ice_x_canopy"] = weather.freezing_rain_proxy * weather.canopy_pct
    weather = label_regimes(weather)
    if weather.canopy_pct.isna().any():
        raise ValueError("canopy_pct is missing; Phase 2 requires canopy_pct_clip.tif")
    return weather


def join_targets(weather: pd.DataFrame, events: pd.DataFrame,
                 hourly: pd.DataFrame) -> pd.DataFrame:
    events = events.copy()
    events["date"] = pd.to_datetime(events.date).dt.tz_localize(None).dt.normalize()
    agg = events.groupby(["fips", "date"]).agg(
        customer_hours=("customer_hours", "sum"),
        restoration_hours=("restoration_hours", "max"),
        peak_frac_out=("peak_frac_out", "max"),
        peak_customers_out=("peak_customers_out", "max"),
        concurrent_state_load=("concurrent_state_load", "max"),
        censored=("censored", "any"), n_events=("event_id", "size"),
    ).reset_index()
    reporting = set(hourly.fips.astype(str))
    merged = weather[weather.fips.astype(str).isin(reporting)].merge(
        agg, on=["fips", "date"], how="left", validate="one_to_one")
    merged["mcc"] = merged.fips.map(load_mcc())
    merged["event"] = merged.customer_hours.notna().astype(int)
    merged["customer_hours"] = merged.customer_hours.fillna(0.0)
    merged["customer_hours_per_customer"] = merged.customer_hours / merged.mcc
    merged["censored"] = merged.censored.fillna(False).astype(bool)
    merged["n_events"] = merged.n_events.fillna(0).astype(int)

    merged = merged.sort_values(["fips", "date"]).reset_index(drop=True)
    event_date = merged.date.where(merged.event.eq(1))
    previous = event_date.groupby(merged.fips).transform(lambda s: s.shift().ffill())
    merged["days_since_last_event"] = (
        pd.to_datetime(merged.date) - pd.to_datetime(previous)).dt.days
    merged["days_since_last_event"] = merged.days_since_last_event.fillna(3650).clip(upper=3650)
    return merged


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "config" / "region.yaml"))
    ap.add_argument("--phase2", default=str(ROOT / "config" / "phase2.yaml"))
    ap.add_argument("--through", choices=["validation", "test"], default="validation")
    ap.add_argument("--acknowledge-test", action="store_true",
                    help="required with --through test; opens held-out outcomes")
    args = ap.parse_args()
    cfg = load_config(args.config, args.phase2)
    if args.through == "test" and not args.acknowledge_test:
        raise SystemExit("Refusing to open 2023. Add --acknowledge-test only after models are frozen.")
    end = pd.Timestamp(cfg["val_end"] if args.through == "validation" else cfg["test_end"])
    paths = output_paths()

    with timed("phase2_event_build", log):
        hourly, unstable = build_hourly(cfg, end)
        events = build_events(hourly, cfg)
    with timed("phase2_weather_build", log):
        weather = build_weather(cfg, end)
        merged = join_targets(weather, events, hourly)

    hourly.to_parquet(paths["hourly"], index=False)
    events.to_parquet(paths["events"], index=False)
    weather.to_parquet(paths["weather"], index=False)
    merged.to_parquet(paths["merged"], index=False)
    paths["coverage"].write_text(json.dumps({
        "excluded_unstable_counties": unstable,
        "through": args.through,
        "end": str(end.date()),
    }, indent=2))
    log.info("Phase 2 tables: %d county-days, %d events, %d hourly rows",
             len(merged), len(events), len(hourly))


if __name__ == "__main__":
    main()
