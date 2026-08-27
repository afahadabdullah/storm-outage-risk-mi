#!/usr/bin/env python
"""Build the multi-year Phase 2 event and county-day feature tables.

The default stops at the configured validation end (2020-07-31). Building 2023
requires an explicit acknowledgement so held-out outcomes are not opened
accidentally.
"""
from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.common.config import PATHS, ROOT, Config, load_config
from src.common.gates import book, set_phase
from src.common.io_outage import normalize_outage_frame
from src.common.logio import get_logger, timed

log = get_logger("phase2_build")


def output_paths(through: str = "validation") -> dict[str, Path]:
    """Artifact paths, namespaced by scope.

    The test build writes its own files rather than overwriting the
    validation-scope table. Overwriting it meant that once the final-test job
    ran, `phase2_train` fit mode refused to start (it correctly rejects a frame
    containing 2023) and the only way back was to repeat a 72-file ERA5
    aggregation to rebuild a table you already had.
    """
    suffix = "" if through == "validation" else "_test"
    return {
        "hourly": PATHS.processed / f"phase2_county_hourly{suffix}.parquet",
        "events": PATHS.processed / f"phase2_events{suffix}.parquet",
        "weather": PATHS.processed / f"phase2_county_day{suffix}.parquet",
        "merged": PATHS.processed / f"phase2_merged{suffix}.parquet",
        "coverage": PATHS.processed / f"phase2_coverage_exclusions{suffix}.json",
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


def build_hourly(cfg: Config, end: pd.Timestamp) -> tuple[pd.DataFrame, dict]:
    model_start = pd.Timestamp(cfg["train_start"], tz="UTC")
    baseline_days = int(cfg.get("baseline_window_days", 30))
    start = model_start - pd.Timedelta(days=baseline_days)
    end_utc = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    years = range(start.year, pd.Timestamp(end).year + 1)
    annual = {year: read_outage_year(cfg, year) for year in years
              if year <= pd.Timestamp(end).year}
    # The prior-year slice is baseline context only. Coverage stability is
    # assessed from the frozen study period, which intentionally starts in 2018.
    annual_sets = {year: set(frame.fips.astype(str))
                   for year, frame in annual.items() if year >= model_start.year}

    # The reporting cohort is fixed by the TRAIN + VALIDATION years and by
    # nothing else. Intersecting across the test year too meant that a county
    # whose utility stopped reporting in 2023 was retroactively dropped from
    # every year -- silently changing the rows the frozen model had been fitted
    # on, at the moment the held-out year was opened, with config_sha256 unable
    # to see it because it hashes config and not data.
    cohort_years = [y for y in annual_sets
                    if model_start.year <= y <= pd.Timestamp(cfg["val_end"]).year]
    cohort_sets = {y: annual_sets[y] for y in sorted(cohort_years)}
    seen = set().union(*cohort_sets.values()) if cohort_sets else set()
    stable = set.intersection(*cohort_sets.values()) if cohort_sets else set()
    unstable = sorted(seen - stable)
    if unstable:
        log.warning("excluding %d counties with changing annual reporting "
                    "coverage across %s: %s", len(unstable),
                    f"{min(cohort_sets)}-{max(cohort_sets)}" if cohort_sets else "-",
                    unstable)
    reporting = sorted(stable)

    # A coverage change confined to the test year is reported, never acted on.
    test_years = sorted(y for y in annual_sets if y > pd.Timestamp(cfg["val_end"]).year)
    test_only_gaps: list[str] = []
    for year in test_years:
        gaps = sorted(set(reporting) - annual_sets[year])
        if gaps:
            log.warning("%d counties in the frozen cohort have NO %d records: "
                        "%s. They are kept (their %d county-days will read as "
                        "zero outage) and named in the coverage file -- this is "
                        "a limitation to state, not a cohort edit.",
                        len(gaps), year, gaps, year)
            test_only_gaps.extend(gaps)
    coverage = {
        "cohort_years": sorted(cohort_sets),
        "reporting_counties": reporting,
        "n_reporting": len(reporting),
        "excluded_unstable_counties": unstable,
        "test_year_coverage_gaps": sorted(set(test_only_gaps)),
    }
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
    return hourly[hourly.time >= model_start].copy(), coverage


def build_events(hourly: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    mod = runpy.run_path(str(ROOT / "src" / "03_build_events.py"))
    gb = book("phase2_events")
    events = mod["detect_events"](hourly, cfg, gb)
    lo, hi = [int(v) for v in cfg.get("expected_event_count", [500, 20_000])]
    if not lo <= len(events) <= hi:
        log.warning("Phase 2 detected %d events, outside the expected %d-%d; "
                    "inspect county/event distributions", len(events), lo, hi)
    gb.flush()
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
    check_units = mod["check_units"]
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
            gb = book(f"era5_{period}")
            # Criterion 4. Units are, by this project's own docstring, the
            # second-highest-probability failure in the pipeline -- and until
            # now they were asserted on five days of 2019 and on none of the 72
            # months the study is actually built from.
            check_units(ds, local, gb)
            frame = county_day_features(ds, local, gb)
            gb.flush()
            frames.append(frame)
    weather = pd.concat(frames, ignore_index=True)
    weather = weather[pd.to_datetime(weather.date) <= pd.Timestamp(end)].copy()
    # derived_features reads canopy_county.csv directly now, so the canopy
    # interactions are correct the first time instead of being computed as NaN
    # and silently recomputed here.
    weather = derived_features(weather, cfg)
    weather = label_regimes(weather)
    if weather.canopy_pct.isna().any():
        missing = sorted(weather.loc[weather.canopy_pct.isna(), "fips"].unique())
        raise SystemExit(
            f"canopy_pct missing for {len(missing)} county(ies): {missing}. "
            "Phase 2 requires data/raw/canopy_county.csv -- run "
            "`make phase2-download-canopy`.")
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
    merged["censored"] = merged.censored.notna() & merged.censored.eq(True)
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
    set_phase(2)
    if args.through == "test" and not args.acknowledge_test:
        raise SystemExit(
            f"Refusing to open {pd.Timestamp(cfg['test_start']).year}. Add "
            "--acknowledge-test only after models are frozen.")
    end = pd.Timestamp(cfg["val_end"] if args.through == "validation" else cfg["test_end"])
    paths = output_paths(args.through)

    with timed("phase2_event_build", log):
        hourly, coverage = build_hourly(cfg, end)
        events = build_events(hourly, cfg)
    with timed("phase2_weather_build", log):
        weather = build_weather(cfg, end)
        merged = join_targets(weather, events, hourly)

    hourly.to_parquet(paths["hourly"], index=False)
    events.to_parquet(paths["events"], index=False)
    weather.to_parquet(paths["weather"], index=False)
    merged.to_parquet(paths["merged"], index=False)
    paths["coverage"].write_text(json.dumps(
        {**coverage, "through": args.through, "end": str(end.date())}, indent=2))
    log.info("Phase 2 tables (%s scope): %d county-days, %d events, %d hourly "
             "rows, %d reporting counties",
             args.through, len(merged), len(events), len(hourly),
             coverage["n_reporting"])
    log.info("wrote %s", paths["merged"].name)


if __name__ == "__main__":
    main()
