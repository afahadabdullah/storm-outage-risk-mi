#!/usr/bin/env python
"""Step 4 -- ERA5 -> county-day features, and the join that proves the premise.

Two aggregations, deliberately different (spec section 5.1 and section 12):

  area-weighted MEAN   smooth fields: precipitation, soil moisture, temperature
  MAX over cells       tail fields:   gust, CAPE

Damage is a tail phenomenon. A county-mean gust destroys the signal, which is
why `gust_max` here is the max over intersecting cells and over the day's hours,
not the mean of anything.

UNITS ARE THE SECOND-HIGHEST-PROBABILITY FAILURE. ERA5 wind is m/s,
precipitation is METRES accumulated (not mm), temperature is Kelvin. Each is
asserted explicitly below rather than trusted because the variable name looked
right.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.common.config import PATHS, base_parser, config_from_args
from src.common.era5_io import open_era5
from src.common.gates import book
from src.common.geo import agg_max, agg_mean, agg_quantile, build_weight_matrix, load_counties
from src.common.logio import get_logger, record, timed

log = get_logger("04_features")
FEATURES_PATH = PATHS.processed / "phase1_county_day.parquet"
MERGED_PATH = PATHS.processed / "phase1_merged.parquet"
HOURLY_PATH = PATHS.processed / "phase1_county_hourly.parquet"

KELVIN = 273.15


# --------------------------------------------------------------------------- #
def check_units(ds, cfg, gb) -> None:
    n_expected = int(cfg["window_days"]) * 24
    gb.check("era5_timesteps", ds.sizes["time"] == n_expected,
             f"{ds.sizes['time']} hourly steps, expected {n_expected} "
             "(a CDS year x month x day cross product returns extra days)")

    gust = ds["i10fg"]
    gb.require("gust_units_ms", str(gust.attrs.get("units", "")).replace(" ", "")
               in ("ms**-1", "m/s", "ms-1", "m s**-1".replace(" ", "")),
               f"i10fg units = {gust.attrs.get('units')!r} -- NOT knots, NOT mph",
               criterion=4)
    gb.require("precip_units_metres",
               str(ds["tp"].attrs.get("units", "")).strip() in ("m", "metres", "m of water equivalent"),
               f"tp units = {ds['tp'].attrs.get('units')!r} -- ERA5 precip is "
               "METRES accumulated, not mm", criterion=4)
    gb.require("temp_units_kelvin", str(ds["t2m"].attrs.get("units", "")).strip() == "K",
               f"t2m units = {ds['t2m'].attrs.get('units')!r}", criterion=4)
    gb.check("temp_range_plausible_kelvin",
             bool(180 < float(ds["t2m"].min()) and float(ds["t2m"].max()) < 340),
             f"t2m in [{float(ds['t2m'].min()):.1f}, {float(ds['t2m'].max()):.1f}] K",
             criterion=4)

    gmax = float(gust.max())
    gb.require("gust_max_stormlike", gmax > 15,
               f"domain max gust {gmax:.1f} m/s -- too low for a storm window; "
               "wrong variable, wrong units, or the window missed the storm")
    nan_vars = [v for v in ds.data_vars if bool(ds[v].isnull().any())]
    gb.check("no_nans_in_era5", not nan_vars, f"variables with NaN: {nan_vars or 'none'}")


def county_day_features(ds, cfg, gb) -> pd.DataFrame:
    lats = ds.lat.values
    lons = ds.lon.values
    W, M, geoids, _ = build_weight_matrix(cfg, lats, lons)

    row_sums = np.asarray(W.sum(axis=1)).ravel()
    gb.require("area_weights_sum_to_one", bool(np.allclose(row_sums, 1.0, atol=1e-6)),
               f"weight row sums in [{row_sums.min():.8f}, {row_sums.max():.8f}], "
               f"computed in {cfg['crs_analysis']}", criterion=5)

    times = pd.to_datetime(ds.time.values)
    if times.tz is None:
        times = times.tz_localize("UTC")
    gb.check("era5_times_utc", times.tz is not None, f"ERA5 tz = {times.tz}", criterion=2)

    with timed("era5_to_county", log):
        gust_h = agg_max(M, ds["i10fg"].values)                 # (t, county)
        gust_p95_h = agg_quantile(M, ds["i10fg"].values, 0.95)
        cape_h = agg_max(M, ds["cape"].values)
        wind_h = agg_max(M, np.hypot(ds["u10"].values, ds["v10"].values))
        wind100_h = agg_max(M, np.hypot(ds["u100"].values, ds["v100"].values))
        precip_h = agg_mean(W, ds["tp"].values) * 1000.0        # m -> mm, once, here
        t2m_h = agg_mean(W, ds["t2m"].values) - KELVIN          # K -> degC, once, here
        soil_h = agg_mean(W, (ds["swvl1"].values + ds["swvl2"].values) / 2)
        snow_h = agg_mean(W, ds["sf"].values) * 1000.0          # m w.e. -> mm w.e.
        gust_mean_h = agg_mean(W, ds["i10fg"].values)

    day = pd.to_datetime(times.date)
    thresholds = [float(t) for t in cfg.get("gust_thresholds_ms", [13, 18, 25])]

    frames = []
    for j, geoid in enumerate(geoids):
        d = pd.DataFrame({
            "date": day, "gust": gust_h[:, j], "gust_p95": gust_p95_h[:, j],
            "gust_mean": gust_mean_h[:, j], "wind": wind_h[:, j],
            "wind100": wind100_h[:, j], "precip": precip_h[:, j],
            "t2m": t2m_h[:, j], "soil": soil_h[:, j], "cape": cape_h[:, j],
            "snow": snow_h[:, j],
        })
        agg = d.groupby("date").agg(
            gust_max=("gust", "max"), gust_p95=("gust_p95", "max"),
            gust_mean=("gust_mean", "mean"),
            wind_sustained_max=("wind", "max"), wind100_max=("wind100", "max"),
            precip_total=("precip", "sum"), precip_max_hourly=("precip", "max"),
            soil_moisture_mean=("soil", "mean"),
            cape_max=("cape", "max"),
            temp_min=("t2m", "min"), temp_max=("t2m", "max"),
        )
        for thr in thresholds:
            agg[f"hours_gust_gt_{int(thr)}"] = d.groupby("date").gust.apply(
                lambda s, t=thr: int((s > t).sum()))
        # freezing rain: precip accumulated while 2m T in [-2, +0.5] degC
        agg["freezing_rain_proxy"] = d.assign(
            fr=d.precip.where(d.t2m.between(-2.0, 0.5), 0.0)).groupby("date").fr.sum()
        # wet snow: snowfall accumulated while 2m T in [-1, +1] degC
        agg["snow_water_equiv_wet"] = d.assign(
            ws=d.snow.where(d.t2m.between(-1.0, 1.0), 0.0)).groupby("date").ws.sum()
        agg["fips"] = geoid
        frames.append(agg.reset_index())

    cd = pd.concat(frames, ignore_index=True)

    n_expected = len(geoids) * cd.date.nunique()
    gb.require("county_day_shape", len(cd) == n_expected,
               f"{len(cd)} rows = {len(geoids)} counties x {cd.date.nunique()} days")
    corr = float(cd.gust_max.corr(cd.gust_p95))
    gb.require("gust_internally_coherent", corr > 0.8,
               f"corr(gust_max, gust_p95) = {corr:.3f}")
    return cd


def derived_features(cd: pd.DataFrame, cfg) -> pd.DataFrame:
    """Section 5.2 -- the interactions carry the physical insight."""
    canopy = load_canopy_pct(cfg, cd.fips.unique())
    cd = cd.copy()
    cd["canopy_pct"] = cd.fips.map(canopy)
    counties = load_counties(cfg)
    land_km2 = (counties.ALAND / 1e6).reindex(cd.fips.unique())
    cd["customer_density"] = cd.fips.map(_customer_density(cfg, land_km2))

    doy = pd.to_datetime(cd.date).dt.dayofyear
    cd["leaf_on"] = ((doy >= 120) & (doy <= 300)).astype(int)
    cd["month"] = pd.to_datetime(cd.date).dt.month
    cd["season"] = ((cd.month % 12) // 3).astype(int)

    cd["gust_x_soil"] = cd.gust_max * cd.soil_moisture_mean
    cd["gust_x_canopy"] = cd.gust_max * cd.canopy_pct.fillna(cd.canopy_pct.mean())
    cd["gust_x_leafon"] = cd.gust_max * cd.leaf_on
    cd["ice_x_canopy"] = cd.freezing_rain_proxy * cd.canopy_pct.fillna(
        cd.canopy_pct.mean())
    cd = cd.sort_values(["fips", "date"])
    cd["antecedent_precip_7d"] = (cd.groupby("fips").precip_total
                                    .transform(lambda s: s.rolling(7, min_periods=1)
                                                          .sum().shift(1).fillna(0)))
    return cd


def _customer_density(cfg, land_km2) -> pd.Series:
    mcc_path = PATHS.raw / "MCC.csv"
    mcc = pd.read_csv(mcc_path, dtype=str)
    mcc.columns = [c.strip().lower() for c in mcc.columns]
    f = next(c for c in mcc.columns if "fips" in c)
    c = next(c for c in mcc.columns if "customer" in c)
    s = (mcc.assign(**{f: mcc[f].str.strip().str.zfill(5)})
            .set_index(f)[c].astype(float))
    return (s / land_km2.reindex(s.index)).dropna()


def load_canopy_pct(cfg, fips_list) -> pd.Series:
    """County-mean NLCD tree canopy. Optional in Phase 1 -- no gate depends on it."""
    tif = PATHS.raw / "canopy_pct_clip.tif"
    if not tif.exists():
        log.warning("no canopy raster; canopy features fall back to NaN. "
                    "Not a Phase 1 gate, but fetch it before Phase 2.")
        return pd.Series(np.nan, index=list(fips_list), dtype=float)
    import rasterio
    from rasterio.mask import mask

    counties = load_counties(cfg)
    out = {}
    with rasterio.open(tif) as src:
        geo = counties.to_crs(src.crs)
        for fips, geom in geo.geometry.items():
            try:
                arr, _ = mask(src, [geom], crop=True, nodata=255)
                vals = arr[(arr >= 0) & (arr <= 100)]
                out[fips] = float(vals.mean()) if vals.size else np.nan
            except Exception:                                  # noqa: BLE001
                out[fips] = np.nan
    return pd.Series(out, dtype=float)


# --------------------------------------------------------------------------- #
def label_regimes(cd: pd.DataFrame) -> pd.DataFrame:
    """Section 5.3. Heuristics, not ground truth -- say so in the write-up."""
    cd = cd.copy()
    reg = pd.Series("benign", index=cd.index, dtype=object)
    long_gust = cd.get("hours_gust_gt_13", pd.Series(0, index=cd.index))
    windy = cd.gust_max > 15
    reg[windy & (cd.cape_max > 500) & (long_gust <= 6)] = "convective_wind"
    reg[windy & (cd.cape_max <= 500) & (long_gust > 6)] = "synoptic_wind"
    reg[cd.snow_water_equiv_wet > 0.5] = "wet_snow"
    reg[cd.freezing_rain_proxy > 0.5] = "ice"       # last: ice dominates labelling
    cd["regime_label"] = reg
    return cd


def join_and_prove(cd, ev, cfg, gb) -> pd.DataFrame:
    """Step 5 -- the correlation that proves the whole premise."""
    ev = ev.copy()
    ev["date"] = pd.to_datetime(ev.date).dt.tz_localize(None).dt.normalize()
    agg = ev.groupby(["fips", "date"]).agg(
        customer_hours=("customer_hours", "sum"),
        restoration_hours=("restoration_hours", "max"),
        peak_frac_out=("peak_frac_out", "max"),
        peak_customers_out=("peak_customers_out", "max"),
        concurrent_state_load=("concurrent_state_load", "max"),
        mcc=("mcc", "first"),
        censored=("censored", "any"),
        n_events=("event_id", "size"),
    ).reset_index()

    merged = cd.merge(agg, on=["fips", "date"], how="left")
    gb.require("join_did_not_duplicate", len(merged) == len(cd),
               f"{len(merged)} rows out, {len(cd)} in")
    gb.require("weather_present_for_all_rows", bool(merged.gust_max.notna().all()),
               "weather missing for some county-days")

    # A county absent from EAGLE-I altogether has an unknown outcome, not a
    # confirmed zero-outage outcome.  Exclude it from target construction and
    # name it in the gate report; criterion 1 remains responsible for deciding
    # whether that coverage gap is acceptable for this study.
    hourly = pd.read_parquet(HOURLY_PATH)
    reporting_fips = set(hourly.fips.astype(str))
    excluded = sorted(set(merged.fips.astype(str)) - reporting_fips)
    if excluded:
        gb.note("nonreporting_counties_excluded_from_targets",
                f"{len(excluded)} counties with no EAGLE-I record excluded: {excluded}")
        log.warning("excluding %d counties with no EAGLE-I record from targets: %s",
                    len(excluded), excluded)
        merged = merged[merged.fips.astype(str).isin(reporting_fips)].copy()

    merged["event"] = merged.customer_hours.notna().astype(int)
    merged["customer_hours"] = merged.customer_hours.fillna(0.0)
    # Total customer-hours is exposure-dependent: a large county can appear
    # consequential even when a much smaller share of its customers is out.
    # The premise gate therefore uses outage-hours per covered customer.
    merged["customer_hours_per_customer"] = (
        merged.customer_hours / merged.mcc).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # ---- criterion 6: THE REAL GATE ----------------------------------------
    peak_date = merged.loc[merged.customer_hours.idxmax(), "date"]
    storm = merged[merged.date == peak_date]
    raw_corr = float(storm[["gust_max", "customer_hours"]].corr().iloc[0, 1])
    corr = float(storm[["gust_max", "customer_hours_per_customer"]].corr().iloc[0, 1])
    thr = float(cfg.get("min_hazard_consequence_corr", 0.30))
    gb.note("raw_customer_hours_correlation",
            f"raw total customer-hours correlation = {raw_corr:.3f} (exposure-confounded)")
    log.info("PEAK DAY %s: corr(gust_max, customer-hours/customer) = %.3f "
             "[raw customer-hours %.3f] over %d reporting counties",
             peak_date.date(), corr, raw_corr, len(storm))
    gb.require(
        "hazard_consequence_correlation", corr > thr,
        f"corr={corr:.3f} for gust_max vs customer-hours/customer on "
        f"{peak_date.date()} across {len(storm)} reporting counties "
        f"(raw customer-hours corr={raw_corr:.3f}; gate: >{thr})",
        criterion=6,
        on_fail="If criteria 1-5 passed, the problem is SCIENTIFIC, not "
                "mechanical. Diagnose in this order: (1) timezone alignment, "
                "(2) whether the peak day is the storm day in BOTH datasets, "
                "(3) whether the utilities serving the hardest-hit counties "
                "report into EAGLE-I, (4) whether ERA5 resolved this storm's "
                "gusts -- a convective event ERA5 smooths away is a real and "
                "reportable finding, and the argument for the HRRR upgrade. Do "
                "not proceed to Phase 2 hoping six years reveals a signal five "
                "days around the year's strongest storm could not.")

    # ---- criterion 2: timezone alignment -----------------------------------
    o_peak = merged.loc[merged.customer_hours.idxmax()]
    g_peak = merged.loc[merged.gust_max.idxmax()]
    lag_h = abs((pd.Timestamp(o_peak.date) - pd.Timestamp(g_peak.date))
                / pd.Timedelta(hours=1))
    gb.check("storm_day_alignment", lag_h <= 24,
             f"outage peak day {o_peak.date.date()} vs gust peak day "
             f"{g_peak.date.date()} ({lag_h:.0f} h apart)", criterion=2)
    return merged


def main() -> None:
    args = base_parser(__doc__).parse_args()
    cfg = config_from_args(args)
    gb = book("04_features")

    ds = open_era5(cfg)
    check_units(ds, cfg, gb)

    with timed("feature_build", log):
        cd = county_day_features(ds, cfg, gb)
        cd = derived_features(cd, cfg)
        cd = label_regimes(cd)
        ev = pd.read_parquet(PATHS.processed / "phase1_events.parquet")
        merged = join_and_prove(cd, ev, cfg, gb)

    cd.to_parquet(FEATURES_PATH)
    merged.to_parquet(MERGED_PATH)
    record("feature_build", n_county_days=len(cd),
           n_features=merged.shape[1], regimes=cd.regime_label.value_counts().to_dict())
    gb.flush()
    log.info("regimes: %s", cd.regime_label.value_counts().to_dict())
    log.info("%d county-days x %d columns -> %s", *merged.shape, MERGED_PATH.name)


if __name__ == "__main__":
    main()
