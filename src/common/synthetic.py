"""Stand-in data generator: `make phase1-synthetic`.

Why this exists. The phase 1 spec is a plumbing test, and plumbing tests should
not be blocked on a CDS queue or a Globus transfer. This module writes files
with the same names, schemas, units and dtypes the real fetchers produce, with a
real (noisy) gust -> outage relationship baked in, so every join, unit
assertion, shape assertion and gate can be exercised offline in seconds.

It is NOT a simulation of anything. No result computed from synthetic data means
anything at all -- it only proves the code executes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PATHS, Config
from .geo import CELL_DEG, counties_path
from .logio import get_logger

log = get_logger("synthetic")
SEED = 20260826
N_COUNTIES = 24


def grid_from_bbox(bbox) -> tuple[np.ndarray, np.ndarray]:
    w, s, e, n = bbox
    lons = np.arange(np.floor(w / CELL_DEG) * CELL_DEG,
                     np.ceil(e / CELL_DEG) * CELL_DEG + 1e-9, CELL_DEG)
    lats = np.arange(np.ceil(n / CELL_DEG) * CELL_DEG,
                     np.floor(s / CELL_DEG) * CELL_DEG - 1e-9, -CELL_DEG)
    return lats, lons


def make_counties(cfg: Config):
    """A tidy lattice of square counties inside the bbox, TIGER-shaped schema."""
    import geopandas as gpd
    from shapely.geometry import box

    out = counties_path(cfg)
    w, s, e, n = cfg["bbox"]
    w, s, e, n = w + 1.0, s + 1.0, e - 1.0, n - 1.0
    ncol, nrow = 6, 4
    xs = np.linspace(w, e, ncol + 1)
    ys = np.linspace(s, n, nrow + 1)
    rows, sf = [], str(cfg["state_fips"][0]).zfill(2)
    k = 0
    for j in range(nrow):
        for i in range(ncol):
            k += 1
            rows.append({
                "GEOID": f"{sf}{2 * k - 1:03d}",          # odd codes, like Michigan
                "NAME": f"Synth{k:02d}",
                "ALAND": 1.5e9 + 5e8 * ((k % 5) - 2),
                "AWATER": 1e8,
                "geometry": box(xs[i], ys[j], xs[i + 1], ys[j + 1]),
            })
    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    gdf.to_parquet(out)
    log.info("synthetic counties -> %s (%d)", out.name, len(gdf))
    return gdf


def _storm_field(lats, lons, times, rng):
    """A gust ridge tracking west -> east across the domain, plus a diurnal floor."""
    lon_g, lat_g = np.meshgrid(lons, lats)
    nt = len(times)
    hours = np.arange(nt)
    track_lon = np.linspace(lons.min() - 1, lons.max() + 1, nt)
    peak_hour = nt // 2
    intensity = 26 * np.exp(-0.5 * ((hours - peak_hour) / 9.0) ** 2)

    gust = np.empty((nt, *lon_g.shape), dtype=np.float32)
    for t in range(nt):
        d = ((lon_g - track_lon[t]) / 1.6) ** 2 + ((lat_g - 44.5) / 3.2) ** 2
        gust[t] = (5.0 + intensity[t] * np.exp(-0.5 * d)
                   + rng.normal(0, 0.8, lon_g.shape)).astype(np.float32)
    return np.clip(gust, 0.5, None)


def make_era5(cfg: Config, start: pd.Timestamp, days: int):
    """NetCDF with ERA5 variable names, CF units and UTC hourly times."""
    import xarray as xr

    rng = np.random.default_rng(SEED)
    lats, lons = grid_from_bbox(cfg["bbox"])
    times = pd.date_range(start, periods=days * 24, freq="1h", tz="UTC").tz_localize(None)
    gust = _storm_field(lats, lons, times, rng)
    shape = gust.shape

    wind = gust * 0.62 + rng.normal(0, 0.4, shape).astype(np.float32)
    ang = rng.uniform(0, 2 * np.pi, shape).astype(np.float32)
    ds = xr.Dataset(
        {
            "i10fg": (("time", "lat", "lon"), gust,
                      {"units": "m s**-1", "long_name": "Instantaneous 10m wind gust"}),
            "u10": (("time", "lat", "lon"), (wind * np.cos(ang)).astype(np.float32),
                    {"units": "m s**-1"}),
            "v10": (("time", "lat", "lon"), (wind * np.sin(ang)).astype(np.float32),
                    {"units": "m s**-1"}),
            "u100": (("time", "lat", "lon"), (1.2 * wind * np.cos(ang)).astype(np.float32),
                     {"units": "m s**-1"}),
            "v100": (("time", "lat", "lon"), (1.2 * wind * np.sin(ang)).astype(np.float32),
                     {"units": "m s**-1"}),
            # total_precipitation is METRES accumulated, not mm. This trips people.
            "tp": (("time", "lat", "lon"),
                   np.clip(gust / 12000 + rng.normal(0, 1e-4, shape), 0, None).astype(np.float32),
                   {"units": "m"}),
            "t2m": (("time", "lat", "lon"),
                    (279.0 + 6 * np.sin(np.arange(shape[0]) / 24 * 2 * np.pi)[:, None, None]
                     + rng.normal(0, 0.7, shape)).astype(np.float32),
                    {"units": "K"}),
            "swvl1": (("time", "lat", "lon"),
                      np.clip(0.28 + rng.normal(0, 0.03, shape), 0, 0.6).astype(np.float32),
                      {"units": "m**3 m**-3"}),
            "swvl2": (("time", "lat", "lon"),
                      np.clip(0.31 + rng.normal(0, 0.02, shape), 0, 0.6).astype(np.float32),
                      {"units": "m**3 m**-3"}),
            "cape": (("time", "lat", "lon"),
                     np.clip(gust * 40 + rng.normal(0, 60, shape), 0, None).astype(np.float32),
                     {"units": "J kg**-1"}),
            "sf": (("time", "lat", "lon"), np.zeros(shape, np.float32),
                   {"units": "m of water equivalent"}),
            "sd": (("time", "lat", "lon"), np.zeros(shape, np.float32),
                   {"units": "m of water equivalent"}),
        },
        coords={"time": times, "lat": lats, "lon": lons},
    )
    out = PATHS.raw / f"era5_{start:%Y%m%d}_{days}d.nc"
    ds.to_netcdf(out)
    log.info("synthetic ERA5 -> %s  %s", out.name, dict(ds.sizes))
    return out, ds


def make_outages(cfg: Config, counties, era5, start: pd.Timestamp, days: int):
    """EAGLE-I schema: fips_code, county, state, sum, run_start_time (UTC, 15-min).

    Outages are driven by that county's max gust with lognormal noise and a
    routine-fault floor, so baseline removal and gate 6 both have something real
    to bite on.
    """
    rng = np.random.default_rng(SEED + 1)
    lats = era5.lat.values
    lons = era5.lon.values
    gust = era5["i10fg"].values

    cen = counties.geometry.centroid
    iy = np.abs(lats[None, :] - cen.y.values[:, None]).argmin(axis=1)
    ix = np.abs(lons[None, :] - cen.x.values[:, None]).argmin(axis=1)

    mcc = (rng.integers(9_000, 260_000, len(counties))).astype(int)
    pd.DataFrame({"County_FIPS": counties.GEOID.values, "Customers": mcc}) \
      .to_csv(PATHS.raw / "MCC.csv", index=False)

    hours = pd.date_range(start, periods=days * 24, freq="1h", tz="UTC")
    recs = []
    for c, (geoid, yy, xx, m) in enumerate(
            zip(counties.GEOID.values, iy, ix, mcc)):
        g = gust[:, yy, xx]
        # sharp response above ~15 m/s; floor of routine faults always present
        frac = 0.0016 + 0.0055 * np.clip(g - 15.0, 0, None) ** 1.35 / 10
        frac = frac * np.exp(rng.normal(0, 0.35, len(g)))
        # restoration tail: outages decay over a few hours, they do not vanish
        decayed = np.maximum.accumulate(np.zeros_like(frac))
        acc = 0.0
        for t in range(len(frac)):
            acc = max(frac[t], acc * 0.72)
            decayed[t] = acc
        out = np.clip(decayed, 0, 0.9) * m
        for t, h in enumerate(hours):
            for q in range(4):                     # 15-minute resolution
                recs.append((geoid, f"Synth{c + 1:02d}", cfg["region_name"],
                             int(round(out[t] * rng.uniform(0.94, 1.06))),
                             h + pd.Timedelta(minutes=15 * q)))
    df = pd.DataFrame(recs, columns=["fips_code", "county", "state", "sum",
                                     "run_start_time"])
    year = start.year
    out = PATHS.raw / f"eaglei_outages_{year}.csv"
    df.to_csv(out, index=False)
    log.info("synthetic EAGLE-I -> %s  (%d rows)", out.name, len(df))
    return out


def make_gefs(cfg: Config, era5, n_members: int, lead_hours):
    """Members = ERA5 truth + member spread + a deliberate warm gust bias.

    The bias is the point: gate 9 checks that quantile mapping removes it. If the
    synthetic members had no bias, the bias-correction test could pass while
    doing nothing.
    """
    import xarray as xr

    rng = np.random.default_rng(SEED + 2)
    sub = era5.isel(time=[h for h in lead_hours if h < era5.sizes["time"]])
    mem = []
    for m in range(n_members):
        g = sub["i10fg"].values * rng.uniform(1.18, 1.34) + rng.normal(
            1.5, 1.2, sub["i10fg"].shape)
        mem.append(np.clip(g, 0.2, None))
    ds = xr.Dataset(
        {
            "gust": (("member", "time", "lat", "lon"),
                     np.stack(mem).astype(np.float32), {"units": "m s**-1"}),
            "tp": (("member", "time", "lat", "lon"),
                   np.stack([sub["tp"].values * rng.uniform(0.7, 1.4)
                             for _ in range(n_members)]).astype(np.float32),
                   {"units": "m"}),
            "t2m": (("member", "time", "lat", "lon"),
                    np.stack([sub["t2m"].values + rng.normal(0, 1.0)
                              for _ in range(n_members)]).astype(np.float32),
                    {"units": "K"}),
        },
        coords={"member": np.arange(n_members), "time": sub.time,
                "lat": sub.lat, "lon": sub.lon},
    )
    out = PATHS.raw / "gefs_synthetic.nc"
    ds.to_netcdf(out)
    log.info("synthetic GEFS -> %s  (%d members)", out.name, n_members)
    return out


def generate_all(cfg: Config, start: pd.Timestamp, days: int, n_members: int,
                 lead_hours):
    counties = make_counties(cfg)
    era5_path, era5 = make_era5(cfg, start, days)
    make_outages(cfg, counties, era5, start, days)
    make_gefs(cfg, era5, n_members, lead_hours)
    return era5_path
