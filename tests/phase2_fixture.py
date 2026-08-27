"""A miniature but structurally faithful Phase 2 input set.

Six counties, a 45-cell ERA5 grid, twelve monthly NetCDFs carrying the
CDS-style ``valid_time``/``latitude``/``longitude`` coordinates the real
downloads use, two annual EAGLE-I CSVs, an MCC table and a canopy table.
Enough to run build -> train -> compose -> forecast -> value end to end in
about twenty seconds, with no downloads and no credentials.

This exists because the Phase 2 test suite was eight rows of synthetic frame
and passed happily while storm-blocked cross-validation was collapsing to a
single group at 83 counties. A fixture that is too small to reach a failure
mode is not covering it.

Run directly to populate a scratch tree:

    python tests/phase2_fixture.py /tmp/p2scratch
"""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
import yaml
from shapely.geometry import box

FIPS = [f"26{i:03d}" for i in (1, 3, 5, 7, 9, 11)]
CUSTOMERS = [40000, 55000, 30000, 90000, 25000, 61000]
CANOPY = [42.1, 55.3, 18.7, 61.0, 33.2, 47.8]
BBOX = [-86.0, 44.0, -84.0, 45.0]

# One year is enough to exercise every code path, and keeps the suite fast.
SPLITS = {
    "train_start": "2018-01-01", "train_end": "2018-06-30",
    "val_start": "2018-07-01", "val_end": "2018-09-30",
    "test_start": "2018-10-01", "test_end": "2018-12-31",
}


def write_config(root: Path, region_src: Path) -> Path:
    """A region.yaml pointed at the fixture's bbox and shortened splits."""
    cfg = yaml.safe_load(region_src.read_text())
    cfg["region_name"] = "MiniMI"
    cfg["bbox"] = BBOX
    cfg.update(SPLITS)
    cfg["case_studies"] = [{"name": "Nov 2018 wind event",
                            "date": "2018-11-15", "type": "convective wind"}]
    out = Path(root) / "config" / "mini_region.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump(cfg, out.open("w"), sort_keys=False)
    return out


def write_fixture(root: Path, seed: int = 7) -> Path:
    raw = Path(root) / "data" / "raw"
    (raw / "era5_monthly").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    # ---- counties: six boxes tiling the mini bbox --------------------------
    boxes = []
    for iy in range(2):
        for ix in range(3):
            x0 = BBOX[0] + ix * 0.6
            y0 = BBOX[1] + iy * 0.5
            boxes.append(box(x0, y0, x0 + 0.6, y0 + 0.5))
    gpd.GeoDataFrame(
        {"GEOID": FIPS, "NAME": [f"C{f}" for f in FIPS],
         "ALAND": [2.5e9] * 6, "AWATER": [1e8] * 6},
        geometry=boxes, crs="EPSG:4326",
    ).to_parquet(raw / "tiger_counties_2023.parquet")

    mcc = pd.DataFrame({"County_FIPS": FIPS, "County": [f"C{f}" for f in FIPS],
                        "Customers": CUSTOMERS})
    mcc.to_csv(raw / "MCC.csv", index=False)
    customers = dict(zip(FIPS, CUSTOMERS))

    pd.DataFrame({"fips": FIPS, "canopy_pct": CANOPY}).to_csv(
        raw / "canopy_county.csv", index=False)

    # ---- a shared storm calendar, so weather and outages actually agree ----
    days = pd.date_range("2017-12-01", "2018-12-31", freq="D")
    stormy = rng.random(len(days)) < 0.12
    intensity = np.where(stormy, rng.uniform(0.6, 1.0, len(days)),
                         rng.uniform(0.0, 0.12, len(days)))
    by_day = dict(zip(days.normalize(), intensity))

    # ---- EAGLE-I annual CSVs ----------------------------------------------
    for year in (2017, 2018):
        start = f"{year}-01-01" if year == 2018 else "2017-12-01"
        hours = pd.date_range(start, f"{year}-12-31 23:00", freq="h")
        amp = np.array([by_day[h.normalize()] for h in hours])
        shape = np.exp(-((np.arange(24) - 15) ** 2) / 12.0)
        profile = np.tile(shape, len(hours) // 24 + 1)[:len(hours)]
        pieces = []
        for fips in FIPS:
            base = rng.uniform(20, 120, len(hours))
            out = base + amp * profile * customers[fips] * 0.06 * \
                rng.uniform(0.6, 1.4, len(hours))
            keep = out > 30
            pieces.append(pd.DataFrame({
                "fips_code": fips, "county": f"C{fips}", "state": "Michigan",
                "sum": out[keep].round().astype(int),
                "run_start_time": hours[keep]}))
        pd.concat(pieces).to_csv(raw / f"eaglei_outages_{year}.csv", index=False)

    # ---- monthly ERA5 ------------------------------------------------------
    lats = np.arange(BBOX[1], BBOX[3] + 0.01, 0.25)
    lons = np.arange(BBOX[0], BBOX[2] + 0.01, 0.25)
    for period in pd.period_range("2018-01", "2018-12", freq="M"):
        times = pd.date_range(period.start_time, period.end_time, freq="h")
        shape3 = (len(times), len(lats), len(lons))
        amp = np.array([by_day[t.normalize()] for t in times])[:, None, None]
        diurnal = np.sin(np.arange(len(times)) % 24 / 24 * 2 * np.pi)[:, None, None]
        gust = 4 + 26 * amp * (0.6 + 0.4 * rng.random(shape3))
        dims = ("valid_time", "latitude", "longitude")
        ds = xr.Dataset(
            {"i10fg": (dims, gust),
             "cape": (dims, 900 * amp * rng.random(shape3)),
             "u10": (dims, gust * 0.6 * rng.random(shape3)),
             "v10": (dims, gust * 0.6 * rng.random(shape3)),
             "u100": (dims, gust * 0.8 * rng.random(shape3)),
             "v100": (dims, gust * 0.8 * rng.random(shape3)),
             "tp": (dims, 0.004 * amp * rng.random(shape3)),
             "t2m": (dims, 273.15
                     + 10 * np.sin((period.month - 4) / 12 * 2 * np.pi)
                     + 4 * diurnal + rng.normal(0, 1.5, shape3)),
             "swvl1": (dims, 0.25 + 0.1 * rng.random(shape3)),
             "swvl2": (dims, 0.28 + 0.1 * rng.random(shape3)),
             "sf": (dims, 0.0015 * amp * rng.random(shape3)),
             "sd": (dims, 0.02 * rng.random(shape3))},
            coords={"valid_time": times, "latitude": lats, "longitude": lons})
        # The unit attributes matter: check_units asserts on exactly these.
        for name, unit in (("i10fg", "m s**-1"), ("tp", "m"), ("t2m", "K")):
            ds[name].attrs["units"] = unit
        ds.to_netcdf(raw / "era5_monthly" /
                     f"era5_{period.year}{period.month:02d}.nc")
    return raw


if __name__ == "__main__":
    target = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    repo = Path(__file__).resolve().parents[1]
    print("fixture written to", write_fixture(target))
    cfg = write_config(target, repo / "config" / "region.yaml")
    print("config written to", cfg)
    print("\nrun the chain against it with:")
    for step in ("phase2_build.py --through validation", "phase2_train.py",
                 "phase2_compose.py", "phase2_forecast.py --synthetic-gefs",
                 "phase2_value.py"):
        print(f"  python {repo / 'src'}/{step} --config {cfg} "
              f"--phase2 {repo / 'config' / 'phase2.yaml'}")
