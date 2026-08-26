"""Opening the ERA5 file, with the coordinate renames the CDS netcdf needs.

Kept out of the numbered fetch script so that step 4 can import it: numbered
modules cannot be imported, only executed.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import PATHS, Config


def era5_path(cfg: Config) -> Path:
    start = pd.Timestamp(cfg.required("window_start"))
    return PATHS.raw / f"era5_{start:%Y%m%d}_{int(cfg['window_days'])}d.nc"


def era5_file_status(path: Path) -> tuple[bool, str]:
    """Check that a purported ERA5 file is an openable, non-empty NetCDF.

    Merely checking ``Path.exists`` is unsafe: the CDS client creates its
    target while bytes are still arriving, so a concurrent pipeline can see a
    partial file and mistake it for a completed cache entry.
    """
    if not path.exists():
        return False, "missing"
    if path.stat().st_size == 0:
        return False, "empty"

    import xarray as xr

    try:
        with xr.open_dataset(path) as ds:
            if not ds.data_vars:
                return False, "contains no data variables"
            time_name = next((name for name in ("valid_time", "time")
                              if name in ds.coords or name in ds.dims), None)
            if time_name is None or ds.sizes.get(time_name, 0) == 0:
                return False, "contains no time samples"
            # Force one value through the backend. Metadata alone can remain
            # readable in a truncated HDF5/NetCDF file.
            sample = ds[next(iter(ds.data_vars))]
            indexers = {dim: 0 for dim in sample.dims}
            sample.isel(indexers).load()
    except Exception as err:                                  # noqa: BLE001
        return False, f"{type(err).__name__}: {err}"
    return True, "ok"


def open_era5(cfg: Config):
    import xarray as xr

    path = era5_path(cfg)
    if not path.exists():
        raise SystemExit(f"{path} missing -- run `make era5-only` and wait for "
                         "the CDS queue, or use --synthetic.")
    valid, reason = era5_file_status(path)
    if not valid:
        raise SystemExit(
            f"{path} is not a completed ERA5 NetCDF ({reason}). If "
            "`make era5-only` is still running, wait for it. Otherwise move "
            "this file aside and rerun `python src/02_fetch_weather.py "
            "--only era5 --force`."
        )
    ds = xr.open_dataset(path)
    ren = {k: v for k, v in {"valid_time": "time", "latitude": "lat",
                             "longitude": "lon"}.items() if k in ds}
    ds = ds.rename(ren)
    for d in ("expver", "number"):                # CDS sometimes adds singletons
        if d in ds.dims and ds.sizes[d] == 1:
            ds = ds.squeeze(d, drop=True)
    return ds.sortby("time")
