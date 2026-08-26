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


def open_era5(cfg: Config):
    import xarray as xr

    path = era5_path(cfg)
    if not path.exists():
        raise SystemExit(f"{path} missing -- run `make era5-only` and wait for "
                         "the CDS queue, or use --synthetic.")
    ds = xr.open_dataset(path)
    ren = {k: v for k, v in {"valid_time": "time", "latitude": "lat",
                             "longitude": "lon"}.items() if k in ds}
    ds = ds.rename(ren)
    for d in ("expver", "number"):                # CDS sometimes adds singletons
        if d in ds.dims and ds.sizes[d] == 1:
            ds = ds.squeeze(d, drop=True)
    return ds.sortby("time")
