#!/usr/bin/env python
"""Step 2 -- ERA5 (fitting), GEFS (forecast), NLCD canopy and terrain (static).

START THE ERA5 REQUEST FIRST and let it queue while you do everything else:
`make era5-only`. The CDS queue is the only unpredictable wait in the project
and it has been known to run hours at peak.

CDS API note: the endpoint and dataset identifiers changed in 2024-25. This uses
the current form -- url https://cds.climate.copernicus.eu/api with a personal
access token as `key`, cdsapi >= 0.7.7, and `data_format` / `download_format`
rather than the old `format`. An old tutorial script fails here with an
unhelpful error.

  ~/.cdsapirc
      url: https://cds.climate.copernicus.eu/api
      key: <YOUR-PERSONAL-ACCESS-TOKEN>

Download scope for phase 1 (see spec section 5): ERA5 is the ONLY source where
the five-day constraint actually saves you anything. EAGLE-I is a whole-year
file, NLCD and TIGER have no temporal dimension at all.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.common.config import PATHS, base_parser, config_from_args
from src.common.era5_io import era5_file_status, era5_path, open_era5  # noqa: F401
from src.common.logio import dir_size_mb, get_logger, record, timed

log = get_logger("02_weather")

# CDS short names -> the names the rest of the pipeline uses.
CDS_TO_SHORT = {
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "instantaneous_10m_wind_gust": "i10fg",
    "100m_u_component_of_wind": "u100",
    "100m_v_component_of_wind": "v100",
    "total_precipitation": "tp",
    "2m_temperature": "t2m",
    "volumetric_soil_water_layer_1": "swvl1",
    "volumetric_soil_water_layer_2": "swvl2",
    "convective_available_potential_energy": "cape",
    "snowfall": "sf",
    "snow_depth": "sd",
}


def fetch_era5(cfg, force: bool = False) -> Path:
    """Exactly 5 days x bbox x 12 variables. This is where the saving is."""
    out = era5_path(cfg)
    tmp = out.with_suffix(out.suffix + ".part")
    if out.exists() and not force:
        valid, reason = era5_file_status(out)
        if valid:
            log.info("cached ERA5 %s (%.1f MB)", out.name, dir_size_mb(out))
            return out
        log.warning("ignoring incomplete/invalid ERA5 cache %s (%s)", out.name, reason)
    if tmp.exists():
        raise SystemExit(
            f"{tmp} exists: another ERA5 download may still be running. Check "
            "`jobs -l` and `tail -f logs/era5.log`; after confirming no downloader "
            "is active, move the stale .part file aside and retry."
        )
    import cdsapi

    start = pd.Timestamp(cfg.required("window_start"))
    days = pd.date_range(start, periods=int(cfg["window_days"]), freq="D")
    w, s, e, n = cfg["bbox"]

    request = {
        "product_type": ["reanalysis"],
        "variable": list(cfg["era5_variables"]),
        "year": sorted({f"{d:%Y}" for d in days}),
        "month": sorted({f"{d:%m}" for d in days}),
        "day": sorted({f"{d:%d}" for d in days}),
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": [round(n, 2), round(w, 2), round(s, 2), round(e, 2)],  # N/W/S/E
        "data_format": "netcdf",
        "download_format": "unarchived",
    }
    log.info("CDS request: %s", request)
    log.info("this queues -- go do step 1, NLCD and TIGER while it runs")
    with timed("era5_download", log):
        cdsapi.Client(url=cfg["sources"]["cds_url"]).retrieve(
            cfg["sources"]["era5_dataset"], request, str(tmp))
    valid, reason = era5_file_status(tmp)
    if not valid:
        raise RuntimeError(
            f"CDS download completed but {tmp} is not a readable NetCDF ({reason}). "
            "Keep it for inspection; do not run Phase 1 with this file."
        )
    tmp.replace(out)
    log.info("validated and published ERA5 cache -> %s", out.name)
    record("era5_download", megabytes=dir_size_mb(out))

    if len(request["month"]) > 1 or len(request["day"]) > 1:
        log.warning("CDS expands year x month x day as a CROSS PRODUCT. If your "
                    "window crosses a month boundary you have extra days. "
                    "Harmless here; in Phase 2 request month by month.")
    return out


# --------------------------------------------------------------------------- #
# GEFS
# --------------------------------------------------------------------------- #
GEFS_KEY = "gefs.{date:%Y%m%d}/{cycle:02d}/atmos/{product}/{member}.t{cycle:02d}z.pgrb2s.0p25.f{fhr:03d}"
GEFS_VARS = {"GUST": "gust", "APCP": "tp", "TMP": "t2m",
             "UGRD": "u10", "VGRD": "v10", "CAPE": "cape"}
GEFS_REFORECAST_KEY = (
    "GEFSv12/reforecast/{date:%Y}/{date:%Y%m%d}00/{member}/Days:1-10/"
    "gust_sfc_{date:%Y%m%d}00_{member}.grib2"
)


def gefs_members(n: int) -> list[str]:
    """Control first, then perturbations. 1 + 30 = the full 31."""
    return (["gec00"] + [f"gep{i:02d}" for i in range(1, 31)])[:n]


def reforecast_member(member: str) -> str:
    """Map operational GEFS member labels onto the reforecast archive labels."""
    return "c00" if member == "gec00" else f"p{member[-2:]}"


def _forecast_ranges(idx_text: str, wanted_hours: list[int]) -> dict[int, tuple[int, int | None]]:
    """Return byte ranges for selected forecast-hour messages in a single-var index."""
    lines = [ln for ln in idx_text.splitlines() if ln.strip()]
    offsets = [int(ln.split(":")[1]) for ln in lines]
    out = {}
    for i, line in enumerate(lines):
        match = re.search(r"(\d+) hour fcst", line)
        if not match:
            continue
        hour = int(match.group(1))
        if hour in wanted_hours:
            out[hour] = (offsets[i], offsets[i + 1] - 1 if i + 1 < len(offsets) else None)
    return out


def fetch_gefs_reforecast(cfg, init: pd.Timestamp, s3, force: bool = False) -> Path:
    """Fetch historical GEFSv12 reforecast gusts when operational files aged out.

    The NOAA operational bucket is a short rolling archive.  Its permanent
    reforecast archive covers 2000--2019, has five daily 00Z members, and stores
    each variable separately.  Phase 1 already uses five members, so write the
    selected gust messages in the same one-member/one-lead GRIB layout that
    ``07_forecast_cases.py`` reads from the operational source.
    """
    if init.year > 2019:
        raise RuntimeError(
            f"operational GEFS files for {init:%Y-%m-%d} have aged out, and the "
            "NOAA GEFSv12 reforecast archive ends in 2019. Choose a 2018--2019 "
            "Phase 1 window or provide an independent historical forecast archive."
        )

    out_dir = PATHS.raw / f"gefs_{init:%Y%m%d}00"
    out_dir.mkdir(exist_ok=True)
    bucket = cfg.get("sources", {}).get("gefs_reforecast_bucket", "noaa-gefs-retrospective")
    requested = [int(h) for h in cfg.get("gefs_lead_hours", [24, 48, 72])]
    n_members = int(cfg["n_gefs_members"])
    if n_members > 5:
        raise ValueError(
            "the daily GEFSv12 reforecast has five members; set n_gefs_members "
            "to 5 for a historical Phase 1 run"
        )
    # Reforecasts begin at +3 h; Phase 1 only needs the daily forecast fields.
    wanted = [h for h in requested if h >= 3]
    if len(wanted) != len(requested):
        log.warning("GEFS reforecast has no f000; skipping it and using leads %s", wanted)

    written = []
    with timed("gefs_reforecast_download", log):
        for member in gefs_members(n_members):
            source_member = reforecast_member(member)
            key = GEFS_REFORECAST_KEY.format(date=init, member=source_member)
            idx = s3.get_object(Bucket=bucket, Key=key + ".idx")["Body"].read().decode()
            ranges = _forecast_ranges(idx, wanted)
            missing = sorted(set(wanted) - set(ranges))
            if missing:
                raise ValueError(f"GEFS reforecast {key} lacks forecast hours {missing}")
            for fhr in wanted:
                dest = out_dir / f"{member}_f{fhr:03d}.grib2"
                if dest.exists() and not force:
                    written.append(dest)
                    continue
                lo, hi = ranges[fhr]
                blob = s3.get_object(
                    Bucket=bucket, Key=key, Range=f"bytes={lo}-{'' if hi is None else hi}"
                )["Body"].read()
                dest.write_bytes(blob)
                written.append(dest)
    record("gefs_reforecast_download", megabytes=dir_size_mb(out_dir), files=len(written))
    log.info("GEFS reforecast: %d gust files, %.1f MB -> %s",
             len(written), dir_size_mb(out_dir), out_dir.name)
    return out_dir


def fetch_gefs(cfg, init: pd.Timestamp, cycle: int = 0, force: bool = False) -> Path:
    """Byte-range subset via the .idx sidecar: pull only the messages we need.

    pgrb2sp25 is the 0.25 degree primary-field product and it does carry GUST at
    the surface, plus UGRD/VGRD 10m, APCP, TMP 2m and CAPE. Downloading whole
    files instead of ranges costs about 20x the bytes for the same result.
    """
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config as BotoConfig

    out_dir = PATHS.raw / f"gefs_{init:%Y%m%d}{cycle:02d}"
    out_dir.mkdir(exist_ok=True)
    bucket = cfg["sources"]["gefs_bucket"]
    product = cfg["sources"]["gefs_product"]
    s3 = boto3.client("s3", config=BotoConfig(signature_version=UNSIGNED))

    members = gefs_members(int(cfg["n_gefs_members"]))
    fhrs = [int(h) for h in cfg.get("gefs_lead_hours", [0, 24, 48, 72])]
    written = []
    with timed("gefs_download", log):
        for member in members:
            for fhr in fhrs:
                key = GEFS_KEY.format(date=init, cycle=cycle, product=product,
                                      member=member, fhr=fhr)
                dest = out_dir / f"{member}_f{fhr:03d}.grib2"
                if dest.exists() and not force:
                    written.append(dest)
                    continue
                try:
                    idx = s3.get_object(Bucket=bucket, Key=key + ".idx")["Body"] \
                            .read().decode()
                except Exception as err:                       # noqa: BLE001
                    code = getattr(err, "response", {}).get("Error", {}).get("Code")
                    if code not in {"NoSuchKey", "404", "NotFound"}:
                        raise
                    log.warning("operational GEFS file has aged out: s3://%s/%s; "
                                "using NOAA's permanent GEFSv12 reforecast archive",
                                bucket, key)
                    return fetch_gefs_reforecast(cfg, init, s3, force=force)
                ranges = _idx_ranges(idx, GEFS_VARS)
                blobs = []
                for lo, hi in ranges:
                    rng = f"bytes={lo}-{'' if hi is None else hi}"
                    blobs.append(s3.get_object(Bucket=bucket, Key=key,
                                               Range=rng)["Body"].read())
                dest.write_bytes(b"".join(blobs))
                written.append(dest)
    record("gefs_download", megabytes=dir_size_mb(out_dir), files=len(written))
    log.info("GEFS: %d files, %.1f MB -> %s",
             len(written), dir_size_mb(out_dir), out_dir.name)
    return out_dir


def _idx_ranges(idx_text: str, wanted: dict[str, str]) -> list[tuple[int, int | None]]:
    """GRIB .idx lines look like:  12:1234567:d=2023...:GUST:surface:3 hour fcst:"""
    lines = [ln for ln in idx_text.splitlines() if ln.strip()]
    offsets = [int(ln.split(":")[1]) for ln in lines]
    out = []
    for i, ln in enumerate(lines):
        parts = ln.split(":")
        var, level = parts[3], parts[4]
        if var not in wanted:
            continue
        if var in ("UGRD", "VGRD") and "10 m" not in level:
            continue
        if var == "TMP" and "2 m" not in level:
            continue
        if var == "CAPE" and "surface" not in level:
            continue
        hi = offsets[i + 1] - 1 if i + 1 < len(offsets) else None
        out.append((offsets[i], hi))
    if not out:
        raise ValueError("no requested variables found in the GRIB index -- "
                         "check the product; pgrb2sp25 should contain GUST")
    return out


# --------------------------------------------------------------------------- #
# Static layers
# --------------------------------------------------------------------------- #
def fetch_canopy(cfg, force: bool = False) -> Path | None:
    """NLCD percent tree canopy, CONUS tile clipped to bbox. Static -- do it once.

    Optional for Phase 1: no gate criterion depends on canopy. It is fetched
    here anyway because doing so removes it from the Phase 2 critical path.
    """
    out = PATHS.raw / "canopy_pct_clip.tif"
    if out.exists() and not force:
        log.info("cached canopy %s", out.name)
        return out
    try:
        import rioxarray  # noqa: F401
        import requests
        import zipfile, io, rasterio
        from rasterio.mask import mask
        from shapely.geometry import box

        url = cfg["sources"]["nlcd_tcc_url"]
        log.info("downloading NLCD canopy (1-2 GB CONUS tile) %s", url)
        zpath = PATHS.raw / "nlcd_tcc_conus.zip"
        if not zpath.exists():
            r = requests.get(url, stream=True, timeout=3600)
            r.raise_for_status()
            with open(zpath, "wb") as fh:
                for chunk in r.iter_content(1 << 22):
                    fh.write(chunk)
        with zipfile.ZipFile(zpath) as z:
            tif = next(n for n in z.namelist() if n.endswith(".tif"))
            z.extract(tif, PATHS.raw)
        src_path = PATHS.raw / tif
        with rasterio.open(src_path) as src:
            import geopandas as gpd
            geom = gpd.GeoSeries([box(*[cfg["bbox"][i] for i in (0, 1, 2, 3)])],
                                 crs="EPSG:4326").to_crs(src.crs)
            arr, transform = mask(src, geom.geometry, crop=True)
            meta = src.meta | {"height": arr.shape[1], "width": arr.shape[2],
                               "transform": transform}
        with rasterio.open(out, "w", **meta) as dst:
            dst.write(arr)
        log.info("canopy clipped -> %s (%.1f MB)", out.name, dir_size_mb(out))
        return out
    except Exception as err:                                   # noqa: BLE001
        log.warning("canopy fetch failed (%s: %s). Phase 1 does not gate on "
                    "canopy; features that use it will be NaN-filled. Get it "
                    "from https://www.mrlc.gov/data/type/nlcd-tree-canopy-cover "
                    "before Phase 2.", type(err).__name__, err)
        return None


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--only", choices=["era5", "gefs", "canopy", "all"], default="all")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = config_from_args(args)

    if args.synthetic:
        log.warning("synthetic mode: weather is generated by run_phase1.")
        return

    if args.only in ("era5", "all"):
        fetch_era5(cfg, force=args.force)
    if args.only in ("gefs", "all"):
        start = pd.Timestamp(cfg.required("window_start"))
        peak_day = start + pd.Timedelta(days=(int(cfg["window_days"]) - 1) // 2)
        init = peak_day - pd.Timedelta(days=int(cfg.get("forecast_lead_days", 2)))
        log.info("GEFS init %s for peak day %s (lead %s d)",
                 init.date(), peak_day.date(), cfg.get("forecast_lead_days", 2))
        fetch_gefs(cfg, init, force=args.force)
    if args.only in ("canopy", "all"):
        fetch_canopy(cfg, force=args.force)


if __name__ == "__main__":
    main()
