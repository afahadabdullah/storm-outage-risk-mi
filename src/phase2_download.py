#!/usr/bin/env python
"""Download the complete Phase 2 inputs in restartable pieces.

Outages are annual EAGLE-I CSVs. ERA5 is monthly because CDS expands year,
month, and day selections as a cross-product and large multi-year requests are
fragile. GEFS is fetched only for the frozen 2023 case studies and four lead
times. Downloading test-year bytes is not model evaluation; ``phase2_train``
still refuses to read the test rows without ``--evaluate-test``.
"""
from __future__ import annotations

import argparse
import calendar
import json
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.common.config import PATHS, Config, ROOT, load_config
from src.common.era5_io import era5_file_status, publish_era5_download
from src.common.geo import fetch_counties, load_counties
from src.common.logio import dir_size_mb, get_logger

log = get_logger("phase2_download")


def load_phase2(region: str, phase2: str) -> Config:
    cfg = load_config(region, phase2)
    if cfg.is_phase1:
        raise SystemExit("Phase 2 downloader cannot run with phase: 1")
    return cfg


def study_years(cfg: Config) -> list[int]:
    return list(range(pd.Timestamp(cfg["train_start"]).year,
                      pd.Timestamp(cfg["test_end"]).year + 1))


def fetch_outages(cfg: Config, years: list[int], force: bool = False) -> None:
    mod = runpy.run_path(str(ROOT / "src" / "01_fetch_outage.py"))
    fetch_eaglei = mod["fetch_eaglei"]
    if force:
        log.warning("--force does not delete annual CSVs; validated cached files are reused")
    fetch_counties(cfg, force=force)
    for year in years:
        outages, _ = fetch_eaglei(cfg, year)
        log.info("EAGLE-I %d ready: %s (%.1f MB)",
                 year, outages.name, dir_size_mb(outages))


def era5_month_path(year: int, month: int) -> Path:
    return PATHS.raw / "era5_monthly" / f"era5_{year}{month:02d}.nc"


def fetch_era5_month(cfg: Config, year: int, month: int,
                     force: bool = False) -> Path:
    out = era5_month_path(year, month)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".part")
    if out.exists() and not force:
        valid, reason = era5_file_status(out)
        if valid:
            log.info("cached ERA5 %s (%.1f MB)", out.name, dir_size_mb(out))
            return out
        log.warning("invalid monthly cache %s (%s); replacing", out, reason)
    if tmp.exists():
        raise SystemExit(
            f"{tmp} exists. Confirm no other downloader owns it, then move the "
            "stale partial file aside before retrying.")

    import cdsapi

    _, n_days = calendar.monthrange(year, month)
    w, s, e, n = cfg["bbox"]
    request = {
        "product_type": ["reanalysis"],
        "variable": list(cfg["era5_variables"]),
        "year": [f"{year:04d}"],
        "month": [f"{month:02d}"],
        "day": [f"{day:02d}" for day in range(1, n_days + 1)],
        "time": [f"{hour:02d}:00" for hour in range(24)],
        "area": [round(n, 2), round(w, 2), round(s, 2), round(e, 2)],
        "data_format": "netcdf",
        "download_format": "unarchived",
    }
    log.info("CDS ERA5 %04d-%02d: %d days x %d variables", year, month,
             n_days, len(request["variable"]))
    cdsapi.Client(url=cfg["sources"]["cds_url"]).retrieve(
        cfg["sources"]["era5_dataset"], request, str(tmp))
    kind = publish_era5_download(tmp, out)
    log.info("published %s (%s, %.1f MB)", out.name, kind, dir_size_mb(out))
    return out


def fetch_era5(cfg: Config, years: list[int], months: list[int],
               force: bool = False) -> None:
    for year in years:
        for month in months:
            fetch_era5_month(cfg, year, month, force=force)


def fetch_case_gefs(cfg: Config, selected: set[str] | None,
                    force: bool = False) -> None:
    mod = runpy.run_path(str(ROOT / "src" / "02_fetch_weather.py"))
    fetch_gefs = mod["fetch_gefs"]
    leads = [int(v) for v in cfg.get("forecast_lead_days", [5, 3, 2, 1])]
    for case in cfg["case_studies"]:
        slug = case["name"].lower().replace(" ", "-")
        if selected and slug not in selected:
            continue
        target = pd.Timestamp(case["date"])
        for lead in leads:
            init = target - pd.Timedelta(days=lead)
            case_cfg = Config(cfg.copy())
            case_cfg["gefs_lead_hours"] = list(
                range(lead * 24, lead * 24 + 24, 3))
            log.info("GEFS %s: init %s, target %s, day-%d",
                     case["name"], init.date(), target.date(), lead)
            fetch_gefs(case_cfg, init, force=force)


def fetch_canopy_required(cfg: Config, force: bool = False) -> None:
    """Fetch county means directly from the official 30 m ImageServer.

    The legacy bulk ZIP moved. Phase 2 only consumes county means, so asking
    the official service for zonal statistics avoids downloading a multi-GB
    CONUS raster while retaining native-pixel calculations.
    """
    import requests
    from shapely.geometry.polygon import orient

    out = PATHS.raw / "canopy_county.csv"
    if out.exists() and not force:
        log.info("cached canopy county table %s", out.name)
        return
    fetch_counties(cfg)
    counties = load_counties(cfg)
    service = cfg["sources"]["nlcd_tcc_image_service"].rstrip("/")
    endpoint = service + "/computeStatisticsHistograms"
    rows = []
    for fips, row in counties.iterrows():
        geom = row.geometry
        polygons = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
        rings = []
        for polygon in polygons:
            # ArcGIS REST expects clockwise exterior and counter-clockwise holes.
            polygon = orient(polygon, sign=-1.0)
            rings.append([list(point) for point in polygon.exterior.coords])
            rings.extend([[list(point) for point in interior.coords]
                          for interior in polygon.interiors])
        geometry = {"rings": rings,
                    "spatialReference": {"wkid": int(cfg["crs_analysis"].split(":")[-1])}}
        payload = {
            "f": "json", "geometryType": "esriGeometryPolygon",
            "geometry": json.dumps(geometry),
            "mosaicRule": json.dumps({"where": "beginyear = 2021"}),
            "renderingRule": json.dumps({"rasterFunction": "NLCDTCC_noBkgrd"}),
        }
        response = requests.post(endpoint, data=payload, timeout=300)
        response.raise_for_status()
        result = response.json()
        if "error" in result or not result.get("statistics"):
            raise RuntimeError(f"canopy service failed for {fips}: {result}")
        mean = float(result["statistics"][0]["mean"])
        if not 0 <= mean <= 100:
            raise ValueError(f"invalid canopy mean for {fips}: {mean}")
        rows.append({"fips": str(fips), "canopy_pct": mean})
        log.info("canopy %s: %.1f%%", fips, mean)
    pd.DataFrame(rows).to_csv(out, index=False)
    log.info("official 2021 NLCD canopy county means -> %s", out)


def parse_int_list(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "config" / "region.yaml"))
    ap.add_argument("--phase2", default=str(ROOT / "config" / "phase2.yaml"))
    ap.add_argument("--only", choices=["all", "outages", "era5", "gefs", "canopy"],
                    default="all")
    ap.add_argument("--years", help="comma-separated subset; default frozen full period")
    ap.add_argument("--months", default="1,2,3,4,5,6,7,8,9,10,11,12")
    ap.add_argument("--case", action="append",
                    help="case-study slug; repeat to select, default all")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_phase2(args.config, args.phase2)
    years = parse_int_list(args.years) if args.years else study_years(cfg)
    months = parse_int_list(args.months)
    allowed = set(study_years(cfg))
    if not set(years) <= allowed:
        raise SystemExit(f"years must be within frozen study period {sorted(allowed)}")
    if not set(months) <= set(range(1, 13)):
        raise SystemExit("months must be in 1..12")

    PATHS.ensure()
    if args.only in ("all", "outages"):
        outage_years = sorted(set(years) | {pd.Timestamp(cfg["train_start"]).year - 1})
        fetch_outages(cfg, outage_years, force=args.force)
    if args.only in ("all", "era5"):
        fetch_era5(cfg, years, months, force=args.force)
    if args.only in ("all", "canopy"):
        fetch_canopy_required(cfg, force=args.force)
    if args.only in ("all", "gefs"):
        fetch_case_gefs(cfg, set(args.case) if args.case else None,
                        force=args.force)


if __name__ == "__main__":
    main()
