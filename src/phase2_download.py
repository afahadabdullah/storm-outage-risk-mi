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
NLCD_PIXEL_M = 30          # NLCD tree-canopy native resolution


def available_eaglei_years(cfg: Config) -> set[int]:
    """Years the figshare article actually carries, from the live listing.

    The config's `eaglei_years_available` is a claim; this is the fact. Checking
    it once, before the first byte, turns "the download died on file eight of
    nine" into a message naming exactly which years exist.
    """
    import re

    mod = runpy.run_path(str(ROOT / "src" / "01_fetch_outage.py"))
    names = mod["figshare_files"](cfg["sources"]["eaglei_figshare_article"])
    return {int(m.group(1)) for name in names
            if (m := re.search(r"eaglei_outages_(\d{4})", name))}


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

    # Preflight the whole requested span against the live listing, so a study
    # period the archive does not cover fails here rather than partway through.
    wanted = sorted(set(years))
    missing_locally = [y for y in wanted
                       if not (PATHS.raw / f"eaglei_outages_{y}.csv").exists()]
    if missing_locally:
        try:
            have = available_eaglei_years(cfg)
        except Exception as err:                              # noqa: BLE001
            log.warning("could not list the figshare article (%s); proceeding "
                        "year by year", err)
        else:
            gaps = sorted(set(missing_locally) - have)
            if gaps:
                raise SystemExit(
                    f"EAGLE-I years {gaps} are not in figshare article "
                    f"{cfg['sources']['eaglei_figshare_article']}. The article "
                    f"carries {min(have)}-{max(have)}. Either the frozen splits "
                    "in region.yaml reach past the archive, or the article "
                    "moved -- fix region.yaml, do not work around this.")
            log.info("figshare carries EAGLE-I %d-%d; study period %d-%d is covered",
                     min(have), max(have), wanted[0], wanted[-1])

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
        wkid = int(cfg["crs_analysis"].split(":")[-1])
        payload = {
            "f": "json", "geometryType": "esriGeometryPolygon",
            "geometry": json.dumps(geometry),
            "mosaicRule": json.dumps({"where": "beginyear = 2021"}),
            "renderingRule": json.dumps({"rasterFunction": "NLCDTCC_noBkgrd"}),
            # Pin the analysis resolution to NLCD's native 30 m. Without this
            # the service picks a pyramid level of its own choosing, so the
            # "native-pixel calculations" claim above was unverified -- and a
            # coarsened mean is still a plausible percentage, so the 0-100
            # bounds check below would never catch it.
            "pixelSize": json.dumps(
                {"x": NLCD_PIXEL_M, "y": NLCD_PIXEL_M,
                 "spatialReference": {"wkid": wkid}}),
        }
        response = requests.post(endpoint, data=payload, timeout=300)
        response.raise_for_status()
        result = response.json()
        if "error" in result or not result.get("statistics"):
            raise RuntimeError(f"canopy service failed for {fips}: {result}")
        stats = result["statistics"][0]
        mean = float(stats["mean"])
        if not 0 <= mean <= 100:
            raise ValueError(f"invalid canopy mean for {fips}: {mean}")
        # Resolution sanity: the pixel count the service reports should be
        # within a factor of ~2 of land area / 900 m^2. Anything far below that
        # means the statistics came from an overview, not native pixels.
        counted = float(stats.get("count") or 0)
        expected = float(row.get("ALAND", 0) or 0) / (NLCD_PIXEL_M ** 2)
        ratio = counted / expected if expected > 0 else float("nan")
        if expected > 0 and counted > 0 and not 0.5 <= ratio <= 2.0:
            log.warning("canopy %s: %.0f pixels vs ~%.0f expected at %dm "
                        "(ratio %.2f) -- the service may have used an overview "
                        "level rather than native resolution", fips, counted,
                        expected, NLCD_PIXEL_M, ratio)
        rows.append({"fips": str(fips), "canopy_pct": mean,
                     "pixel_count": counted, "pixel_count_expected_30m": round(expected)})
        log.info("canopy %s: %.1f%% (%.0f px)", fips, mean, counted)
    frame = pd.DataFrame(rows)
    tmp = out.with_suffix(out.suffix + ".part")
    frame.to_csv(tmp, index=False)
    tmp.replace(out)
    log.info("official 2021 NLCD canopy county means at %dm -> %s (%d counties)",
             NLCD_PIXEL_M, out, len(frame))


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
