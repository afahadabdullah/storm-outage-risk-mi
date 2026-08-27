#!/usr/bin/env python
"""Download the complete Phase 2 inputs in restartable pieces.

Outages are annual EAGLE-I CSVs. Phase 2 ERA5 downloads one Michigan-only,
full-study slice from ECMWF's geo-chunked ARCO store, then combines local
monthly slices with five residual variables from CDS in merged NetCDF files.
GEFS is fetched only for the frozen 2023 case studies and four lead times.
Downloading test-year bytes is not model evaluation; ``phase2_train`` still
refuses to read the test rows without ``--evaluate-test``.
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

from src.common.config import PATHS, ROOT, Config, load_config
from src.common.era5_io import era5_file_status, publish_era5_download
from src.common.geo import fetch_counties, load_counties
from src.common.logio import dir_size_mb, get_logger

log = get_logger("phase2_download")
NLCD_PIXEL_M = 30          # NLCD tree-canopy native resolution

# CDS request names -> the names every downstream feature builder expects.
ERA5_TO_SHORT = {
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

# The ECMWF product table lists descriptive CDS names, while the live Zarr
# arrays use these GRIB-style short keys. Gust is named fg10 in ARCO and i10fg
# in CDS NetCDF, but both represent the same parameter.
ARCO_SOURCE_BY_CDS = {
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "instantaneous_10m_wind_gust": "fg10",
    "100m_u_component_of_wind": "u100",
    "100m_v_component_of_wind": "v100",
    "total_precipitation": "tp",
    "2m_temperature": "t2m",
}


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
        except Exception as err:
            log.warning("could not list the figshare article (%s); proceeding "
                        "year by year", err)
        else:
            gaps = sorted(set(missing_locally) - have)
            if gaps:
                raise SystemExit(
                    f"EAGLE-I years {gaps} are not in figshare article "
                    f"{cfg['sources']['eaglei_figshare_article']}. The article "
                    f"carries {sorted(have)}. Either the frozen splits "
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


def era5_arco_cache_path(cfg: Config) -> Path:
    years = study_years(cfg)
    return PATHS.raw / "era5_arco" / f"era5_arco_{years[0]}_{years[-1]}.nc"


def era5_source_variables(variables: list[str]) -> tuple[list[str], list[str]]:
    """Split requested fields between ECMWF ARCO and residual CDS access."""
    unknown = sorted(set(variables) - set(ERA5_TO_SHORT))
    if unknown:
        raise ValueError(f"ERA5 variables have no output-name mapping: {unknown}")
    arco = [name for name in variables if name in ARCO_SOURCE_BY_CDS]
    cds = [name for name in variables if name not in ARCO_SOURCE_BY_CDS]
    return arco, cds


def _month_bounds(year: int, month: int) -> tuple[pd.Timestamp, pd.Timestamp, int]:
    _, n_days = calendar.monthrange(year, month)
    start = pd.Timestamp(year=year, month=month, day=1)
    end = start + pd.Timedelta(days=n_days) - pd.Timedelta(hours=1)
    return start, end, n_days


def _select_arco_period(ds, cfg: Config, start: pd.Timestamp, end: pd.Timestamp,
                        variables: list[str], expected_hours: int,
                        rename: bool = True):
    """Select fields, time and bbox lazily, before any ARCO bytes are loaded."""
    source_names = [ARCO_SOURCE_BY_CDS[name] for name in variables]
    missing = sorted(set(source_names) - set(ds.data_vars))
    if missing:
        raise RuntimeError(
            f"ECMWF ARCO store is missing expected fields: {missing}; live keys "
            f"are {sorted(ds.data_vars)}")
    for coord in ("time", "latitude", "longitude"):
        if coord not in ds.coords:
            raise RuntimeError(f"ECMWF ARCO store has no {coord!r} coordinate")

    w, s, e, n = (float(value) for value in cfg["bbox"])
    lat = ds["latitude"]
    lat_descends = float(lat.isel(latitude=0)) > float(lat.isel(latitude=-1))
    lat_slice = slice(n, s) if lat_descends else slice(s, n)

    lon = ds["longitude"]
    uses_360 = float(lon.max()) > 180.0
    select_w, select_e = ((w % 360), (e % 360)) if uses_360 else (w, e)
    subset = ds[source_names].sel(
        time=slice(start, end),
        latitude=lat_slice,
        longitude=slice(select_w, select_e),
    )
    if uses_360:
        subset = subset.assign_coords(
            longitude=((subset.longitude + 180) % 360) - 180).sortby("longitude")

    if subset.sizes.get("time") != expected_hours:
        raise RuntimeError(
            f"ECMWF ARCO {start}..{end} returned "
            f"{subset.sizes.get('time', 0)} hours; expected {expected_hours}")
    if subset.sizes.get("latitude", 0) == 0 or subset.sizes.get("longitude", 0) == 0:
        raise RuntimeError(f"ECMWF ARCO bbox selection is empty: {cfg['bbox']}")

    if rename:
        output_names = {ARCO_SOURCE_BY_CDS[name]: ERA5_TO_SHORT[name]
                        for name in variables}
        subset = subset.rename(output_names)
    return subset


def _select_arco_month(ds, cfg: Config, year: int, month: int,
                       variables: list[str]):
    """Select one cached ARCO month and convert its keys to pipeline names."""
    start, end, n_days = _month_bounds(year, month)
    return _select_arco_period(ds, cfg, start, end, variables, n_days * 24)


def _cds_api_token() -> str:
    """Read the bearer token without ever logging it."""
    import os

    import yaml

    token = os.environ.get("CDSAPI_KEY", "").strip()
    if not token:
        rc = Path.home() / ".cdsapirc"
        if not rc.exists():
            raise RuntimeError("~/.cdsapirc is missing; ARCO requires a CDS API token")
        token = str((yaml.safe_load(rc.read_text()) or {}).get("key", "")).strip()
    if not token:
        raise RuntimeError("CDS API token is empty")
    if ":" in token:
        raise RuntimeError(
            "ARCO requires the current bare CDS personal-access token, not UID:KEY")
    return token


def era5_arco_cache_status(path: Path, cfg: Config) -> tuple[bool, str]:
    valid, reason = era5_file_status(path)
    if not valid:
        return valid, reason

    import xarray as xr

    variables, _ = era5_source_variables(list(cfg["era5_variables"]))
    expected = {ARCO_SOURCE_BY_CDS[name] for name in variables}
    start = pd.Timestamp(cfg["train_start"])
    end = pd.Timestamp(cfg["test_end"]) + pd.Timedelta(hours=23)
    expected_hours = int((end - start) / pd.Timedelta(hours=1)) + 1
    try:
        with xr.open_dataset(path) as ds:
            missing = sorted(expected - set(ds.data_vars))
            if missing:
                return False, f"missing ARCO variables {missing}"
            if ds.sizes.get("time") != expected_hours:
                return False, (f"has {ds.sizes.get('time', 0)} hours; "
                               f"expected {expected_hours}")
    except Exception as err:
        return False, f"{type(err).__name__}: {err}"
    return True, "ok"


def fetch_era5_arco_cache(cfg: Config, force: bool = False) -> Path:
    """Fetch the full-study regional ARCO slice once, matching its time chunks."""
    import xarray as xr

    out = era5_arco_cache_path(cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".part")
    if out.exists() and not force:
        valid, reason = era5_arco_cache_status(out, cfg)
        if valid:
            log.info("cached regional ARCO %s (%.1f MB)", out.name, dir_size_mb(out))
            return out
        log.warning("invalid regional ARCO cache %s (%s); replacing", out, reason)
    if tmp.exists():
        raise SystemExit(
            f"{tmp} exists. Confirm no ARCO cache job owns it, then move the "
            "stale partial file aside before retrying.")

    variables, _ = era5_source_variables(list(cfg["era5_variables"]))
    start = pd.Timestamp(cfg["train_start"])
    end = pd.Timestamp(cfg["test_end"]) + pd.Timedelta(hours=23)
    expected_hours = int((end - start) / pd.Timedelta(hours=1)) + 1
    url = cfg["sources"]["era5_arco_geo_url"]
    log.info("ARCO ERA5 %s..%s: one regional slice, %d variables",
             start.date(), end.date(), len(variables))
    ds = xr.open_zarr(
        url,
        consolidated=True,
        storage_options={
            "headers": {"Authorization": f"Bearer {_cds_api_token()}"}
        },
    )
    try:
        subset = _select_arco_period(
            ds, cfg, start, end, variables, expected_hours, rename=False)
        encoding = {name: {"zlib": True, "complevel": 1}
                    for name in subset.data_vars}
        subset.to_netcdf(tmp, engine="netcdf4", encoding=encoding)
    finally:
        ds.close()
    valid, reason = era5_arco_cache_status(tmp, cfg)
    if not valid:
        raise RuntimeError(f"regional ARCO cache is invalid ({reason})")
    tmp.replace(out)
    log.info("published regional ARCO cache %s (%.1f MB)", out.name, dir_size_mb(out))
    return out


def _write_arco_month(cfg: Config, year: int, month: int,
                      variables: list[str], dest: Path) -> None:
    import xarray as xr

    cache = era5_arco_cache_path(cfg)
    valid, reason = era5_arco_cache_status(cache, cfg)
    if not valid:
        raise SystemExit(
            f"Regional ARCO cache {cache} is not ready ({reason}). Run "
            "`sbatch slurm/phase2_download_arco.sbatch` first.")
    log.info("local ARCO cache %04d-%02d: %d variables", year, month,
             len(variables))
    with xr.open_dataset(cache) as ds:
        subset = _select_arco_month(ds, cfg, year, month, variables).load()
    encoding = {name: {"zlib": True, "complevel": 1}
                for name in subset.data_vars}
    subset.to_netcdf(dest, engine="netcdf4", encoding=encoding)
    subset.close()


def _download_cds_month(cfg: Config, year: int, month: int,
                        variables: list[str], dest: Path) -> None:
    import cdsapi

    _, _, n_days = _month_bounds(year, month)
    w, s, e, n = cfg["bbox"]
    request = {
        "product_type": ["reanalysis"],
        "variable": variables,
        "year": [f"{year:04d}"],
        "month": [f"{month:02d}"],
        "day": [f"{day:02d}" for day in range(1, n_days + 1)],
        "time": [f"{hour:02d}:00" for hour in range(24)],
        "area": [round(n, 2), round(w, 2), round(s, 2), round(e, 2)],
        "data_format": "netcdf",
        "download_format": "unarchived",
    }
    log.info("CDS residual ERA5 %04d-%02d: %d days x %d variables",
             year, month, n_days, len(variables))
    cdsapi.Client(url=cfg["sources"]["cds_url"]).retrieve(
        cfg["sources"]["era5_dataset"], request, str(dest))


def _load_era5_part(path: Path):
    import xarray as xr

    with xr.open_dataset(path) as opened:
        ds = opened.load()
    rename = {}
    if "valid_time" in ds.dims and "time" not in ds.dims:
        rename["valid_time"] = "time"
    rename.update({name: short for name, short in ERA5_TO_SHORT.items()
                   if name in ds.data_vars})
    ds = ds.rename(rename)
    for dim in ("expver", "number"):
        if dim in ds.dims and ds.sizes[dim] == 1:
            ds = ds.squeeze(dim, drop=True)
    return ds


def era5_month_status(path: Path, cfg: Config, year: int,
                      month: int) -> tuple[bool, str]:
    valid, reason = era5_file_status(path)
    if not valid:
        return valid, reason

    import xarray as xr

    expected = {ERA5_TO_SHORT[name] for name in cfg["era5_variables"]}
    expected_hours = _month_bounds(year, month)[2] * 24
    try:
        with xr.open_dataset(path) as ds:
            time_name = "time" if "time" in ds.dims else "valid_time"
            missing = sorted(expected - set(ds.data_vars))
            if missing:
                return False, f"missing variables {missing}"
            if ds.sizes.get(time_name) != expected_hours:
                return False, (f"has {ds.sizes.get(time_name, 0)} hours; "
                               f"expected {expected_hours}")
    except Exception as err:
        return False, f"{type(err).__name__}: {err}"
    return True, "ok"


def fetch_era5_month(cfg: Config, year: int, month: int,
                     force: bool = False) -> Path:
    out = era5_month_path(year, month)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".part")
    if out.exists() and not force:
        valid, reason = era5_month_status(out, cfg, year, month)
        if valid:
            log.info("cached ERA5 %s (%.1f MB)", out.name, dir_size_mb(out))
            return out
        log.warning("invalid monthly cache %s (%s); replacing", out, reason)
    arco_tmp = out.with_name(out.name + ".arco.part.nc")
    cds_download = out.with_name(out.name + ".cds.download.part")
    cds_tmp = out.with_name(out.name + ".cds.part.nc")
    partials = [tmp, arco_tmp, cds_download, cds_tmp]
    stale = [path for path in partials if path.exists()]
    if stale:
        raise SystemExit(
            f"Partial ERA5 files exist: {[str(path) for path in stale]}. Confirm "
            "no downloader owns them, then move them aside before retrying.")

    backend = str(cfg.get("era5_backend", "cds")).lower()
    variables = list(cfg["era5_variables"])
    if backend == "cds":
        _download_cds_month(cfg, year, month, variables, tmp)
        kind = publish_era5_download(tmp, out)
    elif backend == "arco":
        import xarray as xr

        arco_variables, cds_variables = era5_source_variables(variables)
        _write_arco_month(cfg, year, month, arco_variables, arco_tmp)
        parts = [_load_era5_part(arco_tmp)]
        if cds_variables:
            _download_cds_month(cfg, year, month, cds_variables, cds_download)
            publish_era5_download(cds_download, cds_tmp)
            parts.append(_load_era5_part(cds_tmp))
        try:
            merged = xr.merge(parts, compat="override", join="exact")
            encoding = {name: {"zlib": True, "complevel": 1}
                        for name in merged.data_vars}
            merged.to_netcdf(tmp, engine="netcdf4", encoding=encoding)
            merged.close()
        finally:
            for part in parts:
                part.close()
        valid, reason = era5_month_status(tmp, cfg, year, month)
        if not valid:
            raise RuntimeError(f"hybrid ARCO/CDS ERA5 file is invalid ({reason})")
        tmp.replace(out)
        for part_path in (arco_tmp, cds_tmp, cds_download):
            if part_path.exists():
                part_path.unlink()
        kind = (f"ARCO {len(arco_variables)} variables + "
                f"CDS {len(cds_variables)} variables")
    else:
        raise ValueError(f"era5_backend must be 'arco' or 'cds', got {backend!r}")

    valid, reason = era5_month_status(out, cfg, year, month)
    if not valid:
        raise RuntimeError(f"published ERA5 month is invalid ({reason})")
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
    ap.add_argument(
        "--only",
        choices=["all", "outages", "era5-arco", "era5", "gefs", "canopy"],
        default="all",
    )
    ap.add_argument("--years", help="comma-separated subset; default frozen full period")
    ap.add_argument("--months", default="1,2,3,4,5,6,7,8,9,10,11,12")
    ap.add_argument("--case", action="append",
                    help="case-study slug; repeat to select, default all")
    ap.add_argument("--era5-backend", choices=["arco", "cds"],
                    help="override phase2.yaml for ERA5 downloads")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_phase2(args.config, args.phase2)
    if args.era5_backend:
        cfg["era5_backend"] = args.era5_backend
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
    if (args.only in ("all", "era5-arco")
            and str(cfg.get("era5_backend", "cds")).lower() == "arco"):
        fetch_era5_arco_cache(cfg, force=args.force)
    if args.only in ("all", "era5"):
        fetch_era5(cfg, years, months, force=args.force)
    if args.only in ("all", "canopy"):
        fetch_canopy_required(cfg, force=args.force)
    if args.only in ("all", "gefs"):
        fetch_case_gefs(cfg, set(args.case) if args.case else None,
                        force=args.force)


if __name__ == "__main__":
    main()
