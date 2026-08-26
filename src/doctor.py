#!/usr/bin/env python
"""Preflight for a fresh machine: `make doctor`.

Checks the things that silently cost you an afternoon on a remote box --
a missing GDAL stack, an unwritten ~/.cdsapirc, a disk too small for the NLCD
tile, an egress rule that blocks the CDS but not PyPI. Exits non-zero if
anything REQUIRED is missing; warnings do not fail.
"""
from __future__ import annotations

import os
import shutil
import socket
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
OK, WARN, BAD = "  ok  ", " warn ", " FAIL "

REQUIRED = ["numpy", "pandas", "scipy", "yaml", "pyarrow", "matplotlib", "joblib",
            "geopandas", "shapely", "pyproj", "xarray", "netCDF4",
            "sklearn", "lightgbm", "lifelines", "requests"]
OPTIONAL = {
    "cdsapi": "ERA5 download (step 2). Required for the real run.",
    "boto3": "GEFS download from AWS (step 2). Required for the real run.",
    "cfgrib": "reading GEFS GRIB2 (step 7). Required for the real run.",
    "rasterio": "NLCD canopy clip (step 2). Optional -- no Phase 1 gate uses canopy.",
    "rioxarray": "NLCD canopy clip. Optional.",
    "ngboost": "magnitude model in Phase 2. Not used by phase1.yaml.",
    "pytest": "the assertion suite (`make test`).",
}
HOSTS = {
    "api.figshare.com": "EAGLE-I outage data",
    "cds.climate.copernicus.eu": "ERA5 (Copernicus CDS)",
    "noaa-gefs-pds.s3.amazonaws.com": "GEFS forecast ensemble",
    "www2.census.gov": "TIGER county boundaries",
    "www.mrlc.gov": "NLCD tree canopy (optional)",
}
# EAGLE-I year ~1-2 GB, NLCD zip 1-2 GB + extracted CONUS tile, ERA5 50-200 MB,
# GEFS subset < 1 GB, processed outputs small.
MIN_FREE_GB = 20


def line(status: str, what: str, detail: str = "") -> None:
    print(f"[{status}] {what}" + (f"   {detail}" if detail else ""))


def main() -> int:
    failures = 0
    print(f"\nstorm-outage-risk preflight -- {ROOT}\n" + "-" * 68)

    v = sys.version_info
    good = (3, 10) <= (v.major, v.minor) <= (3, 12)
    line(OK if good else WARN, f"python {v.major}.{v.minor}.{v.micro}",
         "" if good else "3.11 is what env/environment.yml pins")

    print("\nrequired packages")
    import importlib.util as iu
    for m in REQUIRED:
        if iu.find_spec(m):
            line(OK, m)
        else:
            failures += 1
            line(BAD, m, "conda env create -f env/environment.yml")

    print("\noptional packages")
    for m, why in OPTIONAL.items():
        line(OK if iu.find_spec(m) else WARN, m, "" if iu.find_spec(m) else why)

    print("\ncredentials")
    rc = Path.home() / ".cdsapirc"
    if rc.exists():
        txt = rc.read_text()
        has_url = "cds.climate.copernicus.eu/api" in txt
        has_key = "key:" in txt
        legacy = "UID:" in txt or ":" in txt.split("key:")[-1].split("\n")[0].strip()
        line(OK if (has_url and has_key) else BAD, "~/.cdsapirc",
             "" if has_url else "url must be https://cds.climate.copernicus.eu/api "
                                "(the endpoint changed in 2024-25)")
        if legacy and has_key:
            line(WARN, "~/.cdsapirc key format",
                 "looks like the old UID:KEY form; the current API wants a bare "
                 "personal access token")
        failures += 0 if (has_url and has_key) else 1
    else:
        failures += 1
        line(BAD, "~/.cdsapirc", "create it: url: https://cds.climate.copernicus.eu/api"
                                 "  /  key: <personal access token>. You must also "
                                 "accept the ERA5 licence once, in the CDS web UI.")

    print("\nnetwork (DNS + TCP 443)")
    for host, why in HOSTS.items():
        try:
            socket.create_connection((host, 443), timeout=6).close()
            line(OK, host, why)
        except Exception as err:                                # noqa: BLE001
            line(WARN, host, f"unreachable ({type(err).__name__}) -- {why}")

    print("\ndisk")
    free_gb = shutil.disk_usage(ROOT).free / 1e9
    ok = free_gb >= MIN_FREE_GB
    if not ok:
        failures += 1
    line(OK if ok else BAD, f"{free_gb:,.1f} GB free at {ROOT}",
         "" if ok else f"want >= {MIN_FREE_GB} GB (EAGLE-I year ~1-2 GB, NLCD "
                       "CONUS tile is the big one)")

    print("\nconfig")
    import yaml
    p1 = yaml.safe_load((ROOT / "config" / "phase1.yaml").read_text())
    ws = str(p1.get("window_start"))
    line(WARN if ws == "AUTO" else OK, f"window_start = {ws}",
         "run `make fetch` then `python src/select_window.py --write`"
         if ws == "AUTO" else "")

    print("-" * 68)
    if failures:
        print(f"{failures} required check(s) failed. Fix those before `make fetch`.\n")
        return 1
    print("preflight clean. Next: make phase1-synthetic, then make fetch.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
