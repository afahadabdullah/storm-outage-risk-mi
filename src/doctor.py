#!/usr/bin/env python
"""Preflight for a fresh machine: `make doctor` (Phase 1) / `make doctor-phase2`.

Checks the things that silently cost you an afternoon on a remote box --
a missing GDAL stack, an unwritten ~/.cdsapirc, a disk too small for the NLCD
tile, an egress rule that blocks the CDS but not PyPI. Exits non-zero if
anything REQUIRED is missing; warnings do not fail.

`--phase 2` promotes the packages Phase 2 actually imports from optional to
required. That distinction is not cosmetic: `properscoring` is imported lazily
inside `crps_from_quantiles`, which runs AFTER the model bundle is dumped, so a
node missing it burns the whole training job and leaves a model with no metrics.
"""
from __future__ import annotations

import argparse
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
    "properscoring": "CRPS in Phase 2 validation. Not used by phase1.yaml.",
    "pytest": "the assertion suite (`make test`).",
}
# Phase 2 imports these for real. `properscoring` and `ngboost` are imported
# lazily, deep inside the training run, so a missing one does not surface until
# hours of compute have already been spent.
PHASE2_REQUIRED = {
    "properscoring": "CRPS. Imported inside evaluate(), AFTER the model is dumped.",
    "ngboost": "phase2.yaml sets magnitude_model: ngboost. A missing import "
               "silently falls back to LightGBM quantiles.",
    "cdsapi": "72 monthly ERA5 requests.",
    "boto3": "GEFS case-study download.",
    "cfgrib": "reading the GEFS GRIB2 subset in phase2_forecast.",
    "statsmodels": "logistic and negative-binomial GLM reference models (spec 6.1, 6.2).",
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


def main(phase: int = 1) -> int:
    failures = 0
    print(f"\nstorm-outage-risk preflight (phase {phase}) -- {ROOT}\n" + "-" * 68)

    v = sys.version_info
    good = (3, 10) <= (v.major, v.minor) <= (3, 12)
    line(OK if good else WARN, f"python {v.major}.{v.minor}.{v.micro}",
         "" if good else "3.11 is what env/environment.yml pins")

    print("\nrequired packages")
    import importlib.util as iu
    prefix = Path(sys.prefix).resolve()
    strays = []
    for m in REQUIRED:
        spec = iu.find_spec(m)
        if spec is None:
            failures += 1
            line(BAD, m, "conda env create -f env/environment.yml")
            continue
        origin = Path(spec.origin).resolve() if spec.origin else None
        # A package resolved from ~/.local means user site-packages is shadowing
        # the environment: you are running a different version than you pinned.
        if origin and not str(origin).startswith(str(prefix)):
            strays.append((m, origin))
            line(BAD, m, f"loaded from {origin.parent.parent} -- NOT this env")
            failures += 1
        else:
            line(OK, m)
    if strays:
        line(BAD, "user site-packages is shadowing the env",
             "export PYTHONNOUSERSITE=1  (the Makefile sets this; a bare "
             "`python src/...` outside make does not)")

    if phase == 2:
        print("\nrequired for phase 2")
        for m, why in PHASE2_REQUIRED.items():
            if iu.find_spec(m) is None:
                failures += 1
                line(BAD, m, why)
            else:
                line(OK, m)

    print("\noptional packages")
    for m, why in OPTIONAL.items():
        if phase == 2 and m in PHASE2_REQUIRED:
            continue
        line(OK if iu.find_spec(m) else WARN, m, "" if iu.find_spec(m) else why)

    print("\nknown incompatibilities")
    try:
        import scipy
        import lifelines
        sv = tuple(int(x) for x in scipy.__version__.split(".")[:2])
        lv = tuple(int(x) for x in lifelines.__version__.split(".")[:2])
        bad = sv >= (1, 14) and lv < (0, 29)
        if bad:
            failures += 1
        line(BAD if bad else OK,
             f"scipy {scipy.__version__} + lifelines {lifelines.__version__}",
             "lifelines < 0.29 calls scipy.integrate.trapz, removed in scipy "
             "1.14. Upgrade lifelines to 0.30.3." if bad else "")
    except Exception as err:                                    # noqa: BLE001
        line(WARN, "scipy/lifelines pairing", f"could not check ({err})")

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
    region = yaml.safe_load((ROOT / "config" / "region.yaml").read_text())
    if phase == 1:
        p1 = yaml.safe_load((ROOT / "config" / "phase1.yaml").read_text())
        ws = str(p1.get("window_start"))
        line(WARN if ws == "AUTO" else OK, f"window_start = {ws}",
             "run `make fetch` then `python src/select_window.py --write`"
             if ws == "AUTO" else "")
    else:
        # Phase 2 runs region.yaml + phase2.yaml and never reads phase1.yaml.
        # What matters here is that the frozen splits are intact and that the
        # study period is actually available from the upstream archive.
        p2 = ROOT / "config" / "phase2.yaml"
        if not p2.exists():
            failures += 1
            line(BAD, "config/phase2.yaml", "missing")
        else:
            over = yaml.safe_load(p2.read_text()) or {}
            line(OK if int(over.get("phase", 0)) == 2 else BAD,
                 f"phase2.yaml phase = {over.get('phase')}",
                 "" if int(over.get("phase", 0)) == 2 else "must be 2")
            failures += 0 if int(over.get("phase", 0)) == 2 else 1

        splits = ("train_start", "train_end", "val_start", "val_end",
                  "test_start", "test_end")
        unset = [k for k in splits if str(region.get(k, "AUTO")) in ("AUTO", "None", "")]
        if unset:
            failures += 1
            line(BAD, "frozen splits", f"unset in region.yaml: {unset}")
        else:
            line(OK, "frozen splits",
                 f"train {region['train_start']}..{region['train_end']} / "
                 f"val {region['val_start']}..{region['val_end']} / "
                 f"test {region['test_start']}..{region['test_end']}")

        line(OK if region.get("baseline_method") == "rolling" else BAD,
             f"baseline_method = {region.get('baseline_method')!r}",
             "" if region.get("baseline_method") == "rolling"
             else "region.yaml must say 'rolling'; window_percentile is the "
                  "Phase 1 approximation (spec 9.2)")
        failures += 0 if region.get("baseline_method") == "rolling" else 1

        # O6: the study period is derived from the frozen splits, so the
        # declared archive coverage has to contain it or the download fails
        # partway through instead of here.
        need = list(range(int(str(region["train_start"])[:4]) - 1,
                          int(str(region["test_end"])[:4]) + 1))
        have = [int(y) for y in region.get("sources", {}).get(
            "eaglei_years_available", [])]
        missing = sorted(set(need) - set(have)) if have else []
        if not have:
            line(WARN, "eaglei_years_available", "unset -- cannot validate the study period")
        elif missing:
            failures += 1
            line(BAD, "eaglei_years_available",
                 f"study period needs {need[0]}-{need[-1]}; archive declares "
                 f"{have[0]}-{have[-1]}, missing {missing}. Update region.yaml "
                 "from the figshare article (doi 10.6084/m9.figshare.24237376) "
                 "or move the frozen test year.")
        else:
            line(OK, "eaglei_years_available", f"covers {need[0]}-{need[-1]}")

    print("-" * 68)
    if failures:
        print(f"{failures} required check(s) failed. Fix those before "
              f"{'`make fetch`' if phase == 1 else '`make phase2-submit`'}.\n")
        return 1
    if phase == 1:
        print("preflight clean. Next: make phase1-synthetic, then make fetch.\n")
    else:
        print("preflight clean. Next: make test, make lint, then make phase2-submit.\n")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="storm-outage-risk preflight")
    ap.add_argument("--phase", type=int, choices=(1, 2), default=1)
    sys.exit(main(ap.parse_args().phase))
