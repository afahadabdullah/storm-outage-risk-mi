"""Run log + the section 8 measurements table.

Every step writes one line to logs/phase1_run.log and one timing/volume record
to logs/measurements.json. Section 8 of the phase 1 spec is not optional
paperwork: it is what turns Phase 2 from a guess into a schedule.
"""
from __future__ import annotations

import json
import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path

from .config import PATHS



# Phase 2 scale factors, from the phase 1 spec section 8 table.
PHASE2_FACTOR = {
    "era5_download": 438, "era5_to_county": 438, "event_detection": 438,
    "feature_build": 438, "model_fit": 50, "monte_carlo": 438 * 20,
}


def _log_file():
    return PATHS.logs / "phase1_run.log"


def _meas_file():
    return PATHS.logs / "measurements.json"


def get_logger(name: str) -> logging.Logger:
    log = logging.getLogger(name)
    if log.handlers:
        return log
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)-18s %(message)s",
                            datefmt="%H:%M:%S")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    fh = logging.FileHandler(_log_file())
    fh.setFormatter(fmt)
    log.addHandler(sh)
    log.addHandler(fh)
    log.propagate = False
    return log


def _load() -> dict:
    f = _meas_file()
    return json.loads(f.read_text()) if f.exists() else {}


def record(key: str, **fields) -> None:
    data = _load()
    data.setdefault(key, {}).update(fields)
    _meas_file().write_text(json.dumps(data, indent=2, default=str))


def peak_rss_gb() -> float:
    """Peak resident set size of this process, in GB. Linux/macOS differ in units."""
    import resource
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024 ** 2 if os.uname().sysname == "Darwin" else 1024 ** 2 / 1024
    return round(ru / divisor / 1024, 3) if False else round(
        ru / (1024 ** 3 if os.uname().sysname == "Darwin" else 1024 ** 2), 3)


@contextmanager
def timed(key: str, log: logging.Logger | None = None, **extra):
    t0 = time.perf_counter()
    yield
    dt = round(time.perf_counter() - t0, 2)
    record(key, seconds=dt, peak_rss_gb=peak_rss_gb(), **extra)
    if log:
        proj = PHASE2_FACTOR.get(key)
        tail = f"  (phase 2 projection: {dt * proj / 3600:,.1f} h)" if proj else ""
        log.info("%-18s %8.2f s%s", key, dt, tail)


def dir_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    if path.is_file():
        return round(path.stat().st_size / 1e6, 2)
    return round(sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6, 2)


def measurements_markdown() -> str:
    """Render the section 8 table with the blanks filled in."""
    data = _load()
    rows = [
        ("ERA5 download", "era5_download"),
        ("ERA5 -> county aggregation", "era5_to_county"),
        ("Event detection", "event_detection"),
        ("Feature build", "feature_build"),
        ("Model fit (all stages)", "model_fit"),
        ("Monte Carlo", "monte_carlo"),
    ]
    out = ["| Quantity | Phase 1 measured | Phase 2 projected |",
           "|---|---|---|"]
    for label, key in rows:
        m = data.get(key, {})
        s = m.get("seconds")
        mb = m.get("megabytes")
        if s is None:
            out.append(f"| {label} | _not measured_ | -- |")
            continue
        meas = f"{s:,.1f} s" + (f" / {mb:,.0f} MB" if mb else "")
        if m.get("synthetic"):
            meas += " (synthetic -- no download)"
        f = PHASE2_FACTOR.get(key)
        proj = f"{s * f / 3600:,.1f} h" if f else "--"
        out.append(f"| {label} | {meas} | {proj} |")
    peak = max((v.get("peak_rss_gb", 0) for v in data.values()), default=0)
    disk = dir_size_mb(PATHS.processed) / 1000
    out.append(f"| Peak memory | {peak:,.2f} GB | -- |")
    out.append(f"| Processed data on disk | {disk:,.3f} GB | {disk * 438:,.1f} GB |")
    return "\n".join(out)
