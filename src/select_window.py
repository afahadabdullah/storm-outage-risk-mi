#!/usr/bin/env python
"""Step 0 -- pick the five-day window from the data, not from memory.

  0a  download one full year from the TRAINING window (2018-2021)
  0b  statewide customers-out at hourly resolution for the whole year
  0c  window = largest peak - 2 days  ...  largest peak + 2 days

Why the training window and not 2023: the test year stays untouched until the
final Phase 2 evaluation. Running plumbing over it is technically harmless, but
being able to say the test set was opened exactly once is worth more than the
convenience.

Why +/-2 days: you need pre-storm quiet hours to verify baseline removal, the
storm to verify event detection, and post-storm hours to verify restoration
timing. A window starting at the peak gives a left-censored event.

Deviation from the spec snippet, on purpose: the spec sums every 15-minute
record inside each hour, which counts each county four times. That is
monotonic in the same places so it finds the same peak, but the printed
customer count is 4x too large. Here we take the hourly max per county first,
then sum across counties, so the number on screen is a real customer count.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.common.config import PATHS, base_parser, config_from_args
from src.common.logio import get_logger, record, timed

log = get_logger("select_window")
SANITY_FLOOR = 50_000


def statewide_hourly(df: pd.DataFrame) -> pd.Series:
    hourly = (df.set_index("run_start_time")
                .groupby(["fips_code", pd.Grouper(freq="1h")])["sum"].max())
    return hourly.groupby(level=1).sum().sort_index()


def pick_window(statewide: pd.Series, days: int) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    peak_time = statewide.idxmax()
    pad = (days - 1) // 2
    start = (peak_time - pd.Timedelta(days=pad)).normalize()
    return start, start + pd.Timedelta(days=days), peak_time


def plot_year(statewide: pd.Series, peak: pd.Timestamp, start, end, year: int) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(statewide.index, statewide.values, lw=0.6)
    ax.axvspan(start, end, color="tab:orange", alpha=0.3, label="phase 1 window")
    ax.axvline(peak, color="tab:red", lw=1, label=f"peak {peak:%Y-%m-%d %H:%MZ}")
    ax.set_ylabel("statewide customers out")
    ax.set_title(f"Statewide outages {year} -- window selection")
    ax.legend()
    fig.tight_layout()
    out = PATHS.figures / f"phase1_window_selection_{year}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def write_back(phase1_file: Path, start: pd.Timestamp, end: pd.Timestamp) -> None:
    """Patch window_start/window_end in place, keeping every comment."""
    text = phase1_file.read_text()
    for key, val in (("window_start", start.date()), ("window_end", end.date())):
        text = re.sub(rf"^(\s*{key}:\s*).*$", rf'\g<1>"{val}"', text,
                      count=1, flags=re.M)
    phase1_file.write_text(text)
    log.info("wrote window %s -> %s into %s", start.date(), end.date(), phase1_file.name)


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--year", type=int, default=None)
    ap.add_argument("--write", action="store_true",
                    help="patch window_start/window_end into config/phase1.yaml")
    args = ap.parse_args()
    cfg = config_from_args(args)
    year = args.year or int(cfg.get("window_source_year", 2019))

    train = (int(str(cfg["train_start"])[:4]), int(str(cfg["train_end"])[:4]))
    if not train[0] <= year <= train[1]:
        raise SystemExit(
            f"year {year} is outside the training window {train}. "
            "Window selection must not touch the validation or test years.")

    csv = PATHS.raw / f"eaglei_outages_{year}.csv"
    if not csv.exists():
        raise SystemExit(f"{csv} not found -- run `make fetch` first.")

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from importlib import import_module
    normalize = import_module("src.common.io_outage").normalize_outage_frame

    with timed("window_selection", log):
        df = normalize(pd.read_csv(csv, dtype={"fips_code": "string"}), cfg)
        statewide = statewide_hourly(df)
        start, end, peak = pick_window(statewide, int(cfg["window_days"]))

    peak_val = float(statewide.max())
    log.info("Peak: %s  (%s customers out)", peak, f"{peak_val:,.0f}")
    log.info("Phase 1 window: %s -> %s", start.date(), end.date())
    record("window_selection", year=year, peak_time=str(peak),
           peak_customers=peak_val, window_start=str(start.date()),
           window_end=str(end.date()))

    if peak_val < SANITY_FLOOR:
        log.error("SANITY FLOOR: largest %d peak is %s customers, under %s. "
                  "Either the state is too small or the MCC denominator/state "
                  "filter is wrong. Investigate before proceeding.",
                  year, f"{peak_val:,.0f}", f"{SANITY_FLOOR:,}")

    fig = plot_year(statewide, peak, start, end, year)
    log.info("figure -> %s", fig.name)
    if args.write:
        write_back(Path(args.phase1), start, end)
    else:
        log.info("re-run with --write to patch config/phase1.yaml")


if __name__ == "__main__":
    main()
