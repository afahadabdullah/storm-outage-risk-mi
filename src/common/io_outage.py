"""EAGLE-I CSV normalisation, shared by window selection and ingestion.

Two things happen here and both are load-bearing:

  FIPS  cast to 5-character strings with leading zeros preserved. A CSV reader
        that infers int64 turns 01001 into 1001, the TIGER join then silently
        drops the row, and nothing raises.
  TIME  everything converted to UTC once, at ingestion. A silent local/UTC
        mismatch shifts the storm 4-5 hours and quietly destroys the
        weather-outage correlation; the model just looks weak.
"""
from __future__ import annotations

import pandas as pd

from .config import Config, state_prefixes


def normalize_outage_frame(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df.columns = [c.strip().lower() for c in df.columns]
    if "customers_out" in df.columns and "sum" not in df.columns:
        df = df.rename(columns={"customers_out": "sum"})
    need = {"fips_code", "sum", "run_start_time"}
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(
            f"EAGLE-I CSV missing columns {missing}; got {list(df.columns)}")

    df["fips_code"] = df.fips_code.astype("string").str.strip().str.zfill(5)
    ts = pd.to_datetime(df.run_start_time, errors="coerce", format="mixed")
    df["run_start_time"] = (ts.dt.tz_localize("UTC") if ts.dt.tz is None
                            else ts.dt.tz_convert("UTC"))
    df["sum"] = pd.to_numeric(df["sum"], errors="coerce").fillna(0).clip(lower=0)
    return df[df.fips_code.str[:2].isin(state_prefixes(cfg))].copy()
