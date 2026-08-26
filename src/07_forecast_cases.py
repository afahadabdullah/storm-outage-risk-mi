#!/usr/bin/env python
"""Step 7 -- GEFS-driven forecast, one lead time (spec section 8; phase 1 step 8).

Everything upstream is hindcast on reanalysis. This stage makes it a forecast
system, and that is the half that separates the artifact from a Kaggle-style
analysis.

THE BIAS-CORRECTION STEP IS NOT OPTIONAL. A model fitted on ERA5 and driven with
raw GEFS is being fed inputs from a different distribution than it was trained
on; the gust bias between the two is large enough to swamp the signal. Gate
criterion 9 checks the mapping actually moved the mean -- an inert
"bias correction" that silently does nothing is the failure this catches.

PHASE 1: one lead time, 5 members. Proving the fetch, the quantile mapping, the
regrid, and the model call all connect. Phase 2 runs day-5/-3/-2/-1 x 31 members
x 100 parametric draws = 3,100 realizations, and produces the money plot.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.common.config import PATHS, base_parser, config_from_args
from src.common.era5_io import open_era5
from src.common.gates import book
from src.common.geo import agg_max, agg_mean, build_weight_matrix
from src.common.logio import get_logger, record, timed

log = get_logger("07_forecast")
FCST_PATH = PATHS.processed / "phase1_forecast_realizations.npy"


# --------------------------------------------------------------------------- #
def load_gefs(cfg):
    """Synthetic netcdf if present, otherwise the GRIB subset from step 2."""
    import xarray as xr

    syn = PATHS.raw / "gefs_synthetic.nc"
    if syn.exists():
        log.warning("using synthetic GEFS (%s) -- no forecast skill is implied", syn.name)
        return xr.open_dataset(syn)

    dirs = sorted(PATHS.raw.glob("gefs_*"))
    dirs = [d for d in dirs if d.is_dir()]
    if not dirs:
        raise SystemExit("no GEFS data -- run `make weather` or use --synthetic")
    files = sorted(dirs[-1].glob("*.grib2"))
    log.info("reading %d GEFS GRIB files from %s", len(files), dirs[-1].name)

    mem = {}
    for f in files:
        member, fhr = f.stem.split("_f")
        ds = xr.open_dataset(f, engine="cfgrib",
                             backend_kwargs={"indexpath": "",
                                             "filter_by_keys": {"typeOfLevel": "surface"}})
        name = "gust" if "gust" in ds else next(iter(ds.data_vars))
        mem.setdefault(member, []).append(
            ds[name].expand_dims(time=[int(fhr)]).rename("gust"))
    arrs = [xr.concat(sorted(v, key=lambda a: int(a.time[0])), dim="time")
            for v in mem.values()]
    out = xr.concat(arrs, dim="member").to_dataset(name="gust")
    return out.rename({k: v for k, v in
                       {"latitude": "lat", "longitude": "lon"}.items() if k in out})


def quantile_map(fcst: np.ndarray, ref: np.ndarray, n_q: int = 51) -> np.ndarray:
    """Empirical quantile mapping of GEFS gust onto the ERA5 distribution.

    PHASE 1 SIMPLIFICATION: pooled over all cells and the whole window. Phase 2
    maps per grid cell and per season, as the spec requires -- five days cannot
    populate a per-cell, per-season distribution and pretending otherwise would
    be fitting noise.
    """
    qs = np.linspace(0.01, 0.99, n_q)
    f_q = np.quantile(fcst, qs)
    r_q = np.quantile(ref, qs)
    flat = np.interp(fcst.ravel(), f_q, r_q)
    return flat.reshape(fcst.shape)


# --------------------------------------------------------------------------- #
def main() -> None:
    args = base_parser(__doc__).parse_args()
    cfg = config_from_args(args)
    gb = book("07_forecast")
    rng = np.random.default_rng(int(cfg.get("random_seed", 0)) + 7)

    import joblib
    era5 = open_era5(cfg)
    gefs = load_gefs(cfg)
    bundle = joblib.load(PATHS.models / "phase1_models.joblib")
    feats = bundle["features"]
    merged = pd.read_parquet(PATHS.processed / "phase1_merged.parquet")

    raw = gefs["gust"].values                       # (member, time, lat, lon)
    ref = era5["i10fg"].values
    raw_mean, ref_mean = float(raw.mean()), float(ref.mean())

    with timed("forecast", log):
        mapped = quantile_map(raw, ref)
        map_mean = float(mapped.mean())

        W, M, geoids, _ = build_weight_matrix(cfg, gefs.lat.values, gefs.lon.values)
        n_mem = raw.shape[0]
        county_gust = np.stack([agg_max(M, mapped[m]) for m in range(n_mem)])
        county_gust_mean = np.stack([agg_mean(W, mapped[m]) for m in range(n_mem)])

    # ---- criterion 9: the bias correction must actually do something -------
    before = abs(raw_mean - ref_mean)
    after = abs(map_mean - ref_mean)
    log.info("gust mean  raw GEFS %.2f  mapped %.2f  ERA5 %.2f m/s", raw_mean,
             map_mean, ref_mean)
    gb.require(
        "bias_correction_active", after < before,
        f"|raw-ERA5| {before:.3f} -> |mapped-ERA5| {after:.3f} m/s "
        "(if this does not shrink, the mapping is inert and the forecast stage "
        "is silently broken)", criterion=9)

    day_gust = county_gust.max(axis=1)               # (member, county)
    gb.require("gefs_county_gust_plausible",
               bool(((day_gust >= 0) & (day_gust <= 80)).all()),
               f"county gust in [{day_gust.min():.1f}, {day_gust.max():.1f}] m/s")

    # ---- drive the fitted model with each member ---------------------------
    peak_date = merged.loc[merged.customer_hours.idxmax(), "date"]
    template = (merged[merged.date == peak_date].set_index("fips")
                       .reindex(geoids).reset_index())
    n_par = max(int(cfg["n_mc_draws"]) // 5, 10)
    reals = []
    for m in range(n_mem):
        row = template.copy()
        row["gust_max"] = day_gust[m]
        row["gust_p95"] = day_gust[m] * 0.9
        row["gust_mean"] = county_gust_mean[m].mean(axis=0)
        row["gust_x_soil"] = row.gust_max * row.soil_moisture_mean
        row["gust_x_canopy"] = row.gust_max * row.canopy_pct.fillna(
            row.canopy_pct.mean() if row.canopy_pct.notna().any() else 0.0)
        row["gust_x_leafon"] = row.gust_max * row.leaf_on
        X = row.reindex(columns=feats).fillna(0.0)

        p = bundle["occurrence"].predict_proba(X)[:, 1]
        if bundle.get("calibrator") is not None:
            p = bundle["calibrator"].predict(p)
        mag = _magnitude_median(bundle["magnitude"], X)
        occ = (rng.uniform(size=(len(X), n_par)) < p[:, None])
        draws = occ * mag[:, None] * np.exp(rng.normal(0, 0.5, (len(X), n_par)))
        reals.append(draws.sum(axis=0))              # statewide per realization

    realizations = np.concatenate(reals)
    np.save(FCST_PATH, realizations)
    record("forecast", n_members=n_mem, n_parametric=n_par,
           n_realizations=int(realizations.size))

    # ---- uncertainty decomposition (spec section 8.3) ----------------------
    per_member = np.stack(reals)
    met_var = float(per_member.mean(axis=1).var())
    mod_var = float(per_member.var(axis=1).mean())
    ratio = met_var / max(met_var + mod_var, 1e-12)
    gb.note("uncertainty_share_meteorological", f"{ratio:.2%} at this lead")
    log.info("uncertainty: meteorological %.1f%%, model/parametric %.1f%% "
             "(Phase 2 reports this BY LEAD TIME -- the crossover point is the "
             "operationally useful result)", 100 * ratio, 100 * (1 - ratio))
    gb.flush()
    log.info("%d realizations of statewide customer-hours -> %s",
             realizations.size, FCST_PATH.name)


def _magnitude_median(mag, X) -> np.ndarray:
    qs = mag["quantiles"]
    mid = qs[len(qs) // 2]
    if mag["kind"] == "ngboost":
        return np.expm1(mag["model"].pred_dist(X).ppf(mid)).clip(min=0)
    return np.expm1(mag["models"][mid].predict(X)).clip(min=0)


if __name__ == "__main__":
    main()
