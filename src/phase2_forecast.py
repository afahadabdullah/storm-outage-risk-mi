#!/usr/bin/env python
"""Phase 2 stage 5 -- GEFS-driven forecast application (spec section 8).

Everything upstream is hindcast on reanalysis. This stage makes it a forecast
system, and that is the half that separates the artifact from a Kaggle-style
analysis.

Pipeline (spec 8.1), for each case-study storm and each of the four leads:

  1. read the GEFS members initialised at day-5/-3/-2/-1 before the target
  2. quantile-map gust and precipitation onto the ERA5 climatology,
     PER GRID CELL AND PER SEASON
  3. aggregate to county-day with the same W / M matrices as section 5.1
  4. run each member through the frozen model, drawing
     `forecast_parametric_draws` parametric samples per member
  5. 31 members x 100 draws = 3,100 realizations of statewide customers-out

THE BIAS-CORRECTION STEP IS NOT OPTIONAL (spec 8.2). A model fitted on ERA5 and
driven with raw GEFS is being fed inputs from a different distribution than it
was trained on, and the gust bias between the two is large enough to swamp the
signal. `bias_correction_active` below asserts the mapping actually moved the
distribution toward ERA5 -- an inert "bias correction" that silently does
nothing is the failure that assertion exists to catch.

Outputs:
  data/processed/phase2_forecast_realizations.npz   (lead, member, draw)
  data/processed/phase2_uncertainty_by_lead.csv     spec 8.3 decomposition
  figures/phase2_forecast_<case>.png                spec 8.4, the money plot

Honest limitation, stated here because it belongs in the write-up too: the
GEFS-side distribution of the quantile mapping is pooled over members and
forecast hours for each cell and season, because the case-study downloads are
the only GEFS sample this project has. A mapping trained on the GEFSv12
reforecast archive would be better and is the obvious upgrade.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.common.config import PATHS, ROOT, Config, load_config
from src.common.gates import book, set_phase
from src.common.geo import agg_max, agg_mean, build_weight_matrix
from src.common.logio import get_logger, record, timed

log = get_logger("phase2_forecast")

REAL_PATH = PATHS.processed / "phase2_forecast_realizations.npz"
UNCERT_PATH = PATHS.processed / "phase2_uncertainty_by_lead.csv"
COUNTY_PROB_PATH = PATHS.processed / "phase2_forecast_county_probs.parquet"
CLIM_PATH = PATHS.interim / "era5_forecast_climatology.npz"
TEST_MARKER = PATHS.models / "TEST_YEAR_OPENED.txt"

CLIM_QUANTILES = np.linspace(0.005, 0.995, 199)
# ERA5 name -> the GEFS field it is mapped onto
MAP_VARS = {"i10fg": "gust", "tp": "tp"}


def season_of(month: int) -> int:
    """DJF=0, MAM=1, JJA=2, SON=3 -- the same convention as derived_features."""
    return (month % 12) // 3


# --------------------------------------------------------------------------- #
# ERA5 climatology, per cell and per season
# --------------------------------------------------------------------------- #
def build_era5_climatology(cfg: Config, force: bool = False) -> dict:
    """Per-cell, per-season quantile ladders for the mapped ERA5 variables.

    Spec 8.1 asks for the mapping to be per grid cell and per season, which is
    exactly what Phase 1 could not do on five days and explicitly deferred here.
    Computed once from the monthly ERA5 files and cached -- it is a scan of the
    whole study period and has no business running per case study.
    """
    if CLIM_PATH.exists() and not force:
        z = np.load(CLIM_PATH, allow_pickle=True)
        log.info("cached ERA5 forecast climatology %s", CLIM_PATH.name)
        return {k: z[k] for k in z.files}

    import xarray as xr

    months = sorted((PATHS.raw / "era5_monthly").glob("era5_*.nc"))
    if not months:
        raise SystemExit("no monthly ERA5 files -- run `make phase2-download-era5`")
    # Only the train+validation period may inform the mapping; a climatology
    # that has seen the test year is a leak through the back door.
    limit = pd.Timestamp(cfg["val_end"])
    buckets: dict[tuple[str, int], list[np.ndarray]] = {}
    lats = lons = None
    for path in months:
        stamp = pd.Period(f"{path.stem[5:9]}-{path.stem[9:11]}", freq="M")
        if stamp.start_time > limit:
            continue
        with xr.open_dataset(path) as ds:
            ren = {o: n for o, n in {"valid_time": "time", "latitude": "lat",
                                     "longitude": "lon"}.items() if o in ds}
            ds = ds.rename(ren)
            for dim in ("expver", "number"):
                if dim in ds.dims and ds.sizes[dim] == 1:
                    ds = ds.squeeze(dim, drop=True)
            if lats is None:
                lats, lons = ds.lat.values, ds.lon.values
            season = season_of(stamp.month)
            for era_name in MAP_VARS:
                buckets.setdefault((era_name, season), []).append(
                    ds[era_name].values.reshape(ds.sizes["time"], -1))
        log.info("climatology: read %s", path.name)

    out: dict = {"lats": lats, "lons": lons, "quantiles": CLIM_QUANTILES}
    for (era_name, season), chunks in buckets.items():
        stacked = np.concatenate(chunks, axis=0)          # (time, cell)
        out[f"{era_name}_{season}"] = np.quantile(
            stacked, CLIM_QUANTILES, axis=0).astype(np.float32)   # (q, cell)
        log.info("climatology %s season %d: %d samples x %d cells",
                 era_name, season, stacked.shape[0], stacked.shape[1])
    CLIM_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CLIM_PATH, **out)
    log.info("ERA5 forecast climatology cached -> %s", CLIM_PATH.name)
    return out


def fit_quantile_map(sample: np.ndarray) -> np.ndarray:
    """GEFS-side quantile ladder per cell, from a (sample, cell) training block.

    Fitted ONCE per case study on every lead and member pooled, then applied
    unchanged to each lead. Re-fitting per lead would force each lead's ensemble
    onto the same climatological distribution and erase the lead-dependent
    dispersion -- which is precisely the quantity the section 8.3 decomposition
    exists to measure. The transfer function must be a property of the
    GEFS-vs-ERA5 relationship, not of the lead it happens to be applied to.
    """
    return np.quantile(sample, CLIM_QUANTILES, axis=0)     # (q, cell)


def quantile_map_cellwise(fcst: np.ndarray, src_q: np.ndarray,
                          ref_q: np.ndarray) -> np.ndarray:
    """Apply a fitted per-cell transfer function to a (sample, cell) block."""
    out = np.empty_like(fcst, dtype=np.float64)
    for c in range(fcst.shape[1]):
        src = src_q[:, c]
        # np.interp needs a strictly increasing x; a dry or flat cell is not an
        # error, it just maps to its own reference value.
        if src[-1] <= src[0]:
            out[:, c] = ref_q[:, c].mean()
            continue
        # Outside the fitted range np.interp clamps, which would flatten the
        # extreme members -- the ones that matter. Extend on the end slopes.
        mapped = np.interp(fcst[:, c], src, ref_q[:, c])
        lo_slope = ((ref_q[1, c] - ref_q[0, c]) / (src[1] - src[0])
                    if src[1] > src[0] else 1.0)
        hi_slope = ((ref_q[-1, c] - ref_q[-2, c]) / (src[-1] - src[-2])
                    if src[-1] > src[-2] else 1.0)
        below = fcst[:, c] < src[0]
        above = fcst[:, c] > src[-1]
        mapped[below] = ref_q[0, c] + lo_slope * (fcst[below, c] - src[0])
        mapped[above] = ref_q[-1, c] + hi_slope * (fcst[above, c] - src[-1])
        out[:, c] = mapped
    return np.clip(out, 0.0, None)


# --------------------------------------------------------------------------- #
# GEFS input
# --------------------------------------------------------------------------- #
def gefs_dir(init: pd.Timestamp, cycle: int = 0) -> Path:
    return PATHS.raw / f"gefs_{init:%Y%m%d}{cycle:02d}"


def load_gefs_case(cfg: Config, init: pd.Timestamp, cycle: int = 0):
    """(member, time, lat, lon) arrays for gust and precipitation."""
    import xarray as xr

    directory = gefs_dir(init, cycle)
    files = sorted(directory.glob("*.grib2")) if directory.exists() else []
    if not files:
        raise SystemExit(
            f"no GEFS files in {directory} -- run `make phase2-download-gefs` "
            "(or pass --synthetic-gefs to exercise the plumbing without a "
            "download; no forecast skill is implied by that mode)")
    log.info("reading %d GEFS GRIB files from %s", len(files), directory.name)

    members: dict[str, list] = {}
    for path in files:
        member, fhr = path.stem.split("_f")
        ds = xr.open_dataset(path, engine="cfgrib",
                             backend_kwargs={"indexpath": "",
                                             "filter_by_keys": {"typeOfLevel": "surface"}})
        keep = {}
        for want, names in (("gust", ("gust", "i10fg")), ("tp", ("tp", "unknown"))):
            for name in names:
                if name in ds:
                    keep[want] = ds[name]
                    break
        if "gust" not in keep:
            keep["gust"] = ds[next(iter(ds.data_vars))]
        block = xr.Dataset({k: v.expand_dims(time=[int(fhr)]) for k, v in keep.items()})
        members.setdefault(member, []).append(block)

    per_member = [xr.concat(sorted(v, key=lambda a: int(a.time[0])), dim="time",
                            coords="minimal", compat="override")
                  for v in members.values()]
    out = xr.concat(per_member, dim="member")
    out = out.rename({o: n for o, n in {"latitude": "lat", "longitude": "lon"}.items()
                      if o in out})
    if float(out.lon.max()) > 180:
        out = out.assign_coords(lon=((out.lon + 180) % 360) - 180).sortby("lon")
    w, s, e, n = cfg["bbox"]
    lat_slice = slice(n, s) if float(out.lat[0]) > float(out.lat[-1]) else slice(s, n)
    out = out.sel(lat=lat_slice, lon=slice(w, e))
    if out.sizes.get("lat", 0) == 0 or out.sizes.get("lon", 0) == 0:
        raise ValueError(f"GEFS bbox {cfg['bbox']} selected no grid cells")
    log.info("GEFS subset: %d members x %d leads x %d lat x %d lon",
             out.sizes["member"], out.sizes["time"], out.sizes["lat"], out.sizes["lon"])
    return out


def synthetic_gefs(cfg: Config, target: pd.Timestamp, lead: int, n_members: int,
                   clim: dict, rng):
    """Stand-in members built from the target day's ERA5 field.

    Deliberately biased high and made noisier with lead time, so the quantile
    mapping has a real bias to remove and the section 8.3 decomposition has a
    real spread to partition. This proves the plumbing; it implies no skill.
    """
    import xarray as xr

    month = f"{target.year}{target.month:02d}"
    path = PATHS.raw / "era5_monthly" / f"era5_{month}.nc"
    if not path.exists():
        raise SystemExit(f"{path} missing -- synthetic GEFS is derived from it")
    with xr.open_dataset(path) as ds:
        ren = {o: n for o, n in {"valid_time": "time", "latitude": "lat",
                                 "longitude": "lon"}.items() if o in ds}
        ds = ds.rename(ren).sortby("time")
        day = ds.sel(time=str(target.date()))
        gust = day["i10fg"].values           # (hours, lat, lon)
        tp = day["tp"].values
        lats, lons = ds.lat.values, ds.lon.values
    hours = gust.shape[0]
    spread = 0.12 + 0.06 * lead             # wider at longer lead
    bias = 1.35                              # GEFS gusts run high vs ERA5
    g = np.stack([gust * bias * (1 + rng.normal(0, spread, gust.shape))
                  for _ in range(n_members)])
    p = np.stack([tp * 1.2 * (1 + rng.normal(0, spread, tp.shape)).clip(0)
                  for _ in range(n_members)])
    return xr.Dataset(
        {"gust": (("member", "time", "lat", "lon"), np.clip(g, 0, None)),
         "tp": (("member", "time", "lat", "lon"), np.clip(p, 0, None))},
        coords={"member": np.arange(n_members), "time": np.arange(hours),
                "lat": lats, "lon": lons})


# --------------------------------------------------------------------------- #
# Driving the fitted model
# --------------------------------------------------------------------------- #
def member_realizations(bundle: dict, rows: pd.DataFrame, n_draws: int,
                        rng) -> np.ndarray:
    """One member: statewide customer-hours per draw, and the county probabilities."""
    from src.phase2_compose import sample_magnitude

    X = rows[bundle["features"]]
    raw = bundle["occurrence"].predict_proba(X)[:, 1]
    proba = bundle["calibrator"].predict(raw)
    occ = rng.uniform(size=(len(rows), n_draws)) < proba[:, None]
    mag = sample_magnitude(bundle["magnitude"], X, rng, n_draws)
    return (occ * mag).sum(axis=0), proba   # statewide total per draw, county p


def forecast_feature_fills(merged: pd.DataFrame, bundle: dict,
                           cfg: Config) -> pd.Series:
    """Frozen-training medians for the rare missing forecast covariates.

    LightGBM can route missing values, but the NGBoost magnitude head rejects
    them outright.  Forecast templates contain slowly varying analysis fields,
    and a missing value in one such field should not discard a complete GEFS
    case.  Calculate replacement values from the original training window only:
    using validation, a backtest, or 2023 analysis values here would leak
    information into the forecast application.
    """
    features = bundle["features"]
    dates = pd.to_datetime(merged.date)
    train = merged.loc[dates.between(cfg["train_start"], cfg["train_end"]), features]
    values = train.replace([np.inf, -np.inf], np.nan).apply(pd.to_numeric,
                                                             errors="coerce")
    fills = values.median(axis=0).fillna(0.0)
    return fills.reindex(features).astype(float)


def ensure_finite_forecast_features(rows: pd.DataFrame, bundle: dict,
                                    fills: pd.Series, context: str) -> pd.DataFrame:
    """Use frozen-training medians and fail loudly if a model input stays bad."""
    features = bundle["features"]
    missing_columns = sorted(set(features) - set(rows.columns))
    if missing_columns:
        raise ValueError(f"{context}: forecast rows lack model feature(s) {missing_columns}")
    out = rows.copy()
    values = (out[features].replace([np.inf, -np.inf], np.nan)
              .apply(pd.to_numeric, errors="coerce"))
    missing = values.isna()
    if missing.any().any():
        detail = {col: int(missing[col].sum()) for col in features if missing[col].any()}
        log.warning("%s: imputing missing forecast model inputs from frozen training "
                    "medians: %s", context, detail)
        values = values.fillna(fills)
    remaining = values.isna()
    if remaining.any().any():
        detail = {col: int(remaining[col].sum()) for col in features if remaining[col].any()}
        raise ValueError(f"{context}: non-finite model inputs remain after imputation: {detail}")
    out.loc[:, features] = values
    return out


def forecast_rows(template: pd.DataFrame, geoids: list[str], gust: np.ndarray,
                  gust_mean: np.ndarray, precip: np.ndarray) -> pd.DataFrame:
    """Overwrite the forecastable weather features; keep the static ones.

    Canopy, customer density, leaf-on and soil moisture are static or slowly
    varying, so the analysis values are used. Gust and precipitation come from
    the mapped GEFS member. Every interaction that depends on a replaced field
    is recomputed rather than inherited -- inheriting them was how a "forecast"
    could quietly still contain the analysis gust.
    """
    row = template.set_index("fips").reindex(geoids).reset_index()
    row["gust_max"] = gust
    row["gust_p95"] = gust * 0.92
    row["gust_mean"] = gust_mean
    row["wind_sustained_max"] = gust * 0.65
    row["wind100_max"] = gust * 0.80
    row["precip_total"] = precip
    row["precip_max_hourly"] = precip / 6.0
    for thr in (13, 18, 25):
        col = f"hours_gust_gt_{thr}"
        if col in row:
            # crude but monotone in the forecast gust, and honest about it
            row[col] = np.clip((gust - thr) * 1.5, 0, 24).round()
    canopy = row.canopy_pct.fillna(row.canopy_pct.mean() if row.canopy_pct.notna().any() else 0.0)
    row["gust_x_soil"] = row.gust_max * row.soil_moisture_mean
    row["gust_x_canopy"] = row.gust_max * canopy
    row["gust_x_leafon"] = row.gust_max * row.leaf_on
    row["ice_x_canopy"] = row.freezing_rain_proxy * canopy
    return row


def observed_statewide(cfg: Config, target: pd.Timestamp) -> float | None:
    """Observed customer-hours on the case-study date -- test-year discipline applies.

    The named case studies fall in the held-out year, so the observed overlay is
    only read once the test year has been formally opened. Before that the money
    plot is drawn without its truth line rather than quietly peeking.
    """
    if not TEST_MARKER.exists():
        log.warning("test year not opened (%s absent); the section 8.4 figure "
                    "will be drawn WITHOUT the observed overlay",
                    TEST_MARKER.name)
        return None
    path = PATHS.processed / "phase2_merged_test.parquet"
    if not path.exists():
        return None
    frame = pd.read_parquet(path, columns=["date", "customer_hours"])
    day = frame[pd.to_datetime(frame.date).dt.normalize() == target.normalize()]
    return float(day.customer_hours.sum()) if len(day) else None


def plot_case(case_name: str, per_lead: dict[int, np.ndarray],
              observed: float | None) -> Path:
    """Spec 8.4: one panel per lead, distributions sharpening toward truth."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    leads = sorted(per_lead, reverse=True)
    fig, axes = plt.subplots(1, len(leads), figsize=(3.6 * len(leads), 3.9),
                             sharex=True, squeeze=False)
    everything = np.concatenate(list(per_lead.values()))
    hi = float(np.percentile(everything, 99.5))
    bins = np.linspace(0, max(hi, observed * 1.1 if observed else hi), 45)
    for ax, lead in zip(axes[0], leads):
        vals = per_lead[lead]
        ax.hist(vals, bins=bins, color="#12626F", alpha=.85, edgecolor="none")
        if observed is not None:
            ax.axvline(observed, color="#A8321C", lw=2, label="observed")
            ax.legend(fontsize=8, loc="upper right")
        ax.set_title(f"day-{lead}  (n={vals.size:,})", fontsize=10)
        ax.set_xlabel("statewide customer-hours")
        ax.tick_params(labelsize=8)
    axes[0][0].set_ylabel("realizations")
    fig.suptitle(f"{case_name} — predicted statewide customers-out by lead time",
                 fontsize=12)
    fig.tight_layout()
    slug = case_name.lower().replace(" ", "-")
    out = PATHS.figures / f"phase2_forecast_{slug}.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "config" / "region.yaml"))
    ap.add_argument("--phase2", default=str(ROOT / "config" / "phase2.yaml"))
    ap.add_argument("--case", action="append", help="case-study name substring")
    ap.add_argument("--synthetic-gefs", action="store_true",
                    help="derive stand-in members from ERA5; proves the "
                         "plumbing, implies no forecast skill")
    ap.add_argument("--force-climatology", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config, args.phase2)
    set_phase(2)
    gb = book("phase2_forecast")
    rng = np.random.default_rng(int(cfg.get("random_seed", 0)) + 7)

    import joblib
    model_path = PATHS.models / "phase2_models.joblib"
    if not model_path.exists():
        raise SystemExit("frozen Phase 2 model missing -- run `make phase2-train` first")
    bundle = joblib.load(model_path)
    merged = pd.read_parquet(PATHS.processed / "phase2_merged.parquet")
    feature_fills = forecast_feature_fills(merged, bundle, cfg)

    clim = build_era5_climatology(cfg, force=args.force_climatology)
    W, M, geoids, _ = build_weight_matrix(cfg, clim["lats"], clim["lons"])

    leads = [int(v) for v in cfg["forecast_lead_days"]]
    n_draws = int(cfg.get("forecast_parametric_draws", 100))
    n_members = int(cfg["n_gefs_members"])
    all_out: dict[str, dict] = {}
    uncert_rows: list[dict] = []
    county_prob_rows: list[pd.DataFrame] = []

    for case in cfg["case_studies"]:
        if args.case and not any(c.lower() in case["name"].lower() for c in args.case):
            continue
        target = pd.Timestamp(case["date"])
        season = season_of(target.month)
        # The template supplies the static and slowly-varying covariates. It is
        # taken from the season's median analysis day, never from the target
        # day, so no target-day analysis weather leaks into a "forecast".
        pool = merged[pd.to_datetime(merged.date).dt.month.isin(
            {(3 * season + i - 1) % 12 + 1 for i in range(3)})]
        if pool.empty:
            pool = merged
        template_date = pool.date.iloc[len(pool) // 2]
        template = pool[pool.date == template_date].copy()
        template = ensure_finite_forecast_features(
            template, bundle, feature_fills, f"{case['name']} template")

        per_lead: dict[int, np.ndarray] = {}
        with timed(f"phase2_forecast_{case['name'].replace(' ', '_')}", log):
            # Read every lead first, so the quantile-mapping transfer function
            # can be fitted once on all of them pooled and then applied
            # unchanged -- see fit_quantile_map for why that ordering matters.
            blocks: dict[int, tuple[np.ndarray, np.ndarray]] = {}
            for lead in leads:
                init = target - pd.Timedelta(days=lead)
                gefs = (synthetic_gefs(cfg, target, lead, n_members, clim, rng)
                        if args.synthetic_gefs else load_gefs_case(cfg, init))
                raw_gust = gefs["gust"].values                    # (mem, t, lat, lon)
                raw_tp = (gefs["tp"].values if "tp" in gefs
                          else np.zeros_like(raw_gust))
                blocks[lead] = (raw_gust, raw_tp)

            ref_gust = clim[f"i10fg_{season}"]
            ref_tp = clim[f"tp_{season}"]
            src_gust = fit_quantile_map(np.concatenate(
                [g.reshape(-1, g.shape[-2] * g.shape[-1]) for g, _ in blocks.values()]))
            src_tp = fit_quantile_map(np.concatenate(
                [p.reshape(-1, p.shape[-2] * p.shape[-1]) for _, p in blocks.values()]))
            log.info("quantile map fitted once on %d leads pooled and applied "
                     "unchanged to each", len(blocks))

            for lead in leads:
                raw_gust, raw_tp = blocks[lead]
                n_mem, n_t = raw_gust.shape[0], raw_gust.shape[1]
                flat_gust = raw_gust.reshape(n_mem * n_t, -1)
                flat_tp = raw_tp.reshape(n_mem * n_t, -1)
                mapped_gust = quantile_map_cellwise(flat_gust, src_gust, ref_gust)
                mapped_tp = quantile_map_cellwise(flat_tp, src_tp, ref_tp)

                # ---- spec 8.2: the mapping must actually move the distribution
                ref_mean = float(ref_gust.mean())
                before = abs(float(flat_gust.mean()) - ref_mean)
                after = abs(float(mapped_gust.mean()) - ref_mean)
                gb.require(
                    f"bias_correction_active_day{lead}", after < before,
                    f"|raw-ERA5| {before:.3f} -> |mapped-ERA5| {after:.3f} m/s "
                    f"at day-{lead}",
                    on_fail="The quantile mapping is inert and the forecast "
                            "stage is silently broken: a model fitted on ERA5 "
                            "is being driven with raw GEFS.")
                log.info("day-%d gust mean: raw %.2f -> mapped %.2f (ERA5 %.2f) m/s",
                         lead, flat_gust.mean(), mapped_gust.mean(), ref_mean)

                g4 = mapped_gust.reshape(n_mem, n_t, *raw_gust.shape[2:])
                p4 = mapped_tp.reshape(n_mem, n_t, *raw_gust.shape[2:])
                reals, member_probs = [], []
                for m in range(n_mem):
                    county_gust = agg_max(M, g4[m]).max(axis=0)
                    county_gmean = agg_mean(W, g4[m]).mean(axis=0)
                    county_precip = agg_mean(W, p4[m]).sum(axis=0) * 1000.0
                    rows = forecast_rows(template, geoids, county_gust,
                                         county_gmean, county_precip)
                    rows = ensure_finite_forecast_features(
                        rows, bundle, feature_fills,
                        f"{case['name']} day-{lead} member-{m}")
                    draws, probs = member_realizations(bundle, rows, n_draws, rng)
                    reals.append(draws)
                    member_probs.append(probs)
                block = np.stack(reals)                    # (member, draw)
                per_lead[lead] = block.ravel()
                # Ensemble-mean county probability at this lead. The decision-
                # value stage needs a probability per county per lead; a
                # statewide realization total cannot produce a cost-loss curve.
                county_prob_rows.append(pd.DataFrame({
                    "case": case["name"], "lead_days": lead, "fips": geoids,
                    "date": target, "probability": np.mean(member_probs, axis=0),
                }))

                # ---- spec 8.3: partition predictive variance ---------------
                met_var = float(block.mean(axis=1).var())
                mod_var = float(block.var(axis=1).mean())
                share = met_var / max(met_var + mod_var, 1e-12)
                uncert_rows.append({
                    "case": case["name"], "lead_days": lead,
                    "n_members": n_mem, "n_draws_per_member": n_draws,
                    "n_realizations": block.size,
                    "meteorological_variance": met_var,
                    "model_parametric_variance": mod_var,
                    "meteorological_share": share,
                    "median_statewide_customer_hours": float(np.median(block)),
                    "p10": float(np.percentile(block, 10)),
                    "p90": float(np.percentile(block, 90)),
                })
                log.info("day-%d: %d x %d = %d realizations, meteorological "
                         "share %.1f%%", lead, n_mem, n_draws, block.size,
                         100 * share)

        observed = observed_statewide(cfg, target)
        figure = plot_case(case["name"], per_lead, observed)
        all_out[case["name"]] = {str(k): v for k, v in per_lead.items()}
        log.info("%s -> %s", case["name"], figure.name)

    if not all_out:
        raise SystemExit("no case studies selected")

    np.savez_compressed(REAL_PATH, **{f"{c}|{lead}": arr
                                      for c, d in all_out.items()
                                      for lead, arr in d.items()})
    uncert = pd.DataFrame(uncert_rows)
    uncert.to_csv(UNCERT_PATH, index=False)
    if county_prob_rows:
        pd.concat(county_prob_rows, ignore_index=True).to_parquet(
            COUNTY_PROB_PATH, index=False)
        log.info("per-county forecast probabilities -> %s", COUNTY_PROB_PATH.name)

    # The crossover is the operationally useful result: it tells a utility
    # whether better forecasts or a better damage model would help more at each
    # planning horizon.
    for case_name, group in uncert.groupby("case"):
        ordered = group.sort_values("lead_days", ascending=False)
        shares = dict(zip(ordered.lead_days, ordered.meteorological_share))
        longest, shortest = max(shares), min(shares)
        gb.check(
            "meteorological_share_shrinks_with_lead",
            shares[longest] > shares[shortest],
            f"{case_name}: meteorological share {shares[longest]:.1%} at day-"
            f"{longest} -> {shares[shortest]:.1%} at day-{shortest}",
            warn=True)
        log.info("%s uncertainty share by lead: %s", case_name,
                 {f"day-{k}": f"{v:.1%}" for k, v in sorted(shares.items(), reverse=True)})

    record("phase2_forecast", n_cases=len(all_out), leads=leads,
           n_realizations=int(sum(a.size for d in all_out.values() for a in d.values())))
    (PATHS.processed / "phase2_forecast_summary.json").write_text(json.dumps({
        "cases": list(all_out), "leads": leads, "n_members": n_members,
        "draws_per_member": n_draws,
        "synthetic_gefs": bool(args.synthetic_gefs),
    }, indent=2))
    gb.flush()
    log.info("realizations -> %s ; uncertainty decomposition -> %s",
             REAL_PATH.name, UNCERT_PATH.name)


if __name__ == "__main__":
    main()
