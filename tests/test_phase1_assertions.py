"""The Phase 1 assertion suite, promoted to a test module (spec section 9.5).

Assertions written during smoke testing and then discarded are a waste. Kept,
they catch regressions when you refactor in week three -- which is exactly when
a silent FIPS or unit bug is most expensive.

Tests that need pipeline outputs skip when those outputs are absent, so this
suite is safe to run on a clean checkout and meaningful right after a run.
"""
from __future__ import annotations

import zipfile

import numpy as np
import pandas as pd
import pytest

from src.common.config import PATHS, ROOT, load_config
from src.common.io_outage import normalize_outage_frame

CFG = load_config(ROOT / "config" / "region.yaml", ROOT / "config" / "phase1.yaml")


def _load(name: str) -> pd.DataFrame:
    p = PATHS.processed / name
    if not p.exists():
        pytest.skip(f"{name} not built yet -- run `make phase1-synthetic`")
    return pd.read_parquet(p)


# --------------------------------------------------------------------------- #
# Pure logic: these run anywhere, with no data
# --------------------------------------------------------------------------- #
def test_fips_leading_zeros_survive_ingestion():
    """The highest-probability failure in the entire project."""
    raw = pd.DataFrame({
        "fips_code": [1001, "01003", "26163"],
        "sum": [10, 20, 30],
        "run_start_time": ["2019-07-19 00:00", "2019-07-19 00:15", "2019-07-19 00:30"],
    })
    cfg = dict(CFG) | {"state_fips": ["01", "26"]}
    out = normalize_outage_frame(raw, type(CFG)(cfg))
    assert set(out.fips_code) == {"01001", "01003", "26163"}
    assert out.fips_code.str.len().eq(5).all()


def test_ingestion_is_tz_aware_utc():
    raw = pd.DataFrame({"fips_code": ["26001"], "sum": [5],
                        "run_start_time": ["2019-07-19 12:00"]})
    out = normalize_outage_frame(raw, CFG)
    assert str(out.run_start_time.dt.tz) == "UTC"


def test_cost_loss_value_is_bounded_and_varies():
    import runpy
    mod = runpy.run_path(str(ROOT / "src" / "08_decision_value.py"))
    cost_loss_value = mod["cost_loss_value"]
    rng = np.random.default_rng(0)
    obs = (rng.uniform(size=2000) < 0.1).astype(int)
    prob = np.clip(0.1 + 0.6 * obs + rng.normal(0, 0.15, 2000), 0, 1)
    vals = [cost_loss_value(prob, obs, a) for a in (0.05, 0.1, 0.3)]
    assert all(np.isfinite(v) for v in vals)
    assert max(vals) <= 1.0 + 1e-9
    assert len(set(np.round(vals, 6))) > 1


def test_quantile_mapping_moves_the_mean_toward_the_reference():
    """Gate criterion 9: an inert bias correction must not pass silently."""
    import runpy
    mod = runpy.run_path(str(ROOT / "src" / "07_forecast_cases.py"))
    qmap = mod["quantile_map"]
    rng = np.random.default_rng(1)
    ref = rng.gamma(4.0, 2.0, 5000)
    fcst = ref * 1.3 + 2.0
    mapped = qmap(fcst, ref)
    assert abs(mapped.mean() - ref.mean()) < abs(fcst.mean() - ref.mean())


def test_zipped_cds_response_is_merged_to_netcdf(tmp_path):
    """CDS may zip instant and accumulated variables despite `unarchived`."""
    import xarray as xr
    from src.common.era5_io import era5_file_status, publish_era5_download

    coords = {"valid_time": pd.date_range("2019-07-19", periods=2, freq="1h"),
              "latitude": [42.0], "longitude": [-84.0]}
    instant = xr.Dataset(
        {"i10fg": (("valid_time", "latitude", "longitude"),
                    np.ones((2, 1, 1), dtype=np.float32))}, coords=coords)
    accum = xr.Dataset(
        {"tp": (("valid_time", "latitude", "longitude"),
                np.zeros((2, 1, 1), dtype=np.float32))}, coords=coords)
    instant_path, accum_path = tmp_path / "instant.nc", tmp_path / "accum.nc"
    instant.to_netcdf(instant_path)
    accum.to_netcdf(accum_path)
    archive_path, out = tmp_path / "download.nc.part", tmp_path / "era5.nc"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.write(instant_path, "data_stream-oper_stepType-instant.nc")
        archive.write(accum_path, "data_stream-oper_stepType-accum.nc")

    kind = publish_era5_download(archive_path, out)

    assert kind == "zip (2 NetCDF members merged)"
    assert era5_file_status(out) == (True, "ok")
    with xr.open_dataset(out) as merged:
        assert set(merged.data_vars) == {"i10fg", "tp"}


def test_magnitude_sampler_produces_spread_not_medians():
    """The single most damaging silent bug in the design."""
    import runpy
    mod = runpy.run_path(str(ROOT / "src" / "06_compose_mc.py"))
    sample_magnitude = mod["sample_magnitude"]
    qs = [0.05, 0.25, 0.5, 0.75, 0.95]
    qpred = np.tile(np.log1p(np.array([10, 50, 200, 900, 4000.0])), (7, 1))
    draws = sample_magnitude(qpred, qs, np.random.default_rng(0), 200)
    assert draws.shape == (7, 200)
    assert (draws >= 0).all()
    assert (draws.std(axis=1) > 0).all()


# --------------------------------------------------------------------------- #
# Data-dependent: skip until the pipeline has run
# --------------------------------------------------------------------------- #
def test_frac_out_never_exceeds_one():
    h = _load("phase1_county_hourly.parquet")
    assert (h.frac_out <= 1.0).all(), "frac_out > 1 means the MCC denominator is wrong"
    assert h.mcc.notna().all()


def test_event_table_structure():
    ev = _load("phase1_events.parquet")
    assert len(ev) > 0
    assert (ev.customer_hours > 0).all()
    observed = ~ev.censored
    assert observed.any(), "no event restoration was observed before the window ended"
    assert (ev.loc[observed, "end_time"] > ev.loc[observed, "start_time"]).all()
    assert (ev.restoration_hours >= 0).all()


def test_join_did_not_duplicate_reporting_counties():
    cd = _load("phase1_county_day.parquet")
    merged = _load("phase1_merged.parquet")
    hourly = _load("phase1_county_hourly.parquet")
    reporting = set(hourly.fips.astype(str))
    expected = cd[cd.fips.astype(str).isin(reporting)]
    assert len(merged) == len(expected)
    assert merged.gust_max.notna().all()
    assert merged.mcc.notna().all()
    assert (merged.mcc > 0).all()


def test_hazard_consequence_correlation_on_event_days():
    """Criterion 6 -- the premise, not the plumbing."""
    merged = _load("phase1_merged.parquet")
    event_days = merged[merged.event.eq(1)]
    corr = event_days[["gust_max", "customer_hours_per_customer"]].corr().iloc[0, 1]
    assert corr > float(CFG.get("min_hazard_consequence_corr", 0.3)), (
        f"corr={corr:.3f}. Diagnose in this order: timezone alignment, whether "
        "the peak day is the storm day in both datasets, utility reporting "
        "coverage in the hardest-hit counties, whether ERA5 resolved the gusts."
    )


def test_monte_carlo_carries_uncertainty():
    """Every row that drew at least one event must carry spread.

    Not every row: a county-day with a low occurrence probability can draw zero
    events in all N draws and be legitimately all-zero. Asserting spread on
    those would fail for a reason that has nothing to do with the bug this test
    exists to catch -- composing medians instead of sampling.
    """
    p = PATHS.processed / "phase1_mc_samples.npy"
    if not p.exists():
        pytest.skip("MC samples not built yet")
    s = np.load(p)
    assert s.ndim == 2 and s.shape[1] > 1
    assert (s >= 0).all()
    fired = s.max(axis=1) > 0
    assert fired.any(), "no county-day ever drew an event -- occurrence model is dead"
    assert (s[fired].std(axis=1) > 0).all(), \
        "a fired row with zero spread means point estimates were composed"


def test_area_weights_sum_to_one_in_equal_area_crs():
    from scipy import sparse
    from src.common.geo import weights_path
    p = weights_path(CFG)
    if not p.exists():
        pytest.skip("weight matrix not built yet")
    z = np.load(p, allow_pickle=True)
    W = sparse.csr_matrix((z["w_data"], z["w_indices"], z["w_indptr"]),
                          shape=tuple(z["w_shape"]))
    sums = np.asarray(W.sum(axis=1)).ravel()
    assert np.allclose(sums, 1.0, atol=1e-6)
    assert CFG["crs_analysis"] != "EPSG:4326", "area maths must not be done in degrees"
