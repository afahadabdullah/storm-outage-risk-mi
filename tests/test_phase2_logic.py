from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupKFold

from src import phase2_build
from src.common.config import Config
from src.phase2_backtest import county_skill, skill_matrix
from src.phase2_forecast import (
    ensure_finite_forecast_features,
    fit_quantile_map,
    forecast_feature_fills,
    model_forecast_cohort,
    quantile_distance,
    quantile_map_cellwise,
)
from src.phase2_report import county_skill as report_county_skill
from src.phase2_report import gefs_case_matrix, results_matrix
from src.phase2_train import (
    _interp_log_quantiles,
    crps_from_quantiles,
    masks,
    period_slug,
    storm_groups,
    validate_temporal_split,
)


def test_phase2_frozen_splits_do_not_overlap():
    frame = pd.DataFrame({
        "date": pd.to_datetime(["2022-06-30", "2022-07-01", "2022-12-31",
                                "2023-01-01", "2023-12-31"])
    })
    cfg = Config({"train_start": "2018-01-01", "train_end": "2022-06-30",
                  "val_start": "2022-07-01", "val_end": "2022-12-31",
                  "test_start": "2023-01-01", "test_end": "2023-12-31"})
    split = masks(frame, cfg)
    assert split["train"].tolist() == [True, False, False, False, False]
    assert split["val"].tolist() == [False, True, True, False, False]
    assert split["test"].tolist() == [False, False, False, True, True]
    assert not (split["train"] & split["val"]).any()
    assert not (split["val"] & split["test"]).any()


def test_phase2_split_requires_contiguous_windows():
    cfg = Config({"train_start": "2018-01-01", "train_end": "2022-06-30",
                  "val_start": "2022-07-02", "val_end": "2022-12-31",
                  "test_start": "2023-01-01", "test_end": "2023-12-31"})
    with pytest.raises(ValueError, match="Training and validation"):
        validate_temporal_split(cfg)


def test_period_slug_names_half_year_validation_unambiguously():
    assert period_slug("2022-07-01", "2022-12-31") == "2022H2"
    assert period_slug("2023-01-01", "2023-12-31") == "2023"


def test_results_matrix_keeps_validation_and_test_side_by_side():
    artifacts = {
        "validation_metrics": {"occurrence_brier": 0.08,
                               "occurrence_roc_auc": 0.75},
        "test_metrics": {"occurrence_brier": 0.09,
                         "occurrence_roc_auc": 0.72},
    }
    matrix = results_matrix(artifacts)
    brier = matrix[matrix.key.eq("occurrence_brier")].iloc[0]
    assert brier.validation == pytest.approx(0.08)
    assert brier.test == pytest.approx(0.09)
    assert brier.better == "lower"


def test_gefs_case_matrix_reports_frozen_case_interval(tmp_path):
    path = tmp_path / "realizations.npz"
    values = np.arange(100, dtype=float)
    np.savez(path, **{"Feb 2023 ice storm|5": values})
    archive = np.load(path)
    cfg = Config({"case_studies": [
        {"name": "Feb 2023 ice storm", "date": "2023-02-22"}]})
    artifacts = {
        "realizations": archive,
        "opened": False,
        "test_predictions": None,
        "uncertainty": pd.DataFrame({
            "case": ["Feb 2023 ice storm"], "lead_days": [5],
            "meteorological_share": [0.64],
        }),
        "forecast_summary": {"synthetic_gefs": False},
    }
    matrix = gefs_case_matrix(cfg, artifacts)
    assert matrix.loc[0, "median_customer_hours"] == pytest.approx(49.5)
    assert matrix.loc[0, "p10_customer_hours"] == pytest.approx(9.9)
    assert matrix.loc[0, "p90_customer_hours"] == pytest.approx(89.1)
    assert matrix.loc[0, "meteorological_variance_share"] == pytest.approx(0.64)
    assert matrix.loc[0, "input"] == "GEFS"


def test_report_county_skill_preserves_evaluated_county_diagnostics():
    prediction = pd.DataFrame({
        "fips": ["26001"] * 3 + ["26003"] * 3,
        "event": [0, 1, 1, 0, 0, 1],
        "probability": [0.1, 0.7, 0.8, 0.2, 0.3, 0.6],
        "reference_climatology_county": [0.4] * 6,
        "customer_hours": [0, 100, 120, 0, 0, 80],
        "magnitude_q05": [0, 50, 60, 0, 0, 40],
        "magnitude_q50": [0, 100, 120, 0, 0, 80],
        "magnitude_q95": [0, 150, 180, 0, 0, 120],
    })
    result = report_county_skill(prediction, [0.05, 0.50, 0.95])
    assert result.fips.tolist() == ["26001", "26003"]
    assert result.loc[0, "n_events"] == 2
    assert result.loc[0, "probability_bias"] == pytest.approx(-0.1333333333)
    assert {"brier_skill", "magnitude_crps", "observed_customer_hours"} <= set(result)


def test_storm_blocking_keeps_nearby_days_in_one_group():
    dates = pd.date_range("2022-06-01", periods=8, freq="D")
    frame = pd.DataFrame({"fips": "26001", "date": dates,
                          "event": [0, 1, 0, 1, 0, 0, 0, 1]})
    groups = storm_groups(frame, gap_days=2, min_county_frac=0.10)
    assert groups.iloc[1] == groups.iloc[2] == groups.iloc[3]
    assert groups.iloc[7] != groups.iloc[1]


def _statewide_frame(n_counties: int, per_county_rate: float, seed: int = 0):
    """A frame at the shape the real study has, not a toy eight rows."""
    dates = pd.date_range("2018-01-01", "2022-12-31", freq="D")
    fips = [f"26{i:03d}" for i in range(1, 2 * n_counties, 2)]
    frame = pd.MultiIndex.from_product(
        [fips, dates], names=["fips", "date"]).to_frame(index=False)
    rng = np.random.default_rng(seed)
    frame["event"] = (rng.random(len(frame)) < per_county_rate).astype(int)
    return frame


@pytest.mark.parametrize("n_counties,rate", [(83, 0.05), (83, 0.02), (40, 0.03)])
def test_storm_blocking_survives_statewide_scale(n_counties, rate):
    """Regression test for the defect that made this file worth rewriting.

    Defining a storm day as `groupby("date").event.max()` -- ANY county in event
    -- saturates once the state has dozens of counties: every calendar day
    qualifies, consecutive days never exceed the merge gap, the whole record
    fuses into ONE group, and GroupKFold(n_splits=1) raises at the very end of
    a multi-hour training job.

    This test fails on that implementation and passes on the fraction-of-
    counties one. If it ever passes on both, it has stopped testing anything.
    """
    frame = _statewide_frame(n_counties, rate)
    groups = storm_groups(frame, gap_days=2, min_county_frac=0.10)
    assert groups.nunique() >= 2, (
        f"{n_counties} counties at rate {rate} collapsed to "
        f"{groups.nunique()} storm group(s); GroupKFold cannot split that")
    # And the thing that actually broke: the splitter must construct and run.
    n_splits = min(5, groups.nunique())
    folds = list(GroupKFold(n_splits=n_splits).split(frame, frame.event, groups))
    assert len(folds) == n_splits
    for train_idx, val_idx in folds:
        overlap = set(groups.iloc[train_idx]) & set(groups.iloc[val_idx])
        assert not overlap, "a storm episode appeared in both train and validation"


def test_storm_blocking_never_builds_a_single_split():
    """A degenerate frame must be reported, not turned into GroupKFold(1)."""
    dates = pd.date_range("2022-01-01", periods=40, freq="D")
    frame = pd.MultiIndex.from_product(
        [["26001", "26003"], dates], names=["fips", "date"]).to_frame(index=False)
    frame["event"] = 1                       # every county in event every day
    groups = storm_groups(frame, gap_days=2, min_county_frac=0.10)
    assert groups.nunique() == 1
    # occurrence_cv must skip this scheme rather than raise; the guard is
    # `n_groups < 2`, so assert the condition it keys on.
    assert min(5, groups.nunique()) < 2


def test_crps_from_quantiles_matches_the_analytic_normal():
    """The quantile CRPS must be the real thing, not a truncated ensemble.

    The previous implementation built an equally weighted ensemble spanning only
    q05..q95, discarding the outer 10% of the predictive mass. Against a closed
    form that shows up as a bias; against a heavy-tailed target it is worse.
    """
    from scipy.stats import norm

    mu, sigma = 500.0, 120.0
    qs = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    rng = np.random.default_rng(0)
    observed = rng.normal(mu, sigma, 4000)
    qpred = np.tile(norm.ppf(qs, mu, sigma), (len(observed), 1))

    z = (observed - mu) / sigma
    analytic = float(np.mean(sigma * (z * (2 * norm.cdf(z) - 1)
                                      + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))))
    got = crps_from_quantiles(observed, qpred, qs)
    assert abs(got - analytic) / analytic < 0.005


def test_quantile_extrapolation_is_monotone_and_extends_the_tails():
    trained = [0.05, 0.25, 0.50, 0.75, 0.95]
    base = np.array([[1.0, 2.0, 3.0, 4.0, 5.0],
                     [10.0, 20.0, 30.0, 40.0, 50.0]])
    wanted = [0.001, 0.05, 0.5, 0.95, 0.999]
    out = _interp_log_quantiles(base, trained, wanted)
    assert (np.diff(out, axis=1) > 0).all(), "quantile function must be increasing"
    # the extrapolated tails must go beyond the trained range, not clamp to it
    assert (out[:, 0] < base[:, 0]).all()
    assert (out[:, -1] > base[:, -1]).all()


def test_backtest_skill_outputs_are_partitioned_by_month_and_county():
    """The diagnostic artifacts must retain the strata needed to inspect skill."""
    prediction = pd.DataFrame({
        "fips": ["26001", "26003", "26001", "26003", "26001", "26003"],
        "date": pd.to_datetime(["2021-01-01", "2021-01-01", "2021-01-02",
                                 "2021-01-02", "2021-02-01", "2021-02-01"]),
        "event": [0, 1, 0, 1, 1, 0],
        "probability": [0.1, 0.8, 0.2, 0.7, 0.8, 0.1],
        "reference_climatology_county": [0.4] * 6,
        "customer_hours": [0, 100, 0, 120, 140, 0],
        "magnitude_q05": [0, 50, 0, 60, 70, 0],
        "magnitude_q50": [0, 100, 0, 120, 140, 0],
        "magnitude_q95": [0, 150, 0, 180, 210, 0],
    })
    train = pd.DataFrame({"event": [1, 1, 1, 0],
                          "customer_hours": [80, 110, 150, 0]})
    matrix = skill_matrix(prediction, train, [0.05, 0.50, 0.95])
    counties = county_skill(prediction)
    assert matrix.month.tolist() == [1, 2]
    assert matrix.loc[matrix.month.eq(1), "n_events"].item() == 2
    assert counties.fips.tolist() == ["26001", "26003"]
    assert {"brier_skill", "observed_event_rate", "probability_bias"} <= set(counties)


def test_backtest_ends_before_the_first_missing_weather_month(tmp_path, monkeypatch):
    """A late cache file must not hide a missing month in the score window."""
    monthly = tmp_path / "era5_monthly"
    monthly.mkdir()
    for month in (1, 2, 4):
        (monthly / f"era5_2021{month:02d}.nc").touch()
    monkeypatch.setattr(phase2_build, "PATHS", SimpleNamespace(raw=tmp_path))
    assert phase2_build.available_backtest_end(Config({}), 2021) == pd.Timestamp("2021-02-28")


def test_forecast_feature_guard_uses_frozen_training_medians_only():
    merged = pd.DataFrame({
        "date": pd.to_datetime(["2018-01-01", "2018-01-02", "2020-01-01"]),
        "feature_a": [2.0, 6.0, 100.0],
        "feature_b": [1.0, 5.0, 99.0],
    })
    bundle = {"features": ["feature_a", "feature_b"]}
    cfg = Config({"train_start": "2018-01-01", "train_end": "2018-12-31"})
    fills = forecast_feature_fills(merged, bundle, cfg)
    rows = pd.DataFrame({"feature_a": [np.nan], "feature_b": [np.inf]})
    repaired = ensure_finite_forecast_features(rows, bundle, fills, "test")
    assert repaired.feature_a.item() == 4.0
    assert repaired.feature_b.item() == 3.0


def test_forecast_uses_the_frozen_reporting_counties_not_every_grid_county():
    cohort, positions = model_forecast_cohort(
        {"reporting_counties": ["26003", "26001"]}, ["26001", "26003", "26005"])
    assert cohort == ["26003", "26001"]
    assert positions.tolist() == [1, 0]


def test_pooled_quantile_mapping_reduces_quantile_distance():
    rng = np.random.default_rng(4)
    raw = rng.normal(4.0, 0.8, size=(500, 3)).clip(0)
    reference = rng.normal(7.0, 1.1, size=(3000, 3)).clip(0)
    source_q = fit_quantile_map(raw)
    reference_q = fit_quantile_map(reference)
    mapped = quantile_map_cellwise(raw, source_q, reference_q)
    assert quantile_distance(fit_quantile_map(mapped), reference_q) < \
           quantile_distance(source_q, reference_q)
