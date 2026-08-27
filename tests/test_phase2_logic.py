from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupKFold

from src.common.config import Config
from src.phase2_train import (
    _interp_log_quantiles,
    crps_from_quantiles,
    masks,
    storm_groups,
)


def test_phase2_frozen_splits_do_not_overlap():
    frame = pd.DataFrame({
        "date": pd.to_datetime(["2021-12-31", "2022-01-01", "2022-12-31",
                                "2023-01-01", "2023-12-31"])
    })
    cfg = Config({"train_start": "2018-01-01", "train_end": "2021-12-31",
                  "val_start": "2022-01-01", "val_end": "2022-12-31",
                  "test_start": "2023-01-01", "test_end": "2023-12-31"})
    split = masks(frame, cfg)
    assert split["train"].tolist() == [True, False, False, False, False]
    assert split["val"].tolist() == [False, True, True, False, False]
    assert split["test"].tolist() == [False, False, False, True, True]
    assert not (split["train"] & split["val"]).any()
    assert not (split["val"] & split["test"]).any()


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
