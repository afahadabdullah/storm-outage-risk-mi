from __future__ import annotations

import pandas as pd

from src.common.config import Config
from src.phase2_train import masks, storm_groups


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
    frame = pd.DataFrame({"date": dates, "event": [0, 1, 0, 1, 0, 0, 0, 1]})
    groups = storm_groups(frame, gap_days=2)
    assert groups.iloc[1] == groups.iloc[2] == groups.iloc[3]
    assert groups.iloc[7] != groups.iloc[1]
