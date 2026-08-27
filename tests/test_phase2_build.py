from __future__ import annotations

import pandas as pd

from src.phase2_build import bound_outages_by_mcc


def test_impossible_outage_counts_are_bounded_and_audited():
    hourly = pd.DataFrame({
        "fips": ["26001", "26001", "26003"],
        "customers_out": [50.0, 620.0, 10.0],
        "mcc": [100.0, 100.0, 20.0],
    })
    coverage = {}

    bounded = bound_outages_by_mcc(hourly, coverage)

    assert bounded.customers_out.tolist() == [50.0, 100.0, 10.0]
    assert hourly.customers_out.tolist() == [50.0, 620.0, 10.0]
    assert coverage["outage_rows_capped_to_mcc"] == {
        "n_rows": 1,
        "fraction_of_rows": 1 / 3,
        "counties": ["26001"],
        "max_raw_fraction": 6.2,
    }
