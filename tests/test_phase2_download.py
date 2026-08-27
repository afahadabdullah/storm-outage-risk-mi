from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

import src.phase2_download as downloader
from src.common.config import Config
from src.phase2_download import (
    ARCO_SOURCE_BY_CDS,
    ERA5_TO_SHORT,
    _select_arco_month,
    era5_arco_cache_status,
    era5_source_variables,
)

ERA5_VARIABLES = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "instantaneous_10m_wind_gust",
    "100m_u_component_of_wind",
    "100m_v_component_of_wind",
    "total_precipitation",
    "2m_temperature",
    "volumetric_soil_water_layer_1",
    "volumetric_soil_water_layer_2",
    "convective_available_potential_energy",
    "snowfall",
    "snow_depth",
]


def test_era5_arco_split_keeps_all_required_predictors():
    arco, cds = era5_source_variables(ERA5_VARIABLES)
    assert len(arco) == 7
    assert {ARCO_SOURCE_BY_CDS[name] for name in arco} == {
        "u10", "v10", "fg10", "u100", "v100", "tp", "t2m",
    }
    assert set(cds) == {
        "volumetric_soil_water_layer_1",
        "volumetric_soil_water_layer_2",
        "convective_available_potential_energy",
        "snowfall",
        "snow_depth",
    }
    assert set(arco) | set(cds) == set(ERA5_VARIABLES)
    assert not (set(arco) & set(cds))


def test_arco_month_is_sliced_before_loading():
    times = pd.date_range("2020-01-01", "2020-01-31 23:00", freq="1h")
    latitudes = np.arange(50.0, 39.0, -1.0)
    longitudes = np.arange(-92.0, -80.0, 1.0)
    shape = (len(times), len(latitudes), len(longitudes))
    requested = ["10m_u_component_of_wind", "total_precipitation"]
    ds = xr.Dataset(
        {ARCO_SOURCE_BY_CDS[name]: (
            ("time", "latitude", "longitude"), np.zeros(shape, dtype=np.float32)
        ) for name in requested},
        coords={"time": times, "latitude": latitudes, "longitude": longitudes},
    )
    cfg = Config({"bbox": [-90.5, 41.6, -82.3, 48.3]})

    subset = _select_arco_month(ds, cfg, 2020, 1, requested)

    assert set(subset.data_vars) == {"u10", "tp"}
    assert subset.sizes == {"time": 31 * 24, "latitude": 7, "longitude": 8}
    assert float(subset.latitude.max()) <= 48.3
    assert float(subset.latitude.min()) >= 41.6
    assert float(subset.longitude.max()) <= -82.3
    assert float(subset.longitude.min()) >= -90.5


def test_local_arco_cache_maps_live_keys_to_pipeline_names(tmp_path, monkeypatch):
    times = pd.date_range("2020-01-01", "2020-01-31 23:00", freq="1h")
    coords = {"time": times, "latitude": [44.0], "longitude": [-85.0]}
    shape = (len(times), 1, 1)
    cfg = Config({
        "bbox": [-90.5, 41.6, -82.3, 48.3],
        "train_start": "2020-01-01",
        "test_end": "2020-01-31",
        "era5_variables": ERA5_VARIABLES,
    })
    cache = tmp_path / "era5_arco_2020_2020.nc"
    output = tmp_path / "era5_202001.arco.part.nc"
    arco_variables, _ = era5_source_variables(ERA5_VARIABLES)
    xr.Dataset(
        {ARCO_SOURCE_BY_CDS[name]: (
            ("time", "latitude", "longitude"),
            np.ones(shape, dtype=np.float32),
        ) for name in arco_variables},
        coords=coords,
    ).to_netcdf(cache, engine="netcdf4")
    monkeypatch.setattr(downloader, "era5_arco_cache_path", lambda _cfg: cache)

    valid, reason = era5_arco_cache_status(cache, cfg)
    assert valid, reason
    downloader._write_arco_month(cfg, 2020, 1, arco_variables, output)

    with xr.open_dataset(output) as ds:
        assert set(ds.data_vars) == {
            "u10", "v10", "i10fg", "u100", "v100", "tp", "t2m",
        }
        assert ds.sizes["time"] == 31 * 24


def test_arco_and_cds_parts_publish_one_complete_month(tmp_path, monkeypatch):
    times = pd.date_range("2020-01-01", "2020-01-31 23:00", freq="1h")
    coords = {"time": times, "latitude": [44.0], "longitude": [-85.0]}
    shape = (len(times), 1, 1)
    cfg = Config({
        "bbox": [-90.5, 41.6, -82.3, 48.3],
        "era5_backend": "arco",
        "era5_variables": ERA5_VARIABLES,
    })
    out = tmp_path / "era5_202001.nc"
    monkeypatch.setattr(downloader, "era5_month_path", lambda year, month: out)

    def fake_arco(_cfg, _year, _month, variables, dest):
        names = [ERA5_TO_SHORT[name] for name in variables]
        xr.Dataset(
            {name: (("time", "latitude", "longitude"),
                    np.ones(shape, dtype=np.float32)) for name in names},
            coords=coords,
        ).to_netcdf(dest, engine="netcdf4")

    def fake_cds(_cfg, _year, _month, variables, dest):
        names = [ERA5_TO_SHORT[name] for name in variables]
        cds_coords = dict(coords)
        cds_coords["valid_time"] = cds_coords.pop("time")
        xr.Dataset(
            {name: (("valid_time", "latitude", "longitude"),
                    np.ones(shape, dtype=np.float32)) for name in names},
            coords=cds_coords,
        ).to_netcdf(dest, engine="netcdf4")

    monkeypatch.setattr(downloader, "_write_arco_month", fake_arco)
    monkeypatch.setattr(downloader, "_download_cds_month", fake_cds)

    result = downloader.fetch_era5_month(cfg, 2020, 1)

    assert result == out
    valid, reason = downloader.era5_month_status(out, cfg, 2020, 1)
    assert valid, reason
    with xr.open_dataset(out) as ds:
        assert set(ds.data_vars) == set(ERA5_TO_SHORT.values())
        assert ds.sizes["time"] == 31 * 24
    assert not list(tmp_path.glob("*.part*"))
