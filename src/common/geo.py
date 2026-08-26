"""County geometry and the ERA5 cell -> county weight matrix.

Two matrices come out of here and they are used for different things on purpose
(project spec section 5.1 vs section 12 "county-mean gust destroys the signal"):

  W  area weights, rows sum to 1.0   -> spatial MEAN of smooth fields
                                        (precip, soil moisture, temperature)
  M  membership, 1 where a cell       -> spatial MAX / p95 of tail fields
     intersects the county               (gust, CAPE) -- damage is a tail
                                         phenomenon and the mean erases it

Both are computed once in the equal-area CRS from config and cached. Area maths
in EPSG:4326 is wrong by a latitude-dependent factor -- large enough to matter
in a north-south state like Michigan, small enough to look plausible.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy import sparse

from .config import PATHS, Config, state_prefixes
from .logio import get_logger

log = get_logger("geo")
CELL_DEG = 0.25  # ERA5 single-levels native grid spacing


# --------------------------------------------------------------------------- #
# Counties
# --------------------------------------------------------------------------- #
def counties_path(cfg: Config) -> Path:
    # GeoParquet, not GeoPackage: no SQLite, no GDAL write driver, and it works
    # on network and FUSE mounts where .gpkg writes fail with an opaque error.
    return PATHS.raw / f"tiger_counties_{cfg['sources']['tiger_year']}.parquet"


def fetch_counties(cfg: Config, force: bool = False) -> Path:
    """TIGER/Line counties, clipped to the configured states. Static: cache it."""
    out = counties_path(cfg)
    if out.exists() and not force:
        return out
    import geopandas as gpd

    year = cfg["sources"]["tiger_year"]
    url = cfg["sources"]["tiger_url"].format(year=year)
    log.info("downloading TIGER counties %s", url)
    r = requests.get(url, timeout=300)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        shp = next(n for n in z.namelist() if n.endswith(".shp"))
        z.extractall(PATHS.raw / "tiger_tmp")
    gdf = gpd.read_file(PATHS.raw / "tiger_tmp" / shp)
    gdf = gdf[gdf.STATEFP.isin(state_prefixes(cfg))].copy()
    gdf["GEOID"] = gdf.GEOID.astype(str).str.zfill(5)
    gdf = gdf[["GEOID", "NAME", "ALAND", "AWATER", "geometry"]]
    gdf.to_parquet(out)
    log.info("counties: %d written to %s", len(gdf), out.name)
    return out


def load_counties(cfg: Config):
    """Counties in the analysis (equal-area) CRS, GEOID-indexed, sorted."""
    import geopandas as gpd

    gdf = gpd.read_parquet(counties_path(cfg))
    gdf["GEOID"] = gdf.GEOID.astype(str).str.zfill(5)
    gdf = gdf.sort_values("GEOID").set_index("GEOID")
    return gdf.to_crs(cfg["crs_analysis"])


# --------------------------------------------------------------------------- #
# Weight matrix
# --------------------------------------------------------------------------- #
def cell_polygons(lats: np.ndarray, lons: np.ndarray, crs_analysis: str):
    """Rectangular ERA5 cells in row-major (lat, lon) order -> flat index."""
    import geopandas as gpd
    from shapely.geometry import box

    half = CELL_DEG / 2
    lon_g, lat_g = np.meshgrid(np.asarray(lons), np.asarray(lats))
    flat_lon, flat_lat = lon_g.ravel(), lat_g.ravel()
    geoms = [box(x - half, y - half, x + half, y + half)
             for x, y in zip(flat_lon, flat_lat)]
    gdf = gpd.GeoDataFrame(
        {"cell": np.arange(len(geoms)), "lat": flat_lat, "lon": flat_lon},
        geometry=geoms, crs="EPSG:4326",
    )
    return gdf.to_crs(crs_analysis)


def weights_path(cfg: Config) -> Path:
    return PATHS.interim / f"cell_county_weights_{cfg['region_name'].lower()}.npz"


def build_weight_matrix(cfg: Config, lats, lons, force: bool = False):
    """Return (W, M, geoids, shape). Computed once, reused every timestep.

    Doing the geometric intersection per timestep instead of once is the single
    slowest mistake available in this pipeline (phase 1 spec section 8).
    """
    import geopandas as gpd

    cache = weights_path(cfg)
    shape = (len(lats), len(lons))
    if cache.exists() and not force:
        z = np.load(cache, allow_pickle=True)
        same_coords = ("lats" in z.files and "lons" in z.files
                       and np.allclose(z["lats"], lats) and np.allclose(z["lons"], lons))
        if tuple(z["grid_shape"]) == shape and same_coords:
            W = sparse.csr_matrix((z["w_data"], z["w_indices"], z["w_indptr"]),
                                  shape=tuple(z["w_shape"]))
            M = sparse.csr_matrix((z["m_data"], z["m_indices"], z["m_indptr"]),
                                  shape=tuple(z["m_shape"]))
            return W, M, [str(g) for g in z["geoids"]], shape
        log.warning("cached weight matrix does not match requested coordinates "
                    "(cached shape %s, requested %s) -- rebuilding",
                    tuple(z["grid_shape"]), shape)

    counties = load_counties(cfg)
    cells = cell_polygons(lats, lons, cfg["crs_analysis"])
    log.info("intersecting %d counties x %d cells in %s",
             len(counties), len(cells), cfg["crs_analysis"])

    inter = gpd.overlay(
        counties.reset_index()[["GEOID", "geometry"]],
        cells[["cell", "geometry"]],
        how="intersection", keep_geom_type=True,
    )
    inter["area"] = inter.geometry.area

    geoids = list(counties.index)
    row_of = {g: i for i, g in enumerate(geoids)}
    rows = inter.GEOID.map(row_of).to_numpy()
    cols = inter.cell.to_numpy()

    n_cells = len(cells)
    W = sparse.csr_matrix((inter.area.to_numpy(), (rows, cols)),
                          shape=(len(geoids), n_cells))
    totals = np.asarray(W.sum(axis=1)).ravel()
    if (totals <= 0).any():
        missing = [geoids[i] for i in np.where(totals <= 0)[0]]
        raise ValueError(f"counties with no intersecting ERA5 cell: {missing} "
                         "-- bbox is too tight, pad it in region.yaml")
    W = sparse.diags(1.0 / totals) @ W          # rows now sum to 1.0
    M = (W > 0).astype(np.float64)

    np.savez_compressed(
        cache,
        w_data=W.data, w_indices=W.indices, w_indptr=W.indptr, w_shape=W.shape,
        m_data=M.data, m_indices=M.indices, m_indptr=M.indptr, m_shape=M.shape,
        geoids=np.array(geoids), grid_shape=np.array(shape),
        lats=np.asarray(lats), lons=np.asarray(lons),
    )
    log.info("weight matrix cached -> %s", cache.name)
    return W.tocsr(), M.tocsr(), geoids, shape


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def agg_mean(W: sparse.csr_matrix, field: np.ndarray) -> np.ndarray:
    """(time, lat, lon) -> (time, county) area-weighted mean."""
    t = field.shape[0]
    return (W @ field.reshape(t, -1).T).T


def agg_max(M: sparse.csr_matrix, field: np.ndarray) -> np.ndarray:
    """(time, lat, lon) -> (time, county) max over intersecting cells.

    Sparse matrices have no max-product, so this walks each county's cell list
    once. n_counties is O(100); this is not the bottleneck.
    """
    t = field.shape[0]
    flat = field.reshape(t, -1)
    out = np.empty((t, M.shape[0]), dtype=flat.dtype)
    for i in range(M.shape[0]):
        idx = M.indices[M.indptr[i]:M.indptr[i + 1]]
        out[:, i] = flat[:, idx].max(axis=1)
    return out


def agg_quantile(M: sparse.csr_matrix, field: np.ndarray, q: float) -> np.ndarray:
    t = field.shape[0]
    flat = field.reshape(t, -1)
    out = np.empty((t, M.shape[0]), dtype=np.float64)
    for i in range(M.shape[0]):
        idx = M.indices[M.indptr[i]:M.indptr[i + 1]]
        out[:, i] = np.quantile(flat[:, idx], q, axis=1)
    return out
