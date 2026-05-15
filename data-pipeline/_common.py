"""Shared helpers for the climate data pipeline.

Kept intentionally small — each pipeline step (01–06) imports only the loaders
and zonal-stats utilities it needs. Heavier orchestration logic stays in the
step scripts so they remain CLI-friendly and self-contained.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
import regionmask
import xarray as xr

HYRAS_CRS = "EPSG:3035"  # ETRS89-LAEA — HYRAS NetCDF
# HOSTRADA is published on a Lambert Conformal Conic grid (EPSG:3034), not the
# LAEA grid used by HYRAS. The X/Y coordinate ranges overlap numerically with
# EPSG:3035, so masking with the wrong CRS silently produces NaNs instead of
# raising — verified against rsds/sfcWind file headers (grid_mapping_name=
# lambert_conformal_conic, epsg_code=EPSG:3034).
HOSTRADA_CRS = "EPSG:3034"
# DWD legacy daily Germany grids (EVPOT, SOILMOIST, SOILTEMP `.tgz` archives of
# ESRI ASCII) are published in DHDN / 3-degree Gauss-Krüger zone 3.
LEGACY_GRID_CRS = "EPSG:31467"
DISTRICT_ID_COL = "district_id"


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    return logging.getLogger(name)


def load_districts(shapefile: str | os.PathLike, target_crs: str) -> gpd.GeoDataFrame:
    """Load district polygons and reproject to ``target_crs``.

    The shapefile must contain a stable district identifier. We accept any of
    ``district_id``, ``AGS``, ``NUTS_ID``, ``NUTS3_ID`` and normalise to ``district_id``.
    """
    gdf = gpd.read_file(shapefile)

    id_candidates = [
        DISTRICT_ID_COL,
        "AGS",
        "ags",
        "NUTS_ID",
        "nuts_id",
        "NUTS3_ID",
        "nuts3_id",
        "RS",
    ]
    id_col = next((c for c in id_candidates if c in gdf.columns), None)
    if id_col is None:
        raise ValueError(
            f"District shapefile {shapefile} must contain one of {id_candidates}"
        )
    if id_col != DISTRICT_ID_COL:
        gdf = gdf.rename(columns={id_col: DISTRICT_ID_COL})

    gdf[DISTRICT_ID_COL] = gdf[DISTRICT_ID_COL].astype(str)
    gdf = gdf[[DISTRICT_ID_COL, "geometry"]].dissolve(by=DISTRICT_ID_COL).reset_index()

    if gdf.crs is None:
        raise ValueError(f"District shapefile {shapefile} has no CRS")
    return gdf.to_crs(target_crs)


def build_region_mask(
    districts: gpd.GeoDataFrame, ds: xr.Dataset, lat_dim: str, lon_dim: str
) -> xr.DataArray:
    """Build a regionmask mask aligned to ``ds`` for the given districts.

    ``wrap_lon=False`` is required when the coordinate axis carries projected
    metres (HYRAS/HOSTRADA EPSG:3035). For geographic GCFS2.2 inputs the flag
    is harmless because longitudes are already monotonic.
    """
    regions = regionmask.Regions(
        outlines=list(districts.geometry),
        names=list(districts[DISTRICT_ID_COL]),
        abbrevs=list(districts[DISTRICT_ID_COL]),
        name="districts",
    )
    return regions.mask(ds[lon_dim], ds[lat_dim], wrap_lon=False)


def zonal_mean_daily(
    da: xr.DataArray,
    districts: gpd.GeoDataFrame,
    lat_dim: str,
    lon_dim: str,
    time_dim: str = "time",
) -> pd.DataFrame:
    """Compute district-wise daily mean of a 3-D (time, y, x) DataArray.

    Returns a long-form DataFrame with columns ``date, district_id, value``.
    """
    mask = build_region_mask(districts, da.to_dataset(name="_v"), lat_dim, lon_dim)

    grouped = da.groupby(mask).mean(skipna=True)
    df = grouped.to_dataframe(name="value").reset_index()

    region_col = mask.name or "mask"
    region_to_id = dict(enumerate(districts[DISTRICT_ID_COL].values))
    df[DISTRICT_ID_COL] = df[region_col].astype("Int64").map(region_to_id)
    df = df.drop(columns=region_col)
    df = df.rename(columns={time_dim: "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df[["date", DISTRICT_ID_COL, "value"]].dropna(subset=[DISTRICT_ID_COL])


def detect_lat_lon_names(ds: xr.Dataset) -> tuple[str, str]:
    """Return (lat_dim, lon_dim) for an xarray dataset with common naming.

    Projected dim coords (``y``/``x``) are preferred when present, because
    HYRAS/HOSTRADA carry both 1-D projected dims and 2-D auxiliary geographic
    ``lat``/``lon`` coordinates — using the projected pair avoids reprojecting
    districts back to EPSG:4326 just for masking.
    """
    lat_priority = ("y", "Y", "rlat", "lat", "latitude")
    lon_priority = ("x", "X", "rlon", "lon", "longitude")
    lat = next((n for n in lat_priority if n in ds.coords), None)
    lon = next((n for n in lon_priority if n in ds.coords), None)
    if lat is None or lon is None:
        raise ValueError(
            f"Could not detect lat/lon coordinates in dataset (coords={list(ds.coords)})"
        )
    return lat, lon


def write_parquet(df: pd.DataFrame, path: str | os.PathLike) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def years_range(start: int, end: int) -> Sequence[int]:
    if start > end:
        raise ValueError(f"start year {start} > end year {end}")
    return list(range(start, end + 1))


def cast_float32(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for c in columns:
        if c in df.columns:
            df[c] = df[c].astype(np.float32)
    return df
