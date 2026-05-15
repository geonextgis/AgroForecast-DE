#!/usr/bin/env python
"""Step 7 — Static district-level data (soil quality + topography) → Parquet.

Produces two static lookup tables, one row per district:

    <output_dir>/soil_quality.parquet     # soil_quality_mean, soil_quality_stdDev
    <output_dir>/topography.parquet       # elevation_mean / _stdDev, slope_mean / _stdDev

Sources
-------
* **Soil quality**: BÜK / ESDAC SQR1000 raster (250 m, EPSG:3035 by default).
  Reprojected to EPSG:25832 and aggregated with ``rasterstats.zonal_stats``.
* **Topography**: Copernicus GLO-30 DEM via Google Earth Engine
  (``projects/sat-io/open-datasets/GLO-30``). Slope is computed from elevation
  with ``ee.Terrain.slope``. Districts are pushed to EE on the fly with
  ``geemap.geopandas_to_ee`` so no pre-uploaded EE asset is required.

Districts with missing values (e.g., no raster coverage) are filled by nearest
centroid neighbour to keep the static table complete for every district in the
input shapefile.

Irrigation is intentionally *not* handled here — it lives in a separate step
when/if it is added back to the feature set.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio
import rasterio.mask
from rasterio.warp import Resampling, calculate_default_transform, reproject
from rasterstats import zonal_stats
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    DISTRICT_ID_COL,
    cast_float32,
    get_logger,
    load_districts,
    write_parquet,
)

LOG = get_logger("07_ingest_static")

STATIC_CRS = "EPSG:25832"  # ETRS89 / UTM 32N — matches districts.gpkg native CRS


# ---------------------------------------------------------------------------
# Soil quality
# ---------------------------------------------------------------------------


def reproject_raster(src_path: Path, dst_path: Path, dst_crs: str) -> None:
    with rio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        meta = src.meta.copy()
        meta.update(crs=dst_crs, transform=transform, width=width, height=height)
        with rio.open(dst_path, "w", **meta) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rio.band(src, i),
                    destination=rio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest,
                )


def extract_soil_quality(
    soil_raster: Path,
    districts: gpd.GeoDataFrame,
    interim_dir: Path,
) -> pd.DataFrame:
    interim_dir.mkdir(parents=True, exist_ok=True)
    reproj_path = interim_dir / f"{soil_raster.stem}_{STATIC_CRS.replace(':', '')}.tif"
    if not reproj_path.exists():
        LOG.info("Reprojecting soil raster → %s", reproj_path)
        reproject_raster(soil_raster, reproj_path, STATIC_CRS)

    valid_ids: list[str] = []
    with rio.open(reproj_path) as src:
        for _, row in tqdm(
            districts.iterrows(),
            total=len(districts),
            desc="Checking soil coverage",
        ):
            try:
                masked, _ = rio.mask.mask(src, [row["geometry"]], crop=True)
                if len(np.unique(masked)) > 1:  # has values beyond nodata/background
                    valid_ids.append(row[DISTRICT_ID_COL])
            except Exception as exc:
                LOG.warning(
                    "District %s skipped during soil masking: %s",
                    row[DISTRICT_ID_COL],
                    exc,
                )

    valid = districts[districts[DISTRICT_ID_COL].isin(valid_ids)].copy()
    stats = zonal_stats(valid, str(reproj_path), stats=["mean", "std"])

    out = valid[[DISTRICT_ID_COL]].copy()
    out["soil_quality_mean"] = [s["mean"] for s in stats]
    out["soil_quality_stdDev"] = [s["std"] for s in stats]
    return out


# ---------------------------------------------------------------------------
# Topography (elevation + slope) via Google Earth Engine
# ---------------------------------------------------------------------------


def extract_topography(
    districts: gpd.GeoDataFrame,
    scale: int,
    ee_project: str | None,
) -> pd.DataFrame:
    import ee
    import geemap

    if ee_project:
        ee.Initialize(project=ee_project)
    else:
        ee.Initialize()

    fc = geemap.geopandas_to_ee(districts[[DISTRICT_ID_COL, "geometry"]])

    elev = (
        ee.ImageCollection("projects/sat-io/open-datasets/GLO-30")
        .filterBounds(fc)
        .mosaic()
    )
    slope = ee.Terrain.slope(elev)
    image = elev.rename("elevation").addBands(slope.rename("slope"))

    reducer = ee.Reducer.mean().combine(reducer2=ee.Reducer.stdDev(), sharedInputs=True)
    LOG.info("Reducing GLO-30 elevation/slope per district (scale=%d m)", scale)
    stats_fc = image.reduceRegions(collection=fc, reducer=reducer, scale=scale)
    df = geemap.ee_to_df(stats_fc)

    return df[
        [
            DISTRICT_ID_COL,
            "elevation_mean",
            "elevation_stdDev",
            "slope_mean",
            "slope_stdDev",
        ]
    ]


# ---------------------------------------------------------------------------
# Nearest-neighbour fill for any missing districts
# ---------------------------------------------------------------------------


def fill_missing_with_nearest(
    df: pd.DataFrame, districts: gpd.GeoDataFrame
) -> pd.DataFrame:
    merged = districts[[DISTRICT_ID_COL, "geometry"]].merge(
        df, on=DISTRICT_ID_COL, how="left"
    )
    merged["centroid"] = merged.geometry.centroid

    value_cols = [c for c in df.columns if c != DISTRICT_ID_COL]
    for col in value_cols:
        missing_idx = merged[merged[col].isna()].index
        if len(missing_idx) == 0:
            continue
        valid = merged[merged[col].notna()]
        if valid.empty:
            LOG.warning("No valid values to backfill column %s", col)
            continue
        for idx in missing_idx:
            distances = valid["centroid"].distance(merged.at[idx, "centroid"])
            merged.at[idx, col] = merged.at[distances.idxmin(), col]
        merged[col] = merged[col].round(3)

    return (
        merged.drop(columns=["geometry", "centroid"])
        .sort_values(DISTRICT_ID_COL)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--soil_raster",
        default="/beegfs/halder/DATA/DE_Soil_Quality_Rating_250m/sqr1000_250_v10.tif",
        help="Path to the SQR1000 soil-quality GeoTIFF.",
    )
    p.add_argument(
        "--district_shapefile",
        default="data/raw/districts/districts.gpkg",
    )
    p.add_argument("--output_dir", default="data/interim/static")
    p.add_argument(
        "--gee_scale",
        type=int,
        default=100,
        help="Reduction scale (m) for GLO-30 zonal stats on GEE.",
    )
    p.add_argument(
        "--ee_project",
        default="ee-geonextgis",
        help="Optional Earth Engine cloud project (e.g. ee-yourname).",
    )
    p.add_argument(
        "--skip_soil", action="store_true", help="Skip soil-quality extraction."
    )
    p.add_argument(
        "--skip_topo", action="store_true", help="Skip topography extraction."
    )
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    districts = load_districts(args.district_shapefile, target_crs=STATIC_CRS)
    LOG.info("Loaded %d districts (CRS=%s)", len(districts), STATIC_CRS)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    interim_dir = output_dir / "_interim"

    # --- Soil quality ----------------------------------------------------
    soil_path = output_dir / "soil_quality.parquet"
    if args.skip_soil:
        LOG.info("Skipping soil quality (per --skip_soil)")
    elif soil_path.exists() and not args.overwrite:
        LOG.info("Soil quality already exists: %s", soil_path)
    else:
        soil = extract_soil_quality(Path(args.soil_raster), districts, interim_dir)
        soil = fill_missing_with_nearest(soil, districts)
        soil = cast_float32(soil, ["soil_quality_mean", "soil_quality_stdDev"])
        write_parquet(soil, soil_path)
        LOG.info("Wrote %s (%d rows)", soil_path, len(soil))

    # --- Topography ------------------------------------------------------
    topo_path = output_dir / "topography.parquet"
    if args.skip_topo:
        LOG.info("Skipping topography (per --skip_topo)")
    elif topo_path.exists() and not args.overwrite:
        LOG.info("Topography already exists: %s", topo_path)
    else:
        topo = extract_topography(districts, args.gee_scale, args.ee_project)
        topo = fill_missing_with_nearest(topo, districts)
        topo = cast_float32(
            topo,
            ["elevation_mean", "elevation_stdDev", "slope_mean", "slope_stdDev"],
        )
        write_parquet(topo, topo_path)
        LOG.info("Wrote %s (%d rows)", topo_path, len(topo))


if __name__ == "__main__":
    main()
