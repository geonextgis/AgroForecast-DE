#!/usr/bin/env python
"""Step 2 — HOSTRADA hourly NetCDF → district daily climate (Parquet).

Processes HOSTRADA v1.0 hourly NetCDF files (RSDS, WSPD) for all districts.
Files are organised monthly per variable; for each year we iterate the 12
monthly files, resample hour→day, then take the spatial mean per district.

Hour→day aggregation per variable:
    rsds    → sum   (W/m² hourly → Wh/m²/day daily total irradiation)
    sfcWind → mean  (m/s daily mean, FAO-56 ET0 input convention)

HOSTRADA is on an ETRS89-LCC grid (EPSG:3034), distinct from HYRAS's LAEA
grid (EPSG:3035). The two projections share metre-scale X/Y ranges, so a
wrong-CRS mask silently produces NaNs — districts must be reprojected to
EPSG:3034 here before zonal aggregation.

Output: one intermediate Parquet per (variable, year):
    <output_dir>/<var>/<var>_<year>.parquet
columns: ``date, district_id, <feature>``.

Variable mapping: rsds→rad, sfcWind→wspd. ``wspd`` is stored for completeness
but excluded from ``climate_features`` in the model configs.
"""

from __future__ import annotations

import argparse
import sys
from glob import glob
from pathlib import Path

import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    DISTRICT_ID_COL,
    HOSTRADA_CRS,
    cast_float32,
    detect_lat_lon_names,
    get_logger,
    load_districts,
    write_parquet,
    years_range,
    zonal_mean_daily,
)

LOG = get_logger("02_process_hostrada")

VAR_TO_FEATURE = {
    "rsds": "rad",
    "sfcWind": "wspd",
}

# HOSTRADA stores wind speed under WSPD/, not SFCWIND/, so an upper-cased
# variable name does not always match the directory layout.
VAR_TO_DIR = {
    "rsds": "RSDS",
    "sfcWind": "WSPD",
}

# Hour→day aggregation. rsds is averaged so daily values are mean irradiation
# (Wh/m²/day); wind speed stays also as a daily mean.
VAR_TO_HOURLY_AGG = {
    "rsds": "mean",
    "sfcWind": "mean",
}


def find_monthly_files(hostrada_root: str, var: str, year: int) -> list[Path]:
    """Find all monthly NetCDFs for a (variable, year) pair.

    HOSTRADA filenames look like:
        <var>_1hr_HOSTRADA-v1-0_BE_gn_<YYYYMMDD>HH-<YYYYMMDD>HH.nc
    """
    pattern = (
        f"{hostrada_root}/{VAR_TO_DIR[var]}/"
        f"{var}_1hr_HOSTRADA-v1-0_BE_gn_{year}*-{year}*.nc"
    )
    return sorted(Path(p) for p in glob(pattern))


def hourly_to_daily(ds: xr.Dataset, var: str) -> xr.DataArray:
    """Resample hourly data to daily using the per-variable aggregation rule."""
    da = ds[var] if var in ds.data_vars else ds[list(ds.data_vars)[0]]
    agg = VAR_TO_HOURLY_AGG.get(var, "mean")
    resampler = da.resample(time="1D")
    if agg == "sum":
        return resampler.sum(dim="time", skipna=True)
    return resampler.mean(dim="time", skipna=True)


def process_one(
    var: str,
    year: int,
    hostrada_root: str,
    districts,
    output_dir: Path,
    overwrite: bool,
) -> None:
    feature = VAR_TO_FEATURE[var]
    out_path = output_dir / var / f"{var}_{year}.parquet"
    if out_path.exists() and not overwrite:
        LOG.info("Skip %s %d (exists)", var, year)
        return

    files = find_monthly_files(hostrada_root, var, year)
    if not files:
        LOG.warning("No HOSTRADA monthly files for %s %d", var, year)
        return

    LOG.info("Processing %s %d (%d monthly files)", var, year, len(files))

    monthly_frames: list[pd.DataFrame] = []
    for fp in files:
        with xr.open_dataset(fp) as ds:
            daily = hourly_to_daily(ds, var)
            lat, lon = detect_lat_lon_names(ds)
            df = zonal_mean_daily(daily, districts, lat_dim=lat, lon_dim=lon)
        df = df.rename(columns={"value": feature})
        monthly_frames.append(df)

    df_year = (
        pd.concat(monthly_frames, ignore_index=True)
        .sort_values(["date", DISTRICT_ID_COL])
        .reset_index(drop=True)
    )
    df_year = cast_float32(df_year, [feature])

    write_parquet(df_year, out_path)
    LOG.info(
        "Wrote %s (%d rows, %d districts)",
        out_path,
        len(df_year),
        df_year[DISTRICT_ID_COL].nunique(),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--variables",
        nargs="+",
        default=["rsds", "sfcWind"],
        choices=list(VAR_TO_FEATURE),
    )
    p.add_argument(
        "--years",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        required=True,
    )
    p.add_argument(
        "--hostrada_root",
        default="/data01/FDS/muduchuru/Atmos/DWD/HOSTRADA",
    )
    p.add_argument(
        "--district_shapefile",
        default="data/raw/districts/districts.gpkg",
    )
    p.add_argument(
        "--output_dir",
        default="data/interim/climate/hostrada",
    )
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    districts = load_districts(args.district_shapefile, target_crs=HOSTRADA_CRS)
    LOG.info("Loaded %d districts (CRS=%s)", len(districts), HOSTRADA_CRS)

    output_dir = Path(args.output_dir)
    for var in args.variables:
        for year in years_range(args.years[0], args.years[1]):
            process_one(
                var, year, args.hostrada_root, districts, output_dir, args.overwrite
            )


if __name__ == "__main__":
    main()
