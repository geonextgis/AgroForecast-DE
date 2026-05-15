#!/usr/bin/env python
"""Step 1 — HYRAS NetCDF → district daily climate (Parquet).

Processes HYRAS yearly NetCDF files (TAS, TASMAX, TASMIN, HURS, PR) and the
FAO grass-reference evapotranspiration series (ET0) for all districts. For
each (variable, year), the corresponding NetCDF is opened with xarray,
district polygons are reprojected to the variable's native grid CRS, and a
spatial mean per district per day is computed via ``regionmask``.

The standard HYRAS v6-x variables are on ETRS89-LAEA (EPSG:3035). The ET0
series uses its own projection — DHDN / 3-degree Gauss-Kruger zone 3
(EPSG:31467) — so districts are reprojected separately for that variable.

Output: one intermediate Parquet per (variable, year):
    <output_dir>/<var>/<var>_<year>.parquet
indexed by (date, district_id). These are merged by step 5
(``05_assemble_district_files.py``) into one Parquet per district.

Variables map to model feature names: tas→tavg, tasmax→tmax, tasmin→tmin,
pr→prec, hurs→hurs, evpot→et0.

SOILMOIST and SOILTEMP are only published as monthly ``.tgz`` archives of
daily ASCII grids on the DWD server, not as yearly NetCDF, so they are
processed by step 3 (``03_process_legacy_grids.py``) instead.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    DISTRICT_ID_COL,
    HYRAS_CRS,
    cast_float32,
    detect_lat_lon_names,
    get_logger,
    load_districts,
    write_parquet,
    years_range,
    zonal_mean_daily,
)

LOG = get_logger("01_process_hyras")

VAR_TO_FEATURE = {
    "tas": "tavg",
    "tasmax": "tmax",
    "tasmin": "tmin",
    "hurs": "hurs",
    "pr": "prec",
    "evpot": "et0",
}

# ET0 is published as a separate FAO grass-reference evapotranspiration
# series with its own filename convention, internal variable name, and
# projection (EPSG:31467, DHDN / 3-degree Gauss-Kruger zone 3) — different
# from the standard HYRAS ETRS89-LAEA grid. Districts are reprojected to
# this CRS only for ET0 in main().
ET0_VAR = "evpot"
ET0_DIR = "ET0"
ET0_NC_VAR_CANDIDATES = ("et0", "eta_fao")
ET0_CRS = "EPSG:31467"


def hyras_path(hyras_root: str, var: str, year: int) -> Path | None:
    """Return the highest-version HYRAS file for ``(var, year)``.

    Standard HYRAS variables are named ``<var>_hyras_1_<year>_v*_de.nc``;
    DWD occasionally bumps the version suffix (e.g. ``v6-0`` → ``v6-1`` for
    2026). The FAO ET0 series uses the
    ``grids_germany_daily_evaporation_fao_<year>_v*.nc`` pattern. We match
    any ``v*`` and pick the lexicographically largest, which corresponds to
    the latest version.
    """
    if var == ET0_VAR:
        var_dir = Path(hyras_root) / ET0_DIR
        matches = sorted(
            var_dir.glob(f"grids_germany_daily_evaporation_fao_{year}_v*.nc")
        )
    else:
        var_dir = Path(hyras_root) / var.upper()
        matches = sorted(var_dir.glob(f"{var}_hyras_1_{year}_v*_de.nc"))
    return matches[-1] if matches else None


def process_one(
    var: str,
    year: int,
    hyras_root: str,
    districts,
    output_dir: Path,
    overwrite: bool,
) -> None:
    feature = VAR_TO_FEATURE[var]
    out_path = output_dir / var / f"{var}_{year}.parquet"
    if out_path.exists() and not overwrite:
        LOG.info("Skip %s %d (exists)", var, year)
        return

    nc_path = hyras_path(hyras_root, var, year)
    if nc_path is None:
        LOG.warning(
            "Missing HYRAS file for %s %d under %s/%s/",
            var,
            year,
            hyras_root,
            var.upper(),
        )
        return

    LOG.info("Processing %s %d <- %s", var, year, nc_path)
    # The FAO ET0 series uses two internal variable-name conventions across
    # years (`et0` for 1961-1990 and 2021+, `eta_fao` for 1991-2020), so we
    # try each candidate before falling back to a prefix match.
    if var == ET0_VAR:
        nc_var_candidates = ET0_NC_VAR_CANDIDATES
    else:
        nc_var_candidates = (var,)
    with xr.open_dataset(nc_path) as ds:
        da = None
        for cand in nc_var_candidates:
            if cand in ds.data_vars:
                da = ds[cand]
                break
        if da is None:
            # Fallback: prefix match against the first candidate (handles minor
            # casing/suffix drift in standard HYRAS files).
            primary = nc_var_candidates[0]
            matches = [
                v for v in ds.data_vars if v.lower().startswith(primary.lower())
            ]
            if not matches:
                raise KeyError(
                    f"None of {nc_var_candidates} found in {nc_path} "
                    f"(data_vars={list(ds.data_vars)})"
                )
            da = ds[matches[0]]

        lat, lon = detect_lat_lon_names(ds)
        df = zonal_mean_daily(da, districts, lat_dim=lat, lon_dim=lon)

    df = df.rename(columns={"value": feature})
    df = cast_float32(df, [feature])
    df = df.sort_values(["date", DISTRICT_ID_COL]).reset_index(drop=True)

    write_parquet(df, out_path)
    LOG.info(
        "Wrote %s (%d rows, %d districts)",
        out_path,
        len(df),
        df[DISTRICT_ID_COL].nunique(),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--variables",
        nargs="+",
        default=["tas", "tasmax", "tasmin", "hurs", "pr", "evpot"],
        choices=list(VAR_TO_FEATURE),
        help="HYRAS raw variables to process",
    )
    p.add_argument(
        "--years",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        required=True,
        help="Inclusive year range",
    )
    p.add_argument(
        "--hyras_root",
        default="/data01/FDS/muduchuru/Atmos/DWD/HYRAS",
        help="Root containing TAS/TASMAX/... subdirectories",
    )
    p.add_argument(
        "--district_shapefile",
        default="data/raw/districts/districts.gpkg",
        help="Shapefile with district polygons (must include AGS/NUTS_ID/NUTS3_ID/district_id)",
    )
    p.add_argument(
        "--output_dir",
        default="data/interim/climate/hyras",
        help="Output directory for per-(var,year) intermediate Parquet",
    )
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    districts_by_crs: dict = {}

    def get_districts(crs: str):
        if crs not in districts_by_crs:
            districts_by_crs[crs] = load_districts(
                args.district_shapefile, target_crs=crs
            )
            LOG.info(
                "Loaded %d districts (CRS=%s)", len(districts_by_crs[crs]), crs
            )
        return districts_by_crs[crs]

    output_dir = Path(args.output_dir)
    for var in args.variables:
        crs = ET0_CRS if var == ET0_VAR else HYRAS_CRS
        districts = get_districts(crs)
        for year in years_range(args.years[0], args.years[1]):
            process_one(
                var, year, args.hyras_root, districts, output_dir, args.overwrite
            )


if __name__ == "__main__":
    main()
