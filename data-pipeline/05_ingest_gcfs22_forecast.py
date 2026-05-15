#!/usr/bin/env python
"""Step 6 — DWD GCFS2.2 S2S → district-level seasonal forecast (Parquet).

Processes a single GCFS2.2 initialisation directory ``sfc<YYYYMMDD>/`` for all
ensemble members. For each member we open the per-member daily NetCDFs
(``tas, tasmax, tasmin, hurs, pr, rsds, sfcWind``), compute district zonal
means, and derive ``vpd``, ``et0``, and ``cwb`` consistent with the observed
side. The native temperature unit in GCFS2.2 is **Kelvin**, so ``tas``,
``tasmax`` and ``tasmin`` are converted to °C before any downstream
calculation. Precipitation is published as a mass flux in **kg m-2 s-1**
and is converted to mm/day (× 86400) so it matches the HYRAS PR scale and
keeps ``cwb = prec − et0`` dimensionally consistent. ET0 is computed via
FAO-56 Penman-Monteith using actual wind
speed and **standard atmospheric pressure (101.3 kPa)** as a constant — the
per-member ``sfc<YYYYMMDD>/`` folders do not redistribute ``ps``. VPD uses
the FAO-56 convention ``es = (es(tmin) + es(tmax))/2`` so the formula
matches Step 4 (``04_compute_derived_vars.py``).

Output layout (one Parquet per district per member per init date):
    <output_dir>/<sfc_date>/<member>/<district_id>.parquet

Schema (column order matches data/interim/climate/districts/<id>.parquet):
    ``date, district_id, tavg, tmax, tmin, prec, rad, hurs, et0, wspd, vpd, cwb``.
"""

from __future__ import annotations

import argparse
import re
import sys
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    DISTRICT_ID_COL,
    cast_float32,
    detect_lat_lon_names,
    get_logger,
    load_districts,
    write_parquet,
    zonal_mean_daily,
)

LOG = get_logger("06_ingest_gcfs22_forecast")

GCFS22_CRS = "EPSG:4326"

VAR_TO_FEATURE = {
    "tas": "tavg",
    "tasmax": "tmax",
    "tasmin": "tmin",
    "pr": "prec",
    "rsds": "rad",
    "hurs": "hurs",
    "sfcWind": "wspd",
}

GCFS22_VARS = [
    "tas",
    "tasmax",
    "tasmin",
    "pr",
    "rsds",
    "hurs",
    "sfcWind",
]

# Order matches data/interim/climate/districts/<id>.parquet exactly.
OUTPUT_COLUMNS = [
    "date",
    DISTRICT_ID_COL,
    "tavg",
    "tmax",
    "tmin",
    "prec",
    "rad",
    "hurs",
    "et0",
    "wspd",
    "vpd",
    "cwb",
]

MEMBER_RE = re.compile(r"^r(\d+)i\d+p\d+$")

def list_members(init_dir: Path) -> list[Path]:
    return sorted(
        p for p in init_dir.iterdir() if p.is_dir() and MEMBER_RE.match(p.name)
    )


def member_to_number(member_name: str) -> int:
    """``r{N}i1p1`` → ``number = N - 1`` (raw NetCDFs index members from 0)."""
    m = MEMBER_RE.match(member_name)
    if not m:
        raise ValueError(f"Unrecognized member name: {member_name}")
    return int(m.group(1)) - 1


def member_var_file(member_dir: Path, var: str) -> Path | None:
    pattern = str(member_dir / f"{var}_day_*.nc")
    matches = sorted(glob(pattern))
    return Path(matches[0]) if matches else None


def load_var(member_dir: Path, var: str, districts) -> pd.DataFrame | None:
    fp = member_var_file(member_dir, var)
    if fp is None:
        LOG.warning("Missing GCFS2.2 file: %s/%s_day_*.nc", member_dir, var)
        return None

    with xr.open_dataset(fp) as ds:
        nc_var = var if var in ds.data_vars else list(ds.data_vars)[0]
        da = ds[nc_var]
        lat, lon = detect_lat_lon_names(ds)
        df = zonal_mean_daily(da, districts, lat_dim=lat, lon_dim=lon)

    feature = VAR_TO_FEATURE[var]
    df = df.rename(columns={"value": feature})
    return df


def _sat_vp_kpa(t_c: pd.Series) -> pd.Series:
    return 0.6108 * np.exp((17.27 * t_c) / (t_c + 237.3))


def vpd_kpa(tmin_c: pd.Series, tmax_c: pd.Series, hurs_pct: pd.Series) -> pd.Series:
    es = (_sat_vp_kpa(tmin_c) + _sat_vp_kpa(tmax_c)) / 2.0
    return (1.0 - hurs_pct / 100.0) * es


def et0_penman_monteith(
    tas_c: pd.Series,
    hurs_pct: pd.Series,
    rsds_wm2: pd.Series,
    wspd_ms: pd.Series,
) -> pd.Series:
    """FAO-56 Penman-Monteith reference ET0 [mm/day].

    Surface pressure is fixed at the FAO-56 standard atmosphere value
    (101.3 kPa) because the per-member GCFS2.2 ``sfc<YYYYMMDD>/`` folders
    do not redistribute ``ps``. The ``γ`` term is correspondingly insensitive
    to elevation, but for the German lowlands (≤ ~1000 m) the resulting bias
    in ET0 is well below the spread between ensemble members.

    Inputs (after K → °C conversion of GCFS2.2 temperatures):
        tas_c    [°C]
        hurs_pct [%]
        rsds_wm2 [W/m²]   daily mean shortwave radiation
        wspd_ms  [m/s]    10-m wind speed (converted to 2-m via FAO log profile)
    """
    # 10-m → 2-m wind conversion (FAO-56 eq. 47, z = 10).
    u2 = wspd_ms * (4.87 / np.log(67.8 * 10.0 - 5.42))
    pressure_kpa = 101.3  # FAO-56 standard atmosphere (sea level)

    rs_mj = rsds_wm2 * 0.0864  # W/m² → MJ/m²/day
    rn = 0.77 * rs_mj  # net radiation, grass-reference albedo ≈ 0.23

    es = 0.6108 * np.exp((17.27 * tas_c) / (tas_c + 237.3))
    ea = (hurs_pct / 100.0) * es
    delta = (4098.0 * es) / np.power(tas_c + 237.3, 2)
    gamma = 0.000665 * pressure_kpa

    numerator = 0.408 * delta * rn + gamma * (900.0 / (tas_c + 273.0)) * u2 * (es - ea)
    denominator = delta + gamma * (1.0 + 0.34 * u2)
    return numerator / denominator


def assemble_member(
    member_dir: Path,
    districts,
) -> pd.DataFrame | None:
    LOG.info("Member %s", member_dir.name)
    parts: dict[str, pd.DataFrame] = {}
    for var in GCFS22_VARS:
        df = load_var(member_dir, var, districts)
        if df is None:
            return None
        parts[VAR_TO_FEATURE[var]] = df

    merged: pd.DataFrame | None = None
    for df in parts.values():
        merged = (
            df
            if merged is None
            else merged.merge(df, on=["date", DISTRICT_ID_COL], how="inner")
        )
    assert merged is not None

    # GCFS2.2 air temperatures are stored in Kelvin; convert to °C so the
    # output matches the observed climate Parquet and so vpd_kpa /
    # et0_penman_monteith get inputs in their documented °C units.
    for col in ("tavg", "tmax", "tmin"):
        merged[col] = merged[col] - 273.15

    # GCFS2.2 precipitation is a mass flux in kg m-2 s-1. One kg m-2 of water
    # equals 1 mm of water-equivalent depth, so the daily total in mm is the
    # flux × 86400 s/day. After this conversion `prec` is on the same mm/day
    # scale as HYRAS PR and cwb = prec − et0 is dimensionally consistent.
    merged["prec"] = merged["prec"] * 86400.0

    merged["vpd"] = vpd_kpa(merged["tmin"], merged["tmax"], merged["hurs"])
    merged["et0"] = et0_penman_monteith(
        merged["tavg"],
        merged["hurs"],
        merged["rad"],
        merged["wspd"],
    )
    merged["cwb"] = merged["prec"] - merged["et0"]

    merged = cast_float32(
        merged,
        [
            "tavg",
            "tmax",
            "tmin",
            "prec",
            "rad",
            "hurs",
            "et0",
            "wspd",
            "vpd",
            "cwb",
        ],
    )
    return (
        merged[OUTPUT_COLUMNS]
        .sort_values(["date", DISTRICT_ID_COL])
        .reset_index(drop=True)
    )


def write_per_district(df: pd.DataFrame, member_out_dir: Path, overwrite: bool) -> None:
    member_out_dir.mkdir(parents=True, exist_ok=True)
    for d_id, group in df.groupby(DISTRICT_ID_COL, sort=False):
        out_path = member_out_dir / f"{d_id}.parquet"
        if out_path.exists() and not overwrite:
            continue
        write_parquet(group.reset_index(drop=True), out_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--sfc_date",
        required=True,
        help="Initialisation date YYYYMMDD (matches sfc<YYYYMMDD> directory)",
    )
    p.add_argument(
        "--gcfs22_root",
        default="/data01/FDS/muduchuru/Atmos/DWD/S2S/GCFS22_svh20230401",
        help="Root directory containing sfc<YYYYMMDD>/ subfolders",
    )
    p.add_argument(
        "--district_shapefile",
        default="data/raw/districts/districts.gpkg",
    )
    p.add_argument(
        "--output_dir",
        default="data/interim/gcfs22",
    )
    p.add_argument(
        "--members",
        nargs="*",
        default=None,
        help="Subset of ensemble members (default: all r*i*p* directories)",
    )
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    init_dir = Path(args.gcfs22_root) / f"sfc{args.sfc_date}"
    if not init_dir.is_dir():
        raise FileNotFoundError(init_dir)

    districts = load_districts(args.district_shapefile, target_crs=GCFS22_CRS)
    LOG.info("Loaded %d districts (CRS=%s)", len(districts), GCFS22_CRS)

    members = list_members(init_dir)
    if args.members:
        wanted = set(args.members)
        members = [m for m in members if m.name in wanted]
    LOG.info("Processing %d members for sfc%s", len(members), args.sfc_date)

    out_root = Path(args.output_dir) / f"sfc{args.sfc_date}"
    for member_dir in members:
        df = assemble_member(member_dir, districts)
        if df is None:
            LOG.warning("Skipping %s (incomplete data)", member_dir.name)
            continue
        write_per_district(df, out_root / member_dir.name, args.overwrite)
        LOG.info(
            "Member %s: %d districts × %d days",
            member_dir.name,
            df[DISTRICT_ID_COL].nunique(),
            df["date"].nunique(),
        )


if __name__ == "__main__":
    main()
