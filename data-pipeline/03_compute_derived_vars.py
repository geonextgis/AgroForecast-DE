#!/usr/bin/env python
"""Step 4 — Derived climate variables (VPD, CWB) from district daily outputs.

Reads the per-(variable, year) intermediate Parquets produced by steps 1 & 3
and computes:

    es(T) = 0.6108 * exp(17.27*T / (T + 237.3))                       [kPa]
    es    = (es(tmin) + es(tmax)) / 2                                 [kPa]
    vpd   = (1 - hurs/100) * es                                       [kPa]
    cwb   = prec - et0                                                [mm/day]

VPD follows the FAO-56 convention of averaging saturation vapour pressure at
tmin and tmax (the e_s curve is non-linear, so using tavg underestimates es).

Outputs one Parquet per (derived variable, year):
    <output_dir>/vpd/vpd_<year>.parquet
    <output_dir>/cwb/cwb_<year>.parquet

These formulas are applied identically in hindcast and seasonal-forecast modes
(see step 6 for the GCFS2.2 counterpart).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    DISTRICT_ID_COL,
    cast_float32,
    get_logger,
    write_parquet,
    years_range,
)

LOG = get_logger("04_compute_derived_vars")


def read_var_year(
    interim_dir: Path, source: str, raw_var: str, year: int
) -> pd.DataFrame | None:
    path = interim_dir / source / raw_var / f"{raw_var}_{year}.parquet"
    if not path.exists():
        LOG.warning("Missing %s", path)
        return None
    return pd.read_parquet(path)


def _sat_vp_kpa(t_c: pd.Series) -> pd.Series:
    return 0.6108 * np.exp((17.27 * t_c) / (t_c + 237.3))


def vpd_kpa(
    tmin_c: pd.Series, tmax_c: pd.Series, hurs_pct: pd.Series
) -> pd.Series:
    es = (_sat_vp_kpa(tmin_c) + _sat_vp_kpa(tmax_c)) / 2.0
    return (1.0 - hurs_pct / 100.0) * es


def compute_vpd(
    interim_dir: Path, year: int, output_dir: Path, overwrite: bool
) -> None:
    out_path = output_dir / "vpd" / f"vpd_{year}.parquet"
    if out_path.exists() and not overwrite:
        LOG.info("Skip vpd %d (exists)", year)
        return

    tasmin = read_var_year(interim_dir, "hyras", "tasmin", year)
    tasmax = read_var_year(interim_dir, "hyras", "tasmax", year)
    hurs = read_var_year(interim_dir, "hyras", "hurs", year)
    if tasmin is None or tasmax is None or hurs is None:
        LOG.warning("Cannot compute VPD for %d (missing inputs)", year)
        return

    df = tasmin.merge(tasmax, on=["date", DISTRICT_ID_COL], how="inner")
    df = df.merge(hurs, on=["date", DISTRICT_ID_COL], how="inner")
    df["vpd"] = vpd_kpa(df["tmin"], df["tmax"], df["hurs"])
    df = df[["date", DISTRICT_ID_COL, "vpd"]]
    df = cast_float32(df, ["vpd"])
    df = df.sort_values(["date", DISTRICT_ID_COL]).reset_index(drop=True)

    write_parquet(df, out_path)
    LOG.info("Wrote %s (%d rows)", out_path, len(df))


def compute_cwb(
    interim_dir: Path, year: int, output_dir: Path, overwrite: bool
) -> None:
    out_path = output_dir / "cwb" / f"cwb_{year}.parquet"
    if out_path.exists() and not overwrite:
        LOG.info("Skip cwb %d (exists)", year)
        return

    pr = read_var_year(interim_dir, "hyras", "pr", year)
    evpot = read_var_year(interim_dir, "hyras", "evpot", year)
    if pr is None or evpot is None:
        LOG.warning("Cannot compute CWB for %d (missing inputs)", year)
        return

    df = pr.merge(evpot, on=["date", DISTRICT_ID_COL], how="inner")
    df["cwb"] = df["prec"] - df["et0"]
    df = df[["date", DISTRICT_ID_COL, "cwb"]]
    df = cast_float32(df, ["cwb"])
    df = df.sort_values(["date", DISTRICT_ID_COL]).reset_index(drop=True)

    write_parquet(df, out_path)
    LOG.info("Wrote %s (%d rows)", out_path, len(df))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--years", nargs=2, type=int, metavar=("START", "END"), required=True
    )
    p.add_argument("--interim_dir", default="data/interim/climate")
    p.add_argument("--output_dir", default="data/interim/climate/derived")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    interim_dir = Path(args.interim_dir)
    output_dir = Path(args.output_dir)

    for year in years_range(args.years[0], args.years[1]):
        compute_vpd(interim_dir, year, output_dir, args.overwrite)
        compute_cwb(interim_dir, year, output_dir, args.overwrite)


if __name__ == "__main__":
    main()
