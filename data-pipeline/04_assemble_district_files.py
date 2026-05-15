#!/usr/bin/env python
"""Step 5 — Assemble per-district Parquet files.

Merges the per-(variable, year) intermediate Parquets produced by steps 1–4
into the project's primary data structure: **one Parquet file per district**,
with one row per day and one column per variable, covering the full year range.

Output: ``<output_dir>/<district_id>.parquet``

Schema (matches CLAUDE.md):
    date         datetime64[ns]
    district_id  str
    tavg, tmax, tmin, prec, rad, hurs    float32   (used in model)
    et0,                                 float32   (used in model)
    wspd                                 float32   (stored, not used in model)
    vpd, cwb                             float32   (derived, used in model)

If a district file already exists, this step **updates it in place**: rows for
the requested year range are recomputed and replaced, other rows are kept. Pass
``--overwrite`` to rewrite the whole file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    DISTRICT_ID_COL,
    cast_float32,
    get_logger,
    write_parquet,
    years_range,
)

LOG = get_logger("05_assemble_district_files")

# (intermediate_dir, raw_var, output_column)
SOURCES = [
    ("hyras", "tas", "tavg"),
    ("hyras", "tasmax", "tmax"),
    ("hyras", "tasmin", "tmin"),
    ("hyras", "pr", "prec"),
    ("hyras", "hurs", "hurs"),
    ("hyras", "evpot", "et0"),
    ("hostrada", "rsds", "rad"),
    ("hostrada", "sfcWind", "wspd"),
    ("derived", "vpd", "vpd"),
    ("derived", "cwb", "cwb"),
]

FLOAT_COLS = [
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
SCHEMA_ORDER = ["date", DISTRICT_ID_COL] + FLOAT_COLS


def load_long(interim_dir: Path, years: list[int]) -> pd.DataFrame:
    """Load all (variable, year) intermediates and return one tidy DataFrame.

    Each variable contributes its own column; missing variables are filled with NaN
    so the final per-district schema is always the full set of columns.
    """
    frames: list[pd.DataFrame] = []
    for source_dir, raw_var, out_col in SOURCES:
        for year in years:
            path = interim_dir / source_dir / raw_var / f"{raw_var}_{year}.parquet"
            if not path.exists():
                LOG.warning("Missing %s — column %s will be NaN for %d", path, out_col, year)
                continue
            df = pd.read_parquet(path)
            value_col = next(c for c in df.columns if c not in {"date", DISTRICT_ID_COL})
            if value_col != out_col:
                df = df.rename(columns={value_col: out_col})
            frames.append(df[["date", DISTRICT_ID_COL, out_col]])

    if not frames:
        raise RuntimeError("No intermediate Parquets found — run steps 1–4 first.")

    # Outer-merge in chunks per variable on (date, district_id)
    by_var: dict[str, pd.DataFrame] = {}
    for df in frames:
        col = [c for c in df.columns if c not in {"date", DISTRICT_ID_COL}][0]
        by_var.setdefault(col, []).append(df)

    merged: pd.DataFrame | None = None
    for col, parts in by_var.items():
        df_col = pd.concat(parts, ignore_index=True).drop_duplicates(
            ["date", DISTRICT_ID_COL]
        )
        if merged is None:
            merged = df_col
        else:
            merged = merged.merge(df_col, on=["date", DISTRICT_ID_COL], how="outer")

    assert merged is not None  # guarded by the empty-frames check above

    for col in FLOAT_COLS:
        if col not in merged.columns:
            merged[col] = pd.NA

    merged = cast_float32(merged, FLOAT_COLS)
    merged = merged[SCHEMA_ORDER]
    return merged


def write_per_district(
    df: pd.DataFrame, output_dir: Path, years: list[int], overwrite: bool
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    year_set = set(years)

    for d_id, group in df.groupby(DISTRICT_ID_COL, sort=False):
        out_path = output_dir / f"{d_id}.parquet"
        new = group.sort_values("date").reset_index(drop=True)

        if out_path.exists() and not overwrite:
            existing = pd.read_parquet(out_path)
            existing = existing[~existing["date"].dt.year.isin(year_set)]
            combined = (
                pd.concat([existing, new], ignore_index=True)
                .drop_duplicates(["date", DISTRICT_ID_COL], keep="last")
                .sort_values("date")
                .reset_index(drop=True)
            )
            combined = cast_float32(combined, FLOAT_COLS)
            write_parquet(combined[SCHEMA_ORDER], out_path)
        else:
            write_parquet(new[SCHEMA_ORDER], out_path)

        LOG.info("Wrote %s (%d rows)", out_path, len(new))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--years", nargs=2, type=int, metavar=("START", "END"), required=True)
    p.add_argument("--interim_dir", default="data/interim/climate")
    p.add_argument("--output_dir", default="data/interim/climate/districts")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    years = list(years_range(args.years[0], args.years[1]))

    LOG.info("Loading intermediates for years %d–%d", years[0], years[-1])
    long_df = load_long(Path(args.interim_dir), years)
    LOG.info(
        "Merged frame: %d rows × %d cols, %d districts",
        len(long_df),
        long_df.shape[1],
        long_df[DISTRICT_ID_COL].nunique(),
    )
    write_per_district(long_df, Path(args.output_dir), years, args.overwrite)


if __name__ == "__main__":
    main()
