import json
import os
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Static / phenology / yield loaders
# ---------------------------------------------------------------------------


class StaticFeatureLoader:
    """Loads soil-quality + topography static features per district."""

    def __init__(self, soil_path: str, topography_path: str):
        soil = pd.read_parquet(soil_path).set_index("district_id")
        topo = pd.read_parquet(topography_path).set_index("district_id")
        self.df = soil.join(topo, how="outer")

    def get(self, nuts_id: str, features: List[str]) -> np.ndarray:
        if nuts_id not in self.df.index:
            raise KeyError(f"Static features missing for {nuts_id}")
        return self.df.loc[nuts_id, features].to_numpy(dtype=np.float32)

    @property
    def index(self) -> pd.Index:
        return self.df.index


class PhenologyLoader:
    """Per-crop phenology with median-DOY imputation for missing years."""

    def __init__(self, phenology_path: str, harvest_next_year: bool = True):
        df = pd.read_csv(
            phenology_path,
            usecols=[
                "NUTS_ID",
                "harvest_year",
                "sowing_date",
                "flowering_date",
                "maturity_date",
            ],
            parse_dates=["sowing_date", "flowering_date", "maturity_date"],
        )
        self.df = df.set_index(["NUTS_ID", "harvest_year"]).sort_index()
        self.harvest_next_year = harvest_next_year

    def get_window(
        self, nuts_id: str, year: int
    ) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
        idx = (nuts_id, year)
        if idx in self.df.index:
            row = self.df.loc[idx]
            return (
                pd.to_datetime(row["sowing_date"]),
                pd.to_datetime(row["maturity_date"]),
            )

        if nuts_id not in self.df.index.get_level_values(0):
            return None
        district_rows = self.df.loc[nuts_id]
        median_sow_doy = district_rows["sowing_date"].dt.dayofyear.median()
        median_harv_doy = district_rows["maturity_date"].dt.dayofyear.median()
        if pd.isna(median_sow_doy) or pd.isna(median_harv_doy):
            return None

        sowing_year = year - 1 if self.harvest_next_year else year
        try:
            sowing_date = pd.Timestamp(
                datetime(sowing_year, 1, 1)
                + timedelta(days=int(median_sow_doy) - 1)
            )
            harvest_date = pd.Timestamp(
                datetime(year, 1, 1) + timedelta(days=int(median_harv_doy) - 1)
            )
        except ValueError:
            return None
        return sowing_date, harvest_date


# Maps internal crop names → the `var` code used in the yield CSV.
DEFAULT_CROP_CODE: Dict[str, str] = {
    "winter_wheat": "ww",
    "winter_barley": "wb",
    "winter_rye": "rye",
    "winter_rapeseed": "wrape",
    "silage_maize": "silage_maize",
    "grain_maize": "grain_maize",
    "potato": "potat_tot",
    "sugarbeet": "sugarbeet",
    "oats": "oats",
    "triticale": "triticale",
    "spring_barley": "sb",
}


class YieldLoader:
    """Reads district-level observed yields, filtered by crop and outlier flag."""

    def __init__(self, yield_path: str, crop_code: str, measure: str = "yield"):
        df = pd.read_csv(yield_path)
        df = df[(df["var"] == crop_code) & (df["measure"] == measure)]
        df = df[df["outlier"].fillna(0) == 0]
        df = df.dropna(subset=["value"])
        df["value"] = df["value"].astype(np.float32)
        self.series = (
            df.groupby(["nuts_id", "year"])["value"].mean().sort_index()
        )

    def get(self, nuts_id: str, year: int) -> float:
        return float(self.series.loc[(nuts_id, year)])


# ---------------------------------------------------------------------------
# Year split
# ---------------------------------------------------------------------------


def make_year_split(
    start_year: int = 2001,
    end_year: int = 2024,
    n_val_years: int = 3,
    n_test_years: int = 2,
) -> Dict[str, List[int]]:
    """Chronologically split ``[start_year, end_year]`` into train/val/test years.

    Reserves the last ``n_test_years`` for test, the ``n_val_years``
    immediately before that for validation, and the remainder for
    training. With defaults on 2001–2024 this gives a 19 / 3 / 2 split:
        train: 2001-2019, val: 2020-2022, test: 2023-2024.
    """
    years = list(range(start_year, end_year + 1))
    if len(years) < n_val_years + n_test_years + 1:
        raise ValueError(
            f"Not enough years ({len(years)}) to reserve "
            f"{n_val_years} val + {n_test_years} test years and keep training data."
        )
    test_years = years[-n_test_years:]
    val_years = years[-(n_val_years + n_test_years) : -n_test_years]
    train_years = years[: -(n_val_years + n_test_years)]
    return {"train": train_years, "val": val_years, "test": test_years}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class CropFusionNetDataset(Dataset):
    """PyTorch Dataset for CropFusionNet on the district-Parquet layout.

    Climate variables are pulled from
    ``data/interim/climate/districts/<NUTS>.parquet`` for an extended
    phenological window — by default ``pre_planting_months=3`` months
    of pre-planting context (anchored at the first day of that calendar
    month) up to the last day of the harvest month — then padded /
    truncated to the sequence length expected by the model.

    Forecast mode
    -------------
    When ``add_forecast=True``, days on or after ``sfc_date`` are
    replaced with the matching member from
    ``data/interim/gcfs22/<sfc_date>/<member>/<NUTS>.parquet``. Days
    before ``sfc_date`` still come from the observed parquet, so a
    sample spanning the initialisation date is automatically blended.

    Variable selection
    ------------------
    Feature names are taken from the crop config:
    ``config.climate_features`` and ``config.static_features`` (or
    ``config.soil_features + config.topo_features`` as a fallback).
    Both observed and forecast parquets share the same column schema
    (see CLAUDE.md Step 6), so the same feature list applies to either
    mode.
    """

    DEFAULT_CLIMATE_DIR = "data/interim/climate/districts"
    DEFAULT_GCFS22_ROOT = "data/interim/gcfs22"
    DEFAULT_SOIL_PATH = "data/interim/static/soil_quality.parquet"
    DEFAULT_TOPO_PATH = "data/interim/static/topography.parquet"
    DEFAULT_YIELD_PATH = "data/interim/yield/Final_data_2024.csv"
    DEFAULT_PHENOLOGY_DIR = "data/interim/phenology"

    def __init__(
        self,
        config: Any,
        mode: str = "train",
        scale: bool = False,
        years: Optional[Iterable[int]] = None,
        scalers: Optional[Dict[str, Any]] = None,
        add_forecast: bool = False,
        sfc_date: Optional[str] = None,
        member: str = "r1i1p1",
        gcfs22_root: Optional[str] = None,
        pre_planting_months: Optional[int] = None,
    ):
        self.config = config
        self.mode = mode
        self.scale = scale
        self.pre_planting_months = (
            pre_planting_months
            if pre_planting_months is not None
            else getattr(config, "PRE_PLANTING_MONTHS", 3)
        )

        # ---- Resolve paths ----
        self.climate_dir = getattr(
            config, "CLIMATE_DISTRICTS_DIR", self.DEFAULT_CLIMATE_DIR
        )
        self.soil_path = getattr(config, "STATIC_SOIL_PATH", self.DEFAULT_SOIL_PATH)
        self.topo_path = getattr(config, "STATIC_TOPO_PATH", self.DEFAULT_TOPO_PATH)
        self.yield_path = getattr(config, "YIELD_FILE_PATH", self.DEFAULT_YIELD_PATH)
        self.phenology_path = getattr(config, "PHENOLOGY_FILE_PATH", None) or os.path.join(
            self.DEFAULT_PHENOLOGY_DIR, f"{config.CROP}_phenology.csv"
        )
        self.harvest_next_year = getattr(config, "HARVEST_NEXT_YEAR", True)
        self.crop_code = getattr(
            config, "CROP_CODE", DEFAULT_CROP_CODE.get(config.CROP)
        )
        if self.crop_code is None:
            raise ValueError(
                f"No yield-CSV crop code known for '{config.CROP}'. "
                "Set `CROP_CODE` on the config or extend DEFAULT_CROP_CODE."
            )

        # ---- Features (read from config) ----
        self.climate_features = list(getattr(config, "climate_features", []))
        if not self.climate_features:
            raise ValueError(
                f"Config for '{config.CROP}' has no `climate_features`."
            )

        self.static_features = list(
            getattr(
                config,
                "static_features",
                list(getattr(config, "soil_features", []))
                + list(getattr(config, "topo_features", [])),
            )
        )
        if not self.static_features:
            raise ValueError(
                f"Config for '{config.CROP}' has no `static_features` "
                "(and no `soil_features` + `topo_features` fallback)."
            )

        self.target = config.target
        self.seq_len = config.model_config.get("seq_length")

        # ---- Forecast mode ----
        self.add_forecast = add_forecast
        self.sfc_date = sfc_date
        self.member = member
        self.gcfs22_root = gcfs22_root or getattr(
            config, "GCFS22_ROOT", self.DEFAULT_GCFS22_ROOT
        )
        if self.add_forecast:
            if not self.sfc_date:
                raise ValueError("`sfc_date` (YYYYMMDD) is required when add_forecast=True.")
            self.sfc_timestamp = pd.to_datetime(self.sfc_date, format="%Y%m%d")
        else:
            self.sfc_timestamp = None

        self._observed_cache: Dict[str, pd.DataFrame] = {}
        self._forecast_cache: Dict[str, pd.DataFrame] = {}

        # ---- Year-based split (chronological; last N_TEST_YEARS are test) ----
        split = make_year_split(
            start_year=getattr(config, "START_YEAR", 2001),
            end_year=getattr(config, "END_YEAR", 2024),
            n_val_years=getattr(config, "N_VAL_YEARS", 3),
            n_test_years=getattr(config, "N_TEST_YEARS", 2),
        )
        if years is not None:
            self.years = list(years)
        elif mode in split:
            self.years = split[mode]
        else:
            raise ValueError(f"Unknown mode '{mode}' (expected train/val/test).")

        # ---- Loaders ----
        self.static_loader = StaticFeatureLoader(self.soil_path, self.topo_path)
        self.phenology_loader = PhenologyLoader(
            self.phenology_path, self.harvest_next_year
        )
        self.yield_loader = YieldLoader(self.yield_path, self.crop_code)

        # ---- Build sample index ----
        self.samples: List[Tuple[str, int]] = self._build_sample_index()

        # ---- Scalers (passed in, or pulled from config, or none) ----
        self._set_scalers(scalers if scalers is not None else getattr(config, "scalers", None))

    # ------------------------------------------------------------------ helpers

    def _read_observed(self, nuts_id: str) -> pd.DataFrame:
        if nuts_id not in self._observed_cache:
            path = os.path.join(self.climate_dir, f"{nuts_id}.parquet")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Climate parquet not found: {path}")
            df = pd.read_parquet(path)
            df["date"] = pd.to_datetime(df["date"])
            self._observed_cache[nuts_id] = df.set_index("date").sort_index()
        return self._observed_cache[nuts_id]

    def _read_forecast(self, nuts_id: str) -> pd.DataFrame:
        if nuts_id not in self._forecast_cache:
            path = os.path.join(
                self.gcfs22_root, self.sfc_date, self.member, f"{nuts_id}.parquet"
            )
            if not os.path.exists(path):
                raise FileNotFoundError(f"GCFS2.2 parquet not found: {path}")
            df = pd.read_parquet(path)
            df["date"] = pd.to_datetime(df["date"])
            self._forecast_cache[nuts_id] = df.set_index("date").sort_index()
        return self._forecast_cache[nuts_id]

    def _extended_window(
        self,
        sowing_date: pd.Timestamp,
        harvest_date: pd.Timestamp,
    ) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """Extend (sowing, harvest) to include pre-planting context.

        Start: first day of the month that lies ``pre_planting_months``
        before the sowing month. End: last day of the harvest month.
        """
        start = (
            sowing_date - pd.DateOffset(months=self.pre_planting_months)
        ).normalize().replace(day=1)
        end = (harvest_date + pd.offsets.MonthEnd(0)).normalize()
        return start, end

    def _load_climate(
        self,
        nuts_id: str,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
    ) -> pd.DataFrame:
        observed = self._read_observed(nuts_id)
        if not self.add_forecast or end_date < self.sfc_timestamp:
            return observed.loc[start_date:end_date]
        if start_date >= self.sfc_timestamp:
            return self._read_forecast(nuts_id).loc[start_date:end_date]
        past = observed.loc[start_date : self.sfc_timestamp - pd.Timedelta(days=1)]
        future = self._read_forecast(nuts_id).loc[self.sfc_timestamp : end_date]
        return pd.concat([past, future]).sort_index()

    def _build_sample_index(self) -> List[Tuple[str, int]]:
        valid_districts = set(self.static_loader.index)
        climate_districts = {
            os.path.splitext(f)[0]
            for f in os.listdir(self.climate_dir)
            if f.endswith(".parquet")
        }
        usable = valid_districts & climate_districts

        samples: List[Tuple[str, int]] = []
        dropped_district = 0
        dropped_phenology = 0
        for nuts_id, year in self.yield_loader.series.index:
            year = int(year)
            if year not in self.years:
                continue
            if nuts_id not in usable:
                dropped_district += 1
                continue
            if self.phenology_loader.get_window(nuts_id, year) is None:
                dropped_phenology += 1
                continue
            samples.append((nuts_id, year))

        if dropped_district or dropped_phenology:
            print(
                f"⚠️  [{self.mode}] dropped {dropped_district} samples (no climate/static), "
                f"{dropped_phenology} (no phenology)."
            )
        return samples

    def _set_scalers(self, scalers: Optional[Dict[str, Any]]) -> None:
        if scalers is None:
            self.real_mean = self.real_std = None
            self.static_mean = self.static_std = None
            self.target_mean = self.target_std = None
            return
        self.real_mean = np.asarray(scalers["time_varying_mean"], dtype=np.float32)
        self.real_std = np.asarray(scalers["time_varying_std"], dtype=np.float32)
        self.static_mean = np.asarray(scalers["static_mean"], dtype=np.float32)
        self.static_std = np.asarray(scalers["static_std"], dtype=np.float32)
        self.target_mean = scalers.get(f"{self.target}_mean")
        self.target_std = scalers.get(f"{self.target}_std")

    def set_scalers(self, scalers: Dict[str, Any]) -> None:
        """Inject scalers fitted elsewhere (e.g. on the training split)."""
        self._set_scalers(scalers)

    def _enforce_sequence_length(
        self,
        inputs: np.ndarray,
        valid_mask: np.ndarray,
        variable_mask: np.ndarray,
    ):
        if self.seq_len is None:
            return inputs, valid_mask, variable_mask

        time_steps = inputs.shape[0]
        if time_steps == self.seq_len:
            return inputs, valid_mask, variable_mask
        if time_steps > self.seq_len:
            return (
                inputs[: self.seq_len],
                valid_mask[: self.seq_len],
                variable_mask[: self.seq_len],
            )
        # Pad at the tail so day 0 still aligns with sowing_date.
        pad = self.seq_len - time_steps
        inputs = np.concatenate(
            [inputs, np.zeros((pad, inputs.shape[1]), dtype=inputs.dtype)],
            axis=0,
        )
        valid_mask = np.concatenate(
            [valid_mask, np.zeros(pad, dtype=valid_mask.dtype)]
        )
        variable_mask = np.concatenate(
            [
                variable_mask,
                np.zeros((pad, variable_mask.shape[1]), dtype=variable_mask.dtype),
            ],
            axis=0,
        )
        return inputs, valid_mask, variable_mask

    # ----------------------------------------------------------------- pytorch

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str, int]]:
        nuts_id, year = self.samples[idx]

        # 1) Target -----------------------------------------------------------
        try:
            y = self.yield_loader.get(nuts_id, year)
        except KeyError as exc:
            raise ValueError(f"No yield data found for {nuts_id} in {year}") from exc
        if self.target_mean is not None and self.target_std is not None:
            y = (y - self.target_mean) / self.target_std

        # 2) Phenological window ---------------------------------------------
        window = self.phenology_loader.get_window(nuts_id, year)
        if window is None:
            raise ValueError(
                f"Phenology data missing and cannot be imputed for {nuts_id} in {year}"
            )
        sowing_date, harvest_date = window
        start_date, end_date = self._extended_window(sowing_date, harvest_date)

        # 3) Climate (pre-planting → end of harvest month) -------------------
        ts = self._load_climate(nuts_id, start_date, end_date)
        ts = ts.reindex(pd.date_range(start_date, end_date, freq="D"))
        missing = [c for c in self.climate_features if c not in ts.columns]
        if missing:
            raise KeyError(
                f"Climate features missing from parquet for {nuts_id}: {missing}"
            )
        time_varying_real = ts[self.climate_features].to_numpy(dtype=np.float32)

        valid_mask = (~np.isnan(time_varying_real)).any(axis=1)
        variable_mask = ~np.isnan(time_varying_real)

        if self.scale and self.real_mean is not None:
            time_varying_real = (time_varying_real - self.real_mean) / self.real_std
            time_varying_real = np.nan_to_num(time_varying_real)

        x_inputs, valid_mask, variable_mask = self._enforce_sequence_length(
            time_varying_real, valid_mask, variable_mask
        )

        # 4) Static features --------------------------------------------------
        static_data = self.static_loader.get(nuts_id, self.static_features)
        if self.scale and self.static_mean is not None:
            static_data = (static_data - self.static_mean) / self.static_std
        x_identifier = np.expand_dims(static_data, axis=0)

        return {
            "NUTS_ID": nuts_id,
            "year": year,
            "inputs": torch.tensor(x_inputs, dtype=torch.float32),
            "identifier": torch.tensor(x_identifier, dtype=torch.float32),
            "mask": torch.tensor(valid_mask, dtype=torch.float32),
            "variable_mask": torch.tensor(variable_mask, dtype=torch.float32),
            "target": torch.tensor(y, dtype=torch.float32),
        }

# ---------------------------------------------------------------------------
# Scaler fitting (training split only, observed climate only)
# ---------------------------------------------------------------------------


def fit_scalers(
    config: Any,
    save_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Fit normalization stats on the training split and (optionally) save to JSON.

    Variable names are taken from the crop config (``climate_features``,
    ``static_features``). Always uses observed climate
    (``add_forecast=False``) — the same scalers are reused at inference
    time when GCFS2.2 forecast data is spliced in, since observed and
    forecast parquets share the same variable scales.
    """
    train_ds = CropFusionNetDataset(config, mode="train", scale=False)

    real_chunks: List[np.ndarray] = []
    static_rows: List[np.ndarray] = []
    targets: List[float] = []

    for nuts_id, year in train_ds.samples:
        try:
            window = train_ds.phenology_loader.get_window(nuts_id, year)
            if window is None:
                continue
            sowing_date, harvest_date = window
            start_date, end_date = train_ds._extended_window(sowing_date, harvest_date)
            ts = train_ds._load_climate(nuts_id, start_date, end_date)
            ts = ts.reindex(pd.date_range(start_date, end_date, freq="D"))
            real_chunks.append(
                ts[train_ds.climate_features].to_numpy(dtype=np.float32)
            )
            static_rows.append(
                train_ds.static_loader.get(nuts_id, train_ds.static_features)
            )
            targets.append(train_ds.yield_loader.get(nuts_id, year))
        except (KeyError, FileNotFoundError) as exc:
            print(f"   skipping {nuts_id}/{year}: {exc}")
            continue

    if not real_chunks:
        raise RuntimeError("No training samples available — cannot fit scalers.")

    real = np.concatenate(real_chunks, axis=0)
    static = np.stack(static_rows, axis=0)
    targets_arr = np.asarray(targets, dtype=np.float32)

    scalers = {
        "time_varying_mean": np.round(np.nanmean(real, axis=0), 6).tolist(),
        "time_varying_std": np.round(np.nanstd(real, axis=0), 6).tolist(),
        "static_mean": np.round(np.nanmean(static, axis=0), 6).tolist(),
        "static_std": np.round(np.nanstd(static, axis=0), 6).tolist(),
        f"{config.target}_mean": float(np.round(np.nanmean(targets_arr), 6)),
        f"{config.target}_std": float(np.round(np.nanstd(targets_arr), 6)),
    }

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(scalers, f, indent=2)
        print(f"💾 Saved scalers → {save_path}")

    return scalers
