"""Export per-year observed-vs-predicted distributions and performance metrics.

Reads the train / validation / test pickles produced by
``src/train/train_CropFusionNet.py`` for each crop, denormalizes
predictions and targets with the matching scaler, computes per-year
R^2, RMSE and MAPE, and writes ``docs/data/performance_{crop}.json``
for consumption by the dashboard box-plot card.

Each output file has the shape::

    {
      "crop": "silage_maize",
      "harvest_month": "Sep",
      "units": "t/ha",
      "splits": {
        "train":      {"years": [...], "metrics": {...}, "n": int},
        "validation": {...},
        "test":       {...}
      },
      "years": [
        {
          "year": 2001,
          "split": "train",
          "n": 337,
          "observed":  [...],   # rounded floats, t/ha
          "predicted": [...],
          "metrics":   {"r2": ..., "rmse": ..., "mape": ...}
        },
        ...
      ]
    }
"""

import glob
import json
import os
import pickle

import numpy as np


RESULTS_DIR = "src/train/results"
SCALER_DIR = "src/scaler"
OUTPUT_DIR = "docs/data"

SPLITS = ("train", "validation", "test")


def _metrics(obs, pred):
    """Compute R^2, RMSE, MAPE for one year/split."""
    obs = np.asarray(obs, dtype=float)
    pred = np.asarray(pred, dtype=float)
    n = obs.size
    if n == 0:
        return {"r2": None, "rmse": None, "mape": None, "n": 0}

    rmse = float(np.sqrt(np.mean((pred - obs) ** 2)))

    ss_res = float(np.sum((obs - pred) ** 2))
    ss_tot = float(np.sum((obs - obs.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else None

    nonzero = obs != 0
    if nonzero.any():
        mape = float(np.mean(np.abs((pred[nonzero] - obs[nonzero]) / obs[nonzero])) * 100)
    else:
        mape = None

    return {
        "r2": None if r2 is None else round(r2, 4),
        "rmse": round(rmse, 4),
        "mape": None if mape is None else round(mape, 3),
        "n": int(n),
    }


def _load_split(crop, month, split):
    path = os.path.join(RESULTS_DIR, crop, month, f"{split}_outputs.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def _denormalize(arr, mean, std):
    return np.asarray(arr, dtype=float) * std + mean


def _process_crop(crop):
    months = sorted(
        d for d in os.listdir(os.path.join(RESULTS_DIR, crop))
        if os.path.isdir(os.path.join(RESULTS_DIR, crop, d))
    )
    if not months:
        return None
    month = months[0]

    scaler_path = os.path.join(SCALER_DIR, f"scaler_{crop}.json")
    if not os.path.exists(scaler_path):
        print(f"  [skip] no scaler at {scaler_path}")
        return None
    with open(scaler_path) as f:
        scaler = json.load(f)
    y_mean = float(scaler["yield_mean"])
    y_std = float(scaler["yield_std"])

    out = {
        "crop": crop,
        "harvest_month": month,
        "units": "t/ha",
        "yield_mean": round(y_mean, 4),
        "yield_std": round(y_std, 4),
        "splits": {},
        "years": [],
    }

    by_year = {}  # year -> {"split": str, "obs": [...], "pred": [...]}

    for split in SPLITS:
        data = _load_split(crop, month, split)
        if data is None:
            print(f"  [skip] {crop}/{month}/{split}: no pickle")
            continue

        pred = _denormalize(data["prediction"][:, 1], y_mean, y_std)  # q50
        target = _denormalize(data["target"], y_mean, y_std)
        years = np.asarray(data["year"]).astype(int)

        out["splits"][split] = _metrics(target, pred) | {
            "years": sorted(set(int(y) for y in years.tolist())),
        }

        for yr in np.unique(years):
            mask = years == yr
            entry = by_year.setdefault(int(yr), {"split": split, "obs": [], "pred": []})
            entry["obs"].extend(target[mask].tolist())
            entry["pred"].extend(pred[mask].tolist())

    if not by_year:
        return None

    for yr in sorted(by_year):
        e = by_year[yr]
        obs = np.asarray(e["obs"], dtype=float)
        pred = np.asarray(e["pred"], dtype=float)
        out["years"].append({
            "year": yr,
            "split": e["split"],
            "n": int(obs.size),
            "observed": [round(float(v), 3) for v in obs.tolist()],
            "predicted": [round(float(v), 3) for v in pred.tolist()],
            "metrics": _metrics(obs, pred),
        })

    return out


def main():
    if not os.path.isdir(RESULTS_DIR):
        print(f"Training results dir not found: {RESULTS_DIR}")
        return
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    crops = sorted(
        d for d in os.listdir(RESULTS_DIR)
        if os.path.isdir(os.path.join(RESULTS_DIR, d))
    )
    if not crops:
        print("No crop result directories found.")
        return

    index = []
    for crop in crops:
        print(f"Processing {crop}…")
        payload = _process_crop(crop)
        if payload is None:
            continue
        out_path = os.path.join(OUTPUT_DIR, f"performance_{crop}.json")
        with open(out_path, "w") as f:
            json.dump(payload, f, separators=(",", ":"))
        index.append({
            "crop": crop,
            "harvest_month": payload["harvest_month"],
            "file": f"performance_{crop}.json",
            "n_years": len(payload["years"]),
        })
        print(f"  -> {out_path} "
              f"({len(payload['years'])} years, "
              f"{sum(y['n'] for y in payload['years'])} samples)")

    if index:
        with open(os.path.join(OUTPUT_DIR, "performance_index.json"), "w") as f:
            json.dump(index, f, indent=2)
        print(f"Wrote performance_index.json with {len(index)} crops.")


if __name__ == "__main__":
    main()
