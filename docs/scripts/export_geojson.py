import os
import glob
import json

import fiona
import geopandas as gpd
import pandas as pd


YIELD_CSV = "data/interim/yield/Final_data_2024.csv"

# Forecast crop name -> "var" column code in the yield CSV
CROP_TO_YIELD_VAR = {
    "winter_wheat": "ww",
    "winter_barley": "wb",
    "silage_maize": "silage_maize",
    "grain_maize": "grain_maize",
}


def _clean_columns(gdf):
    """Drop duplicate-suffix columns produced by upstream merges (e.g. NUTS1_NAME_x/_y)."""
    if "NUTS1_NAME_x" in gdf.columns and "NUTS1_NAME" not in gdf.columns:
        gdf = gdf.rename(columns={"NUTS1_NAME_x": "NUTS1_NAME"})
    drop_cols = [c for c in gdf.columns if c.endswith("_y")]
    if drop_cols:
        gdf = gdf.drop(columns=drop_cols)
    return gdf


def _export_layer(gpkg_path, layer, out_path):
    gdf = gpd.read_file(gpkg_path, layer=layer)
    gdf = gdf.to_crs(epsg=4326)
    gdf = _clean_columns(gdf)

    num_cols = gdf.select_dtypes(include=["float64", "float32"]).columns
    gdf[num_cols] = gdf[num_cols].round(3)

    if os.path.exists(out_path):
        os.remove(out_path)
    gdf.to_file(out_path, driver="GeoJSON")


def export_observed_yields(crops, output_dir):
    """Write per-crop observed-yield lookups.

    For each crop, output_dir/observed_{crop}.json maps:
        {
          "districts": { "<NUTS3_ID>": {"year": YYYY, "value": <t/ha>}, ... },
          "states":    { "<NUTS1_ID>": {"year": YYYY, "value": <t/ha>, "n_districts": N}, ... },
          "latest_year": YYYY
        }

    Each district's observed entry uses the *most recent year with a non-NA
    yield for that district*, so coverage is maximized even when the latest
    global year is missing for some districts. State entries aggregate using
    the most recent year that has any observed values.
    """
    if not os.path.exists(YIELD_CSV):
        print(f"Yield CSV not found at {YIELD_CSV}; skipping observed-yield export.")
        return

    df = pd.read_csv(YIELD_CSV)
    df = df[df["measure"] == "yield"].copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value", "nuts_id", "year"])

    for crop in crops:
        var = CROP_TO_YIELD_VAR.get(crop)
        if not var:
            continue
        sub = df[df["var"] == var]
        if sub.empty:
            print(f"  no observed-yield rows for {crop} ({var}); skipping")
            continue

        latest_year = int(sub["year"].max())

        idx = sub.groupby("nuts_id")["year"].idxmax()
        latest_per_dist = sub.loc[idx, ["nuts_id", "year", "value"]]
        districts = {
            row.nuts_id: {"year": int(row.year), "value": round(float(row.value), 3)}
            for row in latest_per_dist.itertuples(index=False)
        }

        sub_latest = sub[sub["year"] == latest_year].copy()
        sub_latest["nuts1"] = sub_latest["nuts_id"].str.slice(0, 3)
        agg = sub_latest.groupby("nuts1")["value"].agg(["mean", "count"]).reset_index()
        states = {
            row.nuts1: {
                "year": latest_year,
                "value": round(float(row["mean"]), 3),
                "n_districts": int(row["count"]),
            }
            for _, row in agg.iterrows()
        }

        out_path = os.path.join(output_dir, f"observed_{crop}.json")
        with open(out_path, "w") as f:
            json.dump(
                {"latest_year": latest_year, "districts": districts, "states": states},
                f,
                separators=(",", ":"),
            )
        print(f"  observed -> observed_{crop}.json (latest {latest_year}, "
              f"{len(districts)} districts, {len(states)} states)")


def export_forecasts():
    """Export district- and state-level forecast layers as GeoJSON for the dashboard."""
    forecast_base_dir = "output/forecasts"
    output_dir = "docs/data"
    os.makedirs(output_dir, exist_ok=True)

    gpkg_files = sorted(glob.glob(os.path.join(forecast_base_dir, "**/*.gpkg"), recursive=True))
    if not gpkg_files:
        print("No forecast .gpkg files found.")
        return

    all_forecasts = []

    for gpkg_path in gpkg_files:
        print(f"Processing {gpkg_path}...")
        parts = gpkg_path.split(os.sep)
        if len(parts) < 4:
            continue

        date_str = parts[-3]  # YYYY-MM
        crop = parts[-2]      # silage_maize, winter_wheat, etc.

        try:
            layers = fiona.listlayers(gpkg_path)
            district_layer = next((l for l in layers if not l.endswith("_state")), None)
            state_layer = next((l for l in layers if l.endswith("_state")), None)

            district_file = None
            state_file = None

            if district_layer:
                district_file = f"{crop}_{date_str}.geojson"
                _export_layer(gpkg_path, district_layer, os.path.join(output_dir, district_file))
                print(f"  district -> {district_file}")

            if state_layer:
                state_file = f"{crop}_{date_str}_state.geojson"
                _export_layer(gpkg_path, state_layer, os.path.join(output_dir, state_file))
                print(f"  state    -> {state_file}")

            all_forecasts.append({
                "crop": crop,
                "date": date_str,
                "file": district_file,
                "state_file": state_file,
            })

        except Exception as e:
            print(f"Error processing {gpkg_path}: {e}")

    with open(os.path.join(output_dir, "forecast_index.json"), "w") as f:
        json.dump(all_forecasts, f, indent=2)

    crops_seen = sorted({entry["crop"] for entry in all_forecasts})
    if crops_seen:
        print("Exporting observed yields…")
        export_observed_yields(crops_seen, output_dir)

    print("Export complete.")


if __name__ == "__main__":
    export_forecasts()
