import os
import glob
import json

import fiona
import geopandas as gpd


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
    print("Export complete.")


if __name__ == "__main__":
    export_forecasts()
