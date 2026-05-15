import json
import os

import torch

CROP = "winter_wheat"
HARVEST_NEXT_YEAR = True

REPO_ROOT = "/beegfs/halder/GITHUB/RESEARCH/AgroForecast-DE"
ROOT_DATA_DIR = os.path.join(REPO_ROOT, "data")

CLIMATE_DISTRICTS_DIR = os.path.join(ROOT_DATA_DIR, "interim/climate/districts")
GCFS22_ROOT = os.path.join(ROOT_DATA_DIR, "interim/gcfs22")
STATIC_SOIL_PATH = os.path.join(ROOT_DATA_DIR, "interim/static/soil_quality.parquet")
STATIC_TOPO_PATH = os.path.join(ROOT_DATA_DIR, "interim/static/topography.parquet")
PHENOLOGY_FILE_PATH = os.path.join(
    ROOT_DATA_DIR, f"interim/phenology/{CROP}_phenology.csv"
)
YIELD_FILE_PATH = os.path.join(ROOT_DATA_DIR, "interim/yield/Final_data_2024.csv")

SCALER_FILE_PATH = os.path.join(REPO_ROOT, f"src/scaler/scaler_{CROP}.json")

# Define features
climate_features = ["tavg", "tmax", "tmin", "prec", "rad", "hurs", "et0", "vpd", "cwb"]
static_features = [
    "soil_quality_mean",
    "soil_quality_stdDev",
    "elevation_mean",
    "elevation_stdDev",
    "slope_mean",
    "slope_stdDev",
]

# Define variable names
time_varying_real = climate_features
time_varying_categorical = []
static_real_variables = static_features
static_categorical_variables = []
target = "yield"

# Scaling parameters for standardization (tolerant on first-time fit)
try:
    with open(SCALER_FILE_PATH, "r") as f:
        scalers = json.load(f)
except FileNotFoundError:
    scalers = None

device = "cuda" if torch.cuda.is_available() else "cpu"

# Define forecasting scenarios
forecast_scenarios = {
    "Mar": 274,
    "Apr": 304,
    "May": 335,
    "Jun": 365,
    "Jul": 396,
}
forecast_month = "Jul"

# Config for TFT
model_config = {
    "device": device,
    "static_categorical_variables": len(static_categorical_variables),
    "static_real_variables": len(static_real_variables),
    "static_embedding_vocab_sizes": [],
    "time_varying_categorical_variables": len(time_varying_categorical),
    "time_varying_embedding_vocab_sizes": [],
    "time_varying_real_variables": len(time_varying_real),
    "lstm_hidden_dimension": 128,
    "lstm_layers": 3,
    "attn_heads": 16,
    "pooling_heads": 4,
    "dropout": 0.4,
    "embedding_dim": 8,
    "seq_length": forecast_scenarios[forecast_month],
    "use_temporal_conv": True,
    "use_lstm": True,
    "use_attention": True,
    "use_static_enrichment": True,
    "use_pyramidal_pooling": True,
    "quantiles": [0.1, 0.5, 0.9],
}

train_config = {
    "device": device,
    "batch_size": 32,
    "lr": 1e-4,
    "weight_decay": 1e-5,
    "num_epochs": 500,
    "early_stopping_patience": 20,
    "exp_name": f"exp_{CROP}_{forecast_month}",
}
