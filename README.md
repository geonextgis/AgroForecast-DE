# AgroForecast-DE

<p align="center">
  <img src="docs/assets/logo.png" alt="AgroForecast-DE Logo" width="200">
</p>

AgroForecast-DE is a district-level crop yield forecasting system for Germany. It utilizes a custom deep learning architecture, **CropFusionNet**, to provide probabilistic yield estimates months before harvest, leveraging seasonal climate forecasts.

## 🌟 Key Features

- **High Resolution:** Forecasts at the NUTS-3 level (approx. 400 German administrative districts).
- **Early Season Prediction:** Generates reliable forecasts as early as April for the autumn harvest.
- **Probabilistic Estimates:** Provides q10/q50/q90 quantiles derived from the 50-member GCFS2.2 climate ensemble.
- **Multi-Crop Support:** Currently supports Silage Maize, Grain Maize, Winter Wheat, and Winter Barley.
- **Interactive Dashboard:** Visualizes results via a web-based geographic dashboard.

## 🏗️ Architecture: CropFusionNet

CropFusionNet is designed to fuse static district-level features with temporal climate data:

- **Variable Selection Network (VSN):** Identifies the most relevant features for each crop and location.
- **Temporal Encoder:** LSTM-based processing with optional Attention and Temporal Convolutional layers.
- **Static Enrichment:** Integrates soil quality and topography data into the temporal representations.
- **Pyramidal Pooling:** Multi-scale temporal aggregation for robust feature extraction.

## 🌾 Supported Crops

| Crop | Harvest Month | Harvest Next Year |
| :--- | :--- | :--- |
| **Silage Maize** | September | No |
| **Grain Maize** | September | No |
| **Winter Wheat** | July | No |
| **Winter Barley** | July | No |

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- PyTorch
- GeoPandas
- (See `GEMINI.md` for detailed technology stack)

### Data Pipeline

The project includes a multi-step pipeline for processing climate, static, and phenology data:

```bash
cd data-pipeline && bash submit_pipeline.sh
```

### Running Forecasts

To generate a new forecast for a specific crop:

```bash
python src/submit/submit_job_forecast.py --crop silage_maize
```

## 📊 Web Dashboard

The forecasts are exported to an interactive dashboard located in the `docs/` folder.

To update the dashboard with new forecast data:

```bash
python3 docs/scripts/export_geojson.py
```

The dashboard can be hosted via GitHub Pages from the `docs/` directory.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Acknowledgments

Developed as part of research into regional crop yield modeling and climate adaptation in Germany.
