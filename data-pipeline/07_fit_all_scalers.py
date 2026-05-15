import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset.dataset import fit_scalers

# "winter_wheat", "winter_barley", "silage_maize", "grain_maize"
for crop in ["grain_maize"]:
    cfg = importlib.import_module(f"src.config.{crop}")
    fit_scalers(cfg, save_path=cfg.SCALER_FILE_PATH)
