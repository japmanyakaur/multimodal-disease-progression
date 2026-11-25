"""Loads configs/config.yaml and defines where every file is stored."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]

PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
PREDICTIONS = RESULTS / "predictions"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
RUNS = RESULTS / "runs"


def load_config(path=None):
    path = Path(path) if path else ROOT / "configs" / "config.yaml"
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["_path"] = str(path)
    return cfg


def data_csv(cfg):
    key = "synthetic_csv" if cfg["data"]["source"] == "synthetic" else "adni_csv"
    return ROOT / cfg["data"][key]


def ensure_dirs():
    for d in (PROCESSED, MODELS, PREDICTIONS, TABLES, FIGURES, RUNS):
        d.mkdir(parents=True, exist_ok=True)
