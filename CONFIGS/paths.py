from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_PATH = PROJECT_ROOT / "BACKEND" / "CONFIGS" / "data_config.json"


def load_config(config_path):
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration not found: {config_path}")

    if not config_path.is_file():
        raise ValueError("Configuration path is not a file.")

    if config_path.suffix.lower() != ".json":
        raise ValueError("Configuration must be a JSON file.")

    try:
        with open(config_path, "r", encoding="utf-8") as file:
            config = json.load(file)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON configuration: {config_path}") from error

    if not isinstance(config, dict):
        raise ValueError("Configuration must be a JSON object.")

    return config


DATA_CONFIG = load_config(CONFIG_PATH)

DATA_DIR = PROJECT_ROOT / "DATA"
MODEL_DIR = PROJECT_ROOT / "BACKEND" / "MODELS"
REPORTS_DIR = PROJECT_ROOT / "BACKEND" / "REPORTS"

RAW_DATA_DIR = PROJECT_ROOT / DATA_CONFIG["directories"]["raw"]
INTERIM_DATA_DIR = PROJECT_ROOT / DATA_CONFIG["directories"]["interim"]
PROCESSED_DATA_DIR = PROJECT_ROOT / DATA_CONFIG["directories"]["processed"]

RAW_DATA_PATH_RISK = RAW_DATA_DIR / DATA_CONFIG["files"]["raw_dataset"]
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / DATA_CONFIG["files"]["processed_dataset"]
REJECTED_DATA_PATH = RAW_DATA_DIR / DATA_CONFIG["files"]["rejected"]

RISK_MODEL = PROJECT_ROOT / DATA_CONFIG["models"]["risk_model"]

LOADING_CONFIG = DATA_CONFIG["loading"]