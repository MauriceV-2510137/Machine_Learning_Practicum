"""Persist and load each tuned model's hyperparameters as plain set_params-ready dicts."""

import json

from src.config import OUTPUT_DIR

STORE_PATH = OUTPUT_DIR / "best_hyperparameters.json"


def save_hyperparameters(model_name, params):
    """Save (overwriting) one model's hyperparameters -- called every time that model's tuning step runs."""
    data = load_hyperparameters()
    data[model_name] = params
    OUTPUT_DIR.mkdir(exist_ok=True)
    STORE_PATH.write_text(json.dumps(data, indent=2))


def load_hyperparameters():
    """All stored hyperparameters, or {} if none have been saved yet."""
    if not STORE_PATH.exists():
        return {}
    return json.loads(STORE_PATH.read_text())