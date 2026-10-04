"""Central settings for the API. Everything configurable lives here."""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Model settings. Change MODEL_VERSION to deploy a new model file.
MODEL_NAME = "titanic-gradient-boosting"
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / f"titanic_gb_{MODEL_VERSION}.joblib"
METADATA_PATH = MODELS_DIR / f"titanic_gb_{MODEL_VERSION}.json"

# The exact columns (and order) the model was trained on
FEATURE_COLUMNS = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]

# Largest number of records allowed in one batch request
MAX_BATCH_SIZE = 100

FRONTEND_FILE = BASE_DIR / "frontend" / "index.html"


# These are functions (not constants) so tests can change environment variables.
def get_log_path():
    """Where prediction logs are written."""
    default_path = BASE_DIR / "logs" / "predictions.log"
    return Path(os.getenv("PREDICTION_LOG_PATH", default_path))


def get_api_key():
    """API key clients must send. Empty means protection is switched off."""
    return os.getenv("API_KEY", "")


def get_rate_limit():
    """Maximum requests per minute for each client."""
    return int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
