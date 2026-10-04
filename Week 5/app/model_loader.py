"""Loads the saved model and its metadata (done once, at startup)."""

import json
import logging

import joblib
import sklearn

from app.config import METADATA_PATH, MODEL_PATH

logger = logging.getLogger("model_loader")


def load_model():
    """Load the fitted pipeline from disk. Raises an error if the file is missing."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}. Run: python -m training.train_model"
        )
    model = joblib.load(MODEL_PATH)
    logger.info("Loaded model from %s", MODEL_PATH.name)
    return model


def load_metadata():
    """Load the small JSON file that describes the model (metrics, features...)."""
    if not METADATA_PATH.exists():
        return {}
    with open(METADATA_PATH) as file:
        metadata = json.load(file)

    # Pickled models can break when scikit-learn versions differ
    trained_with = metadata.get("sklearn_version")
    if trained_with and trained_with != sklearn.__version__:
        logger.warning(
            "Model trained with scikit-learn %s but running %s. Retrain if you see errors.",
            trained_with,
            sklearn.__version__,
        )
    return metadata
