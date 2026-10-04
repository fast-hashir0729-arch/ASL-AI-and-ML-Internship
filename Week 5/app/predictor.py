"""Inference only: turns validated passengers into predictions."""

import pandas as pd

from app.config import FEATURE_COLUMNS

LABELS = {0: "did not survive", 1: "survived"}


def make_dataframe(passengers):
    """Build the table the model expects (same columns and order as training)."""
    rows = [passenger.model_dump() for passenger in passengers]
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS)


def predict_passengers(model, passengers):
    """Return one result dictionary per passenger.

    The model is a full Pipeline, so scaling and encoding are applied
    with the settings learned during training (nothing is refitted here).
    """
    data = make_dataframe(passengers)
    predictions = model.predict(data)
    survive_probabilities = model.predict_proba(data)[:, 1]

    results = []
    for prediction, probability in zip(predictions, survive_probabilities):
        prediction = int(prediction)
        probability = float(probability)
        confidence = probability if prediction == 1 else 1 - probability
        results.append(
            {
                "prediction": prediction,
                "label": LABELS[prediction],
                "probability_survived": round(probability, 4),
                "confidence": round(confidence, 4),
            }
        )
    return results
