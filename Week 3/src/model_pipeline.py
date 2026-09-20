

from sklearn.pipeline import Pipeline

from src.preprocessing import build_preprocessor


def build_full_pipeline(model) -> Pipeline:
    """Wrap the shared preprocessor and a given estimator into one Pipeline."""
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            ("model", model),
        ]
    )
