"""Training-time code: builds, tunes and saves the model.

This runs ONCE, offline. The API (app/) never trains anything.
Run from the project root:  python -m training.train_model
"""

import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.config import (
    BASE_DIR,
    FEATURE_COLUMNS,
    METADATA_PATH,
    MODEL_NAME,
    MODEL_PATH,
    MODEL_VERSION,
)

DATA_PATH = BASE_DIR / "data" / "raw" / "titanic.csv"
TARGET_COLUMN = "survived"
RANDOM_STATE = 42
TEST_SIZE = 0.2

NUMBER_COLUMNS = ["pclass", "age", "sibsp", "parch", "fare"]
TEXT_COLUMNS = ["sex", "embarked"]


def load_data():
    """Read the raw CSV and keep only the columns the API will receive."""
    data = pd.read_csv(DATA_PATH)
    features = data[FEATURE_COLUMNS]
    target = data[TARGET_COLUMN]
    return features, target


def build_pipeline():
    """Preprocessing + model in ONE object, so inference reuses the fitted steps."""
    number_steps = Pipeline(
        [
            ("fill_missing", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    text_steps = Pipeline(
        [
            ("fill_missing", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessing = ColumnTransformer(
        [
            ("numbers", number_steps, NUMBER_COLUMNS),
            ("text", text_steps, TEXT_COLUMNS),
        ]
    )
    return Pipeline(
        [
            ("preprocessing", preprocessing),
            ("model", GradientBoostingClassifier(random_state=RANDOM_STATE)),
        ]
    )


def tune_model(pipeline, x_train, y_train):
    """Randomized search with 5-fold cross-validation (scored by F1)."""
    search_space = {
        "model__n_estimators": randint(50, 300),
        "model__learning_rate": uniform(0.01, 0.2),
        "model__max_depth": randint(2, 5),
        "model__subsample": uniform(0.6, 0.4),
    }
    search = RandomizedSearchCV(
        pipeline,
        search_space,
        n_iter=15,
        scoring="f1",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    search.fit(x_train, y_train)
    return search


def evaluate(model, x_test, y_test):
    """Score the final model on the untouched test set."""
    predictions = model.predict(x_test)
    return {
        "accuracy": round(accuracy_score(y_test, predictions), 4),
        "precision": round(precision_score(y_test, predictions), 4),
        "recall": round(recall_score(y_test, predictions), 4),
        "f1": round(f1_score(y_test, predictions), 4),
    }


def save_model(model, search, test_metrics, row_count):
    """Save the model file and a JSON file describing it."""
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    best_params = {}
    for name, value in search.best_params_.items():
        if isinstance(value, (int, np.integer)):
            best_params[name] = int(value)
        else:
            best_params[name] = round(float(value), 4)
    metadata = {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
        "features": FEATURE_COLUMNS,
        "target": TARGET_COLUMN,
        "training_rows": row_count,
        "best_params": best_params,
        "cv_f1": round(search.best_score_, 4),
        "test_metrics": test_metrics,
    }
    with open(METADATA_PATH, "w") as file:
        json.dump(metadata, file, indent=2)


def main():
    features, target = load_data()
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    print("Tuning model (this takes about a minute)...")
    search = tune_model(build_pipeline(), x_train, y_train)
    best_model = search.best_estimator_

    test_metrics = evaluate(best_model, x_test, y_test)
    print("Best CV F1:", round(search.best_score_, 4))
    print("Test metrics:", test_metrics)

    save_model(best_model, search, test_metrics, len(x_train))
    print("Saved:", MODEL_PATH)


if __name__ == "__main__":
    main()
