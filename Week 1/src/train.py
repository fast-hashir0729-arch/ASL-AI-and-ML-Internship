

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "breast_cancer_clean.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"

TARGET_COLUMN = "diagnosis"
COLUMNS_TO_DROP = ["diagnosis", "diagnosis_label"]
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_features_and_target(df: pd.DataFrame):
    X = df.drop(columns=COLUMNS_TO_DROP)
    y = df[TARGET_COLUMN]
    return X, y


def split_data(X, y):
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )


def build_logistic_regression_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)),
        ]
    )


def build_random_forest() -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)


def train_models(X_train, y_train) -> dict:
    models = {
        "logistic_regression": build_logistic_regression_pipeline(),
        "random_forest": build_random_forest(),
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
        print(f"Trained {name}")
    return models


def save_models(models: dict) -> None:

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        path = MODELS_DIR / f"{name}.joblib"
        joblib.dump(model, path)
        print(f"Saved {name} to {path}")


if __name__ == "__main__":
    dataset = pd.read_csv(PROCESSED_DATA_PATH)
    X, y = load_features_and_target(dataset)
    X_train, X_test, y_train, y_test = split_data(X, y)

    print(f"Train size: {len(X_train)} rows, Test size: {len(X_test)} rows")

    trained_models = train_models(X_train, y_train)
    save_models(trained_models)

    X_test.to_csv(PROCESSED_DIR / "X_test.csv", index=False)
    y_test.to_csv(PROCESSED_DIR / "y_test.csv", index=False)
    print("Saved test split to data/processed/ for evaluation.")
