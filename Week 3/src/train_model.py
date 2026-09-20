
import json
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
import joblib

from src.model_pipeline import build_full_pipeline

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
METRICS_DIR = Path("outputs/metrics")

RANDOM_STATE = 42


def load_train_split():
    """Load the Phase 1 raw (unprocessed) training features and target."""
    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").iloc[:, 0]
    return X_train, y_train


def train_baseline_model(X_train, y_train) -> dict:
    """
    Train the Logistic Regression baseline and cross-validate it (bonus).
    Returns the fitted pipeline plus its cross-validation scores.
    """
    pipeline = build_full_pipeline(
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
    )

    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="f1")

    pipeline.fit(X_train, y_train)

    return {
        "name": "Logistic Regression (baseline)",
        "pipeline": pipeline,
        "cv_f1_mean": cv_scores.mean(),
        "cv_f1_std": cv_scores.std(),
    }


def train_comparative_model(X_train, y_train) -> dict:
    """
    Train a Random Forest as the comparative model, tuned with a small
    GridSearchCV (bonus feature). Returns the best fitted pipeline plus
    the chosen hyperparameters.
    """
    pipeline = build_full_pipeline(
        RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE)
    )

    # Small, deliberately narrow grid -- this is a baseline internship
    # project, not a production hyperparameter search.
    param_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [4, 8, None],
        "model__min_samples_leaf": [1, 3],
    }

    search = GridSearchCV(pipeline, param_grid, cv=5, scoring="f1", n_jobs=-1)
    search.fit(X_train, y_train)

    return {
        "name": "Random Forest (comparative, tuned)",
        "pipeline": search.best_estimator_,
        "best_params": search.best_params_,
        "cv_f1_mean": search.best_score_,
    }


def save_models(baseline: dict, comparative: dict) -> None:
    """Save both fitted pipelines to models/ and a summary of training runs."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(baseline["pipeline"], MODELS_DIR / "baseline_logistic_regression.joblib")
    joblib.dump(comparative["pipeline"], MODELS_DIR / "comparative_random_forest.joblib")

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "baseline": {
            "name": baseline["name"],
            "cv_f1_mean": round(baseline["cv_f1_mean"], 4),
            "cv_f1_std": round(baseline["cv_f1_std"], 4),
        },
        "comparative": {
            "name": comparative["name"],
            "cv_f1_mean": round(comparative["cv_f1_mean"], 4),
            "best_params": comparative["best_params"],
        },
    }
    with open(METRICS_DIR / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def run_training() -> None:
    X_train, y_train = load_train_split()

    print("Training baseline model (Logistic Regression) with 5-fold CV...")
    baseline = train_baseline_model(X_train, y_train)
    print(
        f"  Baseline CV F1: {baseline['cv_f1_mean']:.3f} "
        f"(+/- {baseline['cv_f1_std']:.3f})"
    )

    print("Training comparative model (Random Forest) with GridSearchCV...")
    comparative = train_comparative_model(X_train, y_train)
    print(f"  Best params: {comparative['best_params']}")
    print(f"  Comparative best CV F1: {comparative['cv_f1_mean']:.3f}")

    save_models(baseline, comparative)
    print(f"Both fitted models saved to {MODELS_DIR}/")


if __name__ == "__main__":
    run_training()
