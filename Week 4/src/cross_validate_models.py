"""
Phase 2 — Stratified k-fold cross-validation.

A single train/test split (Week 3's approach) gives one noisy performance
estimate. Here we cross-validate two models — the Week 3 Logistic
Regression baseline and a Random Forest — with 5-fold StratifiedKFold, and
report mean +/- standard deviation for each, so the model-selection
report captures variability across folds rather than a single number.
"""

import json

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from config import N_FOLDS, RANDOM_STATE
from load_data import drop_redundant_columns, load_raw_data
from preprocess import build_preprocessor, get_feature_target_split

SCORING = ["accuracy", "precision", "recall", "f1"]


def make_pipeline(estimator) -> Pipeline:
    """Wrap the shared preprocessor and a given estimator in one Pipeline.

    Because the preprocessor lives inside the Pipeline, cross_validate()
    refits scaling/encoding on each training fold independently -- no
    statistic from a validation fold ever reaches training.
    """
    return Pipeline(
        steps=[("preprocessor", build_preprocessor()), ("model", estimator)]
    )


def cross_validate_model(pipeline: Pipeline, X, y, cv):
    """Run cross_validate and return a summary dict of mean +/- std per metric."""
    results = cross_validate(pipeline, X, y, cv=cv, scoring=SCORING, n_jobs=-1)
    summary = {}
    for metric in SCORING:
        scores = results[f"test_{metric}"]
        summary[metric] = {"mean": round(scores.mean(), 4), "std": round(scores.std(), 4)}
    return summary


def run() -> dict:
    df = drop_redundant_columns(load_raw_data())
    X, y = get_feature_target_split(df)

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(
            class_weight="balanced", random_state=RANDOM_STATE
        ),
    }

    all_results = {}
    for name, estimator in models.items():
        pipeline = make_pipeline(estimator)
        summary = cross_validate_model(pipeline, X, y, cv)
        all_results[name] = summary
        print(f"\n{name} — {N_FOLDS}-fold CV results:")
        for metric, stats in summary.items():
            print(f"  {metric}: {stats['mean']:.4f} +/- {stats['std']:.4f}")

    return all_results


if __name__ == "__main__":
    results = run()
    out_path = "../outputs/cv_baseline_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved cross-validation summary to {out_path}")
