"""
Phase 3 — Hyperparameter search for the ensemble models.

GridSearchCV is used for Random Forest (a modest, fully-enumerable grid)
and RandomizedSearchCV for Gradient Boosting (a larger space, sampled
rather than exhaustively searched, which scales better). Both searches
wrap the full preprocessing + model Pipeline, so every candidate
configuration is refit on training folds only -- tuning never touches the
held-out test set.
"""

import json

import pandas as pd
from scipy.stats import randint, uniform
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold

from config import N_FOLDS, RANDOM_STATE
from cross_validate_models import make_pipeline
from load_data import drop_redundant_columns, load_raw_data
from preprocess import get_feature_target_split, train_test_split_stratified


def tune_random_forest(X_train, y_train, cv):
    """Exhaustive grid search over a modest Random Forest parameter grid."""
    pipeline = make_pipeline(
        RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE)
    )
    param_grid = {
        "model__n_estimators": [100, 200, 400],
        "model__max_depth": [4, 6, 8, None],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
    }
    search = GridSearchCV(
        pipeline, param_grid, scoring="f1", cv=cv, n_jobs=-1, refit=True
    )
    search.fit(X_train, y_train)
    return search


def tune_gradient_boosting(X_train, y_train, cv):
    """Randomized search over a wider Gradient Boosting parameter space."""
    pipeline = make_pipeline(GradientBoostingClassifier(random_state=RANDOM_STATE))
    param_distributions = {
        "model__n_estimators": randint(100, 400),
        "model__learning_rate": uniform(0.01, 0.29),
        "model__max_depth": randint(2, 5),
        "model__subsample": uniform(0.7, 0.3),
        "model__min_samples_leaf": randint(1, 6),
    }
    search = RandomizedSearchCV(
        pipeline,
        param_distributions,
        n_iter=40,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        refit=True,
    )
    search.fit(X_train, y_train)
    return search


def run():
    df = drop_redundant_columns(load_raw_data())
    X, y = get_feature_target_split(df)
    X_train, X_test, y_train, y_test = train_test_split_stratified(X, y)

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    print("Tuning Random Forest with GridSearchCV ...")
    rf_search = tune_random_forest(X_train, y_train, cv)
    print("  Best CV f1:", round(rf_search.best_score_, 4))
    print("  Best params:", rf_search.best_params_)

    print("\nTuning Gradient Boosting with RandomizedSearchCV ...")
    gb_search = tune_gradient_boosting(X_train, y_train, cv)
    print("  Best CV f1:", round(gb_search.best_score_, 4))
    print("  Best params:", gb_search.best_params_)

    best_params = {
        "random_forest": {
            "best_cv_f1": round(rf_search.best_score_, 4),
            "params": rf_search.best_params_,
        },
        "gradient_boosting": {
            "best_cv_f1": round(gb_search.best_score_, 4),
            "params": gb_search.best_params_,
        },
    }
    with open("../models/best_hyperparameters.json", "w") as f:
        json.dump(best_params, f, indent=2, default=str)
    print("\nSaved best_hyperparameters.json")

    return rf_search, gb_search, (X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    run()
