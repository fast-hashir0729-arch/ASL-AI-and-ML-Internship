"""
Bonus feature — log experiment parameters/metrics to a simple CSV, and
compute permutation importance alongside the tree-based built-in
importances for the final selected model.

Run this after evaluate_compare.py (it reads outputs/model_comparison.csv
and models/best_hyperparameters.json).
"""

import json
from datetime import datetime, timezone

import joblib
import pandas as pd
from sklearn.inspection import permutation_importance

from config import (
    BEST_PARAMS_PATH,
    EXPERIMENT_LOG_PATH,
    FINAL_MODEL_PATH,
    RESULTS_TABLE_PATH,
)
from load_data import drop_redundant_columns, load_raw_data
from preprocess import get_feature_target_split, train_test_split_stratified


def append_experiment_log():
    """Append one row per model in the comparison table to a running CSV log."""
    results_df = pd.read_csv(RESULTS_TABLE_PATH)
    results_df["logged_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    if EXPERIMENT_LOG_PATH.exists():
        existing = pd.read_csv(EXPERIMENT_LOG_PATH)
        combined = pd.concat([existing, results_df], ignore_index=True)
    else:
        combined = results_df

    combined.to_csv(EXPERIMENT_LOG_PATH, index=False)
    print(f"Appended {len(results_df)} rows to {EXPERIMENT_LOG_PATH}")


def compute_permutation_importance():
    """Compute permutation importance for the final saved model on the test set."""
    df = drop_redundant_columns(load_raw_data())
    X, y = get_feature_target_split(df)
    _, X_test, _, y_test = train_test_split_stratified(X, y)

    pipeline = joblib.load(FINAL_MODEL_PATH)
    result = permutation_importance(
        pipeline, X_test, y_test, n_repeats=30, random_state=42, scoring="f1"
    )

    importance_df = pd.DataFrame(
        {
            "feature": X_test.columns,
            "perm_importance_mean": result.importances_mean.round(4),
            "perm_importance_std": result.importances_std.round(4),
        }
    ).sort_values("perm_importance_mean", ascending=False)

    print("\nPermutation importance (original feature columns, test set):")
    print(importance_df.to_string(index=False))

    with open(BEST_PARAMS_PATH) as f:
        record = json.load(f)
    record["permutation_importance"] = importance_df.set_index("feature")[
        "perm_importance_mean"
    ].to_dict()
    with open(BEST_PARAMS_PATH, "w") as f:
        json.dump(record, f, indent=2, default=str)

    return importance_df


if __name__ == "__main__":
    append_experiment_log()
    compute_permutation_importance()
