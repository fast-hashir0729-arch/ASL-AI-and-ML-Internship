"""
Phase 4 — Final evaluation, model comparison, and artifact saving.

Model selection uses cross-validation evidence (mean f1 across folds)
gathered in tune_hyperparameters.py, not the held-out test score -- the
test set is touched exactly once at the end, purely to report a final,
unbiased performance estimate for the chosen model. This follows the
Week 4 requirement to "select a final model using validation evidence
rather than test-set peeking."
"""

import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from config import (
    BEST_PARAMS_PATH,
    FIGURES_DIR,
    FINAL_MODEL_PATH,
    RANDOM_STATE,
    RESULTS_TABLE_PATH,
)
from cross_validate_models import make_pipeline, run as run_cross_validation
from tune_hyperparameters import run as run_tuning

# Week 3 baseline, reported here for side-by-side comparison. The dataset,
# target, and train/test split convention are unchanged from Week 3
# (same Titanic data, same random_state=42, same stratified 80/20 split),
# so this figure is directly comparable to this week's test-set results.
WEEK3_BASELINE_TEST_F1 = 0.761
WEEK3_BASELINE_MODEL_NAME = "logistic_regression_week3_baseline"


def evaluate_on_test(pipeline, X_test, y_test) -> dict:
    y_pred = pipeline.predict(X_test)
    return {
        "test_accuracy": round(accuracy_score(y_test, y_pred), 4),
        "test_precision": round(precision_score(y_test, y_pred), 4),
        "test_recall": round(recall_score(y_test, y_pred), 4),
        "test_f1": round(f1_score(y_test, y_pred), 4),
    }


def plot_cv_comparison(cv_results: dict, tuned_results: dict, out_path):
    """Bar chart of mean CV f1 +/- std for every model considered."""
    names, means, stds = [], [], []
    for name, summary in cv_results.items():
        names.append(name.replace("_", " "))
        means.append(summary["f1"]["mean"])
        stds.append(summary["f1"]["std"])
    for name, summary in tuned_results.items():
        names.append(f"{name.replace('_', ' ')} (tuned)")
        means.append(summary["best_cv_f1"])
        stds.append(0)  # best_score_ from search doesn't expose per-fold std directly

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(names, means, yerr=stds, capsize=5, color="#4C72B0")
    ax.set_ylabel("Mean CV F1 score")
    ax.set_title("Cross-Validated F1 by Model (mean \u00b1 std across folds)")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_confusion_matrix(pipeline, X_test, y_test, model_name, out_path):
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_estimator(
        pipeline, X_test, y_test, ax=ax, cmap="Blues", colorbar=False
    )
    ax.set_title(f"Confusion Matrix — {model_name} (test set)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_feature_importance(pipeline, model_name, out_path, top_n=12):
    preprocessor = pipeline.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()
    model = pipeline.named_steps["model"]
    importances = model.feature_importances_

    order = np.argsort(importances)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(
        [feature_names[i] for i in order][::-1],
        [importances[i] for i in order][::-1],
        color="#55A868",
    )
    ax.set_xlabel("Importance")
    ax.set_title(f"Feature Importance — {model_name}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return {feature_names[i]: round(float(importances[i]), 4) for i in order}


def run():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # --- Cross-validated baselines (Logistic Regression, Random Forest) --
    cv_results = run_cross_validation()

    # --- Hyperparameter-tuned Random Forest + Gradient Boosting ----------
    rf_search, gb_search, (X_train, X_test, y_train, y_test) = run_tuning()
    tuned_results = {
        "random_forest": {
            "best_cv_f1": rf_search.best_score_,
            "params": rf_search.best_params_,
        },
        "gradient_boosting": {
            "best_cv_f1": gb_search.best_score_,
            "params": gb_search.best_params_,
        },
    }

    # --- Model selection: based on CV evidence only -----------------------
    candidates = {
        "random_forest_tuned": (rf_search.best_estimator_, rf_search.best_score_),
        "gradient_boosting_tuned": (gb_search.best_estimator_, gb_search.best_score_),
    }
    best_name, (best_pipeline, best_cv_f1) = max(
        candidates.items(), key=lambda item: item[1][1]
    )
    print(f"\nSelected final model by CV f1: {best_name} (CV f1={best_cv_f1:.4f})")

    # --- Test-set evaluation (touched once, for every model, for the table)
    rows = []

    rows.append(
        {
            "model": WEEK3_BASELINE_MODEL_NAME,
            "cv_f1_mean": None,
            "cv_f1_std": None,
            **{"test_accuracy": None, "test_precision": None, "test_recall": None},
            "test_f1": WEEK3_BASELINE_TEST_F1,
            "note": "Reported from Week 3 (unchanged dataset/split)",
        }
    )

    non_tuned_estimators = {
        "logistic_regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(
            class_weight="balanced", random_state=RANDOM_STATE
        ),
    }
    for name, summary in cv_results.items():
        pipeline = make_pipeline(non_tuned_estimators[name])
        pipeline.fit(X_train, y_train)
        test_metrics = evaluate_on_test(pipeline, X_test, y_test)
        rows.append(
            {
                "model": f"{name}_week4_cv",
                "cv_f1_mean": summary["f1"]["mean"],
                "cv_f1_std": summary["f1"]["std"],
                **test_metrics,
                "note": "Baseline / non-tuned, Week 4 5-fold CV",
            }
        )

    for name, (pipeline, cv_f1) in candidates.items():
        test_metrics = evaluate_on_test(pipeline, X_test, y_test)
        rows.append(
            {
                "model": name,
                "cv_f1_mean": round(cv_f1, 4),
                "cv_f1_std": None,
                **test_metrics,
                "note": "Tuned via Grid/RandomizedSearchCV",
            }
        )

    results_df = pd.DataFrame(rows)
    results_df.to_csv(RESULTS_TABLE_PATH, index=False)
    print(f"\nSaved comparison table to {RESULTS_TABLE_PATH}")
    print(results_df.to_string(index=False))

    # --- Plots -------------------------------------------------------------
    plot_cv_comparison(cv_results, tuned_results, FIGURES_DIR / "cv_f1_comparison.png")
    plot_confusion_matrix(
        best_pipeline, X_test, y_test, best_name, FIGURES_DIR / "confusion_matrix_best_model.png"
    )
    top_features = plot_feature_importance(
        best_pipeline, best_name, FIGURES_DIR / "feature_importance_best_model.png"
    )

    # --- Save final model + chosen hyperparameters --------------------------
    joblib.dump(best_pipeline, FINAL_MODEL_PATH)
    final_record = {
        "selected_model": best_name,
        "selection_basis": "cross-validated mean F1 (StratifiedKFold, k=5)",
        "cv_f1": round(best_cv_f1, 4),
        "hyperparameters": {
            k.replace("model__", ""): (v if not hasattr(v, "item") else v.item())
            for k, v in (
                rf_search.best_params_ if best_name == "random_forest_tuned" else gb_search.best_params_
            ).items()
        },
        "test_set_metrics": evaluate_on_test(best_pipeline, X_test, y_test),
        "top_features": top_features,
    }
    with open(BEST_PARAMS_PATH, "w") as f:
        json.dump(final_record, f, indent=2, default=str)
    print(f"\nSaved final model to {FINAL_MODEL_PATH}")
    print(f"Saved final hyperparameter record to {BEST_PARAMS_PATH}")

    return results_df, final_record


if __name__ == "__main__":
    run()
