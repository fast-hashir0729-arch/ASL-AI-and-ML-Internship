
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

CLASS_NAMES = ["Malignant", "Benign"]  # index 0, index 1
POSITIVE_CLASS = 0  # malignant — the clinically important class to catch


def load_test_set():
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").squeeze("columns")
    return X_test, y_test


def load_models() -> dict:
    return {
        "logistic_regression": joblib.load(MODELS_DIR / "logistic_regression.joblib"),
        "random_forest": joblib.load(MODELS_DIR / "random_forest.joblib"),
    }


def evaluate_model(model, X_test, y_test) -> dict:
    predictions = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, pos_label=POSITIVE_CLASS),
        "recall": recall_score(y_test, predictions, pos_label=POSITIVE_CLASS),
        "f1_score": f1_score(y_test, predictions, pos_label=POSITIVE_CLASS),
        "predictions": predictions,
    }


def plot_confusion_matrix(y_test, predictions, model_name: str) -> None:
    cm = confusion_matrix(y_test, predictions, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix: {model_name.replace('_', ' ').title()}")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14)

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    filename = f"confusion_matrix_{model_name}.png"
    fig.savefig(FIGURES_DIR / filename, dpi=150)
    plt.close(fig)
    print(f"Saved {filename}")


def plot_feature_importance(model, feature_names, top_n: int = 10) -> None:
    importances = pd.Series(model.feature_importances_, index=feature_names)
    top_features = importances.sort_values(ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top_features.index, top_features.values, color="#1f77b4")
    ax.set_title(f"Top {top_n} Feature Importances (Random Forest)")
    ax.set_xlabel("Importance")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "feature_importance_random_forest.png", dpi=150)
    plt.close(fig)
    print("Saved feature_importance_random_forest.png")


def print_and_save_comparison(results: dict) -> pd.DataFrame:
    rows = []
    for name, metrics in results.items():
        rows.append(
            {
                "model": name,
                "accuracy": round(metrics["accuracy"], 4),
                "precision": round(metrics["precision"], 4),
                "recall": round(metrics["recall"], 4),
                "f1_score": round(metrics["f1_score"], 4),
            }
        )
    comparison_df = pd.DataFrame(rows)
    print("\nModel Comparison (malignant = positive class):")
    print(comparison_df.to_string(index=False))

    comparison_df.to_csv(OUTPUTS_DIR / "model_comparison.csv", index=False)
    print(f"\nSaved comparison table to outputs/model_comparison.csv")
    return comparison_df


if __name__ == "__main__":
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    X_test, y_test = load_test_set()
    models = load_models()

    results = {}
    for name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test)
        results[name] = metrics
        plot_confusion_matrix(y_test, metrics["predictions"], name)

    print_and_save_comparison(results)

    # Feature importance only applies to the tree-based model.
    plot_feature_importance(models["random_forest"], X_test.columns)
