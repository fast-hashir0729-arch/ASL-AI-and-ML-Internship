

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
FIGURES_DIR = Path("outputs/figures")
METRICS_DIR = Path("outputs/metrics")


def load_test_split():
    """Load the held-out test features and target (Phase 1 split)."""
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").iloc[:, 0]
    return X_test, y_test


def load_trained_models() -> dict:
    """Load both fitted pipelines saved by train_model.py."""
    return {
        "Logistic Regression (baseline)": joblib.load(
            MODELS_DIR / "baseline_logistic_regression.joblib"
        ),
        "Random Forest (comparative, tuned)": joblib.load(
            MODELS_DIR / "comparative_random_forest.joblib"
        ),
    }


def compute_metrics(model, X_test, y_test) -> dict:
    """Compute accuracy, precision, recall, F1, and the confusion matrix."""
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "y_pred": y_pred,
    }


def plot_confusion_matrix(cm, model_name: str, filename: str) -> None:
    """Save a labeled confusion matrix plot for one model."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Did not survive", "Survived"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix -- {model_name}")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=150)
    plt.close(fig)


def plot_model_comparison(results: dict) -> None:
    """Save a grouped bar chart comparing accuracy/precision/recall/F1 across models."""
    metrics = ["accuracy", "precision", "recall", "f1"]
    model_names = list(results.keys())

    fig, ax = plt.subplots(figsize=(7, 5))
    bar_width = 0.35
    x_positions = range(len(metrics))

    for i, name in enumerate(model_names):
        values = [results[name][m] for m in metrics]
        offset = [x + i * bar_width for x in x_positions]
        ax.bar(offset, values, width=bar_width, label=name)

    ax.set_xticks([x + bar_width / 2 for x in x_positions])
    ax.set_xticklabels([m.capitalize() for m in metrics])
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_title("Model Comparison on Test Set")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "model_comparison.png", dpi=150)
    plt.close(fig)


def plot_feature_importance(rf_pipeline) -> None:
    """Save a horizontal bar chart of Random Forest feature importances."""
    preprocessor = rf_pipeline.named_steps["preprocess"]
    model = rf_pipeline.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()
    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    importances.plot(kind="barh", ax=ax, color="#2c7fb8")
    ax.set_xlabel("Importance")
    ax.set_title("Random Forest Feature Importance")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "feature_importance.png", dpi=150)
    plt.close(fig)


def build_comparison_table(results: dict) -> pd.DataFrame:
    """Build a comparison DataFrame of accuracy/precision/recall/F1 per model."""
    rows = []
    for name, metrics in results.items():
        rows.append(
            {
                "model": name,
                "accuracy": round(metrics["accuracy"], 4),
                "precision": round(metrics["precision"], 4),
                "recall": round(metrics["recall"], 4),
                "f1": round(metrics["f1"], 4),
            }
        )
    return pd.DataFrame(rows)


def write_metrics_report(results: dict, comparison_table: pd.DataFrame) -> str:
    """Build the full plain-text evaluation report and save it to disk."""
    lines = ["MODEL EVALUATION REPORT", "=" * 40, ""]

    for name, metrics in results.items():
        lines.append(name)
        lines.append("-" * len(name))
        lines.append(f"Accuracy:  {metrics['accuracy']:.4f}")
        lines.append(f"Precision: {metrics['precision']:.4f}")
        lines.append(f"Recall:    {metrics['recall']:.4f}")
        lines.append(f"F1 score:  {metrics['f1']:.4f}")
        lines.append(f"Confusion matrix (rows=actual, cols=predicted):\n{metrics['confusion_matrix']}")
        lines.append("")

    lines.append("COMPARISON TABLE")
    lines.append(comparison_table.to_string(index=False))
    lines.append("")

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    report_text = "\n".join(lines)
    (METRICS_DIR / "evaluation_report.txt").write_text(report_text, encoding="utf-8")
    comparison_table.to_csv(METRICS_DIR / "model_comparison.csv", index=False)
    return report_text


def choose_and_save_best_model(results: dict, models: dict) -> str:
    """Pick the model with the highest test-set F1 score and save it as the final model."""
    best_name = max(results, key=lambda name: results[name]["f1"])
    best_pipeline = models[best_name]
    joblib.dump(best_pipeline, MODELS_DIR / "final_model.joblib")
    return best_name


def run_evaluation() -> None:
    X_test, y_test = load_test_split()
    models = load_trained_models()

    results = {name: compute_metrics(model, X_test, y_test) for name, model in models.items()}

    for name, metrics in results.items():
        filename = name.split(" ")[0].lower() + "_confusion_matrix.png"
        plot_confusion_matrix(metrics["confusion_matrix"], name, filename)

    plot_model_comparison(results)
    plot_feature_importance(models["Random Forest (comparative, tuned)"])

    comparison_table = build_comparison_table(results)
    report_text = write_metrics_report(results, comparison_table)
    print(report_text)

    best_name = choose_and_save_best_model(results, models)
    print(f"\nBest model by test-set F1: {best_name}")
    print(f"Saved as {MODELS_DIR / 'final_model.joblib'}")


if __name__ == "__main__":
    run_evaluation()
