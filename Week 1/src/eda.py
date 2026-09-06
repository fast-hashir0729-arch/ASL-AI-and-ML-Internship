

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "breast_cancer_clean.csv"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"


def plot_class_balance(df: pd.DataFrame) -> None:

    counts = df["diagnosis_label"].value_counts()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(counts.index, counts.values, color=["#2ca02c", "#d62728"])
    ax.set_title("Class Balance: Benign vs Malignant")
    ax.set_xlabel("Diagnosis")
    ax.set_ylabel("Number of Cases")
    for i, value in enumerate(counts.values):
        ax.text(i, value + 5, str(value), ha="center")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "class_balance.png", dpi=150)
    plt.close(fig)
    print("Saved class_balance.png")


def plot_feature_boxplot(df: pd.DataFrame, feature: str) -> None:

    benign = df[df["diagnosis_label"] == "benign"][feature]
    malignant = df[df["diagnosis_label"] == "malignant"][feature]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot([malignant, benign], tick_labels=["Malignant", "Benign"])
    ax.set_title(f"{feature.title()} by Diagnosis")
    ax.set_ylabel(feature.title())

    fig.tight_layout()
    filename = f"boxplot_{feature.replace(' ', '_')}.png"
    fig.savefig(FIGURES_DIR / filename, dpi=150)
    plt.close(fig)
    print(f"Saved {filename}")


def plot_feature_histogram(df: pd.DataFrame, feature: str) -> None:

    benign = df[df["diagnosis_label"] == "benign"][feature]
    malignant = df[df["diagnosis_label"] == "malignant"][feature]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(benign, bins=25, alpha=0.6, label="Benign", color="#2ca02c")
    ax.hist(malignant, bins=25, alpha=0.6, label="Malignant", color="#d62728")
    ax.set_title(f"Distribution of {feature.title()}")
    ax.set_xlabel(feature.title())
    ax.set_ylabel("Count")
    ax.legend()

    fig.tight_layout()
    filename = f"histogram_{feature.replace(' ', '_')}.png"
    fig.savefig(FIGURES_DIR / filename, dpi=150)
    plt.close(fig)
    print(f"Saved {filename}")


def plot_correlation_heatmap(df: pd.DataFrame, features: list) -> None:

    corr = df[features].corr()

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(features)))
    ax.set_yticks(range(len(features)))
    ax.set_xticklabels([f.title() for f in features], rotation=45, ha="right")
    ax.set_yticklabels([f.title() for f in features])
    ax.set_title("Feature Correlation Heatmap")
    fig.colorbar(im, ax=ax, label="Correlation")

    for i in range(len(features)):
        for j in range(len(features)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=150)
    plt.close(fig)
    print("Saved correlation_heatmap.png")


def run_eda(df: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plot_class_balance(df)
    plot_feature_boxplot(df, "mean radius")
    plot_feature_histogram(df, "mean texture")
    plot_correlation_heatmap(
        df,
        [
            "mean radius",
            "mean texture",
            "mean perimeter",
            "mean area",
            "mean smoothness",
            "diagnosis",
        ],
    )


if __name__ == "__main__":
    dataset = pd.read_csv(PROCESSED_DATA_PATH)
    run_eda(dataset)
