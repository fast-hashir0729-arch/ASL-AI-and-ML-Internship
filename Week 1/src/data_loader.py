"""
data_loader.py

Loads the Breast Cancer Wisconsin (Diagnostic) dataset, a public
tabular dataset bundled with scikit-learn, and saves it as a CSV
file so the rest of the project always works from a fixed,
version-controlled data file instead of re-downloading anything.

Dataset source: scikit-learn's built-in datasets
(sklearn.datasets.load_breast_cancer), originally from the UCI
Machine Learning Repository.
"""

from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer

# Paths are built relative to this file, so the script works on
# any machine without hard-coded absolute paths.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "breast_cancer.csv"


def load_dataset() -> pd.DataFrame:
    """Load the Breast Cancer Wisconsin dataset as a pandas DataFrame.

    Returns:
        A DataFrame with 30 numeric feature columns plus a
        'diagnosis' column (0 = malignant, 1 = benign).
    """
    raw = load_breast_cancer(as_frame=True)
    df = raw.frame
    df = df.rename(columns={"target": "diagnosis"})
    return df


def save_dataset(df: pd.DataFrame, path: Path = RAW_DATA_PATH) -> None:
    """Save a DataFrame to CSV, creating parent folders if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved dataset with {df.shape[0]} rows and {df.shape[1]} columns to {path}")


if __name__ == "__main__":
    dataset = load_dataset()
    save_dataset(dataset)
