"""
data_loader.py

Loads the raw Titanic dataset from data/raw/ and performs light validation
(file exists, not empty, expected target column present).

Dataset: Titanic passenger data (public, sourced from the seaborn-data
GitHub repository: https://github.com/mwaskom/seaborn-data). 891 rows,
15 columns. Target column: 'survived' (0 = did not survive, 1 = survived).
"""

from pathlib import Path

import pandas as pd

RAW_DATA_PATH = Path("data/raw/titanic_raw.csv")
TARGET_COLUMN = "survived"


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at '{path}'. Make sure the CSV is in data/raw/."
        )

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("Loaded dataset is empty.")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Expected target column '{TARGET_COLUMN}' not found in dataset."
        )

    return df


if __name__ == "__main__":
    data = load_raw_data()
    print(f"Loaded {data.shape[0]} rows and {data.shape[1]} columns.")
    print(data.head())
