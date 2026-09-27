"""
Phase 1 — Load the Week 3 dataset and re-run a concise data-quality check.

The Titanic dataset and cleaning decisions are reused unchanged from Week 3
(the assignment asks us to reuse the Week 3 dataset/pipeline). This script
just documents that the raw data still looks the way Week 3 left it before
we build on top of it.
"""

import pandas as pd

from config import COLUMNS_TO_DROP, RAW_DATA_PATH, TARGET_COLUMN


def load_raw_data() -> pd.DataFrame:
    """Load the raw Titanic CSV into a DataFrame."""
    df = pd.read_csv(RAW_DATA_PATH)
    return df


def quality_check(df: pd.DataFrame) -> None:
    """Print a short data-quality summary (shape, dtypes, missing values)."""
    print("Shape:", df.shape)
    print("\nMissing values per column:")
    print(df.isna().sum().sort_values(ascending=False))
    print("\nTarget distribution ('survived'):")
    print(df[TARGET_COLUMN].value_counts(normalize=True).round(3))


def drop_redundant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns identified in Week 3 as redundant or leaky (see config.py)."""
    return df.drop(columns=COLUMNS_TO_DROP)


if __name__ == "__main__":
    raw_df = load_raw_data()
    quality_check(raw_df)
    trimmed_df = drop_redundant_columns(raw_df)
    print("\nColumns kept for modeling:", list(trimmed_df.columns))
