"""
inspect_data.py

A first look at the raw dataset before any cleaning happens.
Reports shape, data types, missing values, duplicates, class
balance, and summary statistics. This helps decide what cleaning
steps (if any) are needed in Day 2.
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "breast_cancer.csv"


def inspect(df: pd.DataFrame) -> None:
    """Print a basic inspection report of the DataFrame."""
    print("=" * 60)
    print("SHAPE")
    print("=" * 60)
    print(f"{df.shape[0]} rows, {df.shape[1]} columns")

    print("\n" + "=" * 60)
    print("COLUMN DATA TYPES")
    print("=" * 60)
    print(df.dtypes)

    print("\n" + "=" * 60)
    print("MISSING VALUES PER COLUMN")
    print("=" * 60)
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.sum() > 0 else "No missing values found.")

    print("\n" + "=" * 60)
    print("DUPLICATE ROWS")
    print("=" * 60)
    print(df.duplicated().sum())

    print("\n" + "=" * 60)
    print("TARGET CLASS BALANCE (diagnosis)")
    print("=" * 60)
    print(df["diagnosis"].value_counts())

    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS (first 5 columns)")
    print("=" * 60)
    print(df.iloc[:, :5].describe())


if __name__ == "__main__":
    dataset = pd.read_csv(RAW_DATA_PATH)
    inspect(dataset)
