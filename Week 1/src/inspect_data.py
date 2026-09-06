
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "breast_cancer.csv"


def inspect(df: pd.DataFrame) -> None:
    print("SHAPE")
    print(f"{df.shape[0]} rows, {df.shape[1]} columns")

    print("COLUMN DATA TYPES")
    print(df.dtypes)

    print("MISSING VALUES PER COLUMN")
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.sum() > 0 else "No missing values found.")

    print("DUPLICATE ROWS")
    print(df.duplicated().sum())
    print("TARGET CLASS BALANCE (diagnosis)")
    print(df["diagnosis"].value_counts())

    print("SUMMARY STATISTICS (first 5 columns)")
    print(df.iloc[:, :5].describe())


if __name__ == "__main__":
    dataset = pd.read_csv(RAW_DATA_PATH)
    inspect(dataset)
