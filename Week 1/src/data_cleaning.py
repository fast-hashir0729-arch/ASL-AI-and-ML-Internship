

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "breast_cancer.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "breast_cancer_clean.csv"

DIAGNOSIS_LABELS = {0: "malignant", 1: "benign"}


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    missing_count = int(df.isnull().sum().sum())
    duplicate_count = int(df.duplicated().sum())
    print(f"Missing values found: {missing_count}")
    print(f"Duplicate rows found: {duplicate_count}")

    if duplicate_count > 0:
        df = df.drop_duplicates()
        print(f"Dropped {duplicate_count} duplicate rows.")
    else:
        print("No duplicates to drop.")

    df["diagnosis_label"] = df["diagnosis"].map(DIAGNOSIS_LABELS)

    return df


def save_dataset(df: pd.DataFrame, path: Path = PROCESSED_DATA_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved cleaned dataset with {df.shape[0]} rows and {df.shape[1]} columns to {path}")


if __name__ == "__main__":
    raw_df = pd.read_csv(RAW_DATA_PATH)
    clean_df = clean_dataset(raw_df)
    save_dataset(clean_df)
