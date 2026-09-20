"""
preprocessing.py

Selects features, drops leakage / redundant columns, splits the data into
train and test sets, and defines the preprocessing pipeline (imputation +
scaling + one-hot encoding) as a scikit-learn ColumnTransformer.

Important: the preprocessor is only ever *fit* here as a sanity check.
The raw train/test splits (not the transformed arrays) are what get saved
to data/processed/. In Phase 2, the ColumnTransformer returned by
build_preprocessor() is placed inside a full sklearn Pipeline together
with the model, so it is fit on the training fold only -> no data leakage.

Column decisions (documented, not silent):
  - 'alive' is dropped: it is a text version of the target ('survived'),
    so keeping it would leak the answer directly to the model.
  - 'who' and 'adult_male' are dropped: both are derived almost entirely
    from 'sex' and 'age', which we already keep as raw features.
  - 'class' is dropped: it duplicates 'pclass' (same information, text vs.
    numeric form).
  - 'embark_town' is dropped: it duplicates 'embarked' (full name vs. code).
  - 'deck' is dropped: ~77% missing, too sparse to impute reliably.
  - 'alone' is dropped: fully derivable from 'sibsp' + 'parch', which we
    already keep.
"""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_loader import TARGET_COLUMN, load_raw_data

PROCESSED_DIR = Path("data/processed")

NUMERIC_FEATURES = ["age", "fare", "sibsp", "parch"]
CATEGORICAL_FEATURES = ["pclass", "sex", "embarked"]
LEAKAGE_OR_REDUNDANT_COLUMNS = [
    "alive",
    "who",
    "adult_male",
    "class",
    "embark_town",
    "deck",
    "alone",
]


def select_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Drop leakage/redundant columns and split the frame into X and y."""
    df = df.drop(columns=LEAKAGE_OR_REDUNDANT_COLUMNS, errors="ignore")
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET_COLUMN]
    return X, y


def split_data(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42
):
    """Stratified train/test split, done before any preprocessing is fit."""
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def build_preprocessor() -> ColumnTransformer:
    """
    Build (but do not fit) the preprocessing ColumnTransformer:
      - numeric columns: median imputation, then standard scaling
      - categorical columns: most-frequent imputation, then one-hot encoding
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )
    return preprocessor


def save_splits(X_train, X_test, y_train, y_test) -> None:
    """Save the raw (unprocessed) train/test splits so Phase 2 can reuse them."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(PROCESSED_DIR / "X_train.csv", index=False)
    X_test.to_csv(PROCESSED_DIR / "X_test.csv", index=False)
    y_train.to_csv(PROCESSED_DIR / "y_train.csv", index=False)
    y_test.to_csv(PROCESSED_DIR / "y_test.csv", index=False)


def run_preprocessing() -> None:
    """End-to-end Phase 1 preprocessing: load, select, split, verify, save."""
    df = load_raw_data()
    X, y = select_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    print(f"Train set: {X_train.shape[0]} rows | Test set: {X_test.shape[0]} rows")

    # Sanity check: fit the preprocessor on the TRAIN split only, then confirm
    # it can transform both splits without error. This proves the pipeline
    # design is leak-free before any model is introduced in Phase 2.
    preprocessor = build_preprocessor()
    X_train_transformed = preprocessor.fit_transform(X_train, y_train)
    X_test_transformed = preprocessor.transform(X_test)
    print(f"Transformed train shape: {X_train_transformed.shape}")
    print(f"Transformed test shape:  {X_test_transformed.shape}")

    save_splits(X_train, X_test, y_train, y_test)
    print(f"Raw train/test splits saved to {PROCESSED_DIR}/")


if __name__ == "__main__":
    run_preprocessing()
