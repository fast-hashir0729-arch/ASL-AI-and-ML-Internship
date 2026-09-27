"""
Phase 1 — Preprocessing pipeline and train/test split.

All scaling/encoding logic lives inside a single ColumnTransformer so that
every model script wraps it in a scikit-learn Pipeline. Fitting happens only
on training folds (via cross_val_score / GridSearchCV / RandomizedSearchCV
internals), so no preprocessing statistic is ever leaked from validation or
test data — this satisfies the Week 4 "no leakage" requirement.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
)


def build_preprocessor() -> ColumnTransformer:
    """Build the numeric/categorical preprocessing pipeline.

    Numeric features: median-impute missing values (age has ~20% missing),
    then standardize. Categorical features: most-frequent-impute, then
    one-hot encode.
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


def get_feature_target_split(df: pd.DataFrame):
    """Split a cleaned DataFrame into X (features) and y (target).

    Categorical columns are cast to string dtype. Without this, a mix of
    int/bool/str columns (pclass, alone, sex, embarked) makes pandas hand
    the ColumnTransformer a block that numpy coerces to a single dtype,
    which breaks the categorical imputer/encoder.
    """
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    X[CATEGORICAL_FEATURES] = X[CATEGORICAL_FEATURES].astype(str)
    y = df[TARGET_COLUMN]
    return X, y


def train_test_split_stratified(X, y):
    """Stratified train/test split, reproducible via RANDOM_STATE."""
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
