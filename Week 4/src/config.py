"""
Project-wide constants and file paths.

Keeping these in one place avoids magic numbers/strings scattered across
scripts and makes the pipeline easy to re-point at a different dataset
or output location.
"""

from pathlib import Path

# --- Paths -------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "titanic.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "titanic_clean.csv"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
RESULTS_TABLE_PATH = PROJECT_ROOT / "outputs" / "model_comparison.csv"
EXPERIMENT_LOG_PATH = PROJECT_ROOT / "outputs" / "experiment_log.csv"
FINAL_MODEL_PATH = MODELS_DIR / "final_tuned_model.joblib"
BEST_PARAMS_PATH = MODELS_DIR / "best_hyperparameters.json"

# --- Reproducibility -----------------------------------------------------
RANDOM_STATE = 42
N_FOLDS = 5
TEST_SIZE = 0.2

# --- Target / feature definitions --------------------------------------
TARGET_COLUMN = "survived"

# Columns dropped before modeling (documented rationale, carried over from
# Week 3): 'alive' duplicates the target as text, 'class' duplicates
# 'pclass', 'who'/'adult_male' leak near-duplicate information derived
# from 'sex'/'age', 'deck' is >75% missing, and 'embark_town' duplicates
# 'embarked'.
COLUMNS_TO_DROP = ["alive", "class", "who", "adult_male", "deck", "embark_town"]

NUMERIC_FEATURES = ["age", "sibsp", "parch", "fare"]
CATEGORICAL_FEATURES = ["pclass", "sex", "embarked", "alone"]
