# AI/ML Internship — Week 3: Supervised Learning on the Titanic Dataset

## Project Objective

An end-to-end supervised machine learning project that predicts Titanic
passenger survival. The project follows a leakage-free workflow: load data,
check its quality, split into train/test, preprocess, train and compare two
models, evaluate them, and document the results.

**Status: Complete** (Phase 1: data loading, quality check, train/test
split, preprocessing pipeline. Phase 2: model training, tuning,
evaluation, and reporting).

## Dataset

- **Source:** Titanic passenger data, from the public
  [seaborn-data](https://github.com/mwaskom/seaborn-data) repository
  (`titanic.csv`).
- **Size:** 891 rows, 15 raw columns.
- **Target:** `survived` (0 = did not survive, 1 = survived) — binary
  classification.

## Project Structure

```
asl-internship-aiml-week3-hashir/
├── data/
│   ├── raw/                    # original, untouched dataset
│   │   └── titanic_raw.csv
│   └── processed/              # train/test splits (generated, gitignored content is raw only)
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       └── y_test.csv
├── src/
│   ├── data_loader.py          # loads + validates the raw dataset
│   ├── data_quality.py         # missing values, duplicates, class balance
│   ├── preprocessing.py        # feature selection, split, ColumnTransformer
│   ├── model_pipeline.py       # combines preprocessor + model into one Pipeline
│   ├── train_model.py          # trains baseline + tuned comparative model
│   └── evaluate.py             # test-set metrics, plots, model comparison
├── models/
│   ├── baseline_logistic_regression.joblib
│   ├── comparative_random_forest.joblib
│   └── final_model.joblib      # best model by test-set F1
├── outputs/
│   ├── figures/
│   │   ├── logistic_confusion_matrix.png
│   │   ├── random_confusion_matrix.png
│   │   ├── model_comparison.png
│   │   └── feature_importance.png
│   └── metrics/
│       ├── data_quality_report.txt
│       ├── training_summary.json
│       ├── evaluation_report.txt
│       └── model_comparison.csv
├── reports/
│   ├── conclusion.md
│   └── Week3_Report.docx
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## How to Run

Run each step from the project root, in order:

```bash
python -m src.data_loader        # sanity-check the raw dataset loads
python -m src.data_quality       # writes outputs/metrics/data_quality_report.txt
python -m src.preprocessing      # splits data, verifies the pipeline, saves splits
python -m src.train_model        # trains baseline + tuned comparative model
python -m src.evaluate           # test-set metrics, plots, picks the final model
```

## Data Quality Notes (documented decisions)

- `age` is missing in ~20% of rows and `embarked` in 2 rows — both are
  median/most-frequent imputed **inside the pipeline**, fit on the training
  split only, so no information from the test set leaks into training.
- `deck` is ~77% missing and is dropped rather than imputed — too sparse to
  fill reliably.
- 107 duplicate rows were found by `data_quality.py`. They are **not**
  dropped: the dataset has no passenger ID column, so two passengers who
  happen to share age, fare, class, etc. look identical on paper without
  actually being the same record. Treating them as errors would silently
  remove real passengers.
- Columns `alive`, `who`, `adult_male`, `class`, `embark_town`, and `alone`
  are dropped before modeling — each either leaks the target directly
  (`alive`) or duplicates information already kept in another feature (full
  reasoning in `src/preprocessing.py`).

## Features Used

- **Numeric:** `age`, `fare`, `sibsp`, `parch`
- **Categorical:** `pclass`, `sex`, `embarked`

## Models & Results

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression (baseline) | 0.810 | 0.740 | 0.783 | **0.761** |
| Random Forest (tuned, comparative) | 0.793 | 0.742 | 0.710 | 0.726 |

**Chosen model: Logistic Regression (baseline)** — it had the higher
test-set F1 and generalized more consistently from cross-validation to the
test set than the tuned Random Forest. Full reasoning is in
`reports/conclusion.md`.

- 5-fold cross-validation (baseline): F1 = 0.728 (± 0.025)
- GridSearchCV best params (Random Forest): `n_estimators=100`,
  `max_depth=8`, `min_samples_leaf=1`
- Both models use `class_weight="balanced"` to address the mild (62/38)
  class imbalance found during the data-quality check

See `outputs/figures/` for confusion matrices, the model comparison chart,
and the Random Forest feature-importance plot, and
`outputs/metrics/evaluation_report.txt` for the full numeric report.

## Implemented Features

- [x] Raw data loading with validation
- [x] Data-quality check (shape, dtypes, missing values, duplicates, class
      balance), saved as a text report
- [x] Feature selection with documented column decisions
- [x] Stratified train/test split (80/20, `random_state=42`), done **before**
      any preprocessing is fit
- [x] Preprocessing pipeline: median/most-frequent imputation → scaling /
      one-hot encoding, via `ColumnTransformer`
- [x] Baseline model (Logistic Regression) + comparative model (Random
      Forest), each wrapped in a full `Pipeline` with the preprocessor
- [x] Evaluation: accuracy, precision, recall, F1, confusion matrix
- [x] Model comparison table and final choice
- [x] Save the final fitted model with `joblib`
- [x] Written conclusion (200–350 words)
- [x] Short DOCX report

## Bonus Features Implemented

- [x] Cross-validation (5-fold, on the baseline model)
- [x] Hyperparameter search (`GridSearchCV` on the Random Forest)
- [x] Feature importance visualization (Random Forest)
- [x] Class imbalance handling (`class_weight="balanced"` on both models)

## Known Limitations

- The dataset is small (891 rows) relative to many production ML datasets,
  and the two models' test-set F1 scores are close enough (0.761 vs. 0.726)
  that the ranking could shift with a different random split.
- `age` imputation uses a simple median strategy rather than a
  group-based estimate (e.g., median age per passenger class), which is
  a reasonable simplification for a baseline project.
- No GUI, so the "screenshots" requested in the submission guide are the
  generated evaluation plots in `outputs/figures/`, which serve as visual
  evidence the pipeline runs end-to-end.
