# AI/ML Internship — Week 3: Supervised Learning on the Titanic Dataset

## Project Objective

An end-to-end supervised machine learning project that predicts Titanic
passenger survival. The project follows a leakage-free workflow: load data,
check its quality, split into train/test, preprocess, train and compare two
models, evaluate them, and document the results.

**Status: Phase 1 complete** (data loading, quality check, train/test split,
preprocessing pipeline). Model training, evaluation, and reporting are
Phase 2.

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
│   └── preprocessing.py        # feature selection, split, ColumnTransformer
├── models/                     # saved fitted model (Phase 2)
├── outputs/
│   ├── figures/                # charts (Phase 2)
│   └── metrics/
│       └── data_quality_report.txt
├── reports/                    # short PDF/DOCX report (Phase 2)
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

## How to Run (Phase 1)

Run each step from the project root, in order:

```bash
python -m src.data_loader        # sanity-check the raw dataset loads
python -m src.data_quality       # writes outputs/metrics/data_quality_report.txt
python -m src.preprocessing      # splits data, verifies the pipeline, saves splits
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

## Implemented So Far (Phase 1)

- [x] Raw data loading with validation
- [x] Data-quality check (shape, dtypes, missing values, duplicates, class
      balance), saved as a text report
- [x] Feature selection with documented column decisions
- [x] Stratified train/test split (80/20, `random_state=42`), done **before**
      any preprocessing is fit
- [x] Preprocessing pipeline: median/most-frequent imputation → scaling /
      one-hot encoding, via `ColumnTransformer`
- [x] Verified the pipeline fits on train only and transforms both splits
      without error

## Coming in Phase 2

- [ ] Baseline model (Logistic Regression) + comparative model (Random
      Forest), each wrapped in a full `Pipeline` with the preprocessor
- [ ] Evaluation: accuracy, precision, recall, F1, confusion matrix
- [ ] Model comparison table and final choice
- [ ] Bonus: cross-validation, hyperparameter search, feature importance
- [ ] Save the final fitted model with `joblib`
- [ ] Written conclusion (200–350 words)
- [ ] Short PDF/DOCX report

## Known Limitations

- The dataset is small (891 rows) relative to many production ML datasets,
  which limits how confidently results generalize.
- `age` imputation uses a simple median strategy rather than a
  group-based estimate (e.g., median age per passenger class), which is
  a reasonable simplification for a baseline project.
