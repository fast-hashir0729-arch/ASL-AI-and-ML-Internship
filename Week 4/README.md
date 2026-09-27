# ASL AI/ML Internship — Week 4: Model Tuning & Ensemble Comparison

**Author:** Hashir Ahmed
**Track:** AI / Machine Learning, Week 4 of 6
**Dataset:** Titanic passenger data (reused from Week 3), via the
[seaborn-data](https://github.com/mwaskom/seaborn-data) repository —
target column: `survived`

## Project Objective

Extend the Week 3 supervised-learning baseline by:
- Cross-validating models with stratified k-fold instead of a single train/test split
- Running a systematic hyperparameter search (GridSearchCV / RandomizedSearchCV)
- Training ensemble models (Random Forest, Gradient Boosting)
- Selecting a final model from cross-validation evidence, then confirming it once on a held-out test set
- Comparing everything against the Week 3 baseline in one results table

## What Was Built

- A reusable preprocessing pipeline (median/most-frequent imputation +
  scaling/one-hot-encoding via `ColumnTransformer`) wrapped in every model's
  `Pipeline`, so no scaling/encoding statistic ever leaks from a validation
  or test fold into training.
- 5-fold `StratifiedKFold` cross-validation for Logistic Regression and
  Random Forest, reporting mean ± standard deviation per metric.
- `GridSearchCV` over a 4-parameter Random Forest grid (108 candidates × 5 folds).
- `RandomizedSearchCV` over a 5-parameter Gradient Boosting space (40 sampled
  configurations × 5 folds).
- Final model selection using cross-validated F1 only; the test set is
  touched exactly once, at the very end, to report an unbiased estimate.
- Feature importance (built-in, tree-based) **and** permutation importance
  on the test set, as a bonus cross-check.
- A simple CSV experiment log (`outputs/experiment_log.csv`) that
  accumulates one row per model per pipeline run.
- Final tuned model saved with `joblib`, alongside its chosen
  hyperparameters and top features in `models/best_hyperparameters.json`.

## Results Summary

| Model | CV F1 (mean ± std) | Test Accuracy | Test Precision | Test Recall | Test F1 |
|---|---|---|---|---|---|
| Logistic Regression (Week 3 baseline, single split) | — | — | — | — | 0.761 |
| Logistic Regression (Week 4, 5-fold CV) | 0.739 ± 0.019 | 0.793 | 0.722 | 0.754 | 0.738 |
| Random Forest (Week 4, untuned, 5-fold CV) | 0.757 ± 0.025 | 0.805 | 0.774 | 0.696 | 0.733 |
| Random Forest (GridSearchCV-tuned) | 0.763 | 0.793 | 0.750 | 0.696 | 0.722 |
| **Gradient Boosting (RandomizedSearchCV-tuned) — final model** | **0.769** | 0.816 | 0.810 | 0.681 | 0.740 |

Full table: `outputs/model_comparison.csv`. Selection was made on the
cross-validated F1 column (validation evidence), not the test columns.

**Chosen hyperparameters (Gradient Boosting):** `n_estimators=197`,
`learning_rate=0.080`, `max_depth=3`, `min_samples_leaf=5`,
`subsample=0.890` — full record in `models/best_hyperparameters.json`.

**Top features** (built-in importance, confirmed by permutation
importance): `sex`, `fare`, `age`, `pclass`.

**Class imbalance:** the target is mildly imbalanced (62% did not
survive vs. 38% who did). `class_weight="balanced"` was applied to
Logistic Regression and Random Forest; scikit-learn's
`GradientBoostingClassifier` has no native `class_weight` parameter, and
given the imbalance is modest, precision/recall/F1 were monitored
directly instead of adding sample-weighting.

## Setup & Run Instructions

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full pipeline (loads data, cross-validates, tunes,
#    evaluates, saves the final model, logs experiments)
cd src
python run_all.py
```

Individual phases can also be run on their own, in this order:
`load_data.py` → `cross_validate_models.py` → `tune_hyperparameters.py`
→ `evaluate_compare.py` → `log_experiments.py`.

Outputs land in `outputs/` (comparison table, figures, experiment log)
and `models/` (final fitted pipeline + hyperparameter record).

## Folder Structure

```
asl-internship-aiml-week4-hashirahmed/
├── data/
│   ├── raw/                     # titanic.csv (unmodified source)
│   └── processed/                # reserved for a cleaned export, if added later
├── src/
│   ├── config.py                 # paths, constants, feature lists
│   ├── load_data.py               # Phase 1: load + quality check
│   ├── preprocess.py              # ColumnTransformer + train/test split
│   ├── cross_validate_models.py   # Phase 2: 5-fold CV, baseline vs Random Forest
│   ├── tune_hyperparameters.py    # Phase 3: GridSearchCV + RandomizedSearchCV
│   ├── evaluate_compare.py        # Phase 4: test evaluation, comparison table, plots, save model
│   ├── log_experiments.py         # Bonus: CSV experiment log + permutation importance
│   └── run_all.py                 # Runs every phase end-to-end
├── models/
│   ├── final_tuned_model.joblib   # Saved final Pipeline (preprocessing + Gradient Boosting)
│   └── best_hyperparameters.json  # Chosen hyperparameters, CV/test metrics, feature importances
├── outputs/
│   ├── figures/                   # cv_f1_comparison.png, confusion_matrix, feature_importance
│   ├── model_comparison.csv       # Full results table
│   ├── cv_baseline_results.json
│   └── experiment_log.csv         # Bonus: appended experiment log
├── reports/
│   ├── Week4_Report.docx          # 1-2 page report for submission
│   └── conclusion.md              # 200-350 word conclusion
├── requirements.txt
├── .gitignore
└── README.md
```

## Implemented Features

- [x] Reused Week 3 dataset and preprocessing, improved with imputation
- [x] Stratified k-fold cross-validation on two models
- [x] Hyperparameter search (GridSearchCV + RandomizedSearchCV) — both used, on different models
- [x] Ensemble models: Random Forest and Gradient Boosting
- [x] All preprocessing inside a Pipeline — no leakage during tuning
- [x] CV mean ± standard deviation reported per model
- [x] Comparison table against the Week 3 baseline
- [x] Feature importance (built-in) for the best model
- [x] Class-imbalance strategy applied and justified
- [x] Final model saved via joblib with recorded hyperparameters
- [x] 200-350 word conclusion

**Bonus features implemented:**
- [x] Permutation importance alongside built-in feature importances
- [x] Simple CSV experiment log

**Bonus not implemented:** XGBoost/LightGBM (not available in this
environment) and nested cross-validation (left as a next step — noted
in the conclusion).

## Known Limitations

- `age` is imputed with the median (≈20% missing) rather than a more
  sophisticated method (e.g. regression imputation) — a simplification
  consistent with Week 3.
- Gradient Boosting has no native `class_weight`; with only mild
  imbalance this was judged an acceptable trade-off rather than adding
  sample-weighting.
- The Week 3 baseline row in the comparison table is a reported figure
  from that week's report, not recomputed in this codebase — the
  dataset, split convention (`random_state=42`, 80/20 stratified), and
  target are identical, so it remains a fair comparison point.

## Git Commit History

Work was completed and committed across three days, mirroring the
Week 1–2 convention:

- **Day 1:** project scaffold, environment setup, data loading/quality
  check, preprocessing pipeline
- **Day 2:** cross-validation of baseline models, hyperparameter search
  for Random Forest and Gradient Boosting
- **Day 3:** final evaluation, model comparison, feature importance,
  experiment logging, final model saved, report and README completed
