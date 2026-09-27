# Conclusion — Week 4: Model Tuning & Ensemble Comparison

This week extended the Week 3 Titanic survival classifier with stratified
5-fold cross-validation, systematic hyperparameter search, and two
ensemble models. Cross-validating the Week 3-style Logistic Regression
baseline and an untuned Random Forest first confirmed that a single
train/test split understates real variability: Logistic Regression scored
0.739 ± 0.019 mean F1 across folds, and Random Forest 0.757 ± 0.025 — a
noticeably wider spread than a single split would ever reveal.

GridSearchCV over Random Forest (n_estimators, max_depth,
min_samples_split, min_samples_leaf) raised its cross-validated F1 to
0.763, and RandomizedSearchCV over Gradient Boosting (40 sampled
configurations across learning_rate, max_depth, subsample, and
min_samples_leaf) reached 0.769 — the best cross-validated score of any
model tried. Because model selection was made on validation evidence
rather than the test set, Gradient Boosting was chosen as the final
model before it was ever evaluated on held-out data. On the untouched
test set it reached accuracy 0.816, precision 0.810, recall 0.681, and
F1 0.740 — slightly below the Week 3 baseline's single-split F1 of
0.761, which is a reasonable outcome: Week 3's number came from one
lucky-or-unlucky split, while this week's figure reflects the model most
consistently strong across five different splits, at some cost to this
particular test partition.

Sex, fare, age, and passenger class dominated feature importance for the
final model, corroborated by permutation importance on the test set,
consistent with the historical "women and children first, and first-class
passengers had better odds" account of the disaster. Class imbalance was
mild (62% did not survive vs. 38% who did), so `class_weight="balanced"`
was applied to Logistic Regression and Random Forest; Gradient Boosting
has no native class_weight parameter, and given the imbalance was modest,
metrics beyond accuracy (precision, recall, F1) were relied on instead of
adding sample-weighting complexity.

Next steps: try XGBoost/LightGBM for a stronger boosting baseline, run
nested cross-validation for a less optimistic performance estimate, and
engineer a family-size feature from sibsp/parch.
