# Conclusion

This project built an end-to-end supervised classification pipeline to
predict Titanic passenger survival, comparing a Logistic Regression
baseline against a tuned Random Forest.

**Performance.** On 5-fold cross-validation over the training set, the Random
Forest scored slightly higher (F1 = 0.756) than the Logistic Regression
baseline (F1 = 0.728 ± 0.025) after GridSearchCV tuning (`max_depth=8`,
`n_estimators=100`, `min_samples_leaf=1`). On the held-out test set,
however, the result flipped: Logistic Regression reached 81.0% accuracy,
76.1% F1, 74.0% precision, and 78.3% recall, versus the Random Forest's
79.3% accuracy and 72.6% F1. The **Logistic Regression baseline is the
better choice here** — it generalized more consistently from cross-validation
to the test set, while the Random Forest's cross-validation edge did not
carry over. On a dataset this small (712 training rows), a simpler,
lower-variance model is a reasonable and slightly safer pick than a more
flexible one, even after tuning.

**What drove predictions.** Feature importance from the Random Forest shows
sex, fare, and passenger class as the strongest predictors of survival,
consistent with the historical "women and children first, first class
priority" evacuation pattern — a good sanity check that the model learned
something real rather than noise.

**Limitations.** The dataset is small (891 rows total) and the two models'
test-set scores differ by only a few points, so this comparison could shift
with a different random split. Age was imputed with a single overall
median rather than a class-aware estimate, which is a simplification.

**Next steps.** With more time, I would: (1) try a class-aware median
imputation for `age` (e.g., median age per `pclass`/`sex` group), (2) add
gradient boosting (e.g., XGBoost) as a third comparative model, and
(3) evaluate stability across several random train/test splits instead of
one, since the two models' scores are close enough that a single split
isn't fully conclusive.
