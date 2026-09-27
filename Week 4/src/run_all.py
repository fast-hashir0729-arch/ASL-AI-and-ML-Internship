"""
Runs the full Week 4 pipeline end-to-end, in order:

  1. load_data       -- quality check on the reused Week 3 dataset
  2. cross_validate_models -- 5-fold CV for the two baseline models
  3. tune_hyperparameters  -- GridSearchCV (Random Forest) +
                              RandomizedSearchCV (Gradient Boosting)
  4. evaluate_compare -- test-set evaluation, comparison table, plots,
                          final model + hyperparameters saved
  5. log_experiments  -- experiment CSV log + permutation importance

Usage:
    python run_all.py
"""

import load_data
import evaluate_compare
import log_experiments


def main():
    print("=" * 70)
    print("PHASE 1 — Load data & quality check")
    print("=" * 70)
    raw_df = load_data.load_raw_data()
    load_data.quality_check(raw_df)

    print("\n" + "=" * 70)
    print("PHASES 2-4 — Cross-validation, tuning, evaluation & comparison")
    print("=" * 70)
    evaluate_compare.run()

    print("\n" + "=" * 70)
    print("BONUS — Experiment log & permutation importance")
    print("=" * 70)
    log_experiments.append_experiment_log()
    log_experiments.compute_permutation_importance()

    print("\nPipeline complete. See outputs/ and models/ for artifacts.")


if __name__ == "__main__":
    main()
