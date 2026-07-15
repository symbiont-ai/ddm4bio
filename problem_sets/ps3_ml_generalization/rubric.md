# PS3 Grading Rubric — Machine Learning and Generalization

Total: **100 points**. The autograder in `tests/` establishes the objective
floor; the remaining judgment is on whether the work is correct, honest, and
reproducible.

## Method correctness — 30 points

Does the machinery do what it claims?

- **Polynomial and NN fits (10 pts).** `fit_polynomial` returns a genuine
  fitted OLS polynomial of the requested degree; `fit_small_nn` scales inputs
  and trains a seeded MLP that converges. Both expose a working `predict`.
- **Interpolation vs. extrapolation (10 pts).** `interp_vs_extrap_error`
  evaluates against the known `true_function` and correctly separates the
  in-window (interpolation) region from the beyond-window (extrapolation)
  region. The polynomial's extrapolation error dwarfs its interpolation error;
  the NN degrades but does not explode.
- **CV and learning-curve wrappers (10 pts).** `kfold_cv` and
  `model_learning_curve` correctly delegate to the `ddm4bio.methods.learning`
  functions and return the documented keys and array shapes.

## Application execution — 25 points

Does the clinical model actually work?

- **Three distinct models (10 pts).** `build_models` returns `linear`, `nn`,
  and `regularized` pipelines that are genuinely different (unregularized vs.
  strongly penalized logistic regression, plus a shallow NN), each standardizing
  features first.
- **Performance clears the floor (10 pts).** Every model's cross-validated
  ROC-AUC exceeds the sensible floor (> 0.95) on the bundled breast-cancer
  fixture the autograder uses; the `get_dataset("heart_uci")` application in
  `main()` is a harder, real-world cohort where a more modest AUC is expected
  and honestly reported.
- **Performance vs. n (5 pts).** `performance_vs_n` produces a curve whose
  validation score rises with `n` and is high at the largest training size.

## Quality control — 25 points

QC is a first-class deliverable, not an afterthought.

- **QC before results (6 pts).** The `ddm4bio.qc` report is produced and
  inspected before any modeling; shape, missingness, and class balance are
  reported.
- **No leakage (7 pts).** `make_splits` yields a stratified train/validation/test
  partition with disjoint indices, verified by `assert_no_leakage`.
- **Generalization gap + calibration (7 pts).** `generalization_gap` reports
  train/val/test AUC with a small gap; `calibration_report` returns a reliability
  curve and a Brier score well below a coin flip.
- **Permutation test (5 pts).** `permutation_beats_chance` shows a small p-value
  (the observed score is significantly better than the shuffled-label null).

## Interpretation & honesty — 15 points

- **Interpretation block (7 pts).** Uses `interpret.interpretation_block` with a
  claim, a defensible confidence level, and evidence that cites the actual
  numbers (CV AUC, permutation p-value, gap, Brier) rather than vibes.
- **Honest limitations (8 pts).** Names real limitations — single dataset, no
  external cohort, the extrapolation failure from Part A, ranking vs. operating
  threshold — and answers "does it generalize?" and "at what `n` does overfitting
  stop?" directly. Overclaiming deployment readiness is penalized; a well-argued
  moderate-confidence answer scores full marks.

## Reproducibility — 5 points

- Runs offline, top to bottom, with no network or downloads.
- Seeded throughout (`seed_everything` / explicit `seed=`); results are stable
  across runs.
- `ruff check` and `ruff format --check` are clean under E, F, I with a
  100-character line limit; heavy imports (sklearn, scipy, pandas) live inside
  function bodies.
