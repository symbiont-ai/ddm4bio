# PS3 — Machine Learning and Generalization

**Reading:** Kutz, *Data-Driven Modeling & Scientific Computation for the Life
Sciences*, Chapter 6 (neural networks) and Chapter 13 (regression, model
selection, and cross-validation).

A model that scores well on the data it was trained on has told you almost
nothing. The only question that matters for a deployed model is whether it works
on data it has never seen. This problem set is about that question from two
angles: first on a synthetic function where you know the ground truth exactly,
then on a real clinical dataset where you do not. Along the way you will build
the honest machinery — cross-validation, learning curves, a leakage-free
train/validation/test split, calibration, and a permutation test — that lets you
say *how confident* you should be, and why.

You will implement your functions in `student/ps3.py`. The public function
signatures are fixed; the autograder in `tests/test_ps3.py` imports them by name
and checks both their return shapes and their performance against thresholds.

## Data (everything offline)

No downloads, no network, no credentials.

- **Part A** uses a synthetic 1-D function generated locally,
  `true_function(x) = sin(1.5·x) + 0.3·x`, sampled with additive Gaussian noise
  by `generate_synthetic_curve(...)`. Both are already implemented for you so the
  ground truth is fixed and shared across the two models.
- **Parts B–C** use scikit-learn's bundled `load_breast_cancer` dataset (569
  samples, 30 numeric features, a binary malignant/benign target). It ships
  inside scikit-learn, so it loads offline. The loader `load_clinical_data()`
  is provided.

Everything is seeded with `GLOBAL_SEED = 20260714`; call
`ddm4bio.seed_everything()` before any stochastic step (the provided `main()`
already does).

## Part A — Method: interpolation vs. extrapolation

Fit **two** models to the *same* noisy samples of `true_function` on the window
`[0, 2π]`:

1. `fit_polynomial(X, y, degree)` — an ordinary least-squares polynomial
   (degree 9 in the demo).
2. `fit_small_nn(X, y, hidden_layer_sizes, seed)` — a small multilayer
   perceptron (scale the inputs first; seed it for determinism).

Then quantify the difference between **interpolation** and **extrapolation** with
`interp_vs_extrap_error(model, x_low, x_high, ...)`: draw fresh points *inside*
the training window and fresh points *just beyond* it, and report the mean
squared error of each against the known true function. You should see the
polynomial fit beautifully inside the window and then diverge violently outside
it, while the bounded neural network degrades far more gently. This is the
generalization lesson in miniature: low training error tells you nothing about
behavior outside the domain you trained on.

Finally, implement the two evaluation tools you will reuse in Part B by wrapping
the course library:

- `kfold_cv(estimator, X, y, cv, scoring, seed)` → `ddm4bio.methods.learning.cross_validate`.
- `model_learning_curve(estimator, X, y, train_sizes, cv, seed)` →
  `ddm4bio.methods.learning.learning_curve`.

## Part B — Application: predict a clinical outcome

On `load_breast_cancer`, build three classifiers in `build_models(seed)`, each a
pipeline that standardizes the features first:

- `"linear"` — a nearly **unregularized** logistic regression (very large `C`).
- `"nn"` — a **shallow** neural network (`MLPClassifier`, one small hidden layer).
- `"regularized"` — a **strongly penalized** logistic regression (small `C`).

Report each model's cross-validated ROC-AUC with your `kfold_cv`. Then use
`performance_vs_n(...)` (built on your learning-curve wrapper) to trace how
validation performance grows as the number of training examples `n` increases —
the empirical answer to "how much data is enough?"

## Part C — Quality control (required)

Quality control is not optional and comes *before* you trust any number.

1. **QC report first.** `run_tabular_qc(X, y, feature_names)` (provided) prints a
   `ddm4bio.qc` report — shape, missingness, duplicates, class balance, outliers —
   before any modeling. No result without QC.
2. **No leakage.** Implement `make_splits(...)` to produce a stratified
   train/validation/test partition and verify with
   `ddm4bio.qc.report.assert_no_leakage` that no sample appears in two splits.
3. **Generalization gap.** `generalization_gap(estimator, splits)` fits on train
   only and reports train, validation, and test AUC plus the train-minus-val
   gap — your estimate of optimism.
4. **Calibration.** `calibration_report(...)` returns a reliability curve and a
   Brier score: are the predicted probabilities honest, or merely well-ranked?
5. **Beats chance.** `permutation_beats_chance(...)` wraps
   `ddm4bio.methods.learning.permutation_test` to show the cross-validated score
   is significantly better than the score under randomly shuffled labels.

## Part D — Interpretation & confidence

Close with an interpretation block built via
`ddm4bio.interpret.interpretation_block(...)` (the provided `main()` shows the
shape). In your own words, answer:

- **Does it generalize?** Cite the CV AUC, the permutation p-value, the
  train/val/test gap, and the Brier score — not just the headline number.
- **At what `n` does overfitting stop mattering?** Read it off your learning
  curve: where does the validation score plateau?
- **Deployment confidence.** State a confidence level (`low`/`moderate`/`high`)
  and name the real limitations — including what Part A taught you about
  extrapolation beyond the observed feature range.

## What to submit

Only edit `student/ps3.py`. Run it directly (`python student/ps3.py`) to see the
full pipeline once your TODOs are filled in, and run the autograder with
`pytest tests/`.
