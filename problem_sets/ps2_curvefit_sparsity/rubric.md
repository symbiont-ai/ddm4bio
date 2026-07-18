# PS2 Grading Rubric — Letting the Data Choose the Model

Total: **100 points**. Grading combines the autograder (`tests/test_ps2.py`)
with a short read of the submitted code and interpretation. All work must run
offline and deterministically; a submission that only passes because it hard-codes
expected numbers, disables a check, or scores a fold on data it was fit on earns no
credit for the affected part.

## Part A — Model-complexity selection — 30 points

- **`poly_cv_mse` (16).** Truly out-of-sample: for each fold, fits the polynomial
  on the complement and scores the held-out points; averages the MSE. A degree far
  above the true one generalizes *worse*, not better.
- **`select_degree` (14).** Returns the whole CV-MSE curve and the degree that
  minimizes it (the true argmax), and recovers the true complexity (± one degree)
  without chasing an overfit high degree.

## Part B — Sparse feature selection — 40 points

- **`lasso_cv_mse` (12).** Out-of-sample MSE of a Lasso at a given penalty (fit with
  the provided `lasso_fit`, predict `X @ coef + intercept`); an over-large penalty
  generalizes worse.
- **`select_alpha` (10).** Returns the CV-MSE curve and the penalty that minimizes
  it; the strongest penalty is not chosen (it underfits).
- **`selected_features` (8).** Correct nonzero-coefficient indices; fewer features
  as the penalty grows, and the true drivers are (nearly) all recovered at a modest
  penalty.
- **`support_scores` (10).** Correct precision and recall against a known driver set
  (verified on hand-checked inputs), with empty sets handled without crashing.

## Quality control — 15 points

- **Out-of-sample discipline (9).** The folds are disjoint and complete (each row
  held out exactly once); no CV score is computed on data the model was fit on. QC
  is printed *before* any result, per the course "QC before results" rule.
- **Honest data (6).** The synthetic recovery is scored against known truth; the
  real WDBC panel is read qualitatively (no ground-truth driver set), and the
  dataset is loaded through the course data layer.

## Interpretation & honesty — 10 points

- A clear `interpretation_block` -- a claim stated with the evidence
  actually generated (the U-shaped CV curves, disjoint-and-complete folds,
  precision/recall vs a known support).
- At least two honest, specific limitations — especially that CV-tuned Lasso
  **over-selects** (high recall, lower precision), that a single CV estimate depends
  on the fold seed, and that the real panel has no ground truth to score against.
  Overclaiming is penalized; calibrated honesty is rewarded.

## Reproducibility — 5 points

- Runs top-to-bottom with a fixed seed; imports follow the repository policy (only
  `numpy` at module top level, heavier libraries inside function bodies); the file
  is `ruff`-clean under `E`, `F`, `I` at line length 100; and no public function
  signature was changed.
