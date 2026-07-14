# PS1 Grading Rubric — Eigen-Recognition

Total: **100 points**. Grading combines the autograder (`tests/test_ps1.py`)
with a short read of the submitted code and interpretation. All work must run
offline and deterministically; a submission that only passes because it hard-
codes expected numbers, disables a check, or peeks at the test split earns no
credit for the affected part.

## Method correctness — 30 points

- **Direct and iterative solvers (10).** `solve_direct` uses an LU
  factorization and returns a machine-precision solution; `solve_iterative`
  runs conjugate gradient, returns `(x, n_iters)`, and genuinely counts CG
  iterations via a callback rather than reporting a constant.
- **Eigen-image pipeline (14).** `eigen_basis` correctly mean-centers and
  extracts the top right singular vectors (orthonormal rows); `project` and
  `reconstruct` are true inverses on a full-rank basis. `compare_solvers`
  reports a correct condition number and per-method residuals.
- **Variance accounting (6).** `modes_for_variance` returns the smallest mode
  count reaching each cumulative-variance threshold, non-decreasing in the
  threshold and capped at the rank.

## Application execution — 25 points

- **Reconstruction curve (12).** `reconstruction_error_curve` fits on train,
  evaluates on the held-out split, is monotonically non-increasing in the number
  of modes, and reaches ≈ 0 at full rank.
- **Classifier (13).** `eigen_nn_accuracy` builds the basis on training data,
  projects both splits, and evaluates a 1-NN classifier whose test accuracy is
  well above the 0.10 chance baseline (threshold: > 0.90).

## Quality control — 25 points

- **Orthonormality (8).** `check_orthonormality` returns correct off-diagonal
  and norm-deviation measures and flags a non-orthonormal basis.
- **Leakage safety (9).** The basis is fit on training data only; the leakage
  guard passes and the student explains why train-only fitting is required.
- **Convergence & balance (8).** The full-rank reconstruction check reports
  ~machine-zero error, and the class-balance QC report is produced and read
  before any result is discussed. QC must appear *before* results, per the
  course "QC before results" rule.

## Interpretation & honesty — 15 points

- A clear `interpretation_block` with an explicit confidence level backed by the
  evidence actually generated (orthonormal basis, monotone convergence,
  leakage-checked split, accuracy vs. chance).
- At least two honest, specific limitations — e.g. that `load_digits` is clean
  and low-resolution, that a single held-out accuracy is not cross-validated, or
  that a linear basis misses nonlinear structure. Overclaiming ("this proves
  cell recognition works") is penalized; calibrated honesty is rewarded.

## Reproducibility — 5 points

- Runs top-to-bottom offline with a fresh seed; imports follow the repository
  policy (only `numpy` at module top level, heavier libraries inside function
  bodies); the file is `ruff`-clean under `E`, `F`, `I` at line length 100; and
  no public function signature was changed.
