# Problem Set 1 — Eigen-Recognition: Linear Systems and the SVD

**Reading:** Kutz, *Data-Driven Modeling & Scientific Computation*, Chapter 2
(linear systems, Gaussian elimination / LU, iterative solvers, and the singular
value decomposition). Skim the sections on conditioning and on low-rank
approximation before you start.

This problem set has two threads that meet in the same idea. First you will
solve a linear system `A x = b` two different ways and watch how the *method*
you choose changes the cost and the numerical behaviour. Then you will use the
singular value decomposition to build an "eigen-image" basis — the same machine
behind eigenfaces — and turn it into a working recognizer for handwritten
digits, which we treat here as a stand-in for cheap, offline "eigen-cells."

Everything is small, deterministic, and runs offline. Fill in the method logic
in `student/ps1.py`; the imports, data loading, and quality-control plumbing are
already wired for you. The autograder in `tests/` checks both the interfaces and
numerical thresholds.

## Data

No downloads, no network. Two offline sources:

- **Synthetic SPD system** — `make_spd_system(n)` (provided) builds a symmetric
  positive-definite matrix `A = M Mᵀ + n·I` and a right-hand side `b`. Symmetry
  and positive-definiteness are what let conjugate gradient apply.
- **`sklearn.datasets.load_digits`** — the bundled 8×8 handwritten-digit images
  (1797 samples, 64 pixel-features, 10 classes) shipped inside scikit-learn.
  `load_digits_split()` (provided) returns a deterministic, stratified
  train/test split. We use it as a small, honest proxy for an imaging-based
  cell-recognition task: low-resolution grayscale images with known labels.

Seed everything through `ddm4bio.seed_everything()` (already called in `main`).

## Part A — Method: two solvers and the eigen-image pipeline

1. **Direct vs. iterative solve.** Implement `solve_direct` (an LU
   factorization via `scipy.linalg.lu_factor` / `lu_solve`) and
   `solve_iterative` (conjugate gradient via `scipy.sparse.linalg.cg`, counting
   iterations with a callback). Then implement `compare_solvers`, which reports
   the condition number of `A`, the residual `‖A x − b‖` for each solution, the
   CG iteration count, the wall-clock time of each method, and whether the two
   solutions agree. Understand *why* CG converges in relatively few iterations
   here and how that would change if `A` were badly conditioned.

2. **Eigen-image pipeline.** Implement the three core steps of a linear
   recognizer: `eigen_basis` (subtract the per-feature mean, then take the top
   `n_modes` right singular vectors via `svd_lowrank` — these are the
   eigen-images), `project` (express centered data in that basis), and
   `reconstruct` (map basis coordinates back to pixel space). The basis rows
   must be orthonormal, and projection followed by reconstruction with a
   full-rank basis must return the original data.

## Part B — Application: eigen-cells on digits

1. Implement `reconstruction_error_curve`: fit the eigen-basis on the **training**
   split only, then measure the relative-L2 reconstruction error on the
   **held-out test** split as the number of modes grows. The curve should fall
   monotonically and reach ≈ 0 at full rank.

2. Implement `eigen_nn_accuracy`: fit the basis on training data, project both
   splits into the eigen-basis, and evaluate a 1-nearest-neighbour classifier on
   the test split. With a modest number of modes this should land far above the
   0.10 random-guess baseline for ten classes.

## Part C — Quality control (required)

The provided `run_qc` prints a QC block **before** any results, and you supply
the pieces it needs:

- **Orthonormality.** Implement `check_orthonormality`, which compares the Gram
  matrix `components · componentsᵀ` to the identity and returns the largest
  off-diagonal entry and the largest deviation of a row norm from one.
- **Reconstruction convergence.** `run_qc` confirms that a full-rank basis drives
  the held-out reconstruction error to ~machine-zero — evidence the pipeline is
  wired correctly.
- **Leakage.** `run_qc` calls `ddm4bio.qc.assert_no_leakage` on the train/test
  index sets. Explain in a comment why fitting the basis on training data only
  (and never on the test split) is the leakage-safe choice.
- **Class balance.** `run_qc` prints a `ddm4bio.qc` tabular report of the label
  distribution. Note whether the ten digit classes are roughly balanced and what
  that means for interpreting accuracy.

## Part D — Interpretation & confidence

Implement `modes_for_variance` to report the smallest number of modes reaching
90%, 95%, and 99% of the cumulative variance (using
`ddm4bio.methods.decomposition.explained_variance_ratio`). Then, in the
interpretation block printed by `main` (via
`ddm4bio.interpret.interpretation_block`), state: how aggressively the digits
compress, how trustworthy the nearest-neighbour classifier is given the QC
evidence, and the honest limitations of reading these results as a claim about
real cell imaging. Pick a confidence level and back it with the evidence you
actually produced.

## Files

- `solution/ps1_solution.py` — reference implementation (instructor copy).
- `student/ps1.py` — your working file; fill in every `# TODO`.
- `tests/test_ps1.py` — autograder (interfaces + numerical thresholds).
- `rubric.md` — how this problem set is graded.

## Running

```bash
python student/ps1.py          # runs until the first unimplemented function
pytest tests/test_ps1.py -q    # autograder
```
