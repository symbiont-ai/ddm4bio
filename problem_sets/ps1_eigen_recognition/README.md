# Problem Set 1 — Eigen-Recognition: Linear Systems and the SVD

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/symbiont-ai/ddm4bio/blob/main/problem_sets/ps1_eigen_recognition/ps1_colab.ipynb)

**Work in the browser:** click the badge to open this problem set in Google Colab — no local setup required. You can also work locally (see below).

**Reading:** Kutz, *Data-Driven Modeling & Scientific Computation*, Chapter 2
(linear systems, Gaussian elimination / LU, iterative solvers, and the singular
value decomposition). Skim the sections on conditioning and on low-rank
approximation before you start.

This problem set has two threads that meet in the same idea. First you will
solve a linear system `A x = b` two different ways and watch how the *method*
you choose changes the cost and the numerical behaviour. Then you will use the
singular value decomposition to build an "eigen-image" basis — the same machine
behind eigenfaces — and turn it into a working recognizer for real
peripheral-blood-cell microscopy crops (BloodMNIST): our literal "eigen-cells."

Everything is small, deterministic, and runs offline. Fill in the method logic
in `student/ps1.py`; the imports, data loading, and quality-control plumbing are
already wired for you. The autograder checks both the interfaces and
numerical thresholds, and runs automatically when you push to GitHub Classroom.

## Data

Two sources, both offline-safe:

- **Synthetic SPD system** — `make_spd_system(n)` (provided) builds a symmetric
  positive-definite matrix `A = M Mᵀ + n·I` and a right-hand side `b`. Symmetry
  and positive-definiteness are what let conjugate gradient apply.
- **BloodMNIST via `get_dataset("bloodmnist")`** — real peripheral-blood-cell
  microscopy crops (MedMNIST v2, CC BY 4.0). `load_digits_split()` (provided)
  pulls the library through the course data layer, converts each crop to
  grayscale (mean over the colour axis), flattens it, and returns a
  deterministic, stratified train/test split. When the download is unavailable
  it transparently falls back to a bundled image stack with the *same payload
  shape*, so the problem set and its autograder run offline and deterministically
  (`get_dataset("bloodmnist", download=False)`). We use it as an honest
  imaging-based cell-recognition task: low-resolution grayscale images with
  known cell-type labels.

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

## Part B — Application: eigen-cells on blood-cell images

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
  distribution. Note whether the cell-type classes are roughly balanced and what
  that means for interpreting accuracy.

## Part D — Interpretation & confidence

Implement `modes_for_variance` to report the smallest number of modes reaching
90%, 95%, and 99% of the cumulative variance (using
`ddm4bio.methods.decomposition.explained_variance_ratio`). Then, in the
interpretation block printed by `main` (via
`ddm4bio.interpret.interpretation_block`), state: how aggressively the images
compress, how trustworthy the nearest-neighbour classifier is given the QC
evidence, and the honest limitations of reading these results as a claim about
real cell imaging. Pick a confidence level and back it with the evidence you
actually produced.

## Files

- `student/ps1.py` — your working file; fill in every `# TODO`.
- `rubric.md` — how this problem set is graded.
- `ps1_colab.ipynb` — one-click Google Colab launcher (badge at the top).
- The reference solution and the autograder (interfaces + numerical thresholds)
  are provided through the course and run automatically by GitHub Classroom.

## Running

```bash
python student/ps1.py          # runs until the first unimplemented function
```

To work in the browser instead, click the Colab badge at the top of this file.
The autograder runs automatically when you push to GitHub Classroom.
