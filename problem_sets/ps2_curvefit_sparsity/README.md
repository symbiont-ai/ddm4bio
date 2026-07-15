# PS2 — Letting the Data Choose the Model: Cross-Validated Selection

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/symbiont-ai/ddm4bio/blob/main/problem_sets/ps2_curvefit_sparsity/ps2_colab.ipynb)

**Work in the browser:** click the badge to open this problem set in Google Colab — no local setup required. You can also work locally (see below).

**Reading:** Kutz, *Data-Driven Modeling & Scientific Computation*, Chapter 4
(least squares & curve fitting) and Chapter 5 (sparsity and the Lasso).

Week 2 fit a model at a complexity *you* chose — a curve of a given form, a Lasso
at a given penalty. This problem set asks the harder question you face when you do
**not** know the truth: how complex a model does the data actually support? The
answer is **cross-validation** — score each candidate on data it was not fit on,
and let held-out error choose. You never look at the ground truth to decide.

The model-fitting primitives (`np.polyfit` / `np.polyval`, the provided `lasso_fit`)
and the fold splitter (`kfold_indices`) are **provided** — this problem set is
about *selecting* with them, not re-deriving them. Fill in the functions marked
`# TODO` in `student/ps2.py`. The autograder checks each on its own seeded
fixtures, so keep the signatures exactly as given.

## Data

- **Synthetic** — `make_response_curve` (a noisy polynomial of known degree) and
  `make_sparse_regression` (a linear model with a known set of nonzero drivers).
  Known ground truth is what lets us check that cross-validation chose correctly.
- **Real** — `load_biomarkers` returns the offline Breast-Cancer-Wisconsin panel
  (569 samples × 30 measurements), used to read off a biomarker panel that
  generalizes. Seed everything through `ddm4bio.seed_everything()` (called in
  `main`).

## Part A — Choose model complexity by cross-validation

Fit polynomials of growing degree to a noisy response curve. Too low a degree
underfits; too high overfits — so the cross-validated MSE is U-shaped and its
minimum is the complexity the data supports.

Implement:

- `poly_cv_mse(x, y, degree, folds)` — mean out-of-sample MSE of a degree-`degree`
  polynomial across the folds (fit on the complement, predict the held-out fold).
- `select_degree(x, y, candidate_degrees, folds)` — the degree minimizing
  cross-validated MSE, and the whole curve.

## Part B — Choose a sparse feature set by cross-validation

Cross-validate the Lasso penalty to pick the feature set that generalizes.

Implement:

- `lasso_cv_mse(x, y, alpha, folds)` — mean out-of-sample MSE of a Lasso at penalty
  `alpha` (use the provided `lasso_fit`).
- `select_alpha(x, y, candidate_alphas, folds)` — the penalty minimizing
  cross-validated MSE, and the curve.
- `selected_features(x, y, alpha)` — indices of the nonzero-coefficient features.
- `support_scores(selected, true_support)` — precision and recall of a selected set
  against the known drivers.

On the synthetic data you measure recovery; on real WDBC you read off a panel
(there is no ground-truth driver set to score, so read it qualitatively).

## Quality control & interpretation (required)

The provided `run_qc` verifies the folds are **disjoint and complete** (each row is
held out exactly once, so every CV score is out-of-sample), printed before any
result. The provided `main` closes with a `ddm4bio.interpret.interpretation_block`:
state how much cross-validation supports the claim, at an honest confidence level,
and name the real limitations — CV-tuned Lasso recovers the true drivers but
**over-selects** (high recall, lower precision), each CV score is a single held-out
estimate, and the real panel has no ground truth to score against.

## Files

- `student/ps2.py` — your working file; fill in every `# TODO`.
- `rubric.md` — how this problem set is graded.
- `ps2_colab.ipynb` — one-click Google Colab launcher (badge at the top).
- The reference solution and the autograder are provided through the course and
  run automatically by GitHub Classroom.

## Running

```bash
python student/ps2.py          # runs until the first unimplemented function
```

To work in the browser instead, click the Colab badge at the top of this file.
The autograder runs automatically when you push to GitHub Classroom.
