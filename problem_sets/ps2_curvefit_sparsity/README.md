# PS2 — Curve Fitting, Regularized Differentiation, and Sparsity

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/symbiont-ai/ddm4bio/blob/main/problem_sets/ps2_curvefit_sparsity/ps2_colab.ipynb)

**Work in the browser:** click the badge to open this problem set in Google Colab — no local setup required. You can also work locally (see below).

**Reading:** Kutz, *Data-Driven Modeling & Scientific Computation*, Chapter 4
(least-squares and nonlinear curve fitting, regression, the coefficient of
determination) and Chapter 5 (sparsity, the Lasso / L1 regularization, and
compressed sensing). This problem set turns three ideas from those chapters into
one honest analysis of pharmacology and biomarker data.

You will fit a sigmoidal dose-response curve and defend the number you extract
from it, watch regularization tame a differentiation problem that naive
differencing wrecks, and use L1 penalties to pull a short, *stable* list of
biomarkers out of a wide feature panel. Throughout, the emphasis is not on
getting a point estimate but on knowing how much to trust it.

You implement your method logic in `student/ps2.py`. The library calls, the
offline data loaders, and the QC/interpretation plumbing are already wired; you
fill only the function bodies marked `# TODO`. The public function signatures are
fixed — the autograder imports them by name.

## Data (everything offline)

No downloads, no network, no credentials.

- **Dose-response assay** — generated locally by `simulate_dose_response`, which
  evaluates a known Hill curve (`ddm4bio.methods.fitting.hill`) and adds
  independent Gaussian replicate noise. Because the true EC50 and Hill
  coefficient are known, you can measure how well you recovered them. (This stays
  synthetic on purpose: it is the ground-truth recovery check.)
- **Differentiation signal** — a noisy sine on a uniform grid, whose analytic
  derivative (a cosine) is known exactly.
- **Biomarker panel** — the Breast Cancer Wisconsin (Diagnostic) dataset (569
  samples, 30 features), loaded through the course data layer as
  `get_dataset("breast_wisconsin", download=False)` and wrapped as a labelled
  DataFrame by `load_breast_cancer_frame` so the tabular QC report can check
  class balance. With `download=False` this serves the scikit-learn *bundled*
  WDBC data deterministically and offline (a synthetic fallback is used only if
  scikit-learn is missing). The related Week 2 lesson also fits a Hill curve to a
  real drug/cell dose-response series from `get_dataset("gdsc")`.

All randomness is seeded from `ddm4bio.config.GLOBAL_SEED`; identical inputs must
give identical outputs.

---

## Part A — Method

Build the three estimators, each on top of the maintained `ddm4bio.methods.fitting`
routines.

1. **Nonlinear dose-response fit.** Implement `fit_dose_response(dose, response,
   seed)`. Accept either a 1-D response vector or a 2-D
   `(n_replicates, n_doses)` array; flatten replicates against their doses and
   fit the four-parameter Hill model with `fit_hill`. Extend the returned dict
   with the coefficient of determination `r_squared`, the flattened `residuals`,
   and the per-dose `mean_residuals` (ordered by dose) used by the QC check.

2. **Regularized vs. finite-difference differentiation.** Implement
   `compare_derivative_methods(y, dx, deriv_true, lam)`. Estimate the derivative
   of the noisy signal two ways — Tikhonov-regularized differentiation
   (`regularized_derivative`) and a plain `np.gradient` finite difference — and
   score each by its L2 distance to the known analytic derivative. A finite
   difference amplifies high-frequency noise; regularization should not.

3. **L1 sparse selection.** Implement `sparse_biomarkers(x, y, alpha,
   standardize, seed)`, a thin wrapper over `lasso_select` that optionally
   z-scores the columns first (essential when features span very different
   scales). The L1 penalty drives uninformative coefficients to exactly zero.

## Part B — Application

1. **EC50 and Hill coefficient with uncertainty.** Run `fit_dose_response` on the
   synthetic assay, then implement `bootstrap_ec50(dose, response, n_boot, ci,
   seed)`: resample the replicates at each dose with replacement, refit each
   resampled mean, and build percentile confidence intervals for the EC50 and
   the Hill coefficient. This propagates replicate measurement noise into the
   parameters you report.

2. **Stable biomarker selection.** Run `sparse_biomarkers` on the breast-cancer
   panel, then implement `stability_selection(x, y, n_boot, alpha, subsample,
   threshold, standardize, seed)`: repeatedly fit the Lasso on random row
   subsamples and record how often each feature is selected. A single Lasso fit
   is notoriously unstable to resampling; the selection *frequency* is the honest
   quantity.

## Part C — Quality control (required)

QC is graded and comes **before** the headline results (the course golden rule).

1. **Goodness of fit.** Report `r_squared`, and implement `residual_structure`
   to summarize the dose-ordered residuals: their lag-1 autocorrelation and the
   number of same-sign runs. A high `r_squared` with structured (autocorrelated,
   few-run) residuals signals a misspecified model, not a good fit — flag it.

2. **Selection stability.** Treat the per-feature selection frequency from
   `stability_selection` as the QC metric for Part B: report which features
   clear the stability threshold and note how many barely miss it.

3. **Tabular QC.** The driver prints `ddm4bio.qc.qc_tabular` on the breast-cancer
   frame (missingness, duplicates, constant columns, class balance, IQR
   outliers) before any selection result. Read it; do not just print it.

## Part D — Interpretation & confidence

Close with an interpretation block emitted through
`ddm4bio.interpret.interpretation_block`. State (i) the EC50 **with its
confidence interval**, (ii) which biomarkers are stable, (iii) a single honest
confidence level (`low` / `moderate` / `high`) backed by named evidence, and
(iv) a named limitations list. The confidence level must reflect the evidence
you actually have — a synthetic assay, a replicate-only bootstrap, a scale- and
`alpha`-sensitive selector, and no held-out biomarker test set all cap how far
you can go. Overstating confidence is the failure mode this part exists to catch.

---

## Running

```bash
python student/ps2.py                 # import cleanly; stop at the first TODO
```

You can also work in the browser: click the **Open in Colab** badge at the top —
no local setup required. The autograder runs automatically when you push to
GitHub Classroom, importing your `student/ps2.py`; a reference solution is
provided through the course. Keep your code `ruff`-clean under `E, F, I` (line
length 100), deterministic, and offline.
