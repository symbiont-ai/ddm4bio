# PS2 Rubric — Curve Fitting, Regularized Differentiation, Sparsity

Total: **100 points**. The autograder checks interfaces and ground-truth
thresholds; the remaining judgment is applied by the grader against the criteria
below. Every deterministic, seeded test must pass against your submission.

## Method correctness — 30 points

The three estimators are implemented correctly on top of the `ddm4bio` library.

- **12 pts** — `fit_dose_response` fits the Hill model on flattened replicates
  and recovers the true EC50 (within 10%) and Hill coefficient (within 0.3);
  `r_squared` and the residual arrays are computed correctly.
- **9 pts** — `compare_derivative_methods` produces both derivative estimates
  and correct L2 errors, and the regularized error is strictly smaller than the
  finite-difference error on the noisy signal.
- **9 pts** — `sparse_biomarkers` / `stability_selection` apply the Lasso
  correctly (standardizing when asked) and recover the exact support of a
  synthetic sparse model.

## Application execution — 25 points

The methods are run on the assigned data and produce the required quantities.

- **13 pts** — `bootstrap_ec50` resamples replicates per dose, refits, and
  returns sensible point estimates and percentile confidence intervals; the
  EC50 interval brackets the known truth.
- **12 pts** — stability selection runs on the breast-cancer panel and yields a
  short, plausible stable set with a valid per-feature frequency vector.

## Quality control — 25 points

QC is first-class and precedes the results.

- **9 pts** — goodness of fit reported honestly: `r_squared` **and** residual
  structure (`residual_structure` correctly computes lag-1 autocorrelation and
  sign-run count), with structured residuals flagged rather than hidden.
- **9 pts** — selection stability reported as the QC metric for Part B, naming
  which features clear the threshold and which barely miss it.
- **7 pts** — the tabular QC report (`qc_tabular`) is produced and read before
  the selection results, not bolted on afterward.

## Interpretation & honesty — 15 points

- **7 pts** — a complete interpretation block via `interpretation_block`: EC50
  with its confidence interval, the stable biomarkers, and named limitations.
- **8 pts** — the confidence level matches the evidence. Claiming `high` on a
  synthetic assay with a replicate-only bootstrap and no held-out biomarker test
  set is penalized; naming those constraints and choosing `moderate` or `low` is
  rewarded.

## Reproducibility — 5 points

- **5 pts** — seeded and deterministic (same seed → identical output), offline
  only (no network, no downloads), and `ruff`-clean under `E, F, I` at line
  length 100.

---

### Automatic caps

- Any network access, download, or non-deterministic output caps the score at
  **60/100** regardless of correctness.
- A missing or non-functional QC section (Part C) caps the score at **75/100** —
  QC is a required, graded deliverable.
- An interpretation block whose stated confidence materially overreaches the
  evidence caps Interpretation & honesty at **5/15**.
