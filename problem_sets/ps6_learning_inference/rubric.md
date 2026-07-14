# PS6 Grading Rubric

Total: **100 points**. The autograder in `tests/test_ps6.py` establishes the
objective floor (interfaces, ground truth, and performance thresholds on seeded
offline data); the remaining judgment concerns method choice, honesty, and
reproducibility. A submission that games a threshold without sound method does
not earn the method points.

---

## Method correctness — 30 points

Does the core machinery implement the right algorithms correctly?

- **Clustering (10).** `cluster_all_methods` returns k-means, Gaussian-mixture,
  and Ward-hierarchical labelings of the correct shape, delegating to the
  course clustering helpers. `cluster_agreement` computes a correct,
  permutation-invariant adjusted Rand index.
- **Model selection (10).** `select_number_of_subtypes` drives both the
  silhouette and BIC criteria, respects their valid k-ranges (silhouette needs
  k ≥ 2), and reports agreement. On separated blobs the silhouette recovers the
  true k.
- **Supervised zoo (10).** `build_classifier` returns a scaler-wrapped pipeline
  for each of LDA, SVM, tree, and NN and raises `ValueError` on an unknown name.
  `evaluate_classifiers` uses stratified cross-validation with scaling inside
  the pipeline (no leakage).

## Application execution — 25 points

Does the pipeline actually run end-to-end on the offline data and produce the
requested quantities?

- **Subtype discovery (12).** Clustering on the synthetic expression matrix
  recovers the planted subtypes with high cross-method and vs-truth ARI
  (> 0.9 on well-separated blobs).
- **Diagnostic model (13).** `diagnostic_auc` performs a leakage-checked
  stratified split, fits a classifier, and reports a held-out ROC-AUC (> 0.9 on
  breast cancer) together with a bootstrap confidence interval. The score path
  works for both probability classifiers and margin classifiers (SVM via the
  decision function).

## Quality control — 25 points

QC is weighted as heavily as application; this is a QC-centered assignment.

- **Cluster stability (13).** `assess_cluster_stability` runs consensus
  resampling, computes a defensible PAC-based stability score, and **warns when
  the solution is not reproducible.** Full credit requires the guard to fire on
  over-clustered, structureless data and stay silent on clean structure.
- **Multiple testing (12).** `per_feature_tests` runs a correct per-feature
  two-sample test (and rejects non-binary labels); `fdr_correct` applies
  Benjamini–Hochberg. Full credit requires demonstrating both power (most true
  signals recovered) and control (false discoveries among the nulls held near
  the target rate).

## Interpretation & honesty — 15 points

- **Calibrated claim (7).** The interpretation block states a claim whose
  confidence level matches the evidence — robustness across methods, the AUC
  interval, and the FDR result. Overconfidence given a synthetic ground truth
  and a single split loses points.
- **Named limitations (8).** The submission explicitly names its real
  limitations: synthetic vs. real data, one split vs. repeated splits, and the
  distinction between controlling an expected error rate and certifying an
  individual feature. Reporting an AUC point estimate without its interval, or
  "significant features" without acknowledging FDR's meaning, is penalized.

## Reproducibility — 5 points

- All results are deterministic under a fixed seed; re-running yields identical
  numbers. Code passes `ruff check` under rules E, F, I at 100 columns, imports
  numpy at the top and heavier libraries inside function bodies, and uses only
  offline data (no network, no downloads). The autograder passes against the
  reference solution.
