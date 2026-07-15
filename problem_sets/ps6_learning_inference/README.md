# PS6 — Unsupervised Discovery and Supervised Inference

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/symbiont-ai/ddm4bio/blob/main/problem_sets/ps6_learning_inference/ps6_colab.ipynb)

**Work in the browser:** click the badge to open this problem set in Google Colab — no local setup required. You can also work locally (see below).

**Theme:** finding structure you did not know was there, then building a model
that predicts something you care about — and, crucially, learning to distrust
both until you have stress-tested them.

**Reading (Kutz, *Data-Driven Modeling & Scientific Computation*).**
Chapters 17–18 for unsupervised learning — clustering, mixture models, model
selection, and the recurring warning that an algorithm will *always* return
clusters whether or not any exist. Chapter 13 for supervised learning —
regression and classification, the bias–variance trade-off, and why
cross-validation is the honest way to estimate out-of-sample performance. Use
the chapters as background; write all of your own code and prose.

This problem set has a single narrative arc. You are handed a cohort with no
labels and asked two questions a biologist actually asks: *Are there subtypes in
here?* and *Can I predict a clinical outcome?* You will answer both, and then
spend at least as much effort deciding whether to believe your own answers.

---

## Data (everything offline)

No downloads, no network, no credentials. Two offline sources only:

- **Synthetic expression matrix** — `load_subtype_data()` wraps
  `sklearn.datasets.make_blobs` into a samples × "genes" matrix with a *known*
  number of latent subtypes. This stays synthetic on purpose: the clustering
  ground truth (adjusted Rand index against the planted assignment) only makes
  sense when the answer is known. Think of the blob centers as molecular
  subtypes and the columns as genes.
- **Breast-cancer diagnostics** — `load_diagnostic_data()` pulls the
  Wisconsin Diagnostic Breast Cancer data through the course data layer,
  `get_dataset("breast_wisconsin", download=False)` (569 samples, 30 features,
  binary malignant/benign label). With `download=False` it resolves
  deterministically to the offline scikit-learn-bundled copy, so it is fully
  offline and needs no network or credentials.

The data-loading and quality-control plumbing is already written in
`student/ps6.py`. You implement only the method logic.

Everything must be **deterministic and seeded** (use `GLOBAL_SEED`). Import
numpy at the top; import scipy / scikit-learn inside your function bodies, the
same convention the course library follows.

---

## (A) Method

Build the core machinery and demonstrate you understand its failure modes.

1. **Three ways to cluster.** Implement `cluster_all_methods` to partition the
   feature matrix with k-means, a Gaussian mixture model, and agglomerative
   (Ward) hierarchical clustering, using `ddm4bio.methods.clustering`. Return
   all three labelings so they can be compared.
2. **Choosing k, two ways.** Implement `select_number_of_subtypes` to pick the
   number of clusters by both the mean silhouette coefficient (`select_k_silhouette`)
   and the Gaussian-mixture BIC (`select_k_bic`). Report whether the two
   criteria agree. When they disagree, that disagreement is *data*, not an
   error — discuss it in part D.
3. **Instability, honestly.** Implement `cluster_agreement` (adjusted Rand
   index) and use it to quantify how much k-means, the GMM, and hierarchical
   clustering agree with one another. Two algorithms that partition the same
   data differently are telling you the structure is soft.
4. **A classifier zoo with proper cross-validation.** Implement
   `build_classifier` to return a `StandardScaler`-wrapped pipeline for each of
   linear discriminant analysis (`"lda"`), a support-vector machine (`"svm"`),
   a decision tree (`"tree"`), and a small neural network (`"nn"`). Implement
   `evaluate_classifiers` to score each by stratified k-fold cross-validation
   through `ddm4bio.methods.learning.cross_validate`. Scaling lives *inside* the
   pipeline so it is refit within every fold — no test-fold information leaks
   into training.

## (B) Application

1. **Discover candidate subtypes.** Run your clustering machinery on the
   synthetic expression matrix. Select k, cluster three ways, and measure
   agreement against the known subtypes and across methods.
2. **Build a diagnostic classifier.** Implement `diagnostic_auc`: make a
   stratified train/test split of the breast-cancer data (guarded by
   `assert_no_leakage`), fit a classifier on the training split, and score the
   held-out test set. Report the ROC-AUC together with a bootstrap **confidence
   interval** via `ddm4bio.methods.learning.roc_with_ci`. A point estimate of
   AUC without an interval is not an answer.

## (C) Quality control *(required)*

Quality control is not an appendix here; it is the point of the assignment.

1. **Is the clustering reproducible?** Implement `assess_cluster_stability`
   using consensus (bootstrap-resampling) clustering from
   `ddm4bio.methods.clustering.consensus_cluster`. Summarize the consensus
   matrix by the Proportion of Ambiguously Clustered pairs (PAC) — the fraction
   of sample pairs whose consensus value lands in the indecisive band between
   0.1 and 0.9 — and turn that into a stability score. **Emit a warning when the
   solution is not reproducible.** Over-clustering structureless data must trip
   this guard.
2. **Control the false-discovery rate.** Implement `per_feature_tests` (a
   two-sample Welch t-test per feature between the two classes) and
   `fdr_correct` (Benjamini–Hochberg via `ddm4bio.methods.learning.bh_fdr`).
   Running 30+ univariate tests without correction manufactures false positives;
   BH-FDR bounds the *expected* fraction of your "significant" features that are
   noise.

## (D) Interpretation & confidence

Close with an honest interpretation block built through
`ddm4bio.interpret.interpretation_block`. State plainly:

- **Which subtypes are robust** — where do all three algorithms, the two
  model-selection criteria, and the consensus stability score agree, and where
  do they not?
- **The AUC with its interval** — quote the confidence interval, not just the
  point estimate, and say what a single train/test split can and cannot support.
- **Where false-discovery risk remains** — how many features survive FDR, and
  why surviving FDR bounds an expected error rate rather than certifying any
  individual feature.

Pick a confidence level (`"low"`, `"moderate"`, `"high"`) that your evidence
actually earns, and name the real limitations — synthetic ground truth, a single
split, and the difference between controlling a rate and proving a fact.

---

## What you submit

Implement every `# TODO` in `student/ps6.py`, keeping the public function
signatures unchanged (the autograder imports them by name). Running
`python student/ps6.py` should print the QC report, the selected k, the ARI
values, the stability verdict, the cross-validation scores, the diagnostic AUC
with its interval, the FDR result, and the interpretation block. Your code must
be deterministic and pass `ruff check` under rules E, F, and I at a 100-column
line length.
