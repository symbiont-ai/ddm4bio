# PS5 — Dimensionality Reduction and Blind Source Separation

**Reading:** Kutz, *Data-Driven Modeling & Scientific Computation*, Ch. 15
(the singular value decomposition and principal component analysis) and Ch. 16
(independent component analysis and image/signal separation). Read for concepts
and derive your own code — do not copy any listing.

Low-dimensional structure is the workhorse of quantitative biology: a handful of
latent axes often explain most of the variation in an expression matrix, a
neural recording, or an imaging stack. But "a component" is not automatically "a
mechanism." The same math that surfaces a real biological program will just as
happily surface a batch artifact, and a scale choice you made three steps
earlier can silently decide which one leads. This problem set asks you to run
the three core linear-decomposition tools, and — more importantly — to build the
habit of checking a method against *known ground truth* before you believe
anything it tells you about real data.

You will implement five functions and wire them into an analysis that is graded
on both correctness and honesty.

## Data

The ground-truth fixtures are synthetic, offline, and deterministic. The
real-data PCA application streams a single-cell matrix through the course data
layer, which caches the genuine download and falls back to a structurally
identical synthetic matrix when offline — so this too runs with no network.

- **Synthetic expression matrix** — `make_expression_matrix()` (provided in the
  template) builds an `(n_samples, n_genes)` matrix with two orthogonal gene
  programs injected on top of Gaussian noise: a strong *biological* program tied
  to a binary condition label, and a weaker *batch* program tied to a nuisance
  label. Because you know the true directions and labels, you can measure
  exactly how well PCA recovers them. This is the PCA **validation** fixture.
- **Real single-cell matrix** — `get_dataset("pbmc3k")`
  (`ddm4bio.datasets.get_dataset`) returns the 10x Genomics PBMC3k assay as an
  `AnnData` (real; use `.X` for the counts) or a labelled fallback `dict` with
  `counts`/`labels`/`gene_names` (offline). `load_single_cell_expression()`
  (provided) log1p-normalizes it and selects the top-variance genes; the
  application applies the same validated PCA to it and prints `ds.source` /
  `ds.provenance` so you can see whether you got real or fallback data.
- **Synthetic mixed sources** — `ddm4bio.datasets.synthetic.make_mixed_sources`
  returns a `MixedSources` fixture: known independent sources (sine, sawtooth,
  square, then Laplacian sources), a known mixing matrix, and the observed
  multichannel mixtures. This is your ICA ground truth.
- **Robust-PCA construction** — a low-rank matrix plus sparse gross corruptions,
  built locally from a seeded RNG.

All ground truth (true directions, labels, sources, mixing) is available, so
every claim you make can be checked against an exact answer.

## Part A — Method

Implement the linear-decomposition machinery on top of the `ddm4bio` wrappers:

1. `svd_decompose(X, center=True)` — compute the (optionally centered) economy
   SVD of a feature matrix and return `U`, the singular values, `Vt`, and the
   explained-variance ratio per component. Be ready to interpret the singular
   values with a **scree plot** and to explain what the right singular vectors
   (`Vt` rows) mean as gene programs / spatial modes.
2. `robust_pca(X)` — separate a matrix into a **low-rank** part plus a
   **sparse** part (principal component pursuit). State plainly what each part
   is meant to capture.
3. `run_ica(observations, n_sources, seed)` — recover independent sources from
   mixed multichannel observations. Remember that ICA is unidentifiable up to
   sign and permutation of the sources.

## Part B — Application

1. **PCA of the expression matrix.** Project the synthetic expression matrix
   onto its leading components with `pca_scores`. Relate the leading components
   back to the injected structure: correlate the PC1 loading with the biological
   program and the PC2 loading with the batch program, and correlate the scores
   with the condition and batch labels. Report which axis is biology and which
   is batch, and how confidently you can tell them apart.
2. **ICA of a multichannel recording.** Run `run_ica` on the mixed-sources
   observations and recover the underlying independent sources.

## Part C — Quality control *(required)*

QC is not optional garnish here; it is the point of the assignment.

1. **Variance explained and reconstruction.** Report the fraction of variance in
   the top components and the relative-L2 error of a rank-2 reconstruction of
   the expression matrix. A component that explains little variance and barely
   changes the reconstruction is not load-bearing.
2. **PCA sensitivity to scaling / normalization.** Using `scaling_sensitivity`,
   show that inflating a single high-variance feature can hijack the leading
   component, and that per-feature standardization changes the answer. Decide
   and justify whether your PCA should run on raw or standardized data.
3. **ICA recovery vs ground truth first.** Compute the max-matched source
   recovery score on the synthetic mixture with
   `ddm4bio.methods.validation.source_recovery_score`. **Do not trust ICA on any
   real recording until this score clears 0.9 on the synthetic case.** This gate
   is the whole discipline: validate on known truth, then extrapolate.

## Part D — Interpretation and confidence

Close with an interpretation block built via
`ddm4bio.interpret.interpretation_block(...)`:

- **How much structure is real vs batch?** Given the loading/score correlations
  and the variance explained, argue which leading component reflects genuine
  biology and which reflects batch, and how separable they are.
- **Is the ICA result physiologically meaningful?** State a confidence level
  (`low` / `moderate` / `high`) justified by the recovery score against ground
  truth, and name the honest limitations — synthetic-vs-real gap, scale
  dependence of PCA, the sign/permutation ambiguity of ICA, and the fact that
  the injected batch and biology were orthogonal by construction while real
  confounds usually are not.

## What to submit

Complete the five `# TODO` functions in `student/ps5.py`. The data loaders, QC
driver, and interpretation block are already wired — you implement only the
method logic. Running `python student/ps5.py` should print the analysis, QC
metrics, and interpretation block once your functions are done. The autograder
in `tests/test_ps5.py` checks interfaces and ground-truth thresholds.
