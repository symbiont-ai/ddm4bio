# PS5 Grading Rubric — Dimensionality Reduction and Blind Source Separation

Total: **100 points**. Each criterion is scored on evidence in the submitted
code, its printed output, and the passing autograder.

## Method correctness — 30 points

- **SVD (10).** `svd_decompose` returns correctly shaped `U`, singular values,
  `Vt`, and explained-variance ratios; the centered reconstruction is
  near-exact and the ratios are non-negative, sum to one, and are non-increasing.
- **Robust PCA (10).** `robust_pca` returns a low-rank plus sparse split that
  reconstructs the true low-rank part on the seeded construction.
- **ICA (10).** `run_ica` returns sources of the right shape and recovers the
  true independent sources; the sign/permutation ambiguity is handled by scoring
  on matched absolute correlation rather than raw equality.

## Application execution — 25 points

- **PCA of the expression matrix (13).** Leading components are computed and
  explicitly related to the injected structure: PC1 loading tied to the
  biological program, PC2 to batch, with score-vs-label correlations reported.
- **ICA of the multichannel recording (12).** ICA is run on the mixed
  observations and the recovered sources are compared to ground truth.

## Quality control — 25 points

- **Variance explained and reconstruction (8).** Top-component variance and a
  rank-2 reconstruction error are reported and interpreted.
- **Scaling / normalization sensitivity (8).** The submission demonstrates that
  PCA is not scale-invariant and justifies the raw-vs-standardized choice.
- **ICA validated on ground truth before trust (9).** The source recovery score
  is computed on the synthetic case and the **> 0.9 gate** is checked and
  honored before any claim of physiological meaning. Skipping this gate caps the
  QC score.

## Interpretation and honesty — 15 points

- **Real vs batch structure (7).** A defensible argument, grounded in the
  reported correlations and variance explained, about which component is biology
  and which is batch, and how separable they are.
- **Confidence and limitations (8).** An `interpretation_block` with a
  confidence level justified by the recovery score and a named-limitations list
  covering the synthetic-vs-real gap, PCA scale dependence, and ICA
  sign/permutation ambiguity. Overclaiming beyond the evidence loses points.

## Reproducibility — 5 points

- Fixed seeds throughout, offline-only data, no hidden global state; repeated
  runs produce identical numbers and the autograder passes deterministically.
