# PS5 — Signal or Noise? A Significance Test for the Rank

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/symbiont-ai/ddm4bio/blob/main/problem_sets/ps5_dimreduction/ps5_colab.ipynb)

**Work in the browser:** click the badge to open this problem set in Google Colab — no local setup required. You can also work locally (see below).

**Reading:** Kutz, *Data-Driven Modeling & Scientific Computation*, Chapters 15–16
(SVD/PCA and independent component analysis), plus any introduction to Horn's
parallel analysis and the Marchenko–Pastur law.

Week 5 computes a PCA/SVD spectrum and **eyeballs a scree plot** to guess how many
components matter. This problem set replaces the guess with a principled stopping
rule — **Horn's parallel analysis**: permute each gene column independently to build a
rank-matched *noise* null, then keep only the leading components whose real eigenvalue
beats the null. You will recover the latent **dimensionality of the feature space**
with a statistical test the lesson never covers, and expose why the popular analytic
shortcut (the Marchenko–Pastur edge) is untrustworthy on real, non-Gaussian data.

> This is a **linear-decomposition** task, not a clustering one: the counted object is
> the number of significant *principal components* (continuous feature-space
> directions), chosen by a spectral permutation null — not a number of sample clusters.

The decomposition mechanics (`pca_eigenvalues`), the null draw (`permute_columns`),
the MP edge (`marchenko_pastur_edge`), and the data (`make_planted_rank`,
`load_pbmc3k_topvar`) are **provided** — this problem set is about the significance
test, not the SVD. Fill in the functions marked `# TODO` in `student/ps5.py`. The
autograder checks each on small seeded matrices of known rank, so keep the signatures
exactly as given.

## Data

- **Planted-rank matrices (synthetic, known ground truth)** — `make_planted_rank(n, p,
  k, noise_sd, seed)` builds `X = scores @ loadings + noise`, whose true number of
  signal components is exactly `k` (0 = pure noise). Because scoring rank *recovery*
  needs the exact truth, the validation fixtures are synthetic by design.
- **Real PBMC3k** — `load_pbmc3k_topvar()` loads the 10x PBMC3k single-cell matrix via
  `get_dataset("pbmc3k")`, library-normalized + log1p + top-variance genes. Choosing
  how many PCs to keep is the first decision in almost every scRNA-seq pipeline.

Seed everything through `ddm4bio.seed_everything()` (called in `main`).

## Part A — The permutation null and the stopping rule

Validate on synthetic matrices of known planted rank. Implement:

- `null_eigenvalue_spectrum(X, n_perm, percentile, seed)` — the rank-matched
  permutation null: per-rank `threshold`, `null_mean`, `null_std`.
- `count_significant_pcs(X, n_perm, percentile, seed)` — the contiguous parallel-
  analysis stopping rule; recovers the planted rank.
- `significance_ratios(X, n_perm, percentile, seed)` — per-PC real/null ratio (crosses
  1 at the significant-PC count).
- `recover_rank_vs_noise(noise_levels, k_true, ...)` — the known-truth validation
  harness: mean recovered rank vs. noise, flat at `k_true` then degrading gracefully.

## Part B — The naive analytic contrast

- `marchenko_pastur_count(X)` — count PCs above the analytic MP edge.
- `compare_selection_rules(X, ...)` — contrast the two rules and state which to trust.
  On clean Gaussian-noise synthetic data they agree; on real PBMC3k the MP edge
  over-counts (its Gaussian assumption breaks), so the data-adaptive permutation null
  is trusted.

## Quality control & interpretation (required)

The provided `run_qc` reports the matrix shape, top-3 variance explained, and aspect
ratio γ = p/n — printed before any rank result. The provided `main` closes with a
`ddm4bio.interpret.interpretation_block`: state how strongly the synthetic recovery and
the PBMC3k contiguous block support the claim, at an honest confidence level, and name
the real limitations — the permutation test's per-rank false-positive rate, the MP
edge's Gaussian assumption, and that a real "true rank" is not a single objective
integer (it shifts with gene selection).

## Files

- `student/ps5.py` — your working file; fill in every `# TODO`.
- `rubric.md` — how this problem set is graded.
- `ps5_colab.ipynb` — one-click Google Colab launcher (badge at the top).
- The reference solution and the autograder are provided through the course and
  run automatically by GitHub Classroom.

## Running

```bash
python student/ps5.py          # runs until the first unimplemented function
```

To work in the browser instead, click the Colab badge at the top of this file.
The autograder runs automatically when you push to GitHub Classroom.
