# PS6 — Valid Inference After Clustering: The Double-Dipping Trap

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/symbiont-ai/ddm4bio/blob/main/problem_sets/ps6_learning_inference/ps6_colab.ipynb)

**Work in the browser:** click the badge to open this problem set in Google Colab — no local setup required. You can also work locally (see below).

**Reading:** Kutz, *Data-Driven Modeling & Scientific Computation*, Chapters 17–18
(unsupervised and supervised learning) and Chapter 13 (statistical methods and
inference), plus an introduction to multiple-testing / FDR and to
selective inference (post-selection / data-thinning).

Module 6 clusters cells and then runs BH-FDR on per-feature t-tests **between the clusters
it just discovered**. That is **circular** — the partition was chosen to separate the
cells, so features look "significant" even in pure noise, and FDR does not fix it (BH
corrects multiplicity, not selection). This problem set makes that failure *measurable*
and builds a procedure that actually controls the error.

> This extends the lesson's own two tools — clustering and BH-FDR — turned against each
> other. The clustering, the t-test, and BH are all **provided**; you build the *testing
> procedure* and its calibration, which the lesson never teaches.

Fill in the functions marked `# TODO` in `student/ps6.py`. The autograder grades
**behavior** (the three protocols' Type-I error regimes and their retained power on a
known-truth null and signal), so it can't be gamed by a constant or degenerate return.

## Data

- **`make_null(n, d, seed)`** — pure Gaussian noise: no clusters, no differential
  features, so the true marker count is exactly **0** and every rejection is a known
  false discovery. The load-bearing Type-I fixture.
- **`make_signal(...)`** — two real groups with a **known informative-feature mask** (a
  positive control for power).
- **Real PBMC3k** via `load_pbmc()` (log-normalized, top-variance genes) — the Part-B
  application, plus `make_gene_permuted_null` to show real marginals suppress the trap.

Seed everything through `ddm4bio.seed_everything()` (called in `main`).

## Part A — The protocols and their calibration

Implement three post-clustering testing protocols and two Monte-Carlo harnesses:

- `cluster_then_test_naive(X, k, alpha, seed)` — cluster all, test between clusters
  (double-dipping).
- `cluster_then_test_splitsample(X, k, alpha, seed)` — cluster split A, test held-out
  split B (the intuitive fix that is **still inflated**).
- `cluster_then_test_datathin(X, sigma, k, alpha, seed)` — Gaussian data-thinning
  (`X1 = X + eps`, `X2 = X - eps`), cluster `X1`, test `X2` (the fix that **works**).
- `null_false_discovery_profile(generate_null, protocol, R, alpha, seed0)` — Type-I
  harness (`mean_fd`, `max_fd`, `prob_any_fd`).
- `power_profile(generate_signal, protocol, R, alpha, seed0)` — retained-power harness.

The headline result: on the null, naive rejects **~21/50** false markers, sample-split
**~5**, data-thin **~0** (nominal α) — while all three keep essentially full power on real
signal.

## Part B — The real-data deliverable

- `inflation_summary(X, sigma, k, alpha, seed)` — run all three on one dataset and
  quantify the naive over-call. On real PBMC3k the two valid protocols agree (~572
  markers) while naive over-calls (~677, **+18%**).

## Quality control & interpretation (required)

Type-I inflation is graded on the synthetic null (true count 0), *not* the real data.
The provided `main` closes with a `ddm4bio.interpret.interpretation_block`: state the claim (the synthetic Type-I sweep is the evidence; real data is
illustrative) and the honest limitations — data-thinning assumes Gaussian noise with an
estimated variance, real single-cell counts double-dip only mildly, and FDR corrects
multiplicity, not selection.

## Files

- `student/ps6.py` — your working file; fill in every `# TODO`.
- `rubric.md` — how this problem set is graded.
- `ps6_colab.ipynb` — one-click Google Colab launcher (badge at the top).
- The reference solution and the autograder are provided through the course and
  run automatically by GitHub Classroom.

## Running

```bash
python student/ps6.py          # runs until the first unimplemented function
```

To work in the browser instead, click the Colab badge at the top of this file.
The autograder runs automatically when you push to GitHub Classroom.
