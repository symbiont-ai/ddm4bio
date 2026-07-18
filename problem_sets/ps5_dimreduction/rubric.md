# PS5 Grading Rubric — Signal or Noise? A Significance Test for the Rank

Total: **100 points**. Grading combines the autograder (`tests/test_ps5.py`)
with a short read of the submitted code and interpretation. All work must run
offline and deterministically; a submission that only passes because it hard-codes
expected numbers or disables a check earns no credit for the affected part.

## Part A — The permutation null and the stopping rule — 60 points

- **`null_eigenvalue_spectrum` (16).** Draws `n_perm` column-permuted copies with a
  seeded RNG, takes each one's descending eigenvalue spectrum, and returns per-rank
  `threshold` (the requested percentile), `null_mean`, and `null_std` as parallel
  arrays of length `min(n, p)`. The threshold is non-increasing and sits at/above the
  null mean.
- **`count_significant_pcs` (18).** Applies the *contiguous* parallel-analysis rule —
  the leading run of PCs whose real eigenvalue exceeds the null threshold at that same
  rank, stopping at the first non-exceedance. Recovers the exact planted rank on
  synthetic fixtures (3, 2, 5 for the graded cases) and returns 0 on pure noise.
- **`significance_ratios` (12).** Per-PC real/null ratio; > 1 for the first `k` PCs and
  < 1 immediately after, and the first-below-1 index equals `count_significant_pcs`.
- **`recover_rank_vs_noise` (14).** Known-truth harness: mean recovered rank per noise
  level, equal to the planted `k_true` across moderate SNR and degrading gracefully
  (staying near `k`, not collapsing) only at extreme noise. Correct per-dataset seeding.

## Part B — The naive analytic contrast — 25 points

- **`marchenko_pastur_count` (12).** Counts the real eigenvalues that exceed the
  provided `marchenko_pastur_edge` `lam_plus`. Full credit uses the real eigenvalues
  and the provided edge correctly.
- **`compare_selection_rules` (13).** Returns `parallel_analysis_k`, `mp_edge_k`, a
  correct `agree` flag, `trusted_rule="parallel_analysis"`, and a `verdict` that
  explains *why* the data-adaptive permutation null is trusted over the MP edge (which
  assumes Gaussian noise and over-counts on real expression data). The two reported
  counts must match the two graded rules.

## Quality control — 10 points

- **QC before results (6).** The provided `run_qc` (matrix shape, top-3 variance
  explained, aspect ratio γ = p/n) is printed before any rank result, per the course
  "QC before results" rule.
- **Honest use of ground truth (4).** The stopping rule is validated against the known
  planted rank on synthetic fixtures; the real PBMC3k count is reported with its
  contiguous-block evidence, not asserted as a universal constant.

## Interpretation & honesty — 5 points

- A clear `interpretation_block` -- a claim stated with the evidence
  actually generated (exact synthetic recovery across the noise sweep, a contiguous
  block of significant PCs with a smooth ratio decay through 1, and the documented
  MP-edge over-count on real data), plus at least two honest limitations — the per-rank
  false-positive rate of the permutation test, the MP edge's Gaussian assumption, and
  that a real "true rank" is not a single objective integer. Overclaiming a universal
  component count is penalized.

## Reproducibility — 5 points

- Runs top-to-bottom with a fixed seed (the permutation null uses a seeded local RNG;
  the SVD is exact); imports follow the repository policy (only `numpy` at module top
  level, heavier libraries inside function bodies); the file is `ruff`-clean under `E`,
  `F`, `I` at line length 100; and no public function signature was changed.

---

### Autograder mapping

| Test | Rubric area |
| --- | --- |
| `test_count_significant_pcs_recovers_the_planted_rank` | Part A: `count_significant_pcs` |
| `test_pure_noise_admits_no_components` | Part A: no false positives |
| `test_null_spectrum_has_valid_shape_and_thresholds` | Part A: `null_eigenvalue_spectrum` |
| `test_significance_ratios_cross_one_at_the_count` | Part A: `significance_ratios` |
| `test_null_is_a_genuine_stochastic_permutation_null` | Part A: the null is a real permutation null (not the analytic MP edge) |
| `test_ratios_and_count_route_through_the_null` | Part A: ratios/count are derived from that null |
| `test_recover_rank_vs_noise_is_flat_then_degrades_gracefully` | Part A: `recover_rank_vs_noise` |
| `test_marchenko_pastur_count_uses_the_edge_correctly` | Part B: `marchenko_pastur_count` |
| `test_compare_selection_rules_structure_and_logic` | Part B: `compare_selection_rules` |

A passing autograder is necessary but not sufficient: the QC and interpretation
credit is awarded for honest, well-argued analysis, not merely green tests.
