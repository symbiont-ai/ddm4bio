# PS6 Grading Rubric — Valid Inference After Clustering (The Double-Dipping Trap)

Total: **100 points**. Grading combines the autograder (`tests/test_ps6.py`)
with a short read of the submitted code and interpretation. All work must run
offline and deterministically; a submission that only passes because it hard-codes
expected numbers or disables a check earns no credit for the affected part.

## Part A — The testing protocols and calibration harnesses — 60 points

- **`cluster_then_test_naive` (12).** Clusters all rows, runs the provided
  `per_feature_ttest` between the two clusters, BH-corrects; returns the full result
  dict. On a genuine two-group signal it recovers the informative features; on the null
  it over-rejects (the trap made concrete).
- **`cluster_then_test_splitsample` (16).** Disjoint A/B split (guarded against leakage),
  cluster A, assign B by nearest centroid, test on **B only**. Full credit requires the
  test to run on the held-out half — and the graded behavior is that it is *still*
  inflated on the null (strictly between naive and data-thinning), because B's labels
  are assigned from B's own values.
- **`cluster_then_test_datathin` (17).** Gaussian data-thinning: `X1 = X + eps`,
  `X2 = X - eps` with `eps ~ N(0, sigma^2)`, cluster `X1`, test `X2`. Full credit
  requires clustering and testing the *two different folds* (not the same matrix), so the
  labels are independent of the tested values and the false-discovery rate returns to
  nominal — while power on real signal is retained.
- **`null_false_discovery_profile` (8).** Monte-Carlo Type-I harness: draws `R` known-null
  datasets, runs the protocol callable on each, returns `mean_fd`, `max_fd`, and
  `prob_any_fd`. Must actually invoke the callable over the seed stream (checked with a
  stub protocol).
- **`power_profile` (7).** Counts rejections that land on the known informative-feature
  mask; returns `mean_recovered` and `frac_true_recovered`. This is the power side of the
  pincer that rules out a "reject nothing" shortcut.

## Part B — The real-data deliverable — 15 points

- **`inflation_summary` (15).** Runs all three protocols on one dataset and returns
  `naive_n`, `split_n`, `thin_n`, `inflation_extra` (`naive_n` minus the larger valid
  count), and `inflation_pct`. Correct ranking (`naive_n >= split_n, thin_n`) and
  arithmetic. Applied to real PBMC3k in `main`.

## Quality control & honesty — 10 points

- **QC / honest use of ground truth (6).** Type-I inflation is measured on the synthetic
  null (where the true marker count is exactly 0), not on the real data; the real PBMC3k
  counts are reported as *illustrative* (naive > valid), with the gene-permuted null
  showing that sparse real marginals suppress the trap.
- **No leakage (4).** The sample-split uses `assert_no_leakage` (or an equivalent
  disjoint-index guarantee); the data-thin folds are genuinely independent.

## Interpretation & honesty — 10 points

- A clear `interpretation_block` -- a claim stated with the evidence actually
  generated (the naive/split/thin Type-I separation, the retained-power sweep, the real
  over-call), plus at least two honest limitations — that thinning assumes Gaussian noise
  with an estimated variance, that real single-cell counts double-dip only mildly, and
  that FDR corrects multiplicity, not selection. Overclaiming (e.g. "BH-FDR makes the
  marker list trustworthy") is penalized.

## Reproducibility — 5 points

- Runs top-to-bottom with fixed seeds (k-means, the A/B split, and the thinning noise are
  all seeded); imports follow the repository policy (only `numpy` at module top level,
  heavier libraries inside function bodies); the file is `ruff`-clean under `E`, `F`, `I`
  at line length 100; and no public function signature was changed.

---

### Autograder mapping

| Test | Rubric area |
| --- | --- |
| `test_protocols_return_valid_structure_and_recover_real_signal` | Part A: the three protocols (structure + signal recovery) |
| `test_type_I_error_separates_the_three_protocols_on_the_null` | Part A: naive inflated / split intermediate / thin controlled |
| `test_all_three_protocols_retain_power_on_real_signal` | Part A: power (the pincer) |
| `test_null_false_discovery_profile_arithmetic_with_a_stub` | Part A: `null_false_discovery_profile` |
| `test_power_profile_arithmetic_with_a_stub` | Part A: `power_profile` |
| `test_inflation_summary_ranks_and_quantifies_the_over_call` | Part B: `inflation_summary` |

A passing autograder is necessary but not sufficient: the QC and interpretation
credit is awarded for honest, well-argued analysis, not merely green tests.
