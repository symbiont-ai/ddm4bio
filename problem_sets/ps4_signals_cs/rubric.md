# PS4 Grading Rubric — How Undersampled Can You Go?

Total: **100 points**. Grading combines the autograder (`tests/test_ps4.py`)
with a short read of the submitted code and interpretation. All work must run
offline and deterministically; a submission that only passes because it hard-codes
expected numbers or disables a check earns no credit for the affected part.

## Part A — The recovery cliff — 55 points

- **`measurement_matrix` (10).** Shape `(m, n)` with `m` rows (measurements) and `n`
  columns (signal length), Gaussian entries scaled by `1/√m`.
- **`recover` (12).** Forms the measurements `y = matrix @ signal`, recovers with an
  L1/Lasso fit (scikit-learn's `Lasso`, `fit_intercept=False`, imported inside the
  body), and returns a length-`n` reconstruction. On a well-measured signal
  (`m ≫ k`) the recovery is close to the truth.
- **`recovery_error` (8).** Relative L2 error `‖recovered − true‖ / ‖true‖`; zero for
  a perfect recovery, one when the estimate is twice the truth.
- **`recovery_error_curve` (13).** Averages the recovery error over `n_trials` random
  measurement matrices per `m` (not a single draw); the curve is high below the
  sampling limit and low above it. The averaging is what makes the cliff sharp.
- **`min_measurements_for_recovery` (12).** Returns the *first* `m` (scanning in
  order) whose averaged error falls below `tol` — the sampling limit — not the `m`
  of least error, with a sane fallback when none qualify.

## Part B — The phase transition — 25 points

- **`phase_transition` (25).** For each sparsity `k`, returns the minimum
  measurements to recover a `k`-sparse signal, built with an independent seed per
  `k`. The result is **non-decreasing** in `k` and strictly larger for a dense signal
  than a very sparse one — a few measurements per nonzero. Correct handling of the
  per-`k` signal construction and reuse of `min_measurements_for_recovery`.

## Quality control — 10 points

- **QC before results (6).** The provided `run_qc` (signal length, nonzero count,
  and how many candidate `m` are genuinely undersampled with `m < n`) is printed
  before any cliff or phase-transition result, per the course "QC before results"
  rule.
- **Honest use of ground truth (4).** The cliff and phase transition are measured
  against the known sparse signals; the real ECG is used only to *motivate*
  compressibility, not to score recovery (which would need a truth it does not have).

## Interpretation & honesty — 5 points

- A clear `interpretation_block` -- a claim stated with the evidence
  actually generated (a sharp error-vs-measurements cliff, a monotone
  minimum-measurements-vs-sparsity curve, a real ECG shown to be compressible), plus
  at least two honest, specific limitations — exact vs. approximate sparsity, a
  single tolerance/grid, and a fixed L1 regularization. Overclaiming (e.g. "any
  signal recovers from a handful of samples") is penalized.

## Reproducibility — 5 points

- Runs top-to-bottom with a fixed seed; imports follow the repository policy (only
  `numpy` at module top level, heavier libraries inside function bodies); the file is
  `ruff`-clean under `E`, `F`, `I` at line length 100; and no public function
  signature was changed.

---

### Autograder mapping

| Test | Rubric area |
| --- | --- |
| `test_measurement_matrix_shape_and_scale` | Part A: `measurement_matrix` |
| `test_recover_and_error_on_a_well_measured_signal` | Part A: `recover`, `recovery_error` |
| `test_recovery_error_curve_falls_as_measurements_grow` | Part A: `recovery_error_curve` |
| `test_min_measurements_is_the_first_success_and_beats_the_smallest_m` | Part A: `min_measurements_for_recovery` |
| `test_phase_transition_grows_with_sparsity` | Part B: `phase_transition` |

A passing autograder is necessary but not sufficient: the QC and interpretation
credit is awarded for honest, well-argued analysis, not merely green tests.
