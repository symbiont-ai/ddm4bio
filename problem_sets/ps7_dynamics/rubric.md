# PS7 Grading Rubric — The Predictability Horizon

Total: **100 points**. Grading combines the autograder (`tests/test_ps7.py`)
with a short read of the submitted code and interpretation. All work must run
offline and deterministically; a submission that only passes because it hard-codes
expected numbers or disables a check earns no credit for the affected part.

## Part A — The four estimators — 60 points

Each function is checked against a **closed-form** answer, so a copied physical number
cannot earn credit.

- **`ensemble_log_divergence` (12).** Averages the **logs** of the separations (not the
  raw separations) — identical rows return `ln(row)` exactly, and the non-identical case
  distinguishes log-then-average from the wrong average-then-log.
- **`finite_time_rate` (10).** A **local** finite-difference slope (not one global slope);
  reads the correct rate in the plateau and exposes the transient/saturation at the ends.
- **`forecast_horizon` (8).** The formula `T = (1/λ)·ln(tol/eps)`, and — mandatory —
  returns `inf` when `λ ≤ 0` (a non-chaotic system never diverges).
- **`empirical_forecast_horizon` (8).** The first time the forecast error exceeds `tol`,
  with correct first-crossing (not last) and a censored endpoint when it never crosses.
- **Correct computation in the driver (22).** The twin-pair separation and the Lyapunov
  slope are *standard NumPy one-liners*, so the driver computes them directly —
  `numpy.linalg.norm(b − a, axis=1)` for the per-timestep separation and
  `numpy.polyfit(t[s:e], mls[s:e], 1)[0]` for the least-squares divergence rate over the
  provided fit window. Graded on using the right call correctly (norm along the state axis;
  slope of the mean-log separation over the window), not on reimplementing either one.

## Part A — The chaotic-vs-non-chaotic contrast — 15 points

- **Lorenz Lyapunov exponent (8).** The twin-trajectory pipeline recovers a clearly
  positive rate within a tolerance band around the known value (~0.905), giving a short,
  finite forecast horizon.
- **The contrast (7).** The non-chaotic systems (FitzHugh–Nagumo limit cycle, stable
  linear) give a rate ~0 or negative and a vastly longer / unbounded horizon. Full credit
  requires all three systems classified correctly, not just Lorenz tuned in isolation.

## Part B — The real forecast — 10 points

- Applies the calibrated `empirical_forecast_horizon` to a real Hankel-DMD forecast of
  COVID incidence, reporting a finite model-limited horizon (~9–13 days) that shrinks near
  epidemic turning points, and honestly diagnosing the near-zero skill on the ECG.

## Quality control & interpretation — 10 points

- QC (data source, length, range) is printed before results; the `interpretation_block`
  distinguishes **intrinsic** predictability (the system's Lyapunov limit) from
  **model-limited** predictability (a particular forecaster's skill), and names the real limitations — the finite-time estimator's
  alignment-transient bias, the Gaussian/linearity assumptions, and COVID non-stationarity.
  Overclaiming a single "the forecast horizon" without a model or regime is penalized.

## Reproducibility — 5 points

- Runs top-to-bottom with a fixed seed; imports follow the repository policy (only `numpy`
  at module top level, heavier libraries inside function bodies); the file is `ruff`-clean
  under `E`, `F`, `I` at line length 100; and no public function signature was changed.

---

### Autograder mapping

| Test | Rubric area |
| --- | --- |
| `test_ensemble_log_divergence_averages_logs_not_raw` | Part A: `ensemble_log_divergence` |
| `test_finite_time_rate_is_a_local_derivative` | Part A: `finite_time_rate` |
| `test_forecast_horizon_formula_and_infinite_for_non_chaotic` | Part A: `forecast_horizon` |
| `test_empirical_forecast_horizon_first_crossing_and_censoring` | Part A: `empirical_forecast_horizon` |
| `test_chaotic_vs_non_chaotic_horizon_contrast` | Part A: the physical contrast + driver `norm`/`polyfit` |

A passing autograder is necessary but not sufficient: the QC and interpretation
credit is awarded for honest, well-argued analysis, not merely green tests.
