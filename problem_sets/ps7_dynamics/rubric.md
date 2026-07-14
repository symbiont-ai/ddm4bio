# PS7 Grading Rubric -- Data-Driven Dynamics

Total: **100 points.** The autograder in `tests/test_ps7.py` establishes the
factual floor (interfaces and ground-truth thresholds); the remaining credit is
for correct method logic, honest quality control, and a defensible reading of
the results.

## Method correctness -- 30 points

Do the three methods actually do what their names claim?

- **DMD (10).** `run_dmd` fits exact DMD on the snapshot matrix and returns
  modes, eigenvalues, and correctly derived growth rates (`log|lambda| / dt`)
  and frequencies (`angle(lambda) / dt`). The recovered spectrum matches the
  true eigenvalues of a synthetic linear system to numerical tolerance.
- **SINDy (10).** `sindy_terms` recovers the correct active library terms on a
  clean system via sequential thresholded least squares; the reported active set
  matches ground truth (precision and recall at low noise both meet the 0.8
  threshold).
- **Kalman filter (10).** `kalman_denoise` implements the matched random-walk
  predict/update recursion with the correct `F`, `H`, `Q`, `R` and returns a
  filtered state of the right shape.

## Application execution -- 25 points

- **Epidemic (13).** `fit_epidemic_dynamics` fits SINDy to SIR case-count data,
  recovers the bilinear infection term and the linear recovery term, and the
  submission interprets what the recovered equation means (transmission vs.
  recovery), not just prints coefficients.
- **Physiological signal (12).** `filter_physiological_signal` filters the noisy
  trace and reports raw vs. filtered L2 error, with the filtered error strictly
  lower than raw on the matched system.

## Quality control -- 25 points

- **Ground truth before real data (12).** `sindy_noise_sensitivity` runs SINDy
  on a known system across a noise sweep and reports precision/recall per level,
  explicitly identifying the noise at which recovery degrades. Testing on
  fixtures with known answers *before* trusting the method is required, not
  optional.
- **Honest held-out forecasting (9).** `dmd_forecast` fits on a training window
  and reports **out-of-sample** forecast error separately from training error.
  Reporting only in-sample error is a QC failure and loses this credit.
- **Signal QC golden rule (4).** The physiological signal is passed through
  `qc_signals` (NaN/flatline/clipping/SNR) before it is analyzed.

## Interpretation & honesty -- 15 points

- Names **which dynamical terms are trustworthy** and ties the claim to the
  recovery scores.
- States the **noise level at which SINDy breaks** as a concrete number from the
  Part C sweep.
- Gives **forecast confidence bounds**: how far the DMD forecast can be trusted,
  and why a linear forecast must not be extrapolated into chaos.
- Uses `interpretation_block(...)` with a matching confidence level and named
  limitations. Overclaiming (e.g. "high" confidence with no evidence, or
  ignoring the noise breakdown) loses points here even if the code is correct.

## Reproducibility -- 5 points

- Runs offline, deterministically (seeds fixed), with no network or downloads.
- `python student/ps7.py` runs end to end and `pytest tests/test_ps7.py` passes.
- Code is ruff-clean under rules E, F, I at line length 100.
