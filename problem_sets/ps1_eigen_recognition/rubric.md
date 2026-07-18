# PS1 Grading Rubric — The Eigen-Subspace as a Model of Normal Cells

Total: **100 points**. Grading combines the autograder (`tests/test_ps1.py`)
with a short read of the submitted code and interpretation. All work must run
offline and deterministically; a submission that only passes because it hard-
codes expected numbers, disables a check, or peeks at the held-out set earns no
credit for the affected part.

## Part A — Denoising — 30 points

- **`snr_db` (8).** Returns `10·log10(‖reference‖² / ‖estimate − reference‖²)`;
  a perfect estimate gives a large SNR and more noise gives a lower one.
- **`denoise` (10).** Projects the noisy images onto the eigen-subspace and
  reconstructs; leaves a signal already in the basis (near-)unchanged and beats
  the noisy input on a low-rank signal.
- **`best_rank_for_denoising` (12).** Returns the SNR-vs-rank curve and the rank
  that maximizes it (the true argmax, not a guess); the curve is single-peaked,
  and enough modes to span the signal beat far too few.

## Part B — Out-of-QC detection — 30 points

- **`reconstruction_anomaly_score` (12).** Per-image relative reconstruction
  error; non-negative, correctly shaped, and cleanly higher for images pushed
  off the subspace than for normal ones.
- **`detection_auc` (8).** Correct ROC-AUC — 1.0 for perfectly separated scores,
  0.5 for identical distributions.
- **`flag_threshold` (10).** The `(1 − max_false_alarm)` quantile of the normal
  scores; empirically bounds the false-alarm rate and tightens as the target
  shrinks.

## Quality control — 20 points

- **Leakage safety (10).** The subspace is fit on the normal-cell library only,
  never the held-out or corrupted images; the leakage guard passes and QC is
  printed *before* any result, per the course "QC before results" rule.
- **Honest data + sanity (10).** The dataset source (`real` vs `fallback`) is
  printed, and the full-rank reconstruction sanity check reports near-zero error.

## Interpretation & honesty — 15 points

- A clear `interpretation_block` -- a claim stated with the
  evidence actually generated (the single-peaked SNR curve, the leakage-checked
  split, the AUC with a false-alarm-bounded threshold).
- At least two honest, specific limitations — e.g. that the noise is synthetic
  additive Gaussian unlike real acquisition artifacts, that the SNR gain and AUC
  are single held-out estimates, or that a linear subspace can miss nonlinear
  novelties. Overclaiming is penalized; calibrated honesty is rewarded.

## Reproducibility — 5 points

- Runs top-to-bottom with a fixed seed; imports follow the repository policy
  (only `numpy` at module top level, heavier libraries inside function bodies);
  the file is `ruff`-clean under `E`, `F`, `I` at line length 100; and no public
  function signature was changed.
