# Problem Set 1 — The Eigen-Subspace as a Model of Normal Cells

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/symbiont-ai/ddm4bio/blob/main/problem_sets/ps1_eigen_recognition/ps1_colab.ipynb)

**Work in the browser:** click the badge to open this problem set in Google Colab — no local setup required. You can also work locally (see below).

**Reading:** Kutz, *Data-Driven Modeling & Scientific Computation*, Chapter 2
(the SVD and low-rank approximation).

Week 1 built an eigen-image basis and used it to *recognize* blood cells. This
problem set keeps that basis but asks a different question: if the top principal
axes capture what a **normal** cell looks like, what can the *residual* — the part
that doesn't fit — tell us? You will put the same subspace to two new uses on real
BloodMNIST.

The eigen-basis primitives (`eigen_basis`, `project`, `reconstruct`) and the data
loader are **provided** — this problem set is about what you *do* with the
subspace, not rebuilding it. Fill in the functions marked `# TODO` in
`student/ps1.py`. The autograder checks each on its own seeded fixtures, so keep
the signatures exactly as given.

## Data

Real **BloodMNIST** peripheral-blood-cell crops via `get_dataset("bloodmnist")`
(MedMNIST v2, CC BY 4.0), converted to grayscale and split into a *normal-cell
library* (train) and a held-out set. Real crops when the source is reachable, a
labelled bundled fallback (same shape) otherwise — the source is printed so you
always know which you got. Seed everything through `ddm4bio.seed_everything()`
(already called in `main`).

## Part A — Denoise by low-rank projection

The clean image lives (approximately) in a few principal axes, while additive
noise spreads across all of them. So projecting a noisy image onto the top-*k*
eigen-subspace and reconstructing it keeps the signal and discards most of the
noise — but only at the right rank: too few modes throw away signal, too many
re-admit noise, so the SNR-vs-rank curve rises, peaks, and falls.

Implement:

- `snr_db(estimate, reference)` — signal-to-noise ratio in decibels,
  `10·log10(‖reference‖² / ‖estimate − reference‖²)`.
- `denoise(x_noisy, mean, components)` — project onto the subspace and reconstruct
  (use the provided `project` / `reconstruct`).
- `best_rank_for_denoising(x_train, x_noisy, x_clean, candidate_ks)` — the rank
  whose denoising maximizes SNR against the clean images, and the SNR-vs-rank
  curve. Understand *why* the curve is single-peaked.

## Part B — Flag out-of-QC images by reconstruction error

An image that does not belong to the normal subspace reconstructs badly, so its
**reconstruction error is a novelty score**. Use it to flag corrupted acquisitions
(sensor noise, saturation/clipping, debris) before they contaminate an analysis.
(Defocus blur is the instructive *failure* case: it is low-pass, so a blurred cell moves
*toward* the smooth subspace and reconstruction error under-detects it.)

Implement:

- `reconstruction_anomaly_score(x, mean, components)` — per-image relative
  reconstruction error against the normal subspace.
- `detection_auc(scores, is_anomaly)` — ROC-AUC of the detector (chance 0.5,
  perfect 1.0).
- `flag_threshold(scores_normal, max_false_alarm)` — a cutoff set at the
  `(1 − max_false_alarm)` quantile of the normal scores, which bounds the
  false-alarm rate.

## Quality control & interpretation (required)

The provided `run_qc` prints a **leakage-checked** block (the normal-cell library
and the held-out set are disjoint) and a full-rank reconstruction sanity check
*before* any results. The provided `main` closes with a
`ddm4bio.interpret.interpretation_block`: state how much the denoising and
detection results support the claim, and name the
real limitations — the noise is synthetic additive Gaussian, the SNR gain and AUC
are single held-out estimates, and the subspace is linear.

## Files

- `student/ps1.py` — your working file; fill in every `# TODO`.
- `rubric.md` — how this problem set is graded.
- `ps1_colab.ipynb` — one-click Google Colab launcher (badge at the top).
- The reference solution and the autograder are provided through the course and
  run automatically by GitHub Classroom.

## Running

```bash
python student/ps1.py          # runs until the first unimplemented function
```

To work in the browser instead, click the Colab badge at the top of this file.
The autograder runs automatically when you push to GitHub Classroom.
