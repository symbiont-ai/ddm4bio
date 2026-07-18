# PS4 — How Undersampled Can You Go? The Compressed-Sensing Limit

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/symbiont-ai/ddm4bio/blob/main/problem_sets/ps4_signals_cs/ps4_colab.ipynb)

**Work in the browser:** click the badge to open this problem set in Google Colab — no local setup required. You can also work locally (see below).

**Reading:** Kutz, *Data-Driven Modeling & Scientific Computation*, Chapter 14
(sparsity and compressed sensing; the FFT and L1 recovery).

Week 4 reconstructs **one** sparse signal from **one** set of compressed
measurements — it shows that compressed sensing *works*. This problem set asks the
question underneath that demo: **how few measurements can you get away with?**
Compressed sensing does not degrade gracefully — below a sharp threshold recovery
fails completely, above it recovery succeeds. You will map that limit, and watch it
move as the signal gets denser.

The sparse-signal generator (`make_sparse`) is **provided** — this problem set is
about mapping the sampling limit, not re-deriving the recovery algorithm. Recovery
itself is a one-call L1/Lasso fit you write in `recover` (scikit-learn's `Lasso`).
Fill in the functions marked `# TODO` in `student/ps4.py`. The autograder checks each
on small seeded synthetic signals, so keep the signatures exactly as given.

## Data

- **Sparse signals (synthetic, known ground truth)** — `make_sparse(n, k, seed)`
  builds a length-`n` signal with exactly `k` nonzero entries. Because scoring
  *recovery* needs the exact truth, the mapped signals are synthetic by design.
- **Real ECG compressibility** — `real_ecg_effective_sparsity()` loads a real
  MIT-BIH segment via `get_dataset("mitbih")` and reports how few Fourier
  coefficients hold 95% of its energy. This is the motivation: real biosignals are
  compressible, which is exactly what lets compressed sensing apply to them.

Seed everything through `ddm4bio.seed_everything()` (called in `main`).

## Part A — The recovery cliff

For a fixed sparsity, find the sharp failure→success transition. Implement:

- `measurement_matrix(m, n, rng)` — a random Gaussian sensing matrix of shape
  `(m, n)`, scaled by `1/√m`.
- `recover(signal, matrix)` — take measurements `y = matrix @ signal` and
  reconstruct them with an L1/Lasso fit (scikit-learn's `Lasso`, `fit_intercept=False`).
- `recovery_error(recovered, true_signal)` — relative L2 error
  `‖recovered − true‖ / ‖true‖`.
- `recovery_error_curve(signal, m_values, seed, n_trials)` — recovery error vs. the
  number of measurements, **averaged over `n_trials` random matrices per `m`**
  (averaging is what turns a noisy scatter into a sharp cliff).
- `min_measurements_for_recovery(signal, m_values, tol, seed)` — the smallest `m`
  whose averaged error drops below `tol`: the sampling limit for that signal.

## Part B — The phase transition

Now sweep sparsity too. Implement:

- `phase_transition(n, sparsities, m_values, tol, seed)` — for each sparsity `k`,
  the minimum measurements to recover a `k`-sparse signal. The result grows with
  `k` — a few measurements per nonzero — which is the boundary that governs every
  compressed acquisition (fewer measurements the sparser the signal).

## Quality control & interpretation (required)

The provided `run_qc` reports the signal length, its nonzero count, and how many of
the candidate measurement counts are genuinely undersampled (`m < n`) — printed
before any result. The provided `main` closes with a
`ddm4bio.interpret.interpretation_block`: state how sharply the cliff and phase
transition support the claim, and name the real
limitations — signals are *exactly* k-sparse in the canonical basis (real signals are
only approximately sparse in a wavelet/Fourier basis, which softens the cliff), the
transition is mapped at one tolerance and one measurement grid, and the L1 solver
uses a fixed regularization.

## Files

- `student/ps4.py` — your working file; fill in every `# TODO`.
- `rubric.md` — how this problem set is graded.
- `ps4_colab.ipynb` — one-click Google Colab launcher (badge at the top).
- The reference solution and the autograder are provided through the course and
  run automatically by GitHub Classroom.

## Running

```bash
python student/ps4.py          # runs until the first unimplemented function
```

To work in the browser instead, click the Colab badge at the top of this file.
The autograder runs automatically when you push to GitHub Classroom.
