# PS4 Grading Rubric — Signals & Compressed Sensing

Total: **100 points**. The autograder (`tests/test_ps4.py`) establishes the
performance floor; the remaining credit is for correct method logic, honest
quality control, and a defensible interpretation. All work must be offline,
seeded, and reproducible.

## Method correctness — 30 points

The Part A primitives are implemented correctly and used honestly.

- **FFT (6 pts).** `compute_fft` returns the one-sided spectrum with the right
  frequency bins and single-sided amplitude scaling; a known tone lands in the
  correct bin.
- **Spectrogram (6 pts).** `compute_spectrogram` returns `(freqs, times,
  power)` with power = squared STFT magnitude and shape `(n_freqs, n_times)`.
- **Wavelet decomposition (6 pts).** `wavelet_decompose` returns the correct
  multilevel coefficient list and is exactly invertible (periodization).
- **Sensing operator + CS solve (12 pts).** `fourier_sensing_matrix` has shape
  `(2·m, n)` and correctly stacks real/imag DFT rows; `cs_reconstruct` performs
  the L1 recovery via the library wrapper. Full credit requires the CS
  reconstruction to meet the error threshold on the seeded fixture.

## Application execution — 25 points

- **Denoising (10 pts).** `snr_db` and `denoise_and_score` are correct; the
  ECG-like segment is denoised with a positive, sensible SNR gain.
- **Accelerated-MRI CS (10 pts).** `mri_undersample_reconstruct` produces both
  the CS and zero-filled reconstructions with correct relative-L2 errors and
  metadata.
- **Error-vs-ratio sweep (5 pts).** `error_vs_sampling_ratio` returns aligned
  CS/zero-filled error curves across the requested ratios.

## Quality control — 25 points

- **Known-SNR validation (8 pts).** QC is run *before* results; the denoising
  gain is validated against the known clean reference and is positive.
- **CS vs. zero-filled (8 pts).** CS is compared against the zero-filled
  baseline and clearly beats it at adequate sampling.
- **Incoherence check (9 pts).** `incoherence_check` demonstrates that CS
  succeeds under incoherent (random Fourier) sampling and fails under coherent
  (identity point) sampling at an identical measurement budget.

## Interpretation & honesty — 15 points

- **Minimum sampling ratio (5 pts).** A specific minimum ratio for acceptable
  reconstruction is reported and matches the sweep.
- **Confidence statement (5 pts).** An explicit confidence level is given and
  justified by ground-truth error / known-SNR gain, produced through
  `interpretation_block`.
- **Named failure modes & honesty (5 pts).** Failure modes (too few
  measurements, coherent sampling, exact-vs-approximate sparsity) are named; any
  method substitution is disclosed. Overstated confidence loses credit here.

## Reproducibility — 5 points

- Runs offline with no downloads; all randomness is seeded with `GLOBAL_SEED`;
  repeated runs give identical numbers; the file is ruff-clean (E, F, I) and
  imports cleanly to the first unimplemented function.

---

### Autograder mapping

| Test | Rubric area |
| --- | --- |
| `test_compute_fft_recovers_known_tone` | Method: FFT |
| `test_compute_spectrogram_shapes` | Method: spectrogram |
| `test_wavelet_decompose_roundtrip` | Method: wavelet |
| `test_fourier_sensing_matrix_shape` | Method: sensing operator |
| `test_denoise_snr_gain_is_positive` | Application + QC: denoising |
| `test_cs_reconstruction_below_threshold_and_beats_zerofilled` | Method/Application/QC: CS |
| `test_error_vs_sampling_ratio_curve` | Application + Interpretation: min ratio |
| `test_incoherent_sampling_recovers_coherent_fails` | QC: incoherence |
| `test_reconstruction_is_deterministic` | Reproducibility |

A passing autograder is necessary but not sufficient: the QC and interpretation
credit is awarded for honest, well-argued analysis, not merely green tests.
