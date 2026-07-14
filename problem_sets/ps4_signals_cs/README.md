# Problem Set 4 — Signals, Time–Frequency Analysis, and Compressed Sensing

**Course:** Data-Driven Modeling & Scientific Computation for the Life Sciences
**Reading:** Kutz, *Data-Driven Modeling & Scientific Computation*, Chapter 14
(Fourier and time–frequency analysis, wavelets, and sparsity / compressed
sensing). Skim the surrounding sections on the FFT and on L1 (sparse) recovery.
**Slug:** `ps4`  ·  **Module you edit:** `student/ps4.py`

Biomedical signals are rarely stationary and are almost never sampled as
densely as we would like. A single global Fourier transform tells you *which*
frequencies are present but not *when*; a wavelet transform localizes structure
in both time and scale; and compressed sensing lets you reconstruct a signal
from far fewer measurements than the classical Nyquist rate — provided the
signal is sparse in some basis and the measurements are incoherent with that
basis. This problem set walks through all three ideas on offline synthetic
fixtures where the ground truth is known exactly, so every claim you make can be
checked against a number rather than a picture.

You will fill in the method logic in `student/ps4.py`. The data loaders, the
quality-control calls, and the `main()` driver are already wired for you — you
implement only the functions marked `# TODO`. Public function names and
signatures must not change; the autograder imports them directly.

## Data (everything offline)

No downloads, no network, no credentials. All fixtures come from the course
library and are deterministic under `GLOBAL_SEED`:

- **Nonstationary signal** — `load_nonstationary_signal()` builds a linear
  chirp (10 → 80 Hz) plus two time-gated tone bursts. Its spectral content
  changes over time, which is exactly what makes an STFT/wavelet view more
  informative than a single FFT.
- **ECG-like segment** — `load_ecg_segment()` returns a clean synthetic beat
  train (Gaussian Q/R/T deflections plus slow baseline wander) *and* a noisy
  copy with additive Gaussian noise of known standard deviation. Because the
  clean reference is known, the denoising SNR gain is measurable.
- **Sparse 1-D field** — `load_sparse_field()` wraps
  `ddm4bio.datasets.synthetic.make_sparse_signal` to produce a k-sparse field
  (a handful of isolated point sources on an empty grid), our stand-in for a
  spatially sparse MRI slice.

You use the library wrappers in `ddm4bio.methods.signals`
(`stft`, `wavelet_denoise`, `compressed_sensing_recon`), the ground-truth
generators in `ddm4bio.datasets.synthetic`, the metric
`ddm4bio.methods.validation.reconstruction_error`, the quality-control report
`ddm4bio.qc.qc_signals`, and the honesty helper
`ddm4bio.interpret.interpretation_block`.

---

## Part A — Method

Implement the core time–frequency and sparse-recovery primitives.

1. **`compute_fft(x, fs)`** — return the one-sided amplitude spectrum
   `(freqs, magnitude)` of a real signal. Use the real FFT and scale the
   magnitude to a single-sided amplitude so a unit-amplitude sinusoid reads off
   near 1.
2. **`compute_spectrogram(x, fs, nperseg)`** — return `(freqs, times, power)`
   from a short-time Fourier transform, where `power` is the squared magnitude
   of the complex STFT coefficients (frequencies in rows, time frames in
   columns). Use the library's `stft`.
3. **`wavelet_decompose(x, wavelet, level)`** — return the multilevel discrete
   wavelet coefficient list `[cA_level, cD_level, …, cD_1]`. Use periodization
   boundary handling so the coefficient count matches the input length and the
   transform is exactly invertible.
4. **`fourier_sensing_matrix(n, kept_indices)`** and **`cs_reconstruct(y, phi,
   alpha, seed)`** — build the real-valued k-space sensing operator (stack the
   real and imaginary parts of the retained DFT rows) and reconstruct the sparse
   field by L1-regularized least squares (Lasso) via
   `compressed_sensing_recon`.

## Part B — Application

Apply the primitives to two biomedical scenarios.

1. **Denoise an ECG-like segment.** Implement `snr_db(reference, estimate)` and
   `denoise_and_score(noisy, clean, wavelet)`. Wavelet-threshold the noisy
   segment and report the SNR before and after denoising along with the gain
   (in dB). Because the clean reference is known, the gain is a hard number, not
   a visual impression.
2. **Simulate accelerated MRI.** Implement `mri_undersample_reconstruct(field,
   ratio, …)`: retain a random fraction `ratio` of the field's Fourier
   coefficients (undersampled k-space) and reconstruct it two ways — by L1
   compressed sensing and by the naive *zero-filled* inverse transform (missing
   k-space set to zero). Then implement `error_vs_sampling_ratio(field, ratios,
   …)` to sweep the sampling ratio and report the relative-L2 reconstruction
   error of both methods as a function of how much data you keep.

## Part C — Quality control (required)

Do not report a reconstruction or a denoised trace without the checks that tell
you whether to trust it.

- **Denoising against a known SNR.** The `main()` driver runs `qc_signals` on
  the noisy segment with the clean reference so the achieved SNR is logged
  before any modeling. Your `denoise_and_score` must show a *positive* SNR gain;
  a nonpositive gain means the threshold/wavelet is wrong, not that the data is
  hopeless.
- **CS vs. zero-filled.** Every CS reconstruction is compared against the
  zero-filled baseline at the same sampling ratio. A CS result that does not
  beat zero-filling is not a compressed-sensing success.
- **Incoherent sampling.** Implement `incoherence_check(field, ratio, …)`, which
  spends the *same* measurement budget two ways: random Fourier (k-space) rows,
  which are highly incoherent with the spike basis in which the field is sparse,
  versus direct spatial point samples (rows of the identity), which are
  maximally coherent with it. Confirm that CS succeeds under incoherent sampling
  and fails under coherent sampling — incoherence is a precondition, not a
  detail.

## Part D — Interpretation & confidence

The driver prints an interpretation block via
`ddm4bio.interpret.interpretation_block`. In your own words (edit the claim if
your numbers differ), state:

- the **minimum sampling ratio** at which CS reconstruction is acceptable
  (relative-L2 error ≤ 0.05) for this fixture, read off your
  `error_vs_sampling_ratio` sweep and `minimum_acceptable_ratio`;
- an explicit **confidence level** (low / moderate / high) backed by the
  ground-truth error and known-SNR gain, not by eyeballing a plot;
- the **failure modes** you can name: too few measurements (below the phase
  transition), coherent sampling, and the gap between an exactly sparse fixture
  and real, only-approximately-sparse physiology.

Honesty norm: if you substitute a simpler method than nominally expected,
say so and say why. Overstating confidence is penalized more than a modest,
well-hedged result.

## How to run

```bash
python student/ps4.py          # loads data, runs QC, stops at your first TODO
pytest tests/test_ps4.py       # autograder (passes against the reference)
```

Everything is seeded and offline; two runs must produce identical numbers.
