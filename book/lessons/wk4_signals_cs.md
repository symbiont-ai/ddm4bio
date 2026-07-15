---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.0
kernelspec:
  display_name: "Python 3"
  language: python
  name: python3
---

# Week 4 - Time-Frequency, Wavelets & Compressed Sensing

Physiological signals rarely hold still. A heart rate drifts with respiration,
a tremor waxes and wanes, an evoked response is a brief transient buried in
ongoing rhythm. The ordinary Fourier transform answers "which frequencies are
present?" but throws away *when* they were present -- exactly the information a
non-stationary biosignal carries. This lesson develops three tools that keep the
timing. The **short-time Fourier transform (STFT)** slides a window along the
signal and reports a local spectrum at each instant, turning a one-dimensional
trace into a two-dimensional time-frequency picture. **Wavelet denoising**
exploits the fact that a real signal concentrates into a few large coefficients
while broadband noise spreads thinly across many small ones, so thresholding the
small coefficients removes noise while sparing structure. **Compressed sensing**
turns sparsity from a nuisance into leverage: if a signal is sparse in some
basis, it can be reconstructed from far fewer measurements than the
Nyquist rate demands.

The unifying discipline of the course applies throughout: we never trust a
method on data whose answer we cannot check. Every technique below is first run
against a synthetic fixture with a *known* ground truth -- a known frequency
sweep, a known input SNR, a known sparse support -- so that the number we report
(a recovered ridge, an SNR gain, a reconstruction error) can be verified rather
than merely asserted.

**Reading.** Kutz, *Data-Driven Modeling & Scientific Computation*, 2nd ed.,
Chapter 14 (time-frequency analysis, wavelets, and compressed sensing). Read it
for the mathematical derivations of the windowed transform and the L1 recovery
guarantees; everything below is developed in our own terms and validated against
our own fixtures.

**Learning goals.**

- Read a spectrogram and recognize a non-stationary frequency component,
  confirming the STFT recovers a known instantaneous-frequency sweep.
- Denoise a signal by wavelet thresholding and *quantify* the improvement as an
  SNR gain measured against a known-SNR noisy input.
- Reconstruct a sparse signal from undersampled measurements and locate the
  minimum sampling ratio below which recovery collapses.
- Close with an explicit confidence-and-limitations statement about that
  minimum sampling ratio.

## Setup

We seed every random number generator and apply the course plotting style so all
figures below are deterministic and reproducible from a cold kernel.

```{code-cell} ipython3
import numpy as np

import ddm4bio
from ddm4bio import seed_everything
from ddm4bio.viz.style import set_style

seed_everything()
set_style()

print(f"ddm4bio version: {ddm4bio.__version__}")
```

## 1. Time-frequency analysis with the STFT

The Fourier transform of a whole recording gives one spectrum for the entire
duration -- fine for a stationary tone, useless for a signal whose frequency
content changes over time. The short-time Fourier transform fixes this by
computing many *local* spectra: it multiplies the signal by a short window,
transforms that snippet, then slides the window forward and repeats. The result
is a matrix of complex coefficients with frequency along the rows and time along
the columns; its squared magnitude is the **spectrogram**.

To show the STFT recovers real timing information, we build a fixture whose
answer we know exactly: a **linear chirp** whose instantaneous frequency sweeps
from 5 Hz to 90 Hz over six seconds -- a stand-in for a physiological rhythm that
accelerates, such as a heart rate ramping during exertion. Because we wrote down
the sweep, we know the true instantaneous frequency at every instant and can
check whether the spectrogram's high-power ridge tracks it.

```{code-cell} ipython3
from ddm4bio.methods.signals import stft

fs = 500.0            # sampling rate (Hz)
duration = 6.0        # seconds
t = np.arange(int(duration * fs)) / fs

f_start, f_end = 5.0, 90.0
# Instantaneous frequency rises linearly; the phase is its running integral.
true_inst_freq = f_start + (f_end - f_start) * t / duration
phase = 2.0 * np.pi * (f_start * t + 0.5 * (f_end - f_start) * t**2 / duration)
chirp = np.sin(phase)

freqs, times, Zxx = stft(chirp, fs=fs, nperseg=256)
power = np.abs(Zxx) ** 2

print(f"Signal:      {chirp.shape[0]} samples over {duration:.0f} s at {fs:.0f} Hz")
print(f"Spectrogram: {power.shape} (frequency bins x time frames)")
```

The panel below shows the spectrogram as a heatmap, with the *known* true
instantaneous frequency overlaid as a dashed line. If the STFT is doing its job,
the bright ridge of power should sit right under that line -- the transform has
localized each frequency to the time at which it actually occurred.

```{code-cell} ipython3
from ddm4bio.viz.plots import time_freq_panel

ax = time_freq_panel(times, freqs, power)
ax.plot(t, true_inst_freq, color="white", linestyle="--", linewidth=1.5,
        label="true instantaneous frequency")
ax.set_ylim(0, 120)
ax.legend(loc="upper left", fontsize=8, framealpha=0.6)
ax.figure;  # end the cell with the Figure so it renders in the notebook output
```

We can turn "the ridge follows the line" into a number. At each time frame the
frequency bin with the most power is the STFT's estimate of the dominant
frequency; comparing it to the known sweep gives a recovery error in Hz.

```{code-cell} ipython3
# Dominant-frequency ridge: the argmax frequency bin in each time column.
ridge_freq = freqs[np.argmax(power, axis=0)]
expected_freq = f_start + (f_end - f_start) * times / duration

# Ignore the first/last frames, where the window straddles the signal edges.
interior = (times > 0.5) & (times < duration - 0.5)
ridge_error_hz = np.median(np.abs(ridge_freq[interior] - expected_freq[interior]))

print(f"Median ridge error over the interior frames: {ridge_error_hz:.2f} Hz")
print(f"Frequency-bin spacing (resolution floor):    {freqs[1] - freqs[0]:.2f} Hz")
```

**QC note.** The ridge tracks the known sweep to within about one frequency-bin
width -- the STFT recovers *when* each frequency occurred, not just that it was
present. The residual error is not a bug: it is the time-frequency uncertainty
principle at work. A shorter window (`nperseg`) sharpens timing but blurs
frequency, and a longer one does the reverse. There is no window that is precise
in both at once, and choosing it is a modeling decision, not a default.

## 2. Wavelet denoising validated against a known SNR

Windowed Fourier analysis uses a fixed window for all frequencies. Wavelets
instead use short windows for high frequencies and long windows for low ones, so
they represent sharp transients and slow trends *both* compactly. That
compactness is the whole basis of wavelet denoising: a structured signal packs
its energy into a handful of large wavelet coefficients, while additive noise
scatters into many small ones. Zeroing the coefficients below a noise-scaled
threshold therefore removes noise while leaving the signal largely intact.

The only honest way to claim a denoiser "works" is to feed it a signal whose
clean version we possess and whose noise level we *set ourselves*. The signal
has to be one wavelets can represent sparsely -- piecewise-smooth stretches
punctuated by abrupt transitions, not a pure sinusoid (a sinusoid spreads across
many wavelet coefficients, so thresholding would gut it). We use the classic
"HeaviSine" test signal: a slow oscillation interrupted by two sharp level
shifts, a fair stand-in for a physiological baseline that drifts and then jumps
when an electrode is bumped. We add white Gaussian noise calibrated to a target
input SNR, so the input SNR is a known fact rather than an estimate.

```{code-cell} ipython3
from ddm4bio.methods.signals import wavelet_denoise
from ddm4bio.qc.signals import qc_signals

fs_d = 360.0
t_d = np.arange(int(4.0 * fs_d)) / fs_d
u = t_d / t_d[-1]  # normalized time in [0, 1]
# HeaviSine: a smooth oscillation with two abrupt jumps (piecewise-smooth).
clean = 4.0 * np.sin(4.0 * np.pi * u) - np.sign(u - 0.3) - np.sign(0.72 - u)

target_snr_db = 5.0
signal_power = float(np.mean(clean**2))
noise_sigma = np.sqrt(signal_power / 10 ** (target_snr_db / 10))

rng = np.random.default_rng(7)
noisy = clean + noise_sigma * rng.standard_normal(clean.shape)

denoised = wavelet_denoise(noisy, wavelet="sym4", mode="soft")
print(f"Target input SNR set by construction: {target_snr_db:.1f} dB")
```

We measure SNR with the library's signal QC helper, which -- given a clean
reference -- reports `10 * log10(signal power / residual power)` in decibels. We
run it on the noisy input and again on the denoised output; the difference is the
SNR gain.

```{code-cell} ipython3
snr_in = qc_signals(noisy, fs=fs_d, reference=clean).summary["snr_db"]
snr_out = qc_signals(denoised, fs=fs_d, reference=clean).summary["snr_db"]
snr_gain = snr_out - snr_in

print(f"Measured input SNR:  {snr_in:6.2f} dB  (matches the {target_snr_db:.1f} dB we set)")
print(f"Output SNR:          {snr_out:6.2f} dB")
print(f"SNR gain:            {snr_gain:+6.2f} dB")

# QC gate: denoising must not make the signal worse than it started.
assert snr_out > snr_in, "denoising reduced SNR -- method failed its validation gate"
print("QC gate passed: output SNR strictly exceeds input SNR.")
```

The measured input SNR lands on the value we dialed in, confirming our noise
calibration, and the output SNR is several decibels higher -- the denoiser
removed noise power without destroying the signal. The plot makes the three
traces visible on a short window.

```{code-cell} ipython3
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 1, figsize=(10, 5.5), sharex=True, sharey=True)
window = slice(None)  # show the full 4-second trace so both jumps are visible
axes[0].plot(t_d[window], clean[window]); axes[0].set_ylabel("clean")
axes[1].plot(t_d[window], noisy[window]); axes[1].set_ylabel("noisy")
axes[2].plot(t_d[window], denoised[window]); axes[2].set_ylabel("denoised")
axes[2].set_xlabel("Time (s)")
fig.suptitle(f"Wavelet denoising: {snr_in:.1f} dB in, {snr_out:.1f} dB out "
             f"({snr_gain:+.1f} dB gain)")
fig;
```

**Why the known SNR matters.** We did not eyeball the denoised trace and declare
it "cleaner." We *set* the noise level, so the input SNR was a fact, not an
estimate, and the gain is measured against that fact. A denoiser that cannot
raise the SNR of a signal whose noise we controlled has no claim on a real
recording, where the clean truth is unavailable and the gain can only be
inferred.

## 3. Biomedical application: a real ECG from MIT-BIH

The validation sections above ran on fixtures we authored. We now point the same
two tools -- the STFT and the wavelet denoiser -- at a *real* recording drawn
through the course data layer. `get_dataset("mitbih")` fetches one channel of
the MIT-BIH Arrhythmia Database from PhysioNet and caches it; when the network
or the optional `wfdb` dependency is unavailable it returns a labeled,
deterministic ECG-like *fallback* with the identical payload shape, so this
section runs the same way online or offline. We print the provenance so the
reader always knows which one they got.

```{code-cell} ipython3
from ddm4bio.datasets import get_dataset

ds = get_dataset("mitbih")  # tries PhysioNet + wfdb, caches; falls back offline
print(f"data source: {ds.source}")
print(f"provenance:  {ds.provenance}")

signal = np.asarray(ds.payload["signal"], dtype=float)  # n_samples x n_channels
fs_ecg = float(ds.payload["fs"])
sig_names = ds.payload["sig_names"]

# One channel, a manageable ~10-second window.
channel = 0
win_n = int(round(10.0 * fs_ecg))
ecg = signal[:win_n, channel]
t_ecg = np.arange(ecg.size) / fs_ecg
print(f"channel {channel} ({sig_names[channel]}): {ecg.size} samples "
      f"= {ecg.size / fs_ecg:.1f} s at {fs_ecg:.0f} Hz")
```

First the time-frequency view. An ECG is aggressively non-stationary -- each QRS
complex is a brief broadband transient punctuating a quieter baseline -- so its
spectrogram shows vertical stripes of power at the beat instants rather than a
single steady ridge. This is the same STFT machinery validated on the chirp,
now with no ground-truth sweep to overlay: the value is the qualitative picture
of *when* the spectral energy arrives.

```{code-cell} ipython3
from ddm4bio.viz.plots import time_freq_panel

f_ecg, t_frames, ecg_power = stft(ecg, fs=fs_ecg, nperseg=256)
ax = time_freq_panel(t_frames, f_ecg, np.abs(ecg_power) ** 2)
ax.set_ylim(0, 40)
ax.set_title(f"MIT-BIH ECG spectrogram ({ds.source} data)")
ax.figure;  # end the cell with the Figure so it renders in the notebook output
```

Now wavelet denoising. A real recording gives us no separately-known clean
version, so -- to keep a *measurable* SNR gain rather than an eyeballed one -- we
treat the loaded ECG window itself as the reference and add white Gaussian noise
calibrated to a target input SNR. The gain we report is therefore how much of
that *known, added* noise the wavelet threshold removes; the recording's own
intrinsic noise is folded into the reference and not counted.

```{code-cell} ipython3
target_snr_db_ecg = 5.0
ecg_power_mean = float(np.mean(ecg**2))
ecg_sigma = np.sqrt(ecg_power_mean / 10 ** (target_snr_db_ecg / 10))

rng_ecg = np.random.default_rng(11)
ecg_noisy = ecg + ecg_sigma * rng_ecg.standard_normal(ecg.shape)
ecg_denoised = wavelet_denoise(ecg_noisy, wavelet="sym4", mode="soft")

ecg_snr_in = qc_signals(ecg_noisy, fs=fs_ecg, reference=ecg).summary["snr_db"]
ecg_snr_out = qc_signals(ecg_denoised, fs=fs_ecg, reference=ecg).summary["snr_db"]
print(f"Real-ECG denoising: input {ecg_snr_in:.2f} dB -> output {ecg_snr_out:.2f} dB "
      f"({ecg_snr_out - ecg_snr_in:+.2f} dB)")
```

```{code-cell} ipython3
fig, axes = plt.subplots(2, 1, figsize=(10, 4.2), sharex=True)
win = slice(0, int(2.5 * fs_ecg))
axes[0].plot(t_ecg[win], ecg_noisy[win], linewidth=0.8)
axes[0].set_ylabel("noisy ECG")
axes[1].plot(t_ecg[win], ecg_denoised[win], linewidth=1.0)
axes[1].plot(t_ecg[win], ecg[win], linewidth=0.8, linestyle="--", alpha=0.7)
axes[1].set_ylabel("denoised")
axes[1].set_xlabel("Time (s)")
fig.suptitle("Wavelet denoising of a real MIT-BIH ECG window")
fig;
```

The recovered trace keeps the R-peaks crisp while suppressing the broadband
noise between beats -- exactly the transient-sparing behavior the wavelet basis
buys us over a naive low-pass filter. But note the honesty ceiling: our
"reference" is the raw recording, which itself carries intrinsic noise, so the
SNR gain measures only the removal of the noise we injected. On a truly
unlabeled clinical recording no such number exists, and the honest report would
fall back to reference-free proxies plus expert inspection.

## 4. Compressed sensing: ground truth first

Compressed sensing inverts the usual sampling logic. If a signal has only `k`
nonzero entries in some basis, it carries far less information than its length
suggests, and it can be recovered from a number of random linear measurements
that scales with `k` -- not with the full length. Recovery works by seeking the
*sparsest* signal consistent with the measurements, which the L1-regularized
least-squares (Lasso) solver approximates efficiently.

Following the course rule, we start from a fixture whose answer is exact:
`make_sparse_signal` returns a length-256 signal with only 8 nonzero
coefficients and hands back the known support and values. We then take
`m` random Gaussian measurements `y = Phi @ x`, reconstruct, and -- because we
hold the true `x` -- measure the relative L2 reconstruction error directly.

```{code-cell} ipython3
from ddm4bio.datasets.synthetic import make_sparse_signal
from ddm4bio.methods.signals import compressed_sensing_recon
from ddm4bio.methods.validation import reconstruction_error

n, k = 256, 8
sparse = make_sparse_signal(n, k, seed=0)
x_true = sparse.signal
print(f"Signal length n = {n}, nonzero coefficients k = {k}")
print(f"True support (indices): {sparse.support}")
```

The heart of compressed sensing is the trade-off between how many measurements
we take and how well we can reconstruct. We sweep the **sampling ratio**
`m / n` from heavy undersampling up to a quarter of the Nyquist count, and at
each ratio draw a fresh random measurement matrix, reconstruct, and record the
error.

```{code-cell} ipython3
ratios = np.linspace(0.05, 0.50, 19)
errors = np.empty_like(ratios)

for idx, ratio in enumerate(ratios):
    m = max(1, int(round(ratio * n)))
    rng_phi = np.random.default_rng(20250714 + m)      # deterministic per m
    Phi = rng_phi.standard_normal((m, n)) / np.sqrt(m)  # random Gaussian sensing
    y = Phi @ x_true
    x_hat = compressed_sensing_recon(y, Phi, alpha=1e-4, max_iter=20000, seed=0)
    errors[idx] = reconstruction_error(x_true, x_hat, kind="rel_l2")

for ratio, err in zip(ratios, errors):
    print(f"ratio={ratio:.3f}  m={int(round(ratio * n)):3d}  rel_L2_error={err:.4f}")
```

The error-versus-ratio curve reveals the signature of compressed sensing: a
sharp **phase transition**. Below a critical ratio the reconstruction is
essentially random; above it the error collapses to near zero and stays there.
We mark the smallest ratio that clears a 5% error tolerance.

```{code-cell} ipython3
tolerance = 0.05
acceptable = ratios[errors < tolerance]
min_ratio = float(acceptable.min()) if acceptable.size else float("nan")
min_m = int(round(min_ratio * n))

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(ratios, errors, marker="o", linewidth=1.5)
ax.axhline(tolerance, color="0.5", linestyle="--", linewidth=1.0,
           label=f"{tolerance:.0%} tolerance")
if acceptable.size:
    ax.axvline(min_ratio, color="C3", linestyle=":", linewidth=1.5,
               label=f"min ratio = {min_ratio:.2f} (m={min_m})")
ax.set_xlabel("Sampling ratio (m / n)")
ax.set_ylabel("Relative L2 reconstruction error")
ax.set_title(f"Compressed-sensing phase transition (n={n}, k={k})")
ax.legend(loc="upper right")
fig;
```

```{code-cell} ipython3
print(f"Minimum sampling ratio for < {tolerance:.0%} error: "
      f"{min_ratio:.3f}  ({min_m} of {n} measurements)")
print(f"That is {n / min_m:.1f}x fewer measurements than the signal length, "
      f"recovering a k={k}-sparse signal.")
```

**Ground truth first, again.** Every error above was computed against the known
`x_true`. This is what lets us state a *specific* minimum sampling ratio rather
than gesturing at "enough" measurements. On a real signal that is only
approximately sparse, the transition softens and the safe operating ratio must
be padded well above this idealized threshold.

## 5. Interpretation

Every ddm4bio analysis closes with an explicit interpretation block: one claim,
an honest confidence level backed by named evidence, and a list of stated
limitations. Here the claim concerns the minimum sampling ratio for acceptable
compressed-sensing recovery.

```{code-cell} ipython3
from ddm4bio.interpret import interpretation_block

block = interpretation_block(
    claim=(
        f"A {k}-sparse length-{n} signal is recovered to within "
        f"{tolerance:.0%} relative L2 error from about {min_ratio:.0%} of the "
        f"Nyquist measurements ({min_m} of {n}); below that ratio recovery "
        "collapses."
    ),
    confidence="high",
    limitations_list=[
        "Result is on a synthetic fixture that is exactly k-sparse in the "
        "canonical basis; real signals are only approximately sparse, which "
        "softens the transition and raises the required ratio.",
        "The measurement matrix is well-conditioned random Gaussian; "
        "structured or coherent sensing matrices need more measurements.",
        "Measurements were noise-free; measurement noise raises the error "
        "floor and shifts the usable ratio upward.",
        "The threshold ratio depends on the chosen 5% tolerance and on the "
        "Lasso regularization strength; both are modeling choices, not "
        "universal constants.",
    ],
    evidence=(
        f"relative L2 error dropped below {tolerance:.0%} at sampling ratio "
        f"{min_ratio:.2f} on a signal with known support, measured against "
        "the true sparse coefficients across a 19-point ratio sweep."
    ),
)
print(block)
```

## Exercises

Your graded work for this week is **Problem Set 4 (PS4)**, distributed and
auto-graded through GitHub Classroom. Building on this lesson, PS4 asks you to:

- Probe the time-frequency trade-off directly: recompute the chirp spectrogram
  across a range of window lengths (`nperseg`) and report how the ridge-error in
  Hz trades against the effective time resolution, then argue for a choice.
- Map wavelet denoising as a function of input SNR: sweep the target input SNR
  and plot the SNR gain, identifying the regime where denoising stops helping
  (and can even hurt), and compare two wavelet families.
- Trace the compressed-sensing phase transition as you vary the sparsity `k`,
  and check the recovered minimum sampling ratio against the theoretical
  `m ~ k log(n / k)` scaling.
- Write an interpretation block for each result using
  `ddm4bio.interpret.interpretation_block`, with a defensible confidence level.

Refer to the PS4 repository README for the submission and auto-grading details.
