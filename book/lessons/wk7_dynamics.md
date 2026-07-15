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

# Week 7 - Discovering Dynamics: DMD, SINDy & Kalman

Biology is rarely static. Neural populations oscillate, gene circuits switch,
epidemics rise and fall, hearts beat. When a measurement is a *time series* --
a voltage trace, a fluorescence movie, a wearable-sensor stream -- the object we
actually want is not a snapshot but the **rule of motion** that generated it:
the operator that pushes the state one step forward in time. This week develops
three complementary data-driven tools for extracting that rule from data alone.
**Dynamic mode decomposition (DMD)** fits the best *linear* operator to a stream
of snapshots and reads its eigenvalues, telling us which spatial patterns grow,
decay, and oscillate. **Sparse identification of nonlinear dynamics (SINDy)**
goes further and writes down the actual *governing equations* -- a short,
human-readable list of terms -- by betting that real dynamics are sparse in a
library of candidate functions. The **Kalman filter** solves the reverse
problem: given a known (or assumed) dynamical model and a stream of *noisy*
observations, it fuses model and measurement into a state estimate cleaner than
either alone.

The through-line of the course holds here more sharply than anywhere else. A
recovered "governing equation" is a strong scientific claim, and the only way to
earn the right to make it on real data is to first show the method recovers a
*known* answer on a synthetic system whose equations we wrote ourselves. Every
method below is therefore exercised against a ground-truth fixture from
`ddm4bio.datasets.synthetic`, scored against the truth, and stress-tested with
noise until it breaks -- so we learn not just that it works, but exactly *where*
it stops working.

**Reading.** Kutz, *Data-Driven Modeling & Scientific Computation*, 2nd ed.,
Chapters 15 (dynamic mode decomposition), 20-21 (sparse regression / SINDy and
data assimilation). Read those chapters for the derivations; everything below is
explained in our own terms and run against our own fixtures.

**Learning goals.**

- Fit a linear operator to snapshot data with DMD and interpret its eigenvalues
  as growth, decay, and oscillation of the underlying system.
- Recover the sparse governing terms of a nonlinear system with SINDy, and
  *quantify* that recovery with precision/recall against a known term set.
- Filter a noisy trajectory with a Kalman filter and demonstrate that the fused
  estimate has lower error than the raw observations.
- Establish the noise level at which each method degrades -- and refuse to trust
  a result whose synthetic ground-truth recovery has already failed.

## Setup

We seed every random number generator and apply the course plotting style so
the figures below are deterministic and reproducible from a cold kernel.

```{code-cell} ipython3
import numpy as np

import ddm4bio
from ddm4bio import seed_everything
from ddm4bio.viz.style import set_style

seed_everything()
set_style()

print(f"ddm4bio version: {ddm4bio.__version__}")
```

## 1. DMD: recovering a spectrum from snapshots

Dynamic mode decomposition asks the simplest possible dynamical question: what
single matrix `A` best advances the state one time step, `x_{k+1} ≈ A x_k`?
Rather than form `A` explicitly (it can be enormous for imaging data), DMD
projects the dynamics onto the leading singular vectors of the snapshot stream
and eigendecomposes the small reduced operator. Its **eigenvalues** are the
prize: a discrete-time eigenvalue `λ` with `|λ| > 1` is a growing mode,
`|λ| < 1` a decaying mode, and a complex `λ` carries oscillation at a rate set
by its phase. In a biological recording -- say a spatiotemporal wave of cortical
activity, or a calcium-imaging movie -- these eigenvalues summarize which
coherent patterns persist and how fast they rotate.

To trust DMD we first hand it a system whose spectrum we *chose*.
`make_linear_dynamics` builds a real system matrix with a prescribed set of
eigenvalues (complex-conjugate pairs are encoded as real rotation-scale blocks),
scrambles it with a random orthogonal transform so no structure is handed to the
algorithm, and iterates a trajectory. We request two decaying oscillatory pairs
just inside the unit circle -- the qualitative signature of damped biological
rhythms.

```{code-cell} ipython3
from ddm4bio.datasets.synthetic import make_linear_dynamics

# Two complex-conjugate eigenvalue pairs: damped oscillations just inside |z|=1.
theta_fast, theta_slow = 0.4, 0.9
true_eigs = np.array([
    0.97 * np.exp(1j * theta_fast), 0.97 * np.exp(-1j * theta_fast),
    0.85 * np.exp(1j * theta_slow), 0.85 * np.exp(-1j * theta_slow),
])

linsys = make_linear_dynamics(true_eigs, n_steps=300, seed=7)

print(f"System matrix A:     {linsys.A.shape}")
print(f"Trajectory:          {linsys.trajectory.shape} (time x state)")
print(f"Prescribed spectrum: {np.round(np.sort_complex(linsys.eigs), 3)}")
```

DMD uses the *snapshot* convention with state variables in rows and time in
columns, so we transpose the `(n_time, state)` trajectory before handing it
over. With no noise and the full rank retained, the recovered eigenvalues should
match the prescribed spectrum to machine precision.

```{code-cell} ipython3
from ddm4bio.methods.dynamics import dmd

result = dmd(linsys.trajectory.T)  # snapshot convention: (state, time)

true_sorted = np.sort_complex(linsys.eigs)
dmd_sorted = np.sort_complex(result.eigenvalues)


def max_match_error(true_vals, est_vals):
    """Largest nearest-neighbour distance from each true eigenvalue to an estimate."""
    remaining = list(est_vals)
    worst = 0.0
    for tv in true_vals:
        j = int(np.argmin([abs(tv - ev) for ev in remaining]))
        worst = max(worst, abs(tv - remaining.pop(j)))
    return worst


print(f"True eigenvalues: {np.round(true_sorted, 4)}")
print(f"DMD eigenvalues:  {np.round(dmd_sorted, 4)}")
print(f"Max eigenvalue match error: {max_match_error(linsys.eigs, result.eigenvalues):.2e}")
```

The complex plane makes the recovery visual. Each true eigenvalue (open circle)
should sit directly under its DMD estimate (cross); the dashed unit circle marks
the stability boundary, and every mode lands *inside* it, correctly reporting a
damped system.

```{code-cell} ipython3
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(5.2, 5.2))

unit = np.exp(1j * np.linspace(0, 2 * np.pi, 400))
ax.plot(unit.real, unit.imag, linestyle="--", color="0.5", linewidth=1.0,
        label="unit circle |z| = 1")
ax.scatter(linsys.eigs.real, linsys.eigs.imag, s=140, facecolors="none",
           edgecolors="#0072B2", linewidths=2.0, label="true")
ax.scatter(result.eigenvalues.real, result.eigenvalues.imag, s=60, marker="x",
           color="#D55E00", linewidths=2.0, label="DMD")
ax.axhline(0, color="0.8", linewidth=0.6)
ax.axvline(0, color="0.8", linewidth=0.6)
ax.set_aspect("equal")
ax.set_xlabel("Re(λ)")
ax.set_ylabel("Im(λ)")
ax.set_title("DMD eigenvalues vs. known spectrum")
ax.legend(loc="lower left", fontsize=8)
fig;
```

**QC note.** The DMD spectrum reproduces the prescribed eigenvalues to roughly
`1e-15`. On clean, genuinely linear data DMD is essentially exact, which is
precisely why any *departure* from the truth on real recordings must be read as
noise, nonlinearity, or rank truncation -- not as a limitation we can excuse
away. DMD gives us an honest baseline: linear structure, recovered perfectly
when it is really there.

## 2. SINDy: recovering governing terms, and finding where it breaks

DMD assumes linearity. Most biology does not oblige. SINDy relaxes that
assumption by regressing the measured derivatives `ẋ` onto a *library* of
candidate nonlinear functions (here, polynomials up to a chosen degree) and then
enforcing **sparsity**: it repeatedly discards small coefficients and refits,
betting that the true dynamics use only a handful of terms. The output is not a
black box but an equation you can read.

Our ground-truth system is the Lorenz model from `make_lorenz` -- three coupled
states whose right-hand sides contain only linear and quadratic terms, so a
degree-2 library can represent them exactly. Written in the library's variable
names (`x0, x1, x2`), the true active terms are

- `dx0/dt = σ(x1 − x0)`  →  terms **`x0`, `x1`**
- `dx1/dt = x0(ρ − x2) − x1`  →  terms **`x0`, `x1`, `x0 x2`**
- `dx2/dt = x0 x1 − β x2`  →  terms **`x0 x1`, `x2`**

so the union of governing terms we must recover is
`{x0, x1, x2, x0 x1, x0 x2}`. Beyond biology, Lorenz is the standard stress-test
for equation discovery; here it stands in for any low-dimensional nonlinear
biological oscillator whose mechanism we hope to read off from a clean recording.

```{code-cell} ipython3
from ddm4bio.datasets.synthetic import make_lorenz
from ddm4bio.methods.dynamics import sindy_fit

lorenz = make_lorenz(t_max=20.0, n_steps=4000, noise=0.0)
true_terms = {"x0", "x1", "x2", "x0 x1", "x0 x2"}

fit = sindy_fit(lorenz.states, lorenz.t, poly_degree=2, threshold=0.5)

print("Recovered active terms per state equation:")
for state, active in enumerate(fit.active_terms_by_state):
    print(f"  dx{state}/dt : {sorted(active)}")
print(f"\nUnion of active terms: {sorted(fit.active_terms)}")
```

Reading the equation back is satisfying, but *reading* is not *scoring*. The
`term_recovery` validator compares the recovered active-term set against the
known truth and returns precision (are the terms we kept real?), recall (did we
find all the real terms?), and their harmonic mean, F1. This is the QC gate: on
a system whose equations we wrote ourselves, we demand F1 = 1.0 before believing
SINDy on anything we did not.

```{code-cell} ipython3
from ddm4bio.methods.validation import term_recovery

scores = term_recovery(true_terms, fit)
print(f"precision = {scores['precision']:.3f}")
print(f"recall    = {scores['recall']:.3f}")
print(f"F1        = {scores['f1']:.3f}")

assert scores["f1"] == 1.0, "SINDy failed to recover the known Lorenz terms on clean data"
print("\nQC PASS: exact term recovery on clean ground truth -- SINDy is licensed to proceed.")
```

Clean data is a fantasy. Real derivatives are estimated from noisy states by
finite differencing, which *amplifies* noise, so the honest question is: how
much observation noise can SINDy absorb before its equation is wrong? We sweep
the Lorenz observation-noise level and rescore term recovery at each step. The
`make_lorenz` fixture adds seeded Gaussian noise to every state sample, so the
sweep is fully deterministic.

```{code-cell} ipython3
noise_levels = np.array([0.0, 0.1, 0.5, 1.0, 2.0, 3.0, 5.0])
f1_curve = []
for noise in noise_levels:
    noisy = make_lorenz(t_max=20.0, n_steps=4000, noise=float(noise), seed=0)
    noisy_fit = sindy_fit(noisy.states, noisy.t, poly_degree=2, threshold=0.5)
    f1_curve.append(term_recovery(true_terms, noisy_fit)["f1"])
f1_curve = np.array(f1_curve)

for noise, f1 in zip(noise_levels, f1_curve):
    print(f"noise = {noise:>4} -> term-recovery F1 = {f1:.3f}")
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(6.5, 4.0))
ax.plot(noise_levels, f1_curve, marker="o", linewidth=1.8, color="#009E73")
ax.axhline(1.0, linestyle=":", color="0.5", linewidth=1.0)
ax.set_ylim(-0.05, 1.08)
ax.set_xlabel("Observation-noise standard deviation")
ax.set_ylabel("Term-recovery F1")
ax.set_title("Where SINDy breaks: term recovery vs. noise")
ax.annotate("exact recovery", xy=(0.1, 1.0), xytext=(0.5, 0.55),
            fontsize=9, arrowprops=dict(arrowstyle="->", color="0.5"))
fig;
```

**QC note.** SINDy recovers the Lorenz equations *exactly* up to a noise
standard deviation of about 0.1 (F1 = 1.0), and holds most of the structure --
full recall, one spurious term -- through roughly 1.0. Past that, finite-
difference derivative noise swamps the sparse regression: terms drop out, recall
collapses, and by a noise level of 2 the recovered equation is simply wrong.
This is the number to carry to real data: SINDy is trustworthy only when the
measurement noise is small relative to the dynamics, and here that boundary sits
near a noise-to-signal scale of order 0.1.

## 3. Kalman filtering: fusing a known model with noisy observations

DMD and SINDy *learn* a model. The Kalman filter *uses* one. Given a linear
state-transition model and a stream of noisy measurements, it runs a
predict/update recursion that optimally weights what the model predicts against
what the sensor reports, producing a filtered state estimate. This is the
workhorse of real-time biomedical signal processing -- continuous glucose
monitors, wearable ECG, eye-tracking, motion capture -- wherever a noisy sensor
streams data faster than we can average it.

The Kalman filter we use is exactly linear, so we assimilate a system it can
model exactly: a persistent oscillation from `make_linear_dynamics` with
eigenvalues on the unit circle, standing in for a steady biological rhythm. We
observe every state directly (`H = I`) but through additive Gaussian sensor
noise, and hand the filter the *true* transition matrix as its model `F`. The
process-noise covariance `Q` is tiny (the model is trusted) and the measurement
covariance `R` is set to the true noise level.

```{code-cell} ipython3
from ddm4bio.methods.dynamics import kalman_filter
from ddm4bio.methods.validation import reconstruction_error

# Marginally-stable rotation: a clean, persistent oscillation to assimilate.
theta = 0.30
osc_eigs = np.array([np.exp(1j * theta), np.exp(-1j * theta)])
system = make_linear_dynamics(osc_eigs, n_steps=200, seed=3)

F = system.A                    # the filter's model = the true transition matrix
truth = system.trajectory       # (time, 2) clean state trajectory

sigma_obs = 0.25
obs_rng = np.random.default_rng(0)
observations = truth + obs_rng.normal(0.0, sigma_obs, size=truth.shape)

H = np.eye(2)                   # observe both states directly
Q = 1e-5 * np.eye(2)            # trust the model (small process noise)
R = sigma_obs**2 * np.eye(2)    # measurement noise matched to the truth

filtered = kalman_filter(observations, F, H, Q, R, x0=truth[0], P0=np.eye(2))

raw_err = reconstruction_error(truth, observations, kind="rel_l2")
filt_err = reconstruction_error(truth, filtered, kind="rel_l2")
print(f"Raw observation error   (rel. L2): {raw_err:.4f}")
print(f"Filtered estimate error (rel. L2): {filt_err:.4f}")
print(f"Error reduction factor:            {raw_err / filt_err:.1f}x")
```

The reduction is the whole point: by folding in the dynamical model, the filter
drives the state error well below the raw sensor error. The panels below show
one state coordinate over time -- the noisy observations scattered around the
truth, and the filtered estimate tracking it closely -- alongside the per-step
error, which is visibly smaller and steadier for the filtered estimate.

```{code-cell} ipython3
fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
t = np.arange(truth.shape[0])

axes[0].plot(t, observations[:, 0], ".", color="0.6", markersize=4, label="noisy obs")
axes[0].plot(t, truth[:, 0], color="#0072B2", linewidth=2.0, label="truth")
axes[0].plot(t, filtered[:, 0], color="#D55E00", linewidth=1.4, linestyle="--",
             label="Kalman filtered")
axes[0].set_xlabel("Time step")
axes[0].set_ylabel("State 0")
axes[0].set_title("Assimilating a noisy oscillation")
axes[0].legend(loc="upper right", fontsize=8)

raw_step = np.linalg.norm(observations - truth, axis=1)
filt_step = np.linalg.norm(filtered - truth, axis=1)
axes[1].plot(t, raw_step, color="0.6", linewidth=1.2, label="raw obs error")
axes[1].plot(t, filt_step, color="#D55E00", linewidth=1.6, label="filtered error")
axes[1].set_xlabel("Time step")
axes[1].set_ylabel("Per-step L2 error")
axes[1].set_title("Filtered error stays below observation error")
axes[1].legend(loc="upper right", fontsize=8)
fig.suptitle("Kalman filter: model + noisy data > either alone")
fig;
```

**QC note.** With a correct linear model and a well-specified noise level, the
Kalman filter cuts the state error by roughly an order of magnitude relative to
the raw observations. That gain is contingent on the model being right: `F`
here *is* the true operator, and `R` *is* the true noise. On real data both are
estimated, and a mis-specified model would bias the filter toward its own wrong
predictions -- which is exactly why the DMD/SINDy step of learning an honest
model comes first.

## 4. Application: DMD on a real epidemic curve

The synthetic sections earned the methods their license; now we spend it on data
whose governing equations nobody wrote down. We pull an archived Johns Hopkins
CSSE COVID-19 confirmed-case time series through the course data layer. The
loader tries the real frozen archive first and caches it; with no network (or no
`pandas`) it returns a deterministic, clearly-labelled synthetic epidemic curve
with the *same* `(date, cases)` shape, so the analysis below runs identically
either way. We print the provenance so the reader always knows which they got.

```{code-cell} ipython3
from ddm4bio.datasets import get_dataset

covid = get_dataset("jhu_covid")  # payload: DataFrame with columns (date, cases)
print(f"source = {covid.source}")
print(f"provenance: {covid.provenance}")

cases = np.asarray(covid.payload["cases"], dtype=float)
print(f"cumulative confirmed-case series: {cases.shape[0]} daily points")
```

The raw series is *cumulative* confirmed cases. The dynamical object is its rate
of change, so we difference it into daily new cases, apply a one-week moving
average to suppress reporting-day artefacts, and take `log(1 + ·)` so the
exponential rise becomes an approximately linear ramp. We then isolate the
**early-onset window** -- a fixed span starting when incidence first lifts off --
because that is where a single linear operator is a defensible local model; a
full multi-wave series is emphatically not one linear system. A scalar time
series carries no spatial modes for DMD to find, so we embed it in a **time-delay
(Hankel)** coordinate: stacking `n_delays` shifted copies turns the 1-D signal
into a multi-row snapshot matrix whose linear operator encodes the local
dynamics (the delay-embedding trick Kutz develops alongside DMD).

```{code-cell} ipython3
new_cases = np.clip(np.diff(cases), 0.0, None)
smoothed = np.convolve(new_cases, np.ones(7) / 7.0, mode="valid")
signal = np.log1p(smoothed)


def hankel(x, n_delays):
    """Stack n_delays shifted copies into a (n_delays, n_time) snapshot matrix."""
    cols = x.size - n_delays + 1
    return np.stack([x[i:i + cols] for i in range(n_delays)])


# Onset = first day incidence exceeds 1% of its peak; take a fixed early window.
onset = int(np.argmax(smoothed > 0.01 * smoothed.max()))
window_len = min(60, signal.size - onset)
early = signal[onset:onset + window_len]

n_delays = 10
snapshots = hankel(early, n_delays)
n_time = snapshots.shape[1]
n_train = int(0.7 * n_time)  # fit on the first 70%, forecast the rest
print(f"onset day {onset}, window {window_len} days -> Hankel snapshots {snapshots.shape}")
```

Fitting DMD to the training portion gives a dominant discrete-time eigenvalue
whose magnitude is the per-day multiplicative factor of the leading delay mode:
`|lambda| > 1` is a growing epidemic, `< 1` a receding one, and `log|lambda|`
the continuous growth rate whose reciprocal (times `ln 2`) is a doubling time an
epidemiologist can act on. Because the operator is linear it is trustworthy only
for a *short* forecast, so we fit on the first 70% of the window and score
held-out relative-L2 error at 7- and 14-day horizons on the remainder.

```{code-cell} ipython3
epi_dmd = dmd(snapshots[:, :n_train], r=6)
lead = epi_dmd.eigenvalues[np.argmax(np.abs(epi_dmd.eigenvalues))]
rate = float(np.log(np.abs(lead)))

print(f"dominant |lambda|   : {np.abs(lead):.4f}")
print(f"per-day growth rate : {rate:+.4f}")
if rate > 0.002:
    print(f"leading mode growing: doubling time ~ {np.log(2) / rate:.1f} days")
elif rate < -0.002:
    print(f"leading mode receding: halving time ~ {np.log(2) / -rate:.1f} days")
else:
    print("leading mode ~ neutral (|lambda| ~ 1): window straddles rise and plateau")

time_index = np.arange(n_time)
dynamics = np.power(epi_dmd.eigenvalues[:, np.newaxis], time_index[np.newaxis, :])
recon = (epi_dmd.modes @ (dynamics * epi_dmd.amplitudes[:, np.newaxis])).real

for horizon in (7, 14):
    stop = min(n_train + horizon, n_time)
    err = reconstruction_error(snapshots[:, n_train:stop], recon[:, n_train:stop], "rel_l2")
    print(f"forecast horizon {horizon:>2}d : held-out rel-L2 = {err:.3f}")
```

The overlay tells the same story visually. On the current-value row of the Hankel
embedding we plot the smoothed early incidence against the DMD reconstruction;
the fit is tight through the training window and the short forecast, then drifts
once the linear model is extrapolated past the regime it was fit on.

```{code-cell} ipython3
current = early[n_delays - 1:]
days = np.arange(current.size) + onset

fig, ax = plt.subplots(figsize=(7.5, 4.2))
ax.plot(days, np.expm1(current), color="#0072B2", linewidth=1.8,
        label="smoothed daily new cases")
ax.plot(days, np.expm1(recon[-1]), color="#D55E00", linewidth=1.6, linestyle="--",
        label="DMD reconstruction / forecast")
ax.axvline(onset + n_train, color="0.5", linestyle=":", linewidth=1.0,
           label="end of training window")
ax.set_xlabel("Day of series")
ax.set_ylabel("Daily new cases")
ax.set_title(f"DMD on a real epidemic curve (source: {covid.source})")
ax.legend(loc="upper left", fontsize=8)
fig;
```

**QC note.** DMD does not "know" epidemiology; it reports the best local linear
operator over the window it is given, and its dominant eigenvalue reads off the
leading mode's per-day amplification. Held-out error stays small a week or two
out and grows past that -- the honest short-horizon boundary. On the clean
synthetic curve the leading mode is a clear growth mode with a sensible doubling
time; on real multi-wave data the same window may straddle a rise and its
plateau, so the leading mode can read near-neutral -- a caution the print-out
makes explicit rather than papering over. Either way the claim is about the near
future of a locally linear regime, never a forecast through the epidemic's
nonlinear turnover.

## 5. Application: Kalman filtering a real ECG

For the filter we assimilate a genuinely noisy physiological recording: a single
lead from the MIT-BIH Arrhythmia Database, pulled through the same data layer.
The real fetch uses PhysioNet via `wfdb`; offline it falls back to a
deterministic ECG-like waveform with the identical payload shape (`signal`,
`fs`, `sig_names`), so the filter and its plot render regardless.

```{code-cell} ipython3
ecg = get_dataset("mitbih")  # payload: {"signal": (n, ch), "fs": Hz, "sig_names": [...]}
print(f"source = {ecg.source}")
print(f"provenance: {ecg.provenance}")

fs = float(ecg.payload["fs"])
raw = np.asarray(ecg.payload["signal"], dtype=float)[:, :1]  # first lead
window = raw[:1500]  # a few seconds at 360 Hz
print(f"lead '{ecg.payload['sig_names'][0]}' at {fs:g} Hz, window = {window.shape[0]} samples")
```

Unlike the synthetic oscillation of Section 3, here we have no ground-truth clean
trace and no known transition matrix. The pragmatic, honest model is the
**random-walk** Kalman filter: each sample is assumed to persist (`F = H = I`),
observed through additive noise. The ratio of process variance `Q` to
measurement variance `R` sets the smoothing strength -- trusting the model more
(smaller `Q/R`) smooths harder. Because we cannot score against a clean signal,
we measure denoising as the drop in sample-to-sample **roughness** (mean absolute
first difference), a reference-free proxy for high-frequency noise.

```{code-cell} ipython3
identity = np.eye(1)
filtered = kalman_filter(window, F=identity, H=identity,
                         Q=0.02 * identity, R=0.5 * identity)


def roughness(z):
    """Mean absolute first difference -- a reference-free high-frequency-noise proxy."""
    return float(np.mean(np.abs(np.diff(z, axis=0))))


print(f"raw roughness      : {roughness(window):.4f}")
print(f"filtered roughness : {roughness(filtered):.4f}")
print(f"roughness reduction: {roughness(window) / roughness(filtered):.1f}x")
```

The panel overlays the raw lead and the filtered estimate over the same window.
The filter tracks the large QRS excursions while shaving the jitter between beats
-- the visible signature of a filter that trusts the data during fast transients
and the model during quiet stretches.

```{code-cell} ipython3
t_sec = np.arange(window.shape[0]) / fs

fig, ax = plt.subplots(figsize=(9.0, 3.6))
ax.plot(t_sec, window[:, 0], color="0.6", linewidth=0.9, label="raw ECG")
ax.plot(t_sec, filtered[:, 0], color="#D55E00", linewidth=1.3, label="Kalman filtered")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude (mV)")
ax.set_title(f"Random-walk Kalman filter on a real ECG lead (source: {ecg.source})")
ax.legend(loc="upper right", fontsize=8)
fig;
```

**QC note.** With no clean reference we cannot claim an error reduction the way
Section 3 could; we can only show the filter suppresses high-frequency roughness
while preserving beat morphology. That is the correct posture for real data: the
random-walk model is a deliberate, mis-specified simplification (a real ECG is
not a random walk), so it is a reasonable denoiser but not a validated state
estimator. The validated claim lives in Section 3, on synthetic ground truth;
this section only *applies* the tool.

## 6. Interpretation

Every ddm4bio analysis closes with an explicit interpretation block: a single
claim, an honest confidence level backed by named evidence, and a list of stated
limitations. For a week about *discovering* dynamics, the load-bearing questions
are which recovered terms we should actually trust and at what noise level the
discovery stops being reliable.

```{code-cell} ipython3
from ddm4bio.interpret import interpretation_block

break_noise = float(noise_levels[np.argmax(f1_curve < 1.0)])

block = interpretation_block(
    claim="On synthetic ground truth, DMD recovers a known linear spectrum to "
          "machine precision, SINDy recovers the exact Lorenz governing terms "
          "(F1 = 1.0) on clean data, and a Kalman filter cuts state error about "
          f"{raw_err / filt_err:.0f}x below the raw noisy observations.",
    confidence="high",
    limitations_list=[
        "All three results are on synthetic fixtures with known answers; real "
        "recordings are noisier, only approximately low-dimensional, and have "
        "no ground truth against which to score recovery.",
        f"SINDy's exact term recovery survives only up to observation noise of "
        f"about 0.1 and begins failing by ~{break_noise:g}; finite-difference "
        "derivatives amplify noise, so the trustworthy terms are the ones that "
        "persist across the low-noise regime, not any single fit.",
        "DMD assumes linear dynamics and a fixed operator; genuinely nonlinear "
        "or nonstationary biology violates that assumption and its eigenvalues "
        "then describe a linearization, not the true system.",
        "The Kalman result used the TRUE transition matrix and TRUE noise "
        "covariance; a mis-specified model or noise level degrades the filter "
        "and can bias it toward confident but wrong predictions.",
        "SINDy recovery depends on the candidate library containing the true "
        "terms (here a degree-2 polynomial basis exactly spans Lorenz); a "
        "wrong library cannot recover terms it does not contain.",
        "The real-data sections (a JHU COVID-19 curve for DMD, a MIT-BIH ECG "
        "lead for the Kalman filter) have no ground truth to score against; each "
        "prints its data source so a fallback run is never mistaken for real "
        "data, and their claims are deliberately narrower than the synthetic "
        "ones -- a short-horizon growth rate and a roughness reduction, not a "
        "validated recovery.",
    ],
    evidence=f"DMD max eigenvalue error ~1e-15; SINDy term-recovery F1 = "
             f"{scores['f1']:.2f} on clean Lorenz with precision and recall = 1.0; "
             f"Kalman relative-L2 error {filt_err:.3f} filtered vs. {raw_err:.3f} raw.",
)
print(block)
```

## Exercises

Your graded work for this week is **Problem Set 7 (PS7)**, distributed and
auto-graded through GitHub Classroom. Building on this lesson, PS7 asks you to:

- Push DMD past the clean case: add observation noise to the
  `make_linear_dynamics` trajectory, sweep the SVD truncation rank `r`, and
  report how the recovered eigenvalues drift as noise rises and rank is
  truncated -- then decide how many modes you would honestly keep.
- Map SINDy's failure boundary more finely than the lesson does: vary both the
  noise level and the sparsity `threshold`, and plot term-recovery F1 as a
  surface to find the threshold that best trades precision against recall near
  the breakdown noise.
- Assimilate a partially-observed system with the Kalman filter (observe only
  one of two states via a non-identity `H`) and quantify how state-estimate
  error grows as you hide more of the state.
- Write an interpretation block for each result using
  `ddm4bio.interpret.interpretation_block`, stating a defensible confidence
  level and naming the noise regime in which your conclusion holds.

Refer to the PS7 repository README for the submission and auto-grading details.
