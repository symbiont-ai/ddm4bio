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

# Week 2 - Curve Fitting, Noisy Differentiation & Sparsity

Biology hands us measurements, and we want *parameters*: the potency of a drug,
the rate at which a signal is changing, the short list of features that actually
carry the diagnosis. Each of those is an inverse problem -- we observe a noisy
output and must reason backward to the compact quantity that produced it. This
week develops three of the workhorse tools for that job. First, **nonlinear
curve fitting**: we fit a dose-response model and, crucially, report not just a
point estimate of the potency but its *uncertainty*. Second, **regularized
differentiation**: taking a derivative of noisy data is the textbook example of
an ill-posed problem, where the naive answer is dominated by noise, and we show
how a little regularization rescues it. Third, **sparse selection**: from thirty
correlated tumor measurements we ask the Lasso to keep only the handful that
matter, and then we check whether that handful is *stable*.

The connective tissue across all three is the same honesty discipline as the
rest of the course. A fit without an error bar is a guess dressed up as a
measurement; a derivative without regularization is noise dressed up as signal;
a feature list without a stability check is an accident dressed up as biology.
Every method below is first pointed at a fixture whose true answer we already
know, so we can quantify how well it recovers ground truth before we trust it on
real data.

**Reading.** Kutz, *Data-Driven Modeling & Scientific Computation*, 2nd ed.,
Chapters 4-5 (least-squares curve fitting, polynomial and spline
interpolation, and the numerical differentiation of noisy data). Read those for
the derivations and the finite-difference stencils; everything below is
explained in our own terms and run against our own fixtures.

**Learning goals.**

- Fit a Hill dose-response curve by nonlinear least squares and read the EC50
  and Hill coefficient *with* their standard errors.
- Judge a fit by its residual structure and goodness-of-fit, not by eye alone.
- Understand why differentiating noisy data is ill-posed, and use Tikhonov
  regularization to beat a naive finite difference against a known derivative.
- Select a sparse biomarker panel with the Lasso and test its stability by
  resampling, distinguishing robust features from lucky ones.
- Close every analysis with an explicit confidence-and-limitations statement.

## Setup

We seed all random number generators and apply the course plotting style so the
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

## 1. Fitting a dose-response curve with known ground truth

A dose-response experiment measures how a readout (cell viability, receptor
occupancy, fluorescence) changes as we sweep the concentration of a compound
across several orders of magnitude. The **Hill equation** is the standard
sigmoid model for that curve:

$$ y(x) = \text{bottom} + (\text{top} - \text{bottom})\,\frac{x^{n}}{\text{EC50}^{n} + x^{n}}. $$

Two of its four parameters are the ones pharmacologists actually report. The
**EC50** is the concentration producing a half-maximal response -- the standard
summary of potency. The **Hill coefficient** $n$ controls the steepness of the
transition and is often read as a measure of cooperativity. Fitting the curve is
easy; the discipline is reporting how *precisely* those two numbers are pinned
down by noisy data.

To test the fitter honestly we build a synthetic experiment whose true
parameters we choose ourselves. We lay out ten concentrations spaced
logarithmically from 0.01 to 100 (a typical five-log dose range), take three
replicate measurements at each concentration, and add Gaussian measurement noise.
An honest fitter should recover the true EC50 and Hill coefficient to within the
uncertainty it reports.

```{code-cell} ipython3
from ddm4bio.methods.fitting import hill, fit_hill

rng = np.random.default_rng(1)

# Ground-truth Hill parameters we will try to recover.
true = {"bottom": 5.0, "top": 95.0, "ec50": 1.2, "n": 1.8}

concentrations = np.geomspace(0.01, 100.0, 10)  # 10 doses over 5 logs
n_replicates = 3
x = np.repeat(concentrations, n_replicates)       # 3 replicates per dose

y_clean = hill(x, true["bottom"], true["top"], true["ec50"], true["n"])
y = y_clean + rng.normal(0.0, 2.5, size=x.size)   # additive measurement noise

print(f"Design: {concentrations.size} concentrations x {n_replicates} replicates "
      f"= {x.size} measurements")
print(f"True EC50 = {true['ec50']}, true Hill coefficient = {true['n']}")
```

Now fit the Hill model by nonlinear least squares. `fit_hill` wraps
`scipy.optimize.curve_fit`, which returns both the best-fit parameters and their
covariance matrix; the square roots of its diagonal are the 1-sigma standard
errors on each parameter. We print the fitted EC50 and Hill coefficient next to
their true values, each with a standard error.

```{code-cell} ipython3
fit = fit_hill(x, y, seed=0)

ec50_hat, ec50_se = fit["ec50"], fit["std_errors"][2]
n_hat, n_se = fit["hill_coeff"], fit["std_errors"][3]

print(f"fit converged: {fit['success']}")
print(f"EC50:            {ec50_hat:6.3f}  +/- {ec50_se:.3f}   (true {true['ec50']})")
print(f"Hill coeff n:    {n_hat:6.3f}  +/- {n_se:.3f}   (true {true['n']})")
print(f"bottom:          {fit['bottom']:6.3f}            (true {true['bottom']})")
print(f"top:             {fit['top']:6.3f}            (true {true['top']})")
```

A point estimate near the truth is reassuring, but the honest question is
whether the *true* value sits inside the confidence interval the fit reports. We
express the gap between fitted and true as a z-score -- the number of standard
errors between them. A z-score comfortably inside +/-2 means the true value is
consistent with the fit at roughly the 95% level; a large z-score would warn
that the fit is biased or the error bar is too optimistic.

```{code-cell} ipython3
z_ec50 = (ec50_hat - true["ec50"]) / ec50_se
z_n = (n_hat - true["n"]) / n_se

print(f"EC50 recovered within {z_ec50:+.2f} standard errors of truth")
print(f"Hill coefficient recovered within {z_n:+.2f} standard errors of truth")
```

The figure below overlays the fitted curve on the replicate measurements. Dose
is plotted on a log axis (the natural scale for a five-log concentration sweep),
the true curve is drawn as a dashed reference, and a vertical marker shows the
fitted EC50 with its uncertainty shaded.

```{code-cell} ipython3
import matplotlib.pyplot as plt

xx = np.geomspace(concentrations.min(), concentrations.max(), 300)
y_fit = hill(xx, *fit["params"])
y_true_curve = hill(xx, true["bottom"], true["top"], true["ec50"], true["n"])

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.scatter(x, y, s=28, alpha=0.7, label="measurements", zorder=3)
ax.plot(xx, y_fit, linewidth=2, label="fitted Hill curve")
ax.plot(xx, y_true_curve, linestyle="--", linewidth=1.5, label="true curve")
ax.axvspan(ec50_hat - ec50_se, ec50_hat + ec50_se, alpha=0.15,
           color="0.4", label="EC50 +/- 1 s.e.")
ax.axvline(ec50_hat, color="0.4", linewidth=1)
ax.set_xscale("log")
ax.set_xlabel("Concentration (arb. units, log scale)")
ax.set_ylabel("Response (% of maximum)")
ax.set_title("Hill dose-response fit vs. ground truth")
ax.legend(loc="upper left", fontsize=9)
fig
```

**QC: goodness of fit and residual structure.** A good fit is not just close on
average -- its residuals should look like structureless noise. If the residuals
curved systematically with concentration, that would signal a *model*
mismatch (wrong functional form) rather than mere measurement scatter. We report
the coefficient of determination $R^2$ and plot the residuals against dose,
looking for the flat, patternless band that signals a well-specified model.

```{code-cell} ipython3
y_hat = hill(x, *fit["params"])
residuals = y - y_hat
ss_res = float(np.sum(residuals**2))
ss_tot = float(np.sum((y - y.mean())**2))
r_squared = 1.0 - ss_res / ss_tot
rmse = float(np.sqrt(np.mean(residuals**2)))

print(f"R^2  = {r_squared:.4f}")
print(f"RMSE = {rmse:.3f} response units "
      f"(noise was drawn with sigma = 2.5)")

fig, ax = plt.subplots(figsize=(7, 3))
ax.axhline(0.0, color="0.5", linewidth=1)
ax.scatter(x, residuals, s=28, alpha=0.7)
ax.set_xscale("log")
ax.set_xlabel("Concentration (log scale)")
ax.set_ylabel("Residual")
ax.set_title("Residuals vs. dose (should be a structureless band)")
fig
```

**QC note.** The RMSE lands close to the 2.5-unit noise we injected, and the
residuals scatter around zero with no visible trend across the dose range --
exactly the signature of a correctly specified model whose only error is the
measurement noise we put in. The fit has not "used up" the data explaining
structure that isn't there.

## 2. Differentiating a noisy signal

Many biological questions are really questions about a *rate*: the velocity of a
growth curve, the acceleration of a tumor volume, the slope of a metabolic
trace. That means differentiating measured data -- and differentiation is the
canonical ill-posed problem. The derivative operator amplifies high frequencies,
and measurement noise is almost all high frequency, so a naive finite difference
turns a barely-visible wiggle in the data into a wild oscillation in the
estimated derivative.

To see this cleanly we again start from ground truth: a pure sine wave, whose
derivative we know analytically ($\frac{d}{dt}\sin(2\pi f t) = 2\pi f\cos(2\pi f t)$).
We sample it on a uniform grid, corrupt it with a small amount of noise, and
then compare two derivative estimates against the exact answer -- a plain
finite difference (`numpy.gradient`) versus the Tikhonov-regularized derivative
from `regularized_derivative`, which finds the smooth function whose integral
best matches the data.

```{code-cell} ipython3
from ddm4bio.methods.fitting import regularized_derivative

rng = np.random.default_rng(0)

n = 200
t = np.linspace(0.0, 1.0, n, endpoint=False)
dt = t[1] - t[0]
freq = 2.0

clean = np.sin(2.0 * np.pi * freq * t)
true_derivative = 2.0 * np.pi * freq * np.cos(2.0 * np.pi * freq * t)  # analytic
noisy = clean + rng.normal(0.0, 0.05, size=n)  # only 5% noise on the signal

finite_diff = np.gradient(noisy, dt)
reg_deriv = regularized_derivative(noisy, dt, lam=1e-2)

def rel_error(estimate):
    """Relative L2 error of a derivative estimate against the analytic truth."""
    return float(np.linalg.norm(estimate - true_derivative)
                 / np.linalg.norm(true_derivative))

print(f"Signal noise level: 5% of amplitude")
print(f"Finite-difference   relative error: {rel_error(finite_diff):.3f}")
print(f"Regularized         relative error: {rel_error(reg_deriv):.3f}")
```

The numbers tell the whole story: a 5% wiggle in the *signal* becomes a roughly
70% error in the naive derivative, while regularization keeps the error near
10%. The plot makes the mechanism visible. The finite difference (top) is buried
in noise; the regularized estimate (bottom) tracks the true cosine closely.

```{code-cell} ipython3
fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

axes[0].plot(t, true_derivative, linewidth=2, label="true derivative")
axes[0].plot(t, finite_diff, linewidth=1, alpha=0.8, label="finite difference")
axes[0].set_ylabel("dy/dt")
axes[0].set_title("Naive finite difference amplifies the noise")
axes[0].legend(loc="upper right", fontsize=9)

axes[1].plot(t, true_derivative, linewidth=2, label="true derivative")
axes[1].plot(t, reg_deriv, linewidth=1.5, linestyle="--",
             label="regularized derivative")
axes[1].set_xlabel("t")
axes[1].set_ylabel("dy/dt")
axes[1].set_title("Tikhonov regularization recovers the derivative")
axes[1].legend(loc="upper right", fontsize=9)
fig
```

**Choosing the regularization strength.** The smoothing parameter `lam` is a
dial between two failure modes: too little and noise leaks through, too much and
genuine features get flattened. Sweeping it shows a broad basin of good values
rather than a single knife-edge -- reassuring, because it means we do not have
to tune the parameter to unrealistic precision.

```{code-cell} ipython3
lambdas = [1e-4, 1e-3, 1e-2, 3e-2, 1e-1]
errors = [rel_error(regularized_derivative(noisy, dt, lam=lam)) for lam in lambdas]

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(lambdas, errors, marker="o", linewidth=1.5, label="regularized")
ax.axhline(rel_error(finite_diff), color="0.5", linestyle="--",
           label="finite difference")
ax.set_xscale("log")
ax.set_xlabel("Regularization strength lambda")
ax.set_ylabel("Relative L2 error vs. truth")
ax.set_title("A broad basin of good regularization strengths")
ax.legend(loc="center right", fontsize=9)
fig
```

## 3. Sparse biomarker selection on a real dataset

We now leave synthetic fixtures for a biomedical dataset that ships inside
scikit-learn: the Wisconsin Diagnostic Breast Cancer measurements. Each of 569
tumors is described by 30 features computed from a digitized image of a
fine-needle aspirate -- radius, texture, concavity and so on, each summarized
as a mean, a standard error, and a "worst" (largest) value -- and labeled
benign or malignant. Many of these 30 features are strongly correlated with one
another. A clinician does not want thirty numbers; they want the *few* that
carry the diagnostic signal.

The **Lasso** is built for exactly this. By penalizing the sum of absolute
coefficients it drives most of them to *exactly* zero, keeping only a compact
subset. We standardize the features first (so the penalty treats them on equal
footing) and fit the Lasso at a penalty strong enough to force a small panel.

```{code-cell} ipython3
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

from ddm4bio.methods.fitting import lasso_select

data = load_breast_cancer()
feature_names = data.feature_names
X = StandardScaler().fit_transform(data.data)  # zero-mean, unit-variance columns
y = data.target.astype(float)                   # 0 = malignant, 1 = benign

print(f"Design matrix: {X.shape[0]} tumors x {X.shape[1]} features")

result = lasso_select(X, y, alpha=0.05, seed=0)
selected = result["selected"]

print(f"\nLasso kept {selected.size} of {X.shape[1]} features "
      f"(alpha = {result['alpha']}):")
for idx in selected:
    print(f"  {feature_names[idx]:24s}  coef = {result['coefficients'][idx]:+.3f}")
```

**QC: stability of the selected features.** A single Lasso fit is one draw from
one sample. If we had collected a slightly different cohort, would we get the
same panel? The honest way to find out is to *resample*: we draw many bootstrap
replicates of the patients, re-run the selection on each, and count how often
each feature is chosen. Features selected in nearly every replicate are robust;
features that flicker in and out are artifacts of this particular sample and
should not be trusted as biomarkers.

```{code-cell} ipython3
n_boot = 50
n_patients = X.shape[0]
selection_counts = np.zeros(X.shape[1])

boot_rng = np.random.default_rng(0)
for _ in range(n_boot):
    idx = boot_rng.integers(0, n_patients, size=n_patients)  # sample with replacement
    boot_result = lasso_select(X[idx], y[idx], alpha=0.05, seed=0)
    selection_counts[boot_result["selected"]] += 1

selection_freq = selection_counts / n_boot

# Rank features by how often they survived resampling.
order = np.argsort(-selection_freq)
print(f"Selection frequency across {n_boot} bootstrap resamples:")
print("(* marks features also chosen on the full sample)\n")
for idx in order[:10]:
    star = "*" if idx in selected else " "
    print(f"  {star} {feature_names[idx]:24s}  {selection_freq[idx]:.2f}")
```

The bar chart below makes the split obvious. A few features -- the "worst" (i.e.
largest-region) concave-points, radius, and texture measurements -- are selected
in essentially every resample, marking them as the stable diagnostic core.
Others hover near a coin flip and would be irresponsible to report as
established biomarkers on this evidence.

```{code-cell} ipython3
top = order[:10]
fig, ax = plt.subplots(figsize=(8, 4.5))
colors = ["#0072B2" if idx in selected else "#D55E00" for idx in top]
ax.barh(range(len(top)), selection_freq[top], color=colors)
ax.set_yticks(range(len(top)))
ax.set_yticklabels([feature_names[idx] for idx in top])
ax.invert_yaxis()  # most frequent at the top
ax.axvline(0.5, color="0.5", linestyle="--", linewidth=1, label="coin flip")
ax.set_xlabel("Fraction of bootstrap resamples that selected the feature")
ax.set_title("Feature-selection stability under resampling")
ax.legend(loc="lower right", fontsize=9)
fig
```

```{code-cell} ipython3
stable = [feature_names[idx] for idx in order if selection_freq[idx] >= 0.8]
print(f"Features selected in >= 80% of resamples ({len(stable)}):")
for name in stable:
    print(f"  - {name}")
```

## 4. Interpretation

Every ddm4bio analysis closes with an explicit interpretation block: a single
claim, an honest confidence level backed by named evidence, and a list of stated
limitations. Here we make two claims -- one about the recovered potency, one
about which biomarkers are trustworthy -- and pin each to the evidence we
actually generated above.

```{code-cell} ipython3
from ddm4bio.interpret import interpretation_block

n_stable = int(np.sum(selection_freq >= 0.8))

block = interpretation_block(
    claim=(
        f"The Hill fit recovers the true potency (EC50 = {ec50_hat:.2f} +/- "
        f"{ec50_se:.2f}, true 1.20) and cooperativity within its reported "
        f"uncertainty, and the breast-cancer biomarker panel has a stable core "
        f"of {n_stable} features (led by worst concave points, worst radius, "
        f"and worst texture) that survive resampling."
    ),
    confidence="high",
    limitations_list=[
        f"The EC50 confidence interval assumes Gaussian, homoscedastic noise; "
        f"real assays often have concentration-dependent variance that would "
        f"widen the true interval beyond the reported +/-{ec50_se:.2f}.",
        "Standard errors from curve_fit are asymptotic (linearized) and can "
        "understate uncertainty for a small design like 10 doses x 3 replicates.",
        "The Hill fit was validated on a synthetic curve with a known form; a "
        "genuinely biphasic or model-mismatched response would fit poorly "
        "despite a clean-looking single-model R^2.",
        "Lasso stability was assessed at one fixed penalty (alpha = 0.05); a "
        "different penalty changes the panel size, and the borderline features "
        "(worst smoothness, mean texture) are penalty-sensitive.",
        "Selection frequency measures reproducibility on THIS cohort, not "
        "biological causality or generalization to a new patient population.",
    ],
    evidence=(
        f"EC50 recovered within {z_ec50:+.2f} s.e. of truth with R^2 = "
        f"{r_squared:.3f} and structureless residuals; regularized "
        f"differentiation cut relative error from {rel_error(finite_diff):.2f} "
        f"to {rel_error(reg_deriv):.2f} against the analytic derivative; "
        f"{n_stable} features selected in >= 80% of {n_boot} bootstrap resamples."
    ),
)
print(block)
```

## Exercises

Your graded work for this week is **Problem Set 2 (PS2)**, distributed and
auto-graded through GitHub Classroom. Building on this lesson, PS2 asks you to:

- Push the dose-response fit past the easy case: study how the EC50 standard
  error grows as you thin the number of concentrations or replicates, and find
  the design at which the true value starts falling outside the reported
  interval more often than it should.
- Stress-test regularized differentiation by raising the noise level and by
  differentiating a signal with a sharp feature (not a pure sinusoid); report
  the relative error of `regularized_derivative` versus a plain finite
  difference as a function of `lam`, and locate the bias-variance sweet spot.
- Trace the full Lasso path on `load_breast_cancer` by sweeping `alpha`, plot
  how many features survive at each penalty, and use bootstrap selection
  frequency (as in Section 3) to report a defensible minimal biomarker panel.
- Write an interpretation block for each result using
  `ddm4bio.interpret.interpretation_block`, with a confidence level you can
  defend from the evidence you generated.

Refer to the PS2 repository README for the submission and auto-grading details.
