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

# Week 5 - Dimensionality Reduction: SVD, PCA, Robust PCA & ICA

High-dimensional life-science measurements -- gene-expression matrices, imaging
stacks, multichannel physiological recordings -- are almost never as complex as
their raw dimensionality suggests. The interesting biology usually lives on a
low-dimensional surface embedded inside a much larger measurement space. This
lesson develops the linear-algebra machinery that finds that surface. You will
learn to read a **scree curve** to decide how many components a dataset truly
needs, to separate a signal into a **low-rank structure plus a sparse
corruption** with robust PCA, and -- the centerpiece -- to unmix statistically
independent sources with **independent component analysis (ICA)**. The
overarching skill is not running the algorithms (a few library calls do that)
but *validating* them: every method here is first exercised against a synthetic
fixture whose answer we already know, so that when we later point it at real
biological data we can state honestly how much to trust the result.

**Reading.** Kutz, *Data-Driven Modeling & Scientific Computation*, 2nd ed.,
Chapters 15-16 (the singular value decomposition and its use in PCA/ICA). Read
those chapters for the derivations; everything below is explained in our own
terms and run against our own fixtures.

**Learning goals.**

- Connect the singular value decomposition to principal component analysis and
  read explained-variance ratios off a scree plot.
- Distinguish PCA (maximize captured variance) from ICA (maximize statistical
  independence) and know which question each one answers.
- Use a known-ground-truth fixture to *quantify* recovery quality before
  trusting a method on real data.
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

## 1. PCA via the SVD

Principal component analysis rotates the coordinate axes of a dataset so that
the first axis points along the direction of greatest variance, the second along
the greatest remaining variance orthogonal to the first, and so on. Numerically
this is just the singular value decomposition of the mean-centered data matrix:
the right singular vectors are the principal axes, and the squared singular
values are proportional to the variance captured along each axis.

To see PCA work, we build a dataset that is *genuinely* low-dimensional. We draw
100 samples that live on a random 2-dimensional plane inside 8-dimensional
feature space, then sprinkle a little isotropic measurement noise on top. Because
the underlying generator has rank 2, an honest dimensionality-reduction method
should report that two components explain nearly all of the variance.

```{code-cell} ipython3
from ddm4bio.methods.decomposition import pca_reduce, explained_variance_ratio

rng = np.random.default_rng(0)

n_samples, n_features, true_rank = 100, 8, 2

# Rank-2 generator: latent scores (100 x 2) times loadings (2 x 8).
latent = rng.standard_normal((n_samples, true_rank))
loadings = rng.standard_normal((true_rank, n_features))
noise = 0.05 * rng.standard_normal((n_samples, n_features))
X = latent @ loadings + noise

scores = pca_reduce(X, n_components=2)
evr = explained_variance_ratio(X)

print(f"Data matrix X: {X.shape} (samples x features)")
print(f"PCA scores:    {scores.shape} (samples x components)")
print(f"Explained-variance ratio (all components):\n{np.round(evr, 4)}")
```

The scree plot below shows how quickly the explained variance decays. A sharp
"elbow" after the second component is the visual signature of a rank-2 dataset:
components beyond the elbow are capturing noise, not structure.

```{code-cell} ipython3
from ddm4bio.viz.plots import scree_plot

ax = scree_plot(evr)
ax.figure  # end the cell with the Figure so it renders in the notebook output
```

```{code-cell} ipython3
captured = float(evr[:true_rank].sum())
print(f"QC note: the first {true_rank} components capture "
      f"{captured:.1%} of total variance -- consistent with a rank-{true_rank} "
      f"generator plus small isotropic noise.")
```

**QC note.** Two components capture roughly 99.9% of the variance, exactly what
we expect when a rank-2 signal is buried under a thin layer of noise. The
remaining components each explain a negligible slice -- they are describing the
noise floor, not biology.

## 2. ICA with known ground truth

PCA asks "which directions carry the most variance?" ICA asks a different and
often more biologically relevant question: "which underlying signals are
*statistically independent*?" When several independent processes are linearly
mixed at each sensor -- think of overlapping fluorophores in an imaging channel,
or independent neural sources summed at a scalp electrode -- PCA will happily
find high-variance directions, but those directions are generally *mixtures*.
ICA is designed to invert the mixing and hand back the original sources.

The only honest way to trust an unmixing algorithm is to test it on data whose
sources we already know. The fixture below generates three independent,
non-Gaussian sources (ICA relies on non-Gaussianity), linearly mixes them
through a random well-conditioned matrix, and returns both the true sources and
the observed mixtures.

```{code-cell} ipython3
from ddm4bio.datasets.synthetic import make_mixed_sources

m = make_mixed_sources(3, 2000, seed=0)

print(f"True sources:   {m.sources.shape} (sources x samples)")
print(f"Mixing matrix:  {m.mixing.shape}")
print(f"Observations:   {m.observations.shape} (channels x samples)")
```

First, look at what we are given versus what we want to recover. The top row is
the ground truth (three cleanly separated waveforms); the bottom row is what a
sensor actually records -- three tangled mixtures in which no individual source
is visible.

```{code-cell} ipython3
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 3, figsize=(11, 4.5), sharex=True)
window = slice(0, 400)  # show a short window so the waveforms are legible

for j in range(3):
    axes[0, j].plot(m.sources[j, window])
    axes[0, j].set_title(f"True source {j + 1}")
    axes[1, j].plot(m.observations[j, window])
    axes[1, j].set_title(f"Mixed observation {j + 1}")
axes[0, 0].set_ylabel("True")
axes[1, 0].set_ylabel("Observed")
fig.suptitle("Blind source separation: ground truth (top) vs. mixtures (bottom)")
fig
```

Now unmix. `ica_unmix` runs FastICA on the observations and returns one
estimated source per row. We then score the recovery with
`source_recovery_score`, which optimally matches each estimated source to a true
source and reports the mean absolute correlation across the matched pairs -- a
number in `[0, 1]` where 1 means perfect recovery up to the sign and ordering
ambiguities that are inherent to ICA.

```{code-cell} ipython3
from ddm4bio.methods.decomposition import ica_unmix
from ddm4bio.methods.validation import source_recovery_score

est = ica_unmix(m.observations, 3, seed=0)
score = source_recovery_score(m.sources, est)

print(f"Estimated sources: {est.shape}")
print(f"Source-recovery score: {score:.4f}")
```

A score near 0.999 tells us the unmixing is essentially exact on this fixture.
The plot below confirms it visually: each recovered source is overlaid on its
matched true source. ICA does not preserve the *sign* or *order* of sources, so
before plotting we align each estimate to its best-matching truth by correlation
sign -- a cosmetic fix that changes nothing about the recovery quality.

```{code-cell} ipython3
from scipy.optimize import linear_sum_assignment

# Optimally match estimated sources to true sources by absolute correlation.
n_src = m.sources.shape[0]
corr = np.zeros((n_src, n_src))
for i in range(n_src):
    for j in range(n_src):
        a = m.sources[i] - m.sources[i].mean()
        b = est[j] - est[j].mean()
        corr[i, j] = np.corrcoef(a, b)[0, 1]

true_idx, est_idx = linear_sum_assignment(-np.abs(corr))

fig, axes = plt.subplots(1, 3, figsize=(11, 3), sharex=True)
for panel, (ti, ei) in enumerate(zip(true_idx, est_idx)):
    sign = np.sign(corr[ti, ei])  # flip estimate to match the truth's sign
    axes[panel].plot(m.sources[ti, window], label="true", linewidth=2)
    axes[panel].plot(sign * est[ei, window], label="recovered",
                     linewidth=1, linestyle="--")
    axes[panel].set_title(f"Source {panel + 1}")
    axes[panel].legend(loc="upper right", fontsize=8)
fig.suptitle("Recovered sources overlaid on ground truth")
fig
```

**Why this ordering matters.** We validated recovery on a fixture with a *known*
answer before ever touching real biological data. This is the non-negotiable
course rule: a method that cannot recover known synthetic sources has no business
being trusted on messy experimental measurements, where there is no ground truth
to check against. The synthetic score is your license to proceed -- or your
warning to stop.

## 3. Interpretation

Every ddm4bio analysis closes with an explicit interpretation block: a single
claim, an honest confidence level backed by named evidence, and a list of stated
limitations. This forces us to write down not just *what* we found but *how much*
we should believe it and *where* it could break.

```{code-cell} ipython3
from ddm4bio.interpret import interpretation_block

block = interpretation_block(
    claim="FastICA recovers the three independent sources from their linear "
          "mixtures with essentially perfect fidelity on this fixture.",
    confidence="high",
    limitations_list=[
        f"Result is on a synthetic fixture (score={score:.3f}); real data are "
        "noisier and only approximately linear mixtures.",
        "Sources were constructed to be non-Gaussian and independent -- the "
        "exact assumptions ICA needs; violate them and recovery degrades.",
        "The mixing matrix was well-conditioned by construction; near-singular "
        "mixing would make the inverse problem ill-posed.",
        "Sign and ordering of recovered sources are arbitrary and must be "
        "resolved by external reference, not by ICA itself.",
    ],
    evidence=f"source-recovery score = {score:.4f} against known ground truth, "
             "optimally matched across the three source pairs.",
)
print(block)
```

## Exercises

Your graded work for this week is **Problem Set 5 (PS5)**, distributed and
auto-graded through GitHub Classroom. Building on this lesson, PS5 asks you to:

- Push PCA past the easy case: study how the scree elbow blurs as you raise the
  noise level, and decide where you would honestly stop keeping components.
- Stress-test ICA by degrading its assumptions -- add a nearly-Gaussian source,
  or make the mixing matrix progressively more ill-conditioned -- and report the
  source-recovery score as a function of how badly the assumptions are violated.
- Apply robust PCA (`ddm4bio.methods.decomposition.rpca`) to separate a
  low-rank background from sparse corruptions, and validate the split against a
  fixture whose low-rank and sparse parts are known.
- Write an interpretation block for each result using
  `ddm4bio.interpret.interpretation_block`, with a defensible confidence level.

Refer to the PS5 repository README for the submission and auto-grading details.
