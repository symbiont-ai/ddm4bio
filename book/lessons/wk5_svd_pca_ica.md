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

To make PCA concrete we point it at real single-cell data. The PBMC3k dataset
is a classic 10x Genomics assay of peripheral-blood mononuclear cells from a
healthy donor: thousands of cells, each a sparse vector of gene counts. We pull
it through the course data layer, which fetches the genuine 10x matrix when it
can and otherwise returns a structurally identical synthetic single-cell matrix
so the analysis runs anywhere. The provenance line printed below tells you which
one you received; the analysis is written to work for either.

```{code-cell} ipython3
from ddm4bio.datasets import get_dataset
from ddm4bio.methods.decomposition import explained_variance_ratio, pca_reduce

ds = get_dataset("pbmc3k")
payload = ds.payload

# get_dataset returns a real AnnData (with .X) or a labelled fallback dict.
if hasattr(payload, "X"):
    counts = payload.X
    labels = None
    obs = getattr(payload, "obs", None)
    if obs is not None:
        for col in ("cell_type", "cell_types", "louvain", "leiden", "bulk_labels"):
            if col in obs:
                labels = np.asarray(obs[col])
                break
else:
    counts = payload["counts"]
    labels = np.asarray(payload["labels"])

# 10x data ships as a sparse matrix; densify for the linear algebra below.
counts = np.asarray(counts.toarray() if hasattr(counts, "toarray") else counts, dtype=float)

print(f"[pbmc3k] source={ds.source}: {ds.provenance}")
groups = "unlabelled" if labels is None else f"{np.unique(labels).size} label groups"
print(f"expression matrix: {counts.shape[0]} cells x {counts.shape[1]} genes ({groups})")
```

Raw UMI counts are heavy-tailed and vary in sequencing depth from cell to cell,
so we apply the standard single-cell transform before any linear algebra:
normalize each cell to a common library size, then take `log1p`. We then keep
the most variable genes, which concentrates the biological signal and keeps the
full-width real matrix (tens of thousands of genes) tractable.

```{code-cell} ipython3
# Library-size normalize, then log1p-compress the counts.
library = counts.sum(axis=1, keepdims=True)
library[library == 0] = 1.0
target = float(np.median(counts.sum(axis=1)))
log_counts = np.log1p(counts / library * target)

# Restrict to the top-variance genes (all of them when the matrix is already small).
n_keep = min(1000, log_counts.shape[1])
top_var = np.argsort(log_counts.var(axis=0))[::-1][:n_keep]
expr = log_counts[:, top_var]

scores = pca_reduce(expr, n_components=2)
evr = explained_variance_ratio(expr)

print(f"kept {expr.shape[1]} high-variance genes; PCA scores {scores.shape}")
print(f"leading explained-variance ratios: {np.round(evr[:5], 4)}")
```

Real expression data does not collapse to an exact low rank the way a clean
synthetic fixture does; instead the scree curve decays smoothly, and the "elbow"
is a judgement call about where genuine structure fades into the long tail of
biological and technical noise. We plot only the leading components, since the
tail is a near-flat noise floor.

```{code-cell} ipython3
from ddm4bio.viz.plots import scree_plot

n_show = min(20, evr.size)
ax = scree_plot(evr[:n_show])
ax.figure;  # end the cell with the Figure so it renders in the notebook output
```

Projecting the cells onto their first two principal components gives the
familiar single-cell scatter. When the payload carries cell labels we colour by
them; the real 10x matrix ships unlabelled, so the same cell then renders a
single-colour cloud whose structure we read from the loadings instead.

```{code-cell} ipython3
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(5.5, 4.5))
if labels is not None:
    for g in np.unique(labels):
        sel = labels == g
        ax.scatter(scores[sel, 0], scores[sel, 1], s=12, label=str(g))
    ax.legend(title="label", fontsize=8, loc="best")
else:
    ax.scatter(scores[:, 0], scores[:, 1], s=12)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title("PBMC cells in principal-component space")
fig;
```

Finally we quantify how much of each leading component lines up with the provided
grouping, using the correlation ratio (eta): the fraction of a component's
spread that is explained by label membership, a number in `[0, 1]`.

```{code-cell} ipython3
def label_separation(values, group_labels):
    """Correlation ratio (eta) of a 1-D score against categorical labels."""
    values = np.asarray(values, dtype=float)
    grand = values.mean()
    total = ((values - grand) ** 2).sum()
    if total == 0.0:
        return 0.0
    between = sum(
        int((group_labels == g).sum()) * (values[group_labels == g].mean() - grand) ** 2
        for g in np.unique(group_labels)
    )
    return float(np.sqrt(between / total))


captured = float(evr[:2].sum())
if labels is not None:
    eta1 = label_separation(scores[:, 0], labels)
    eta2 = label_separation(scores[:, 1], labels)
    print(f"QC note ({ds.source} data): the top two PCs capture {captured:.1%} of "
          f"variance; label separation is eta(PC1)={eta1:.2f}, eta(PC2)={eta2:.2f}.")
else:
    print(f"QC note ({ds.source} data): the top two PCs capture {captured:.1%} of "
          "variance; this payload ships without cell labels, so the leading axes "
          "must be read from their gene loadings, not a provided grouping.")
```

**QC note.** On the labelled fallback the leading component already pulls the
synthetic cell types apart (a high eta), and a handful of components carry most
of the variance -- exactly the low-dimensional structure PCA is meant to
surface. On the real unlabelled matrix we instead lean on the scree shape and
the gene loadings, and we resist over-reading components buried in the noise
tail.

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
fig;
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
fig;
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
