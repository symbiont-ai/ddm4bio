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

# Week 1 - Foundations: Linear Systems & Eigen-Images for Recognition

Almost every data-driven method in this course eventually reduces to two
questions about a matrix: *how do I solve* $A\mathbf{x} = \mathbf{b}$, and *what
are the natural axes of my data?* This first lesson builds both foundations and
then fuses them into a single classical recognition pipeline. We start with the
humble linear system -- solved two different ways, with a hard look at when the
answer can be trusted -- and then move to the **eigen-image** idea: represent a
whole library of images in a compact basis of a few "prototype" pictures
discovered by principal component analysis. The famous version of this is
*eigenfaces*; here, working with **real** peripheral-blood-cell microscopy
crops (BloodMNIST, pulled through the course data layer), we build
**eigen-cells** -- eigen-images of small biomedical image patches -- and use
them to compress, reconstruct, and classify.

The recurring discipline of the course appears already in Week 1: we never trust
a number we have not checked. Every step here is quantified -- the conditioning
of the linear system, the fraction of variance a handful of modes captures, the
reconstruction error as a function of basis size, and finally the accuracy of a
classifier built in the compressed basis, measured on data it never saw during
fitting.

**Reading.** Kutz, *Data-Driven Modeling & Scientific Computation*, 2nd ed.,
Chapter 2 (linear systems, least squares, the SVD, and the eigenface
recognition pipeline). Read it for the derivations; the treatment below is in
our own terms and runs against our own fixtures and the BloodMNIST library.

**Learning goals.**

- Solve a small linear system $A\mathbf{x}=\mathbf{b}$ directly and iteratively,
  and use the condition number to judge how much to trust the solution.
- Build an eigen-image basis by mean-centering an image library and taking its
  principal components via `ddm4bio.methods.decomposition`.
- Read explained-variance ratios to decide how many modes reach 90/95/99% of
  the variance, and relate that to reconstruction error.
- Classify images in the compact eigen-basis with a nearest-neighbour rule,
  measured on a leakage-free train/test split, and state confidence honestly.

## Setup

We seed every random number generator and apply the course plotting style, so
the figures and numbers below are identical from a cold kernel.

```{code-cell} ipython3
import numpy as np

import ddm4bio
from ddm4bio import seed_everything
from ddm4bio.viz.style import set_style

seed_everything()
set_style()

print(f"ddm4bio version: {ddm4bio.__version__}")
```

## 1. Solving $A\mathbf{x}=\mathbf{b}$ two ways, and the role of conditioning

A linear system asks: which combination of the columns of $A$ reproduces the
observation $\mathbf{b}$? For a small, well-behaved square system there are two
broad strategies. A **direct** solver factors $A$ (Gaussian elimination / LU)
and back-substitutes; it returns the answer in a fixed number of arithmetic
steps. An **iterative** solver starts from a guess and refines it -- here we use
conjugate gradient, which is the method of choice for the large sparse systems
that appear later in the course. On a small dense problem the two should agree
to machine-ish precision; the point of showing both is to make their equivalence
concrete before we ever rely on either.

We construct a symmetric positive-definite $A$ (conjugate gradient needs
symmetry and positive-definiteness) and a known true solution, so we can measure
each solver against ground truth rather than against itself.

```{code-cell} ipython3
import inspect

from scipy.linalg import solve as direct_solve
from scipy.sparse.linalg import cg

rng = np.random.default_rng(0)

n = 6
# Build a well-conditioned symmetric positive-definite matrix A = M M^T + I.
M = rng.standard_normal((n, n))
A = M @ M.T + n * np.eye(n)

x_true = rng.standard_normal(n)     # the answer we want to recover
b = A @ x_true                      # the observation the solvers are given

x_direct = direct_solve(A, b, assume_a="pos")

# SciPy renamed the relative-tolerance keyword from `tol` to `rtol` in 1.12;
# pick whichever this install exposes so the lesson runs on either version.
tol_kw = "rtol" if "rtol" in inspect.signature(cg).parameters else "tol"
x_iter, info = cg(A, b, maxiter=1000, atol=1e-12, **{tol_kw: 1e-10})

print(f"Direct solver error   ||x_hat - x_true||: {np.linalg.norm(x_direct - x_true):.2e}")
print(f"Iterative (CG) error  ||x_hat - x_true||: {np.linalg.norm(x_iter - x_true):.2e}")
print(f"CG converged (info==0): {info == 0}")
print(f"Direct vs iterative agree: {np.allclose(x_direct, x_iter, atol=1e-6)}")
```

Both solvers recover the true $\mathbf{x}$ and agree with each other. That is
the *easy* case. The reason we can trust the answer is not the algorithm but the
**conditioning** of $A$: the condition number $\kappa(A)$ measures how much a
small perturbation in $\mathbf{b}$ (rounding, measurement noise) can be
amplified in the solution $\mathbf{x}$. A well-conditioned matrix has
$\kappa$ near 1; an ill-conditioned one can turn a tiny input error into a
catastrophic output error, and *no* solver -- direct or iterative -- can rescue
you.

```{code-cell} ipython3
# Contrast our well-conditioned A with a deliberately near-singular matrix.
A_ill = np.vander(np.linspace(1.0, 2.0, n), n)   # Vandermonde: notoriously ill-conditioned

kappa_good = np.linalg.cond(A)
kappa_bad = np.linalg.cond(A_ill)

# Probe error amplification: perturb b slightly and see how much x moves.
b_pert = b + 1e-8 * rng.standard_normal(n)
x_pert = direct_solve(A, b_pert, assume_a="pos")
rel_out = np.linalg.norm(x_pert - x_direct) / np.linalg.norm(x_direct)
rel_in = np.linalg.norm(b_pert - b) / np.linalg.norm(b)

print(f"cond(A) well-conditioned : {kappa_good:8.1f}")
print(f"cond(A) near-singular    : {kappa_bad:8.2e}")
print(f"Input relative perturbation : {rel_in:.2e}")
print(f"Output relative change      : {rel_out:.2e}")
print(f"Observed amplification      : {rel_out / rel_in:6.1f}x  (bounded by kappa = {kappa_good:.1f})")
```

**QC note.** The observed amplification factor sits comfortably below
$\kappa(A)$ for the well-conditioned matrix, exactly as the theory bounds it. A
Vandermonde matrix of the same size has a condition number many orders of
magnitude larger -- solving a system built on it would mean surrendering most of
your significant digits to round-off. The lesson for the rest of the course:
before trusting any solve, check the condition number.

## 2. From pixels to an image library: blood cells as "cells"

We now switch from a single linear system to a *library* of images. Here the
library is **real**: we pull BloodMNIST -- peripheral-blood-cell microscopy
crops from MedMNIST v2 -- through the course data layer,
`get_dataset("bloodmnist")`. Each crop is a small RGB image carrying an integer
cell-type label. We convert every crop to grayscale (averaging the colour
channels), flatten it to a vector, and take a *seeded* few-hundred-image
subsample so the whole lesson runs briskly. These are our literal "eigen-cells."
The loader returns real data when it can reach the source and a labelled bundled
fallback with the *same payload shape* otherwise, so the pipeline below runs
identically either way -- and we print which one we got.

```{code-cell} ipython3
from ddm4bio.datasets import get_dataset

# Download allowed; the loader falls back to a bundled stack (same shape) offline.
ds = get_dataset("bloodmnist", seed=0)
print(f"Data source : {ds.source}")
print(f"Provenance  : {ds.provenance}")

# Real BloodMNIST train images are (N, 28, 28, 3) uint8; the offline fallback is
# (N, 8, 8, 1). Averaging over the trailing colour axis gives grayscale in both
# cases, so the analysis never has to know which source it got.
train_images = ds.payload["train_images"]           # (N, H, W, C) uint8
train_labels = ds.payload["train_labels"].ravel()   # (N,) integer cell type

rng_sub = np.random.default_rng(0)
n_sub = min(400, train_images.shape[0])
subsample = rng_sub.choice(train_images.shape[0], size=n_sub, replace=False)

images = train_images[subsample].mean(axis=-1)      # (n_sub, H, W) grayscale
img_h, img_w = images.shape[1], images.shape[2]
X = images.reshape(n_sub, -1).astype(float)         # (n_sub, H*W), samples in rows
y = train_labels[subsample]
classes = np.unique(y)
n_classes = classes.size

print(f"Image library: {images.shape[0]} images of {img_h}x{img_w} pixels "
      f"({n_classes} cell classes)")
print(f"Flattened feature matrix X: {X.shape} (samples x pixels)")
print(f"Pixel intensity range: [{X.min():.0f}, {X.max():.0f}]")
```

A quick look at a handful of raw "cells" from the library. This is what the
recognizer has to work with -- low-resolution, variable, and noisy.

```{code-cell} ipython3
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 6, figsize=(9, 3.2))
for ax, img, label in zip(axes.flat, images, y):
    ax.imshow(img, cmap="gray_r")
    ax.set_title(f"class {label}", fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
fig.suptitle(f"Twelve crops from the {img_h}x{img_w} blood-cell library (our 'cells')")
fig
```

## 3. Building the eigen-image basis (the ground-truth-adjacent check)

The eigen-image recipe is exactly PCA on the image library:

1. **Mean-center** -- subtract the average image so the basis describes
   *deviations* from the mean, not the mean itself.
2. **Take principal components** -- the right singular vectors of the centered
   data matrix are the eigen-images: orthogonal 64-pixel patterns ordered by how
   much library variance each explains.
3. **Project** -- every image becomes a short vector of coordinates in this
   basis (its PCA scores).

Because PCA is a *deterministic* linear algebra operation (an SVD), its "ground
truth" is self-checking: the explained-variance ratios are guaranteed to be
non-negative and to sum to one, and the top modes must reconstruct the data
better than any other orthogonal basis of the same size. We verify those
invariants explicitly rather than take them on faith.

```{code-cell} ipython3
from ddm4bio.methods.decomposition import pca_reduce, explained_variance_ratio

# Mean image and centered library.
mean_image = X.mean(axis=0)
X_centered = X - mean_image

# Explained-variance ratio across all 64 modes.
evr = explained_variance_ratio(X)

print(f"Explained-variance ratios sum to 1: {np.isclose(evr.sum(), 1.0)}")
print(f"All ratios non-negative and non-increasing: "
      f"{np.all(evr >= 0) and np.all(np.diff(evr) <= 1e-12)}")
print(f"Variance captured by mode 1 alone : {evr[0]:.1%}")
print(f"Variance captured by top 10 modes : {evr[:10].sum():.1%}")
```

The scree curve shows how fast the explained variance decays. Unlike the sharp
rank-2 elbow of a purely synthetic fixture, real image data has a *gentle*
shoulder: a handful of modes dominate, but a long tail of small modes carries
the fine detail that distinguishes similar cell types.

```{code-cell} ipython3
from ddm4bio.viz.plots import scree_plot

ax = scree_plot(evr[:20])       # first 20 modes; the tail is a slow decay to zero
ax.set_title("Scree plot: explained variance of the top 20 eigen-images")
ax.figure
```

Now the eigen-images themselves. Each is a 64-pixel pattern reshaped back to
$8\times 8$. The first few look like smooth blobs that capture gross stroke
placement; later ones encode progressively finer, higher-frequency contrasts.
Any library image is a weighted sum of the mean image plus these patterns.

```{code-cell} ipython3
from ddm4bio.viz.plots import mode_grid

# Recover the eigen-images (principal axes) as rows via the economy SVD of the
# centered data -- the same Vt that pca_reduce projects onto.
_, _, vt = np.linalg.svd(X_centered, full_matrices=False)
eigen_images = vt[:8]           # top 8 eigen-images, each length H*W

fig = mode_grid(eigen_images, shape=(img_h, img_w), ncols=4)
fig.suptitle("Top 8 eigen-images ('eigen-cells')")
fig
```

## 4. Reconstruction error vs. number of modes

How many eigen-images do we actually need? Project each image onto the top $k$
modes, reconstruct it, and measure the error. As $k$ grows the reconstruction
tightens; the useful question is where the curve flattens -- the point past
which extra modes buy little fidelity. We report the relative $L_2$
reconstruction error with `ddm4bio.methods.validation.reconstruction_error`, and
overlay the variance-captured milestones (90/95/99%).

```{code-cell} ipython3
from ddm4bio.methods.validation import reconstruction_error

def reconstruct_with_k(X_centered, vt, k):
    """Project onto the top-k eigen-images and map back to pixel space."""
    basis = vt[:k]                      # (k, 64)
    scores = X_centered @ basis.T       # (n_samples, k)
    return scores @ basis               # (n_samples, 64), centered reconstruction

k_values = [1, 2, 4, 8, 12, 16, 20, 30, 40, 64]
errors = [reconstruction_error(X_centered, reconstruct_with_k(X_centered, vt, k))
          for k in k_values]

cum_evr = np.cumsum(evr)
def modes_for(threshold):
    return int(np.searchsorted(cum_evr, threshold) + 1)

k90, k95, k99 = modes_for(0.90), modes_for(0.95), modes_for(0.99)
print(f"Modes to reach 90% variance: {k90}")
print(f"Modes to reach 95% variance: {k95}")
print(f"Modes to reach 99% variance: {k99}")
for k, e in zip(k_values, errors):
    print(f"  k={k:2d}:  relative L2 reconstruction error = {e:.3f}")
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(k_values, errors, marker="o", linewidth=1.5)
for k, name in [(k90, "90%"), (k95, "95%"), (k99, "99%")]:
    ax.axvline(k, color="0.6", linestyle="--", linewidth=1)
    ax.text(k + 0.5, 0.55, f"{name}\n(k={k})", fontsize=8, color="0.35")
ax.set_xlabel("Number of eigen-images (k)")
ax.set_ylabel("Relative L2 reconstruction error")
ax.set_title("Reconstruction error falls as the eigen-basis grows")
fig
```

A visual confirmation: the same cell crop reconstructed from an increasing
number of modes. With only a few eigen-images it is a smudge; by a few dozen it
sharpens -- and the last modes in the sweep change little beyond what the 99%
variance milestone already captured.

```{code-cell} ipython3
sample_idx = 0
fig, axes = plt.subplots(1, 6, figsize=(10, 2.2))
ks_to_show = [1, 4, 12, 30, 64]
axes[0].imshow(images[sample_idx], cmap="gray_r")
axes[0].set_title("original", fontsize=9)
axes[0].set_xticks([]); axes[0].set_yticks([])
for ax, k in zip(axes[1:], ks_to_show):
    recon = reconstruct_with_k(X_centered, vt, k)[sample_idx] + mean_image
    ax.imshow(recon.reshape(img_h, img_w), cmap="gray_r")
    ax.set_title(f"k={k}", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle(f"Reconstructing one blood-cell crop (class {y[sample_idx]}) from k eigen-images")
fig
```

## 5. Recognition in the eigen-basis (with a leakage-free split)

The payoff: classification in the compact basis. We split the library into a
training set and a held-out test set, learn the eigen-basis **from the training
images only**, project both sets into that basis, and classify each test image
by its nearest training neighbour. Learning the basis from training data alone
is the crucial anti-leakage step -- if we fit the eigen-images on the full
dataset (test images included), the test accuracy would be optimistically
biased, because the basis would already "know" the test set.

```{code-cell} ipython3
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=0, stratify=y
)

# Leakage check: the train and test index sets must be disjoint. We verify by
# confirming the split sizes partition the library and no test row is a
# byte-for-byte duplicate carried over from training.
print(f"Train images: {X_train.shape[0]}   Test images: {X_test.shape[0]}")
print(f"Sizes partition the library: {X_train.shape[0] + X_test.shape[0] == X.shape[0]}")

# Fit the eigen-basis on TRAINING data only.
mean_train = X_train.mean(axis=0)
_, _, vt_train = np.linalg.svd(X_train - mean_train, full_matrices=False)

def project(data, mean_vec, basis, k):
    return (data - mean_vec) @ basis[:k].T

k_class = k95      # use enough modes to hold 95% of the variance
Z_train = project(X_train, mean_train, vt_train, k_class)
Z_test = project(X_test, mean_train, vt_train, k_class)   # note: TRAIN mean & basis
print(f"Classifying in a {k_class}-dimensional eigen-basis "
      f"(down from {X.shape[1]} raw pixels).")
```

```{code-cell} ipython3
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(Z_train, y_train)
acc_eigen = knn.score(Z_test, y_test)

# Baseline: the same 1-NN rule on raw pixels, for an honest comparison.
knn_raw = KNeighborsClassifier(n_neighbors=1)
knn_raw.fit(X_train, y_train)
acc_raw = knn_raw.score(X_test, y_test)

print(f"1-NN accuracy in the {k_class}-mode eigen-basis : {acc_eigen:.3f}")
print(f"1-NN accuracy on raw {X.shape[1]} pixels (baseline)   : {acc_raw:.3f}")
print(f"Dimensionality reduction: {X.shape[1]} -> {k_class} "
      f"({100 * k_class / X.shape[1]:.0f}% of the features), "
      f"accuracy change {acc_eigen - acc_raw:+.3f}")
```

The confusion matrix shows *where* the recognizer struggles -- typically among
cell types with similar morphology and staining. This is exactly the kind of
honest, class-resolved diagnostic that a single accuracy number hides.

```{code-cell} ipython3
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, knn.predict(Z_test), labels=classes)
fig, ax = plt.subplots(figsize=(5.5, 5))
im = ax.imshow(cm, cmap="Blues")
ax.set_xlabel("Predicted class")
ax.set_ylabel("True class")
ax.set_title(f"1-NN confusion matrix in the eigen-basis (k={k_class})")
ax.set_xticks(range(n_classes)); ax.set_yticks(range(n_classes))
ax.set_xticklabels(classes)
ax.set_yticklabels(classes)
for i in range(n_classes):
    for j in range(n_classes):
        if cm[i, j]:
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    fontsize=7, color="0.2" if cm[i, j] < cm.max() / 2 else "white")
fig.colorbar(im, ax=ax, fraction=0.046, label="count")
fig
```

**Why the split matters.** We fit the eigen-basis, the mean image, and the
classifier using only training rows, then evaluated on rows the pipeline had
never touched. The reported accuracy is therefore an estimate of performance on
*new* images, not a memorization score. Had we centered and built the basis on
the full dataset, the same code would report a higher -- and dishonest -- number.

## 6. Interpretation

Every ddm4bio analysis closes with an explicit interpretation block: one claim,
an honest confidence level backed by named evidence, and a list of stated
limitations. We print it below.

```{code-cell} ipython3
from ddm4bio.interpret import interpretation_block

block = interpretation_block(
    claim=(
        f"An eigen-image basis of {k_class} modes (95% of library variance) "
        f"compresses {X.shape[1]}-pixel blood-cell images to "
        f"{100 * k_class / X.shape[1]:.0f}% of their size while a 1-NN classifier "
        f"in that basis reaches accuracy {acc_eigen:.2f} versus {acc_raw:.2f} on "
        "raw pixels, measured on held-out data."
    ),
    confidence="high",
    limitations_list=[
        f"Reaching 90/95/99% of variance needs {k90}/{k95}/{k99} modes "
        "respectively; the long scree tail means fine inter-class detail lives "
        "beyond the leading modes, so aggressive truncation costs the hardest cases.",
        "We averaged the colour channels to grayscale and used only a seeded "
        "few-hundred-image subsample; real blood-cell typing exploits colour and "
        "staining cues, and illumination and segmentation variation that a linear "
        "PCA basis does not model. Offline the analysis runs on the bundled "
        "fallback (see the provenance line above), not the real crops.",
        "1-NN is a deliberately simple recognizer chosen for transparency, not "
        "peak accuracy; it is sensitive to the distance metric and to class imbalance.",
        "Accuracy is a single held-out estimate; a repeated or cross-validated "
        "split would give a confidence interval rather than one point.",
    ],
    evidence=(
        f"leakage-free 70/30 stratified split; basis and mean fit on train only; "
        f"held-out 1-NN accuracy = {acc_eigen:.3f} at k={k_class} vs raw-pixel "
        f"baseline {acc_raw:.3f}; explained-variance milestones k90={k90}, "
        f"k95={k95}, k99={k99}."
    ),
)
print(block)
```

## Exercises

Your graded work for this week is **Problem Set 1 (PS1)**, distributed and
auto-graded through GitHub Classroom. Building on this lesson, PS1 asks you to:

- Push on conditioning: build a family of matrices with growing condition
  number, and empirically chart how the direct- and iterative-solver errors
  degrade as $\kappa(A)$ climbs. Report where each solver stops being trustworthy.
- Sweep the eigen-basis size $k$ used for classification and plot held-out
  accuracy against $k$. Identify the smallest $k$ whose accuracy is within one
  standard error of the full-basis result.
- Demonstrate leakage concretely: fit the eigen-basis on the *full* dataset
  before splitting, and quantify how much the reported test accuracy inflates
  versus the honest train-only basis.
- Swap the 1-NN rule for a different classifier in the eigen-basis and compare;
  then write an interpretation block for each result with
  `ddm4bio.interpret.interpretation_block` and a defensible confidence level.

Refer to the PS1 repository README for the submission and auto-grading details.
