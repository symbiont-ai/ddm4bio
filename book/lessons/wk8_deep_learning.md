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

# Week 8 - Deep Learning for Representations: from PCA to Autoencoders

Deep learning can feel like a discontinuity in the course -- as though everything
we built with the singular value decomposition were suddenly replaced by an
opaque network of weights. This lesson makes the opposite case. The simplest
neural network that learns a representation, a **linear autoencoder** trained to
minimize reconstruction error, is not a rival to PCA: it *is* PCA. The
Eckart-Young theorem guarantees that the best rank-$k$ approximation of a matrix
is its truncated SVD, and a linear autoencoder minimizing squared reconstruction
error can do no better than that optimum -- so gradient descent walks the network
straight to the principal subspace we already know how to compute in closed form.

Establishing that equivalence exactly, on data whose true rank we control, is the
anchor of the lesson. Once the equivalence is concrete, the interesting question
becomes what changes when we relax the "linear" assumption: nonlinear
autoencoders bend the latent surface, **SHRED** couples a sparse sensor network
to a decoder, and **transformers** learn context-dependent representations of
sequences. We discuss where each earns its complexity on biomedical data and,
just as important, where a linear latent is already all the signal there is --
and we stay honest that training the genuinely deep models needs a GPU and a
tensor framework this offline notebook deliberately does not use.

**Reading.** Kutz, *Data-Driven Modeling & Scientific Computation* / *Data-Driven
Science and Engineering*, Chapters 15 (neural networks and deep learning),
19 (SHRED and sensor-based reconstruction), and 22 (advances including
transformers and attention). Read those chapters for the architectures and their
derivations; everything below is developed in our own terms and validated
against our own fixtures.

**Learning goals.**

- State and *demonstrate numerically* the equivalence between a linear
  autoencoder trained by gradient descent and PCA / the truncated SVD.
- Read a training-loss curve and confirm it descends to the closed-form
  Eckart-Young optimum rather than beating it.
- Apply the same linear-representation machinery to a real biomedical feature
  matrix and interpret the resulting two-dimensional latent space.
- Reason about when nonlinear autoencoders, SHRED, and transformers add real
  value on biological data -- and when a linear latent already captures the
  structure.
- Close with an explicit, calibrated confidence statement about what a linear
  latent can and cannot represent.

## Setup

We seed every random number generator and apply the course plotting style so the
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

## 1. The equivalence, on synthetic ground truth

A **linear autoencoder** is the smallest network that learns a representation. It
compresses each centered input row $x \in \mathbb{R}^d$ to a $k$-dimensional code
$z = x W_e$ with an encoder matrix $W_e \in \mathbb{R}^{d \times k}$, then expands
it back with a decoder matrix $W_d \in \mathbb{R}^{k \times d}$ to a
reconstruction $\hat{x} = z W_d$. There are no nonlinear activations anywhere --
that omission is the whole point. Training minimizes the mean squared
reconstruction error $\frac{1}{N d}\sum_i \lVert \hat{x}_i - x_i \rVert^2$ by
gradient descent on the two weight matrices.

Here is the claim we will verify. Among *all* rank-$k$ linear maps, the
Eckart-Young theorem says the best squared-error reconstruction of a centered
data matrix is its truncated SVD -- exactly what PCA computes. A linear
autoencoder can only ever realize a rank-$k$ linear map, so the lowest loss it can
possibly reach is the PCA reconstruction error. Gradient descent, given enough
steps, converges to that floor. The encoder/decoder it lands on spans the **same
subspace** as the top-$k$ principal components (up to a rotation and scaling
inside the code, which the reconstruction is blind to).

To test this cleanly we build data whose true rank we control: 300 samples living
on a random 3-dimensional plane inside 12-dimensional feature space, plus a thin
layer of isotropic noise. An honest rank-3 reconstruction should capture almost
everything.

```{code-cell} ipython3
from ddm4bio.methods.decomposition import explained_variance_ratio, svd_lowrank
from ddm4bio.methods.validation import reconstruction_error

rng = np.random.default_rng(0)

n_samples, n_features, true_rank = 300, 12, 3

# Rank-3 generator: latent scores (300 x 3) times loadings (3 x 12), plus noise.
latent = rng.standard_normal((n_samples, true_rank))
loadings = rng.standard_normal((true_rank, n_features))
noise = 0.02 * rng.standard_normal((n_samples, n_features))
X = latent @ loadings + noise

# Center once; both PCA and the autoencoder operate on the centered matrix.
Xc = X - X.mean(axis=0, keepdims=True)

evr = explained_variance_ratio(X)
print(f"Data matrix X: {X.shape} (samples x features)")
print(f"Explained-variance ratio (first 6 components):\n{np.round(evr[:6], 4)}")
print(f"First {true_rank} components capture {evr[:true_rank].sum():.4%} of variance.")
```

### 1a. The PCA reconstruction (closed form)

The rank-$k$ PCA reconstruction is the truncated SVD of the centered matrix:
keep the top $k$ singular triplets and multiply them back together. This is the
Eckart-Young optimum -- no rank-$k$ linear map reconstructs `Xc` with lower
squared error.

```{code-cell} ipython3
k = true_rank

U, s, Vt = svd_lowrank(Xc, k)
X_pca = U @ np.diag(s) @ Vt  # best rank-k reconstruction (Eckart-Young)

err_pca = reconstruction_error(Xc, X_pca, kind="rel_l2")
print(f"PCA (rank-{k}) relative-L2 reconstruction error: {err_pca:.6f}")
```

### 1b. The linear autoencoder (gradient descent)

Now the network. We initialize small random encoder and decoder matrices and run
plain full-batch gradient descent on the mean-squared reconstruction error. The
gradients are elementary -- no autodiff framework, no GPU, just two matrix
multiplies per step -- which is exactly why this fits in an offline notebook.

```{code-cell} ipython3
d = n_features
N = Xc.shape[0]

# Small random initial weights (a fresh, seeded generator for reproducibility).
init_rng = np.random.default_rng(1)
W_e = 0.1 * init_rng.standard_normal((d, k))  # encoder: features -> code
W_d = 0.1 * init_rng.standard_normal((k, d))  # decoder: code -> features

lr = 0.05
n_epochs = 4000
losses = np.empty(n_epochs)

for epoch in range(n_epochs):
    Z = Xc @ W_e            # (N, k) latent codes
    X_hat = Z @ W_d         # (N, d) reconstruction
    R = X_hat - Xc          # residual
    losses[epoch] = np.mean(R**2)

    # Gradients of the mean-squared error w.r.t. each weight matrix.
    grad_W_d = (Z.T @ R) * (2.0 / (N * d))
    grad_W_e = (Xc.T @ (R @ W_d.T)) * (2.0 / (N * d))

    W_e -= lr * grad_W_e
    W_d -= lr * grad_W_d

X_ae = (Xc @ W_e) @ W_d
err_ae = reconstruction_error(Xc, X_ae, kind="rel_l2")
print(f"Autoencoder (rank-{k}) relative-L2 reconstruction error: {err_ae:.6f}")
print(f"Final training MSE: {losses[-1]:.3e}")
```

The training loss should fall and then flatten onto the PCA reconstruction error.
It cannot dip below it: the dashed line is the Eckart-Young floor, and the
network is asymptotically pinned to it.

```{code-cell} ipython3
import matplotlib.pyplot as plt

pca_mse = np.mean((X_pca - Xc) ** 2)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(losses, linewidth=1.5, label="autoencoder training MSE")
ax.axhline(pca_mse, linestyle="--", color="0.4",
           label=f"PCA (Eckart-Young) MSE = {pca_mse:.2e}")
ax.set_yscale("log")
ax.set_xlabel("Gradient-descent epoch")
ax.set_ylabel("Mean squared reconstruction error")
ax.set_title("Linear autoencoder descends to the PCA optimum")
ax.legend(loc="upper right")
fig
```

### 1c. Do the two reconstructions actually match?

A loss curve landing on the right floor is suggestive; the decisive test is
whether the two methods reconstruct the *same data* the same way. We compare the
reconstructions directly, and -- more stringently -- measure the principal angles
between the subspace the autoencoder's decoder spans and the top-$k$ PCA
subspace. Angles near zero mean the network found the principal subspace itself,
not merely a comparable error.

```{code-cell} ipython3
# Direct agreement of the two reconstructions.
recon_gap = reconstruction_error(X_pca, X_ae, kind="rel_l2")

# Subspace agreement: principal angles between decoder row-space and PCA components.
Q_ae, _ = np.linalg.qr(W_d.T)   # orthonormal basis for the autoencoder subspace
Q_pca, _ = np.linalg.qr(Vt.T)   # orthonormal basis for the top-k PCA subspace
cos_angles = np.clip(np.linalg.svd(Q_ae.T @ Q_pca, compute_uv=False), -1.0, 1.0)
principal_angles_deg = np.degrees(np.arccos(cos_angles))

print(f"Autoencoder-vs-PCA reconstruction difference (rel-L2): {recon_gap:.4f}")
print(f"Principal angles between subspaces (deg): "
      f"{np.round(principal_angles_deg, 3)}")
```

The reconstructions agree to a fraction of a percent and the principal angles are
essentially zero. That is the equivalence made concrete: a network trained only
to reconstruct its input, with no knowledge of eigenvectors, has rediscovered the
principal subspace. A scatter of one reconstructed feature against the other
method makes the point visually -- the points sit on the identity line.

```{code-cell} ipython3
feat = 0  # inspect a single feature dimension across all samples

fig, ax = plt.subplots(figsize=(5.2, 5))
lo = min(X_pca[:, feat].min(), X_ae[:, feat].min())
hi = max(X_pca[:, feat].max(), X_ae[:, feat].max())
ax.plot([lo, hi], [lo, hi], linestyle="--", color="0.5", label="identity")
ax.scatter(X_pca[:, feat], X_ae[:, feat], s=14, alpha=0.6, label="samples")
ax.set_xlabel(f"PCA reconstruction (feature {feat})")
ax.set_ylabel(f"Autoencoder reconstruction (feature {feat})")
ax.set_title("Per-sample reconstructions coincide")
ax.legend(loc="upper left")
fig
```

## 2. A biomedical latent space

The synthetic fixture proved the equivalence; now we point the same machinery at
a real biomedical measurement. The Wisconsin **breast-cancer** dataset bundled
with scikit-learn describes 569 tumor samples by 30 morphological features
computed from digitized fine-needle-aspirate images (cell radius, texture,
concavity, and so on), each labeled benign or malignant. This is the kind of
wide, correlated feature matrix where a low-dimensional latent is genuinely
useful: many of the 30 features are near-redundant descriptors of a few
underlying tumor properties.

Because the features live on wildly different numeric scales, we standardize each
to zero mean and unit variance first -- otherwise PCA (and the autoencoder) would
simply chase whichever feature happens to have the largest raw units.

```{code-cell} ipython3
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
X_raw = data.data                       # (569, 30) morphological features
y = data.target                         # 0 = malignant, 1 = benign

# Standardize features, then center for the decomposition.
X_std = (X_raw - X_raw.mean(axis=0)) / X_raw.std(axis=0)
X_bio = X_std - X_std.mean(axis=0, keepdims=True)

evr_bio = explained_variance_ratio(X_std)
print(f"Feature matrix: {X_raw.shape} (tumors x features)")
print(f"Class balance (malignant, benign): {np.bincount(y)}")
print(f"Top-2 components capture {evr_bio[:2].sum():.2%} of variance.")
```

We compress to a two-dimensional latent with both methods and confirm they agree
on real data too, then read the latent space. Unlike the synthetic fixture, no
two components capture *all* the variance here -- a real 30-feature tumor
description is not exactly rank 2 -- so the reconstruction error is substantial
and honest. The question is whether the two dimensions we keep are biologically
organized.

```{code-cell} ipython3
k_bio = 2

# Closed-form PCA reconstruction.
U_b, s_b, Vt_b = svd_lowrank(X_bio, k_bio)
X_bio_pca = U_b @ np.diag(s_b) @ Vt_b
err_bio_pca = reconstruction_error(X_bio, X_bio_pca, kind="rel_l2")

# Linear autoencoder trained the same way as before.
d_b, N_b = X_bio.shape[1], X_bio.shape[0]
ae_rng = np.random.default_rng(2)
We_b = 0.1 * ae_rng.standard_normal((d_b, k_bio))
Wd_b = 0.1 * ae_rng.standard_normal((k_bio, d_b))
lr_b, epochs_b = 0.02, 8000
for _ in range(epochs_b):
    Zb = X_bio @ We_b
    Rb = Zb @ Wd_b - X_bio
    Wd_b -= lr_b * (Zb.T @ Rb) * (2.0 / (N_b * d_b))
    We_b -= lr_b * (X_bio.T @ (Rb @ Wd_b.T)) * (2.0 / (N_b * d_b))
X_bio_ae = (X_bio @ We_b) @ Wd_b
err_bio_ae = reconstruction_error(X_bio, X_bio_ae, kind="rel_l2")

print(f"PCA (k=2) reconstruction error:        {err_bio_pca:.4f}")
print(f"Autoencoder (k=2) reconstruction error: {err_bio_ae:.4f}")
print(f"Method-to-method difference (rel-L2):   "
      f"{reconstruction_error(X_bio_pca, X_bio_ae, kind='rel_l2'):.4f}")
```

The two latents reconstruct the tumors almost identically -- the equivalence is
not an artifact of the clean synthetic data. Now the biomedical payoff: we plot
each tumor by its two PCA latent scores and color by diagnosis. A linear latent
is only worth keeping if the biology organizes along it.

```{code-cell} ipython3
from ddm4bio.methods.decomposition import pca_reduce

scores = pca_reduce(X_std, n_components=2)

fig, ax = plt.subplots(figsize=(7, 5.5))
for label, name in [(0, "malignant"), (1, "benign")]:
    sel = y == label
    ax.scatter(scores[sel, 0], scores[sel, 1], s=18, alpha=0.65, label=name)
ax.set_xlabel("Latent dimension 1")
ax.set_ylabel("Latent dimension 2")
ax.set_title("Two-dimensional linear latent of breast-cancer morphology")
ax.legend(title="diagnosis", loc="upper right")
fig
```

**What the picture shows.** The two diagnostic classes separate cleanly along the
first latent dimension even though the latent was built with no knowledge of the
labels -- the compression is fully unsupervised. That is the honest strength of a
linear representation on this dataset: the dominant axis of morphological
variation is essentially the malignant-versus-benign axis. It is also the honest
limit. A linear latent can only draw a straight boundary between clusters; the
residual overlap in the middle is structure a straight axis cannot untangle, and
that is precisely the gap nonlinear models are built to close.

## 3. Beyond linear: where nonlinearity earns its keep

The equivalence above is a floor, not a ceiling. Everything that makes modern deep
learning powerful is a principled way of relaxing the "linear" assumption we held
fixed. Three extensions matter for biomedical data, and each buys something at a
stated cost.

**Nonlinear autoencoders.** Insert nonlinear activations between the encoder and
decoder layers and the latent surface is no longer a flat subspace -- it can bend
to follow a curved manifold. Single-cell transcriptomes are the canonical case:
cells along a differentiation trajectory trace a curve through gene-expression
space, and a nonlinear encoder can straighten that curve into an interpretable
latent coordinate where PCA would only find the chord across it. The cost is real:
nonlinear autoencoders have no closed-form optimum, so there is no Eckart-Young
guarantee, results depend on initialization and architecture, and a latent that
"looks structured" can be an artifact of overfitting rather than biology. The
validation discipline from this course -- test recovery on synthetic ground truth
before trusting real data -- becomes *more* important, not less, once the
closed-form check is gone.

**SHRED (shallow recurrent decoder).** SHRED targets a different problem:
reconstructing a full high-dimensional field from a handful of sparse sensors
measured over time. A small recurrent network compresses the sensor time-series
into a latent state, and a shallow decoder expands that state back to the entire
spatial field. The biomedical analogue is reconstructing a whole physiological or
imaging field from a few electrodes or probes -- exactly the regime where you
cannot instrument everything but the dynamics are low-dimensional. SHRED's leverage
comes from *time*: the history at a few sensors constrains the global state far
more than a single snapshot does. Its limit is that it assumes the field really
does live near a low-dimensional attractor and that the sensors, sparse as they
are, actually observe it.

**Transformers and attention.** Transformers learn representations of *sequences*
in which each element's encoding depends on the others through attention weights,
rather than a fixed linear projection. In biology this is the machinery behind
protein-language models that read an amino-acid sequence and predict structure or
function, and behind genomic models that learn context-dependent representations
of DNA. What attention adds over a linear (or even a fixed nonlinear) encoder is
*context-dependence*: the same token can be represented differently depending on
its neighbors, which is essential when meaning is positional -- a residue's role
depends on the rest of the fold. The cost is steep. Transformers need large
labeled or self-supervised corpora, careful regularization, and real compute; on a
small, tabular biomedical dataset they will happily overfit and underperform the
two-line PCA we ran above.

**The honest hardware caveat.** Everything in Section 1 and 2 ran on the CPU in
plain numpy because a *linear* autoencoder is just matrix algebra. The genuinely
deep models in this section -- multilayer nonlinear autoencoders, SHRED,
transformers -- need a tensor framework (PyTorch or JAX) and, in practice, a GPU
to train in reasonable time. That is an intentional boundary of this offline
course notebook, not a claim that those models are unimportant. Treat the GPU/torch
track as the optional course extra it is: the concepts here are the foundation you
would carry into it.

## 4. Interpretation

Every ddm4bio analysis closes with an explicit interpretation block: a single
claim, an honest confidence level backed by named evidence, and a list of stated
limitations. Here the claim is the equivalence itself, and the limitations are
specifically about what a *linear* latent can and cannot represent.

```{code-cell} ipython3
from ddm4bio.interpret import interpretation_block

block = interpretation_block(
    claim="A linear autoencoder trained by gradient descent to minimize "
          "reconstruction error converges to PCA: it recovers the same "
          "principal subspace and the same rank-k reconstruction as the "
          "truncated SVD, on both synthetic and real biomedical data.",
    confidence="high",
    limitations_list=[
        f"The equivalence is exact only for the LINEAR autoencoder and squared "
        f"error; on the synthetic fixture the two subspaces agreed to within "
        f"{principal_angles_deg.max():.2f} degrees and reconstructions to "
        f"{recon_gap:.3f} relative-L2.",
        "A linear latent can only capture a flat subspace: it represents the "
        "top directions of variance and draws straight boundaries. Curved "
        "manifolds (e.g. differentiation trajectories) and context-dependent "
        "structure are beyond it by construction.",
        "On the breast-cancer data the k=2 reconstruction error is large "
        f"(rel-L2 = {err_bio_pca:.2f}) because 30 correlated features are not "
        "truly rank 2; the latent summarizes, it does not reconstruct.",
        "Class separation in the latent is unsupervised and descriptive, not a "
        "validated classifier; the residual overlap is real and a straight axis "
        "cannot resolve it.",
        "Nonlinear autoencoders, SHRED, and transformers relax these limits but "
        "lose the closed-form Eckart-Young guarantee and, in practice, require a "
        "GPU/torch stack this offline notebook does not use.",
    ],
    evidence=f"synthetic: PCA rel-L2 = {err_pca:.4f} vs autoencoder "
             f"rel-L2 = {err_ae:.4f}, max principal angle "
             f"{principal_angles_deg.max():.2f} deg; biomedical: PCA "
             f"{err_bio_pca:.3f} vs autoencoder {err_bio_ae:.3f} rel-L2.",
)
print(block)
```

## Exercises

Week 8 has no separate problem set. Your graded work is the **capstone project**,
in which you carry a dataset of your choice through the full ddm4bio pipeline --
honest QC, a validated method, and an interpretation block -- end to end. Building
on this lesson, the capstone asks you to:

- Choose a representation method appropriate to your data and *justify the choice*
  against the linear baseline: show what a PCA / linear-autoencoder latent
  captures, and argue explicitly whether your data need a nonlinear model or
  whether the linear latent is already sufficient.
- Validate recovery on a synthetic or held-out ground truth before interpreting
  the real result, exactly as Section 1 validates the equivalence before Section 2
  trusts it on tumors.
- If you use a nonlinear autoencoder, SHRED, or a transformer (the optional
  GPU/torch track), report not just the reconstruction quality but the stability
  of the latent across initializations -- the closed-form guarantee is gone and
  you must earn trust empirically.
- Close with an `interpretation_block` stating a calibrated confidence level and
  named limitations, with special attention to what your chosen latent can and
  cannot represent.

Refer to the capstone project brief for scope, deliverables, and the grading
rubric.
