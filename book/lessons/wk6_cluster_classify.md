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

# Week 6 - Learning with Valid Inference: Clustering, Classification & Multiple Testing

Machine learning is easy to *run* and hard to *trust*. A clustering routine will
always return clusters, a classifier will always return a decision boundary, and
a batch of statistical tests will always hand back some "significant" hits --
whether or not any of it reflects real biology. This week is about the second
half of every learning pipeline: the inference that tells you which of those
outputs are robust and which are artifacts of noise, chance, or an over-eager
model. We work through the three failure modes that most often turn a
respectable-looking analysis into a false discovery. First, **unsupervised
learning**: we cluster data, choose the number of clusters honestly, and use
resampling to flag clusters that fall apart when the data are perturbed. Second,
**supervised learning**: we train a diagnostic classifier and refuse to report a
single AUC without a confidence interval around it. Third, **multiple testing**:
we screen many features at once and control the false-discovery rate so that our
list of "significant" features is not mostly wishful thinking.

As always in this course, every method meets a synthetic fixture with a known
answer before it is allowed near real data, and every analysis ends with an
explicit statement of confidence and limitations.

**Reading.** Kutz, *Data-Driven Modeling & Scientific Computation*, 2nd ed.,
Chapters 17-18 (unsupervised and supervised learning) and Chapter 13 (statistical
methods and inference). Read those for the underlying theory; everything below is
explained in our own terms and validated against our own fixtures.

**Learning goals.**

- Cluster data with k-means and Gaussian mixtures, and choose the number of
  clusters with silhouette and BIC rather than by eye.
- Use consensus (resampling) clustering to distinguish stable structure from
  clusters that are artifacts of a single fit.
- Train a diagnostic classifier and report its AUC with a bootstrap confidence
  interval instead of a bare point estimate.
- Screen many features at once and control the false-discovery rate with the
  Benjamini-Hochberg procedure.
- Close every result with an honest confidence-and-limitations statement.

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

## 1. Clustering on known ground truth

Clustering is unsupervised: we hand the algorithm a feature matrix with no
labels and ask it to find groups. The hard part is not running k-means -- it is
knowing whether the groups it returns are real. The only honest way to build
that intuition is to start where we *do* know the answer. We generate four
well-separated Gaussian blobs in two dimensions; the generator records which
blob each point came from, and we hold those labels aside as ground truth to
score recovery against.

```{code-cell} ipython3
from sklearn.datasets import make_blobs

X, y_true = make_blobs(
    n_samples=300,
    centers=4,
    cluster_std=1.0,
    n_features=2,
    random_state=7,
)

print(f"Feature matrix X: {X.shape} (samples x features)")
print(f"True number of clusters: {np.unique(y_true).size}")
```

In a real study we would *not* know that there are four groups. So before
clustering we let the data tell us how many clusters to look for. Two criteria
answer that question from different directions. The **silhouette** score measures
how well-separated the clusters are (higher is better) and tends to reward the
partition that carves the data at its natural gaps. The **Bayesian Information
Criterion (BIC)** scores a Gaussian-mixture fit and penalizes extra components
(lower is better), so it resists splitting the data into more clusters than the
likelihood can justify. We sweep candidate cluster counts and let each criterion
vote.

```{code-cell} ipython3
from ddm4bio.methods.clustering import select_k_silhouette, select_k_bic

sil = select_k_silhouette(X, range(2, 8), seed=0)
bic = select_k_bic(X, range(2, 8), seed=0)

ks = [int(k) for k in sil["ks"]]
print(f"Silhouette picks k = {sil['best_k']}")
print(f"  scores per k {ks}: {np.round(sil['scores'], 3)}")
print(f"BIC picks       k = {bic['best_k']}")
print(f"  BIC per k    {ks}: {np.round(bic['bic'], 1)}")
```

The two criteria agree, and they agree with the truth: both point to four
clusters. When they *disagree* -- which happens whenever the clusters overlap --
that disagreement is itself information and a signal to distrust any single
answer. The plot below shows why they agree here: silhouette peaks and BIC
bottoms out at the same value.

```{code-cell} ipython3
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))

axes[0].plot(sil["ks"], sil["scores"], marker="o")
axes[0].axvline(sil["best_k"], color="0.6", linestyle="--", linewidth=1)
axes[0].set_xlabel("Number of clusters k")
axes[0].set_ylabel("Mean silhouette")
axes[0].set_title("Silhouette (higher is better)")

axes[1].plot(bic["ks"], bic["bic"], marker="o", color="#D55E00")
axes[1].axvline(bic["best_k"], color="0.6", linestyle="--", linewidth=1)
axes[1].set_xlabel("Number of clusters k")
axes[1].set_ylabel("BIC")
axes[1].set_title("BIC (lower is better)")

fig.suptitle("Choosing the number of clusters from the data")
fig;
```

Now we cluster at the chosen `k` with both k-means and a Gaussian mixture, and we
*score* the result against the held-aside truth. The adjusted Rand index (ARI)
measures agreement between two labelings up to arbitrary relabeling: 1.0 is a
perfect match and 0.0 is chance-level. Because the blobs are cleanly separated,
we expect near-perfect recovery -- our license to trust these methods on data
where no ground truth exists.

```{code-cell} ipython3
from sklearn.metrics import adjusted_rand_score
from ddm4bio.methods.clustering import kmeans_cluster, gmm_cluster

best_k = sil["best_k"]
labels_km = kmeans_cluster(X, best_k, seed=0)
labels_gmm = gmm_cluster(X, best_k, seed=0)

ari_km = adjusted_rand_score(y_true, labels_km)
ari_gmm = adjusted_rand_score(y_true, labels_gmm)

print(f"k-means      ARI vs. ground truth: {ari_km:.3f}")
print(f"GMM          ARI vs. ground truth: {ari_gmm:.3f}")
```

```{code-cell} ipython3
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharex=True, sharey=True)

axes[0].scatter(X[:, 0], X[:, 1], c=y_true, s=14, cmap="tab10")
axes[0].set_title("Ground-truth labels")
axes[1].scatter(X[:, 0], X[:, 1], c=labels_km, s=14, cmap="tab10")
axes[1].set_title(f"k-means labels (ARI = {ari_km:.2f})")
for ax in axes:
    ax.set_xlabel("Feature 1")
axes[0].set_ylabel("Feature 2")
fig.suptitle("Recovered clusters overlaid on ground truth")
fig;
```

**QC note.** An ARI at or near 1.0 means the recovered partition is essentially
the true one. This is what honest validation looks like: we confirmed the method
reproduces a known answer *before* interpreting any cluster as a biological
subtype. On real data the ARI is unavailable -- there is no truth column -- which
is exactly why the next step matters.

## 2. Cluster stability: which clusters survive resampling?

A single clustering run hands back a partition with no error bars. But a cluster
that only appears because of the particular points we happened to sample, or the
particular random initialization we happened to draw, is not a discovery -- it is
noise wearing a label. **Consensus clustering** stress-tests the result: it
repeatedly subsamples the data, re-clusters each subsample, and records how often
each pair of points lands together. Pairs that cluster together in *every*
resample are stable; pairs that cluster together only sometimes mark the fault
lines where the structure is fragile.

We run consensus clustering twice. First at the correct `k = 4`, then -- to see
what instability looks like -- at `k = 5`, deliberately asking for one cluster
more than the data contain. Over-clustering has to split a real group in two, and
that artificial split is exactly what resampling exposes.

```{code-cell} ipython3
from ddm4bio.methods.clustering import consensus_cluster

con_ok = consensus_cluster(X, best_k, n_boot=25, subsample=0.8, seed=0)
con_over = consensus_cluster(X, best_k + 1, n_boot=25, subsample=0.8, seed=0)

print(f"Consensus at k={best_k}:   labels over {np.unique(con_ok['labels']).size} clusters")
print(f"Consensus at k={best_k + 1}:   labels over {np.unique(con_over['labels']).size} clusters")
```

To turn the consensus matrix into a QC gate we quantify its *ambiguity*: the
fraction of point pairs whose co-clustering frequency is stuck in the murky
middle (between 0.1 and 0.9) rather than a confident 0 or 1. A clean, stable
partition drives almost every pair to the extremes, so its ambiguity is near
zero. A high ambiguity is a warning that the requested clusters are not
reproducible.

```{code-cell} ipython3
def consensus_ambiguity(consensus_matrix: np.ndarray) -> float:
    """Fraction of off-diagonal pairs with intermediate (unstable) consensus."""
    n = consensus_matrix.shape[0]
    off_diagonal = consensus_matrix[~np.eye(n, dtype=bool)]
    return float(np.mean((off_diagonal > 0.1) & (off_diagonal < 0.9)))


def stability_warnings(ambiguity: float, k: int, threshold: float = 0.05) -> list[str]:
    """Emit a QC warning when a partition's consensus is too ambiguous."""
    warnings: list[str] = []
    if ambiguity > threshold:
        warnings.append(
            f"UNSTABLE: k={k} clustering has ambiguity {ambiguity:.1%} "
            f"(> {threshold:.0%}); clusters are not reproducible under resampling."
        )
    return warnings


amb_ok = consensus_ambiguity(con_ok["consensus_matrix"])
amb_over = consensus_ambiguity(con_over["consensus_matrix"])

print(f"Ambiguity at k={best_k}:   {amb_ok:.1%}")
print(f"Ambiguity at k={best_k + 1}:   {amb_over:.1%}")
print()
for w in stability_warnings(amb_ok, best_k) or ["OK: k=%d partition is stable." % best_k]:
    print(w)
for w in stability_warnings(amb_over, best_k + 1) or ["OK: k=%d partition is stable." % (best_k + 1)]:
    print(w)
```

The consensus matrices below make the difference visible. Rows and columns are
ordered by the consensus labels, so a stable partition shows crisp block-diagonal
squares (points inside a block always cluster together; points across blocks
never do). The over-clustered matrix instead shows a smeared, low-contrast block
that never fully commits -- the visual signature of a cluster the data do not
actually support.

```{code-cell} ipython3
def order_by_labels(matrix: np.ndarray, labels: np.ndarray) -> np.ndarray:
    order = np.argsort(labels)
    return matrix[np.ix_(order, order)]


fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))

im0 = axes[0].imshow(
    order_by_labels(con_ok["consensus_matrix"], con_ok["labels"]),
    cmap="magma", vmin=0, vmax=1,
)
axes[0].set_title(f"Stable: k={best_k} (ambiguity {amb_ok:.0%})")
im1 = axes[1].imshow(
    order_by_labels(con_over["consensus_matrix"], con_over["labels"]),
    cmap="magma", vmin=0, vmax=1,
)
axes[1].set_title(f"Over-clustered: k={best_k + 1} (ambiguity {amb_over:.0%})")
for ax in axes:
    ax.set_xlabel("Sample (reordered)")
    ax.set_ylabel("Sample (reordered)")
fig.colorbar(im1, ax=axes, label="Co-clustering frequency", shrink=0.8)
fig;
```

**QC note.** Adding a fifth cluster does not add information; it manufactures an
unstable split that resampling immediately flags. The lesson generalizes: on real
data, a cluster you cannot reproduce under subsampling should be reported as
tentative at best, never as a discovered subtype.

## 3. Clustering real single-cell data

The blob fixtures earned our trust in the machinery; now we point it at data with
no answer key. We load the 10x Genomics PBMC3k peripheral-blood single-cell
RNA-seq matrix through the course data layer. `get_dataset` returns the true 10x
matrix when it can reach the source and a labelled synthetic fallback with the
same payload shape otherwise, so the analysis below runs identically either way --
the provenance line tells the reader which one they got.

```{code-cell} ipython3
from sklearn.decomposition import PCA

from ddm4bio.datasets import get_dataset


def unpack_singlecell(payload):
    """Return (counts, labels) from an AnnData or a plain-dict payload."""
    if hasattr(payload, "X"):  # AnnData: counts in .X, no ground-truth labels
        return payload.X, None
    return payload["counts"], payload.get("labels")


ds_sc = get_dataset("pbmc3k")
print(f"pbmc3k source: {ds_sc.source}")
print(f"provenance: {ds_sc.provenance}")

counts_raw, sc_labels = unpack_singlecell(ds_sc.payload)
print(f"Raw counts: {counts_raw.shape[0]} cells x {counts_raw.shape[1]} genes")
```

Single-cell counts are not a feature matrix yet. We cap the cell count so the
consensus matrix stays affordable for either payload, normalize each cell to a
common library size and take `log1p`, keep the most variable genes, and reduce to
a handful of principal components -- the standard route from raw UMIs to a
geometry k-means can work in.

```{code-cell} ipython3
rng = np.random.default_rng(0)
n_keep = min(600, counts_raw.shape[0])
cells = np.sort(rng.choice(counts_raw.shape[0], size=n_keep, replace=False))

sub = counts_raw[cells]
counts_sc = sub.toarray() if hasattr(sub, "toarray") else np.asarray(sub, dtype=float)
counts_sc = np.asarray(counts_sc, dtype=float)
labels_sc = None if sc_labels is None else np.asarray(sc_labels)[cells]

library = counts_sc.sum(axis=1, keepdims=True)
library[library == 0] = 1.0
logn = np.log1p(counts_sc / library * 1e4)

n_top = min(1000, logn.shape[1])
top = np.argsort(logn.var(axis=0))[::-1][:n_top]
embed = PCA(
    n_components=min(30, n_keep - 1, n_top), random_state=0
).fit_transform(logn[:, top])
print(f"Working embedding: {embed.shape} (cells x PCs)")
```

We now run the *same* model-selection and clustering steps validated on the
blobs. With no truth column we lean on the two label-free signals the earlier
sections built: whether k-means and the Gaussian mixture agree, and how ambiguous
the consensus matrix is. When the payload is the labelled fallback we can also
score recovery against its planted cell types.

```{code-cell} ipython3
sil_sc = select_k_silhouette(embed, range(2, 8), seed=0)
k_sc = sil_sc["best_k"]

km_sc = kmeans_cluster(embed, k_sc, seed=0)
gmm_sc = gmm_cluster(embed, k_sc, seed=0)
con_sc = consensus_cluster(embed, k_sc, n_boot=25, subsample=0.8, seed=0)
amb_sc = consensus_ambiguity(con_sc["consensus_matrix"])

print(f"Silhouette picks k = {k_sc}")
print(f"k-means vs. GMM ARI:        {adjusted_rand_score(km_sc, gmm_sc):.3f}")
print(f"Consensus ambiguity at k={k_sc}: {amb_sc:.1%}")
if labels_sc is not None:
    print(f"k-means ARI vs. provided labels: {adjusted_rand_score(labels_sc, km_sc):.3f}")
else:
    print("No ground-truth labels in this payload; agreement and stability are the QC.")
```

The left panel shows the cells in their first two principal components, coloured
by the k-means partition; the right panel is the consensus matrix, reordered by
consensus label. Tight, well-separated colour groups and crisp block-diagonal
consensus are the label-free stand-ins for the ARI we no longer have.

```{code-cell} ipython3
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

axes[0].scatter(embed[:, 0], embed[:, 1], c=km_sc, s=10, cmap="tab10")
axes[0].set_title(f"PBMC cells in PC space (k-means, k={k_sc})")
axes[0].set_xlabel("PC 1")
axes[0].set_ylabel("PC 2")

im = axes[1].imshow(
    order_by_labels(con_sc["consensus_matrix"], con_sc["labels"]),
    cmap="magma", vmin=0, vmax=1,
)
axes[1].set_title(f"Consensus (ambiguity {amb_sc:.0%})")
axes[1].set_xlabel("Cell (reordered)")
axes[1].set_ylabel("Cell (reordered)")
fig.colorbar(im, ax=axes[1], shrink=0.8, label="Co-clustering frequency")

fig.suptitle(f"Clustering real single-cell data (source: {ds_sc.source})")
fig;
```

**QC note.** On real cells there is no ARI to hide behind, so the reportable
evidence is exactly what survived resampling: the number of clusters the
silhouette settles on, the agreement between two different algorithms, and a
consensus ambiguity low enough to trust the partition. If the loader fell back to
synthetic data the provenance line above says so -- and the planted labels then
let us confirm the pipeline end to end before a real cohort arrives with none.

## 4. A diagnostic classifier with a confidence interval

We now switch to supervised learning on a real-ish biomedical dataset: the
Wisconsin breast-cancer dataset bundled with scikit-learn, in which 30 features
computed from digitized tumor images are used to classify each sample as
malignant or benign. The diagnostic question is: *can these features separate
malignant from benign tumors, and how sure are we?* We frame malignant as the
positive class -- the case a screening tool must not miss.

We split off a held-out test set, standardize the features, and fit a logistic
regression. Keeping training and test data strictly separate is what makes the
performance estimate honest; a model scored on data it trained on will always
look better than it is.

```{code-cell} ipython3
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

ds_bc = get_dataset("breast_wisconsin")
print(f"breast_wisconsin source: {ds_bc.source}")
print(f"provenance: {ds_bc.provenance}")

X_bc = np.asarray(ds_bc.payload["X"], dtype=float)
y_bc = np.asarray(ds_bc.payload["y"], dtype=int)  # 0 = malignant, 1 = benign
bc_feature_names = np.asarray(ds_bc.payload["feature_names"])

X_train, X_test, y_train, y_test = train_test_split(
    X_bc, y_bc, test_size=0.30, stratify=y_bc, random_state=0
)

clf = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=5000, random_state=0),
)
clf.fit(X_train, y_train)

# Positive class = malignant (target 0); score = predicted P(malignant).
y_test_pos = (y_test == 0).astype(int)
scores_test = clf.predict_proba(X_test)[:, 0]

print(f"Training samples: {X_train.shape[0]}, test samples: {X_test.shape[0]}")
print(f"Test-set positives (malignant): {int(y_test_pos.sum())} of {y_test_pos.size}")
```

A single AUC number is not a result -- it is a point estimate with unstated
uncertainty. `roc_with_ci` bootstraps the test set to put a confidence interval
around the AUC, so we can say not just *how good* the classifier looks but *how
precisely* we know that. The width of the interval is set mostly by the size of
the test set, which is why the CI matters most on small biomedical cohorts.

```{code-cell} ipython3
from ddm4bio.methods.learning import roc_with_ci

roc = roc_with_ci(y_test_pos, scores_test, n_boot=1000, alpha=0.05, seed=0)
lo, hi = roc["auc_ci"]

print(f"Test AUC: {roc['auc']:.3f}")
print(f"95% bootstrap CI: [{lo:.3f}, {hi:.3f}]")
```

```{code-cell} ipython3
from ddm4bio.viz.plots import roc_with_ci as plot_roc

ax = plot_roc(roc["fpr"], roc["tpr"])
ax.set_title(f"Malignant-vs-benign ROC (AUC = {roc['auc']:.3f})")
ax.figure;  # end the cell with the Figure so it renders in the notebook output
```

**QC note.** The classifier separates the classes almost perfectly, and -- more
importantly -- the confidence interval is narrow and sits far above the 0.5
chance line, so the performance is not a fluke of one lucky split. Had the
interval reached down toward 0.5 we would have reported the result as
inconclusive regardless of how high the point estimate looked.

## 5. Screening many features while controlling false discovery

A common first-pass analysis asks, for each feature separately, whether it
differs between malignant and benign tumors. With 30 features that is 30
hypothesis tests, and multiple testing is where false discoveries breed: at a
0.05 threshold, even features with no real difference will cross the line about
5% of the time by chance alone. The Benjamini-Hochberg (BH) procedure controls
the **false-discovery rate** -- the expected fraction of your "significant" hits
that are actually null -- by adjusting each p-value into a q-value.

We run a Welch t-test per feature, then correct.

```{code-cell} ipython3
from scipy import stats
from ddm4bio.methods.learning import bh_fdr

malignant = X_bc[y_bc == 0]
benign = X_bc[y_bc == 1]

pvalues = np.array([
    stats.ttest_ind(malignant[:, j], benign[:, j], equal_var=False).pvalue
    for j in range(X_bc.shape[1])
])

fdr = bh_fdr(pvalues, alpha=0.05)

n_raw = int(np.sum(pvalues < 0.05))
n_fdr = int(np.sum(fdr["reject"]))
print(f"Features significant at raw p < 0.05:        {n_raw} / {pvalues.size}")
print(f"Features significant after BH FDR control:   {n_fdr} / {pvalues.size}")
```

On this dataset the malignant and benign groups differ so strongly across most
features that FDR correction changes little -- a reassuring sign that the signal
is real rather than a multiple-testing mirage. The value of the correction shows
when signal is weak: then the gap between the raw and corrected counts is where
the false discoveries would have lived. The plot ranks features by q-value; bars
above the dashed line clear the FDR threshold.

```{code-cell} ipython3
order = np.argsort(fdr["qvalues"])
q_sorted = fdr["qvalues"][order]
reject_sorted = fdr["reject"][order]
feature_names = bc_feature_names[order]

colors = ["#D55E00" if r else "#999999" for r in reject_sorted]
neglog_q = -np.log10(q_sorted + 1e-300)

fig, ax = plt.subplots(figsize=(10, 4.4))
ax.bar(np.arange(q_sorted.size), neglog_q, color=colors)
ax.axhline(-np.log10(0.05), color="0.3", linestyle="--", linewidth=1,
           label="FDR = 0.05 threshold")
ax.set_xlabel("Feature (ranked by q-value)")
ax.set_ylabel(r"$-\log_{10}(q)$")
ax.set_title("Per-feature significance after BH FDR control")
ax.legend(loc="upper right")
fig;
```

**QC note.** We report the FDR-corrected count, not the raw one. The four
features that fail correction here are exactly the ones a naive `p < 0.05` scan
would have been most likely to over-sell. Reporting corrected significance is not
optional bookkeeping -- it is the difference between a feature list you can defend
and one that is partly chance.

## 6. Interpretation

Every ddm4bio analysis closes with an explicit interpretation block: a single
claim, an honest confidence level backed by named evidence, and a list of stated
limitations. Here we pull the whole week together -- what clustered robustly, how
well the classifier discriminates, and where false-discovery risk still lives.

```{code-cell} ipython3
from ddm4bio.interpret import interpretation_block

block = interpretation_block(
    claim="The four blob clusters are recovered exactly and are stable under "
          "resampling; the breast-cancer classifier separates malignant from "
          "benign with high, tightly-bounded AUC; and most per-feature "
          "differences survive false-discovery-rate control.",
    confidence="high",
    limitations_list=[
        f"Cluster recovery (ARI={ari_km:.2f}) is on well-separated synthetic "
        "blobs; overlapping real subpopulations would lower silhouette/BIC "
        "agreement and raise consensus ambiguity.",
        f"Over-clustering at k={best_k + 1} produced an unstable partition "
        f"(ambiguity {amb_over:.0%}) -- a reminder that the clustering algorithm "
        "always returns clusters, stable or not.",
        f"The classifier AUC CI [{lo:.3f}, {hi:.3f}] comes from one train/test "
        "split; a small or biased cohort would widen it, and image-derived "
        "features may not transfer across scanners or sites.",
        f"{pvalues.size - n_fdr} of {pvalues.size} features fail BH correction; "
        "reporting raw p < 0.05 would over-state significance, and per-feature "
        "tests ignore correlations among the 30 features.",
    ],
    evidence=f"ARI={ari_km:.3f} vs. ground truth; consensus ambiguity "
             f"{amb_ok:.0%} at k={best_k} vs. {amb_over:.0%} at k={best_k + 1}; "
             f"test AUC={roc['auc']:.3f} (95% CI [{lo:.3f}, {hi:.3f}]); "
             f"{n_fdr}/{pvalues.size} features significant at FDR 0.05.",
)
print(block)
```

## Exercises

Your graded work for this week is **Problem Set 6 (PS6)**, distributed and
auto-graded through GitHub Classroom. Building on this lesson, PS6 asks you to:

- Push clustering into the hard regime: increase the blob overlap
  (`cluster_std`) until silhouette and BIC disagree, and report the consensus
  ambiguity at which you would stop trusting the partition.
- Compare `kmeans_cluster` and `gmm_cluster` on non-spherical clusters, and
  explain from the consensus matrices why one degrades faster than the other.
- Report the breast-cancer classifier's AUC with `roc_with_ci` across several
  random splits, and relate the spread of the point estimates to the width of a
  single split's bootstrap CI.
- Run per-feature tests on a harder target where signal is weak, and quantify how
  many raw "hits" the Benjamini-Hochberg correction with `bh_fdr` removes.
- Write an interpretation block for each result using
  `ddm4bio.interpret.interpretation_block`, with a defensible confidence level.

Refer to the PS6 repository README for the submission and auto-grading details.
