"""PS6: valid inference after clustering (the double-dipping trap).

Module 6 clusters cells and then runs BH-FDR on per-feature t-tests *between the clusters
it just discovered*. That is **circular**: the partition was chosen to separate the
cells, so features look "significant" even in pure noise -- and FDR does not fix it (BH
corrects multiplicity, not selection). This problem set makes that failure measurable,
and builds a procedure that actually controls the error.

- Part A -- **validate the trap and the fix on synthetic data with KNOWN truth**: on a
  pure-noise null (no clusters, no differential features, so the true marker count is
  exactly 0) quantify the false discoveries of three workflows: (i) naive double-dipping,
  (ii) sample-splitting -- the intuitive fix that is *still* inflated, and (iii) Gaussian
  data-thinning, which restores the nominal error rate. A power check confirms the valid
  methods keep real signal.
- Part B -- **apply the calibrated protocols to real PBMC3k and interpret honestly**.

Fill in every function body marked ``# TODO``. The clustering (`sklearn`'s `KMeans`), the
t-test (`per_feature_ttest`), BH-FDR (`bh_fdr`), the fixtures, `assign_to_centroids`,
and `estimate_sigma` are provided -- this problem set is about the testing *procedure*,
not the machinery. The autograder imports these functions by name, so keep the
signatures exactly as given. Run with ``python ps6.py``; it stops at the first
unimplemented function.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ddm4bio.config import GLOBAL_SEED, seed_everything
from ddm4bio.interpret import interpretation_block
from ddm4bio.methods.learning import bh_fdr  # noqa: F401  (use in the protocols)

# --------------------------------------------------------------------------- #
# Provided: fixtures, the t-test, and small helpers (do not edit)              #
# --------------------------------------------------------------------------- #


def make_null(n: int = 300, d: int = 50, seed: int = GLOBAL_SEED) -> np.ndarray:
    """(provided) Pure-noise data: ``X ~ N(0, I)`` with NO clusters and NO markers.

    The true number of differential features is exactly 0, so every rejection a testing
    procedure makes on this data is a known false discovery.

    Note: the RNG is namespaced (``[seed, 101]``) so the generated data is drawn from a
    different stream than a data-thinning noise draw at the same ``seed`` -- otherwise the
    thinning noise could reconstruct the data exactly and silently break the fold split.
    """
    return np.random.default_rng([seed, 101]).standard_normal((n, d))


def make_signal(
    n: int = 300,
    n_informative: int = 20,
    n_noise: int = 30,
    separation: float = 2.2,
    seed: int = GLOBAL_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """(provided) Two real groups with a KNOWN informative-feature mask (positive control).

    The first ``n_informative`` features carry a two-group mean shift; the remaining
    ``n_noise`` features are pure noise. Returns ``(X, informative_mask)`` where
    ``informative_mask`` is a boolean array marking the truly differential features.

    The RNG is namespaced (``[seed, 202]``) for the same reason as ``make_null``: the data
    stream must not coincide with a data-thinning noise draw at the same ``seed``.
    """
    rng = np.random.default_rng([seed, 202])
    group = rng.integers(0, 2, size=n)  # true labels (unknown to the procedures)
    shift = np.where(group[:, None] == 1, separation / 2, -separation / 2)
    informative = shift + rng.standard_normal((n, n_informative))
    noise = rng.standard_normal((n, n_noise))
    x = np.hstack([informative, noise])
    mask = np.zeros(n_informative + n_noise, dtype=bool)
    mask[:n_informative] = True
    return x, mask


def make_gene_permuted_null(x_real: np.ndarray, seed: int = GLOBAL_SEED) -> np.ndarray:
    """(provided) A real-flavored null: permute each gene column of real data independently.

    Destroys all cross-cell structure while preserving each gene's (sparse, heavy-tailed)
    marginal -- used only to *illustrate* that real single-cell marginals suppress the
    trap; Type-I error is graded on ``make_null``, not here.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x_real, dtype=float)
    out = np.empty_like(x)
    for j in range(x.shape[1]):
        out[:, j] = x[rng.permutation(x.shape[0]), j]
    return out


def load_pbmc(n_cells: int = 800, n_genes: int = 1000, seed: int = GLOBAL_SEED) -> np.ndarray:
    """(provided) Real PBMC3k, library-normalized + log1p + top-variance genes, subsampled."""
    from ddm4bio.datasets import get_dataset

    ds = get_dataset("pbmc3k")
    payload = ds.payload
    counts = payload.X if hasattr(payload, "X") else payload["counts"]
    counts = np.asarray(counts.toarray() if hasattr(counts, "toarray") else counts, dtype=float)
    library = counts.sum(1, keepdims=True)
    library[library == 0] = 1.0
    log_counts = np.log1p(counts / library * float(np.median(counts.sum(1))))
    top = np.argsort(log_counts.var(0))[::-1][: min(n_genes, log_counts.shape[1])]
    x = log_counts[:, top]
    if n_cells < x.shape[0]:
        idx = np.sort(np.random.default_rng(seed).choice(x.shape[0], size=n_cells, replace=False))
        x = x[idx]
    return x


def per_feature_ttest(X: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """(provided) Per-feature Welch two-sample t-test p-values between the two clusters.

    Compares the two most populous label groups feature by feature. Returns a p-value per
    feature (1.0 for a degenerate/constant feature or a group too small to test).
    """
    from scipy.stats import ttest_ind

    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    values, counts = np.unique(labels, return_counts=True)
    top_two = values[np.argsort(counts)[::-1][:2]]
    g0 = X[labels == top_two[0]]
    g1 = X[labels == top_two[1]] if top_two.size > 1 else X[:0]
    if g0.shape[0] < 2 or g1.shape[0] < 2:
        return np.ones(X.shape[1])
    p = ttest_ind(g0, g1, axis=0, equal_var=False).pvalue
    return np.nan_to_num(np.asarray(p, dtype=float), nan=1.0)


def assign_to_centroids(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """(provided) Nearest-centroid label for each row of ``X``."""
    X = np.asarray(X, dtype=float)
    centroids = np.asarray(centroids, dtype=float)
    dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
    return np.argmin(dists, axis=1)


def estimate_sigma(X: np.ndarray) -> np.ndarray:
    """(provided) Per-feature noise standard deviation, for data-thinning on real data."""
    return np.asarray(X, dtype=float).std(axis=0, ddof=1)


def _centroids_from_labels(X: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """(provided) Group-mean centroids for labels 0..k-1 (empty groups -> the global mean)."""
    X = np.asarray(X, dtype=float)
    grand = X.mean(axis=0)
    return np.array(
        [X[labels == c].mean(axis=0) if np.any(labels == c) else grand for c in range(k)]
    )


# --------------------------------------------------------------------------- #
# Part A -- The three post-clustering testing protocols  (you implement)       #
# --------------------------------------------------------------------------- #


def cluster_then_test_naive(
    X: np.ndarray, k: int = 2, alpha: float = 0.05, seed: int | None = None
) -> dict:
    """The double-dipping baseline: cluster ALL the data, then test between the clusters.

    Cluster every row with k-means (`sklearn`'s `KMeans`), run ``per_feature_ttest`` between
    the two clusters, BH-correct at ``alpha``. Because the partition was fit to separate the
    very rows being tested, this over-rejects on data with no real groups.
    """
    # TODO: import KMeans from sklearn.cluster inside this function;
    # labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(X);
    # pvalues = per_feature_ttest(X, labels); fdr = bh_fdr(pvalues, alpha). Return a dict with
    # keys "labels", "pvalues", "qvalues", "reject" (fdr["reject"]), and
    # "n_reject" (int(reject.sum())).
    raise NotImplementedError("Implement cluster_then_test_naive.")


def cluster_then_test_splitsample(
    X: np.ndarray, k: int = 2, alpha: float = 0.05, seed: int | None = None
) -> dict:
    """The intuitive but INSUFFICIENT fix: cluster on split A, test on held-out split B.

    Randomly split the rows into disjoint halves A and B, cluster A, assign B by nearest
    centroid, and test between the B groups. Still inflated on a null, because B's labels
    are assigned from B's own feature values -- the circularity survives the split.
    """
    # TODO: rng = np.random.default_rng(seed); perm = rng.permutation(len(X)); split into
    # disjoint idx_a, idx_b at the midpoint. Import KMeans from sklearn.cluster inside this
    # function and cluster X[idx_a] with
    # KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(X[idx_a]);
    # centroids = _centroids_from_labels(X[idx_a], labels_a, k); labels_b =
    # assign_to_centroids(X[idx_b], centroids); pvalues = per_feature_ttest(X[idx_b],
    # labels_b); BH-correct. Return {"labels_test", "pvalues", "qvalues", "reject",
    # "n_reject"}.
    raise NotImplementedError("Implement cluster_then_test_splitsample.")


def cluster_then_test_datathin(
    X: np.ndarray,
    sigma: np.ndarray | float,
    k: int = 2,
    alpha: float = 0.05,
    seed: int | None = None,
) -> dict:
    """The protocol that actually controls the error: Gaussian data-thinning.

    Draw ``eps ~ N(0, sigma^2)`` elementwise and form ``X1 = X + eps``, ``X2 = X - eps``;
    under Gaussian noise these are independent. Cluster ``X1``, test on ``X2``. The labels
    (from ``X1``) are independent of the values tested (``X2``), so a null grouping is
    random with respect to the test data and the false-discovery rate stays at nominal.
    """
    # TODO: rng = np.random.default_rng(seed); eps = rng.standard_normal(X.shape) * sigma;
    # X1 = X + eps, X2 = X - eps. Import KMeans from sklearn.cluster inside this function;
    # labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(X1);
    # pvalues = per_feature_ttest(X2, labels); BH-correct. Return {"labels", "pvalues",
    # "qvalues", "reject", "n_reject"}.
    raise NotImplementedError("Implement cluster_then_test_datathin.")


# --------------------------------------------------------------------------- #
# Part A -- Monte-Carlo calibration harnesses  (you implement)                 #
# --------------------------------------------------------------------------- #


def null_false_discovery_profile(
    generate_null: Callable[[int], np.ndarray],
    protocol: Callable[[np.ndarray, int], np.ndarray],
    R: int = 200,
    alpha: float = 0.05,
    seed0: int = 0,
) -> dict:
    """Type-I harness: how many false discoveries does ``protocol`` make on the null?

    Draw ``R`` independent known-null datasets from ``generate_null(seed)``, run
    ``protocol(X, seed)`` (which returns a boolean reject mask) on each, and summarize.
    Every rejection is by construction a false discovery. Returns ``mean_fd``, ``max_fd``,
    and ``prob_any_fd`` (the family-wise error rate).
    """
    # TODO: for r in range(R): s = seed0 + r; reject = protocol(generate_null(s), s);
    # record the number of rejections. Return {"mean_fd", "max_fd", "prob_any_fd"}
    # (prob_any_fd = fraction of the R runs with at least one rejection).
    raise NotImplementedError("Implement null_false_discovery_profile.")


def power_profile(
    generate_signal: Callable[[int], tuple[np.ndarray, np.ndarray]],
    protocol: Callable[[np.ndarray, int], np.ndarray],
    R: int = 30,
    alpha: float = 0.05,
    seed0: int = 0,
) -> dict:
    """Power harness: does ``protocol`` still recover the truly informative features?

    Over ``R`` datasets from ``generate_signal(seed) -> (X, informative_mask)``, run the
    protocol and count rejections that land on truly informative features. Returns
    ``mean_recovered`` and ``frac_true_recovered`` (fraction of informative features found).
    """
    # TODO: for r in range(R): s = seed0 + r; X, mask = generate_signal(s);
    # reject = protocol(X, s); count reject & mask (true positives) and the fraction of
    # informative features recovered. Return {"mean_recovered", "frac_true_recovered"}.
    raise NotImplementedError("Implement power_profile.")


# --------------------------------------------------------------------------- #
# Part B -- The real-data deliverable  (you implement)                         #
# --------------------------------------------------------------------------- #


def inflation_summary(
    X: np.ndarray,
    sigma: np.ndarray | float,
    k: int = 2,
    alpha: float = 0.05,
    seed: int | None = None,
) -> dict:
    """Run all three protocols on one dataset and quantify the naive over-call.

    Returns ``naive_n``, ``split_n``, ``thin_n`` (rejections each), ``inflation_extra``
    (``naive_n`` minus the larger of the two valid counts), and ``inflation_pct``.
    """
    # TODO: run the three protocols on X (n_reject each). valid = max(split_n, thin_n);
    # inflation_extra = naive_n - valid; inflation_pct = 100 * inflation_extra / max(valid, 1).
    # Return the dict with all five integer/float fields.
    raise NotImplementedError("Implement inflation_summary.")


# --------------------------------------------------------------------------- #
# Provided: driver                                                             #
# --------------------------------------------------------------------------- #


def main() -> None:
    """Show the double-dipping trap and the fix on known-truth data, then on real PBMC3k."""
    seed_everything()
    alpha = 0.05

    # Protocol callables sharing a (X, seed) -> reject-mask signature for the harnesses.
    def p_naive(x, s):
        return cluster_then_test_naive(x, k=2, alpha=alpha, seed=s)["reject"]

    def p_split(x, s):
        return cluster_then_test_splitsample(x, k=2, alpha=alpha, seed=s)["reject"]

    def p_thin(x, s):
        return cluster_then_test_datathin(x, 1.0, k=2, alpha=alpha, seed=s)["reject"]

    print("== Part A: the trap and the fix on synthetic data (true marker count = 0) ==")
    x0 = make_null(n=300, d=50, seed=0)
    print(
        f"    one null draw:  naive={cluster_then_test_naive(x0, seed=0)['n_reject']}  "
        f"split={cluster_then_test_splitsample(x0, seed=0)['n_reject']}  "
        f"thin={cluster_then_test_datathin(x0, 1.0, seed=0)['n_reject']}"
    )

    print("\n    Type-I over 200 null datasets (mean false discoveries / P(any) ):")
    for name, proto in (("naive ", p_naive), ("split ", p_split), ("thin  ", p_thin)):
        prof = null_false_discovery_profile(
            lambda s: make_null(300, 50, s), proto, R=200, alpha=alpha
        )
        print(
            f"      {name}: mean_fd={prof['mean_fd']:6.2f}  max_fd={prof['max_fd']:3d}  "
            f"P(any FD)={prof['prob_any_fd']:.3f}"
        )
    print(
        "    -> naive massively inflated; sample-splitting is STILL inflated; only "
        "data-thinning matches nominal alpha."
    )

    print("\n    Power over 30 signal datasets (informative features recovered / 20):")
    for name, proto in (("naive ", p_naive), ("split ", p_split), ("thin  ", p_thin)):
        pw = power_profile(lambda s: make_signal(seed=s), proto, R=30, alpha=alpha)
        print(
            f"      {name}: mean_recovered={pw['mean_recovered']:5.1f}  "
            f"frac={pw['frac_true_recovered']:.2f}"
        )
    print("    -> the valid methods keep their power; they only discard the fake signal.")

    print("\n== Part B: apply the calibrated protocols to real PBMC3k ==")
    x_real = load_pbmc(n_cells=800, n_genes=1000)
    sigma_hat = estimate_sigma(x_real)
    summ = inflation_summary(x_real, sigma_hat, k=2, seed=0)
    print(
        f"    naive={summ['naive_n']} markers @FDR{alpha}   "
        f"split={summ['split_n']}   thin={summ['thin_n']}"
    )
    print(
        f"    naive over-call vs the agreeing valid methods: +{summ['inflation_extra']} "
        f"({summ['inflation_pct']:.0f}%)"
    )
    perm = make_gene_permuted_null(x_real, seed=0)
    print(
        f"    gene-permuted real null: naive={cluster_then_test_naive(perm, seed=0)['n_reject']} "
        f"vs split={cluster_then_test_splitsample(perm, seed=0)['n_reject']} "
        f"-> real marginals suppress the trap (Type-I is graded on the synthetic null)."
    )

    print("\n== Interpretation ==")
    block = interpretation_block(
        claim=(
            "Testing marker genes between clusters discovered from the same cells is circular: "
            "on a known null it produces many false discoveries that BH-FDR does not prevent, and "
            "even sample-splitting stays inflated -- only data-thinning, which makes the discovery "
            "and test sets independent by construction, restores the nominal error rate while "
            "retaining power on real signal."
        ),
        limitations_list=[
            "Type-I control is demonstrated on Gaussian synthetic nulls; on real single-cell data "
            "the sparse, heavy-tailed gene marginals suppress the trap, so the real counts are "
            "illustrative (naive > valid), not a clean null.",
            "Data-thinning assumes additive Gaussian noise with a known/estimated variance; on "
            "genuinely count-distributed data the independence of the two folds is approximate.",
            "The synthetic power check uses a planted informative-feature mask; real effect sizes "
            "and dependence structure are richer.",
        ],
    )
    print(block)


if __name__ == "__main__":
    main()
