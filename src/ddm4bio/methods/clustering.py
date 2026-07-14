"""Clustering with stability diagnostics for Week-6 teaching.

Wrappers around scikit-learn clustering (k-means, Gaussian mixtures,
agglomerative) plus model-selection helpers (silhouette, BIC) and a consensus
clustering routine. The emphasis is on *stability*: silhouette and BIC guard
against choosing too many clusters, while consensus clustering resamples the
data so that spurious, unstable clusters can be flagged rather than trusted.

All functions take a scikit-learn feature matrix ``X`` of shape
(n_samples, n_features) and return integer label arrays of shape (n_samples,)
as numpy arrays. scikit-learn is imported inside each function body so that
``import ddm4bio`` works with numpy alone.
"""

from __future__ import annotations

import numpy as np


def kmeans_cluster(X: np.ndarray, k: int, seed: int | None = None) -> np.ndarray:
    """Partition samples into ``k`` clusters with k-means.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Feature matrix (scikit-learn orientation).
    k : int
        Number of clusters.
    seed : int, optional
        Seed for reproducible centroid initialization.

    Returns
    -------
    np.ndarray, shape (n_samples,)
        Integer cluster labels in ``[0, k)``.
    """
    from sklearn.cluster import KMeans

    x = np.asarray(X, dtype=float)
    model = KMeans(n_clusters=k, random_state=seed, n_init=10)
    return np.asarray(model.fit_predict(x))


def gmm_cluster(X: np.ndarray, k: int, seed: int | None = None) -> np.ndarray:
    """Cluster samples with a ``k``-component Gaussian mixture model.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Feature matrix (scikit-learn orientation).
    k : int
        Number of mixture components (clusters).
    seed : int, optional
        Seed for reproducible initialization.

    Returns
    -------
    np.ndarray, shape (n_samples,)
        Integer cluster labels in ``[0, k)`` (hard MAP assignment).
    """
    from sklearn.mixture import GaussianMixture

    x = np.asarray(X, dtype=float)
    model = GaussianMixture(n_components=k, random_state=seed)
    return np.asarray(model.fit_predict(x))


def hierarchical_cluster(X: np.ndarray, k: int, linkage: str = "ward") -> np.ndarray:
    """Agglomerative (hierarchical) clustering into ``k`` clusters.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Feature matrix (scikit-learn orientation).
    k : int
        Number of flat clusters to cut the dendrogram into.
    linkage : str, default "ward"
        Linkage criterion ("ward", "complete", "average", "single").

    Returns
    -------
    np.ndarray, shape (n_samples,)
        Integer cluster labels in ``[0, k)``.
    """
    from sklearn.cluster import AgglomerativeClustering

    x = np.asarray(X, dtype=float)
    model = AgglomerativeClustering(n_clusters=k, linkage=linkage)
    return np.asarray(model.fit_predict(x))


def select_k_silhouette(
    X: np.ndarray, k_range: object, seed: int | None = None
) -> dict[str, object]:
    """Choose the number of clusters by mean silhouette score.

    Fits k-means for each candidate ``k`` and scores the partition with the
    mean silhouette coefficient (higher is better). Guards against splitting a
    homogeneous dataset into too many weakly separated clusters.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Feature matrix (scikit-learn orientation).
    k_range : iterable of int
        Candidate cluster counts; each must be >= 2 for a defined silhouette.
    seed : int, optional
        Seed forwarded to k-means for reproducibility.

    Returns
    -------
    dict
        Keys: "best_k" (int), "ks" (np.ndarray of candidates), and "scores"
        (np.ndarray of mean silhouette per candidate).
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    x = np.asarray(X, dtype=float)
    ks = np.asarray(list(k_range), dtype=int)
    scores = np.empty(ks.shape[0], dtype=float)
    for i, k in enumerate(ks):
        labels = KMeans(n_clusters=int(k), random_state=seed, n_init=10).fit_predict(x)
        if np.unique(labels).size < 2:
            scores[i] = -1.0
        else:
            scores[i] = silhouette_score(x, labels)
    best_k = int(ks[int(np.argmax(scores))])
    return {"best_k": best_k, "ks": ks, "scores": scores}


def select_k_bic(X: np.ndarray, k_range: object, seed: int | None = None) -> dict[str, object]:
    """Choose the number of clusters by Gaussian-mixture BIC.

    Fits a Gaussian mixture for each candidate ``k`` and records its Bayesian
    Information Criterion (lower is better). BIC penalizes model complexity, so
    it resists over-fitting the data with excess components.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Feature matrix (scikit-learn orientation).
    k_range : iterable of int
        Candidate component counts (clusters).
    seed : int, optional
        Seed forwarded to the mixture model for reproducibility.

    Returns
    -------
    dict
        Keys: "best_k" (int, minimizing BIC), "ks" (np.ndarray of candidates),
        and "bic" (np.ndarray of BIC values per candidate).
    """
    from sklearn.mixture import GaussianMixture

    x = np.asarray(X, dtype=float)
    ks = np.asarray(list(k_range), dtype=int)
    bic = np.empty(ks.shape[0], dtype=float)
    for i, k in enumerate(ks):
        model = GaussianMixture(n_components=int(k), random_state=seed)
        model.fit(x)
        bic[i] = model.bic(x)
    best_k = int(ks[int(np.argmin(bic))])
    return {"best_k": best_k, "ks": ks, "bic": bic}


def consensus_cluster(
    X: np.ndarray,
    k: int,
    n_boot: int = 20,
    subsample: float = 0.8,
    seed: int | None = None,
) -> dict[str, object]:
    """Consensus clustering to flag unstable clusters via resampling.

    Repeatedly subsamples the rows of ``X``, runs k-means on each subsample,
    and accumulates a co-association (consensus) matrix: entry (i, j) is the
    fraction of resamples in which samples i and j, when both drawn, landed in
    the same cluster. Final labels come from agglomerative clustering on the
    consensus distance (1 - consensus). Pairs with intermediate consensus mark
    unstable structure that should not be over-trusted.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Feature matrix (scikit-learn orientation).
    k : int
        Number of clusters.
    n_boot : int, default 20
        Number of bootstrap subsamples.
    subsample : float, default 0.8
        Fraction of samples drawn (without replacement) per bootstrap.
    seed : int, optional
        Seed for the local random generator.

    Returns
    -------
    dict
        Keys: "consensus_matrix" (np.ndarray, shape (n_samples, n_samples),
        symmetric with unit diagonal), and "labels" (np.ndarray, shape
        (n_samples,)) of consensus cluster assignments.
    """
    from sklearn.cluster import AgglomerativeClustering, KMeans

    x = np.asarray(X, dtype=float)
    n_samples = x.shape[0]
    rng = np.random.default_rng(seed)

    m = max(1, int(round(subsample * n_samples)))
    co_assign = np.zeros((n_samples, n_samples), dtype=float)
    co_sampled = np.zeros((n_samples, n_samples), dtype=float)

    for _ in range(n_boot):
        idx = rng.choice(n_samples, size=m, replace=False)
        sub = x[idx]
        # Draw a fresh integer seed so each bootstrap fit is deterministic yet
        # varied, without touching global numpy state.
        km_seed = int(rng.integers(0, 2**31 - 1))
        labels = KMeans(n_clusters=k, random_state=km_seed, n_init=10).fit_predict(sub)

        co_sampled[np.ix_(idx, idx)] += 1.0
        same = labels[:, None] == labels[None, :]
        co_assign[np.ix_(idx, idx)] += same.astype(float)

    with np.errstate(invalid="ignore", divide="ignore"):
        consensus = np.where(co_sampled > 0, co_assign / co_sampled, 0.0)
    np.fill_diagonal(consensus, 1.0)

    distance = 1.0 - consensus
    np.fill_diagonal(distance, 0.0)
    model = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average")
    labels = np.asarray(model.fit_predict(distance))

    return {"consensus_matrix": consensus, "labels": labels}
