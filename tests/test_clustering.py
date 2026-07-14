"""Ground-truth tests for the clustering and model-selection helpers.

k-means and the Gaussian mixture must recover well-separated blobs (high
adjusted Rand index against the true labels), and both model-selection routines
must identify the true number of blobs. All tests are deterministic and offline.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

from ddm4bio.methods.clustering import (
    gmm_cluster,
    kmeans_cluster,
    select_k_bic,
    select_k_silhouette,
)


def _blobs(seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Three compact, well-separated Gaussian blobs in 2-D."""
    x, y = make_blobs(
        n_samples=300,
        centers=3,
        cluster_std=0.6,
        random_state=seed,
    )
    return x, y


def test_kmeans_recovers_blobs():
    x, y_true = _blobs()
    labels = kmeans_cluster(x, k=3, seed=0)

    assert labels.shape == (x.shape[0],)
    assert adjusted_rand_score(y_true, labels) > 0.9


def test_gmm_recovers_blobs():
    x, y_true = _blobs()
    labels = gmm_cluster(x, k=3, seed=0)

    assert labels.shape == (x.shape[0],)
    assert adjusted_rand_score(y_true, labels) > 0.9


def test_select_k_silhouette_finds_true_k():
    x, _ = _blobs()
    result = select_k_silhouette(x, range(2, 7), seed=0)
    assert result["best_k"] == 3


def test_select_k_bic_finds_true_k():
    x, _ = _blobs()
    result = select_k_bic(x, range(1, 7), seed=0)
    assert result["best_k"] == 3
