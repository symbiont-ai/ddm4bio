"""Extra coverage for clustering + reproducibility helpers.

Exercises the currently-uncovered paths: the consensus (co-association) matrix
and its labels, hierarchical recovery of separated blobs, the full per-k scoring
loops of ``select_k_silhouette`` / ``select_k_bic``, and the determinism/seeding
edge cases (non-deterministic callable, RNG restoration after an exception).
All tests are offline and deterministic (seeded).
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

from ddm4bio.methods.clustering import (
    consensus_cluster,
    hierarchical_cluster,
    select_k_bic,
    select_k_silhouette,
)
from ddm4bio.utils.seeds import check_determinism, with_seed


def _blobs(seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Three compact, well-separated Gaussian blobs in 2-D."""
    x, y = make_blobs(
        n_samples=150,
        centers=3,
        cluster_std=0.6,
        random_state=seed,
    )
    return x, y


# --------------------------------------------------------------------------- #
# consensus_cluster
# --------------------------------------------------------------------------- #
def test_consensus_matrix_shape_and_bounds():
    x, _ = _blobs()
    result = consensus_cluster(x, k=3, n_boot=15, subsample=0.8, seed=0)

    consensus = result["consensus_matrix"]
    labels = result["labels"]

    n = x.shape[0]
    assert consensus.shape == (n, n)
    # All entries are valid probabilities in [0, 1].
    assert np.all(consensus >= 0.0)
    assert np.all(consensus <= 1.0)
    # Symmetric with a unit diagonal.
    assert np.allclose(consensus, consensus.T)
    assert np.allclose(np.diag(consensus), 1.0)

    assert labels.shape == (n,)
    assert np.unique(labels).size == 3


def test_consensus_strong_within_cluster_coassociation():
    x, y_true = _blobs()
    result = consensus_cluster(x, k=3, n_boot=25, subsample=0.8, seed=1)
    consensus = result["consensus_matrix"]

    # Mean co-association among same-true-cluster pairs (off-diagonal) should
    # dominate that of different-cluster pairs for well-separated blobs.
    same = y_true[:, None] == y_true[None, :]
    off_diag = ~np.eye(x.shape[0], dtype=bool)
    within = consensus[same & off_diag]
    between = consensus[~same]

    assert within.mean() > 0.9
    assert between.mean() < 0.1
    assert within.mean() > between.mean()


def test_consensus_labels_recover_blobs():
    x, y_true = _blobs()
    result = consensus_cluster(x, k=3, n_boot=25, subsample=0.8, seed=2)
    assert adjusted_rand_score(y_true, result["labels"]) > 0.9


def test_consensus_is_deterministic_under_seed():
    x, _ = _blobs()
    a = consensus_cluster(x, k=3, n_boot=10, seed=7)
    b = consensus_cluster(x, k=3, n_boot=10, seed=7)
    assert np.array_equal(a["consensus_matrix"], b["consensus_matrix"])
    assert np.array_equal(a["labels"], b["labels"])


# --------------------------------------------------------------------------- #
# hierarchical_cluster
# --------------------------------------------------------------------------- #
def test_hierarchical_recovers_blobs():
    x, y_true = _blobs()
    labels = hierarchical_cluster(x, k=3)

    assert labels.shape == (x.shape[0],)
    assert np.unique(labels).size == 3
    assert adjusted_rand_score(y_true, labels) > 0.9


def test_hierarchical_average_linkage_recovers_blobs():
    x, y_true = _blobs()
    labels = hierarchical_cluster(x, k=3, linkage="average")
    assert adjusted_rand_score(y_true, labels) > 0.9


# --------------------------------------------------------------------------- #
# select_k_silhouette / select_k_bic -- exercise the full scoring collections
# --------------------------------------------------------------------------- #
def test_select_k_silhouette_scores_collection():
    x, _ = _blobs()
    k_range = range(2, 7)
    result = select_k_silhouette(x, k_range, seed=0)

    ks = result["ks"]
    scores = result["scores"]

    assert np.array_equal(ks, np.asarray(list(k_range)))
    assert scores.shape == ks.shape
    # Silhouette is bounded in [-1, 1].
    assert np.all(scores >= -1.0)
    assert np.all(scores <= 1.0)
    # best_k is the argmax over the score collection and equals the true k.
    assert result["best_k"] == int(ks[int(np.argmax(scores))])
    assert result["best_k"] == 3


def test_select_k_bic_scores_collection():
    x, _ = _blobs()
    k_range = range(1, 7)
    result = select_k_bic(x, k_range, seed=0)

    ks = result["ks"]
    bic = result["bic"]

    assert np.array_equal(ks, np.asarray(list(k_range)))
    assert bic.shape == ks.shape
    assert np.all(np.isfinite(bic))
    # best_k is the argmin over the BIC collection and equals the true k.
    assert result["best_k"] == int(ks[int(np.argmin(bic))])
    assert result["best_k"] == 3


# --------------------------------------------------------------------------- #
# seeds: check_determinism
# --------------------------------------------------------------------------- #
def test_check_determinism_true_for_deterministic_callable():
    # A seeded draw is reproducible when both runs share the same seed context.
    result = check_determinism(lambda: np.random.rand(8), seed=123)
    assert result is True


def test_check_determinism_raises_for_nondeterministic_callable():
    # A counter that advances on every call yields different outputs each run,
    # even under a shared seed context (it ignores the RNG).
    counter = itertools.count()

    def nondeterministic() -> int:
        return next(counter)

    with pytest.raises(AssertionError):
        check_determinism(nondeterministic, seed=0)


def test_check_determinism_raises_for_unseeded_random_callable():
    # Without a seed context, two independent random draws almost surely differ.
    with pytest.raises(AssertionError):
        check_determinism(lambda: np.random.rand(8), seed=None)


# --------------------------------------------------------------------------- #
# seeds: with_seed restores global RNG state even on exception
# --------------------------------------------------------------------------- #
def test_with_seed_restores_rng_state_after_exception():
    np.random.seed(321)
    np.random.rand(3)  # advance the global stream

    # An exception raised inside the block must not leak the block's RNG state.
    with pytest.raises(ValueError):
        with with_seed(999):
            np.random.rand(50)
            raise ValueError("boom")

    after = np.random.rand(4)

    # Control: identical setup, never entering the with_seed block.
    np.random.seed(321)
    np.random.rand(3)
    control_after = np.random.rand(4)

    assert np.array_equal(after, control_after)


def test_with_seed_restores_python_random_state_after_exception():
    import random

    random.seed(321)
    random.random()  # advance the global stdlib stream

    with pytest.raises(ValueError):
        with with_seed(999):
            random.random()
            raise ValueError("boom")

    after = random.random()

    random.seed(321)
    random.random()
    control_after = random.random()

    assert after == control_after
