"""Unit tests for the decomposition / dimensionality-reduction wrappers.

Includes the key ground-truth test: ICA on ``make_mixed_sources`` must recover
the true sources with a max-matched correlation above 0.9. All tests are
deterministic and run offline.
"""

from __future__ import annotations

import numpy as np

from ddm4bio.datasets.synthetic import make_mixed_sources
from ddm4bio.methods.decomposition import (
    explained_variance_ratio,
    ica_unmix,
    pca_reduce,
    rpca,
    svd_lowrank,
)
from ddm4bio.methods.validation import reconstruction_error, source_recovery_score


def test_pca_reduce_recovers_low_rank_structure():
    rng = np.random.default_rng(0)
    # Rank-2 data embedded in 8 features: (100 samples, 8 features).
    scores_true = rng.standard_normal((100, 2))
    basis = rng.standard_normal((2, 8))
    x = scores_true @ basis

    scores = pca_reduce(x, 2)
    assert scores.shape == (100, 2)

    # For centered rank-2 data, the 2D projection captures ~all the variance.
    x_centered = x - x.mean(axis=0, keepdims=True)
    total_var = x_centered.var(axis=0, ddof=0).sum()
    captured_var = scores.var(axis=0, ddof=0).sum()
    assert np.isclose(captured_var, total_var, rtol=1e-6)


def test_explained_variance_ratio_sums_to_one():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((50, 6))
    evr = explained_variance_ratio(x)
    assert np.isclose(evr.sum(), 1.0)
    # Non-increasing ordering.
    assert np.all(np.diff(evr) <= 1e-12)


def test_explained_variance_ratio_low_rank_concentrates():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((100, 2)) @ rng.standard_normal((2, 8))
    evr = explained_variance_ratio(x)
    # Rank-2 data: first two ratios account for ~all variance.
    assert evr[:2].sum() > 0.999


def test_svd_lowrank_reconstructs_rank_k_matrix():
    rng = np.random.default_rng(3)
    k = 4
    x = rng.standard_normal((30, k)) @ rng.standard_normal((k, 40))

    u, s, vt = svd_lowrank(x, k)
    assert u.shape == (30, k)
    assert s.shape == (k,)
    assert vt.shape == (k, 40)

    recon = (u * s) @ vt
    assert reconstruction_error(x, recon, kind="rel_l2") < 1e-8


def test_ica_recovers_true_sources_ground_truth():
    m = make_mixed_sources(3, 2000, seed=0)
    est = ica_unmix(m.observations, 3, seed=0)
    assert est.shape == (3, 2000)

    score = source_recovery_score(m.sources, est)
    assert score > 0.9


def test_rpca_recovers_low_rank_plus_sparse():
    rng = np.random.default_rng(0)
    n, r = 50, 3
    low_true = rng.standard_normal((n, r)) @ rng.standard_normal((r, n))

    sparse_true = np.zeros((n, n))
    mask = rng.random((n, n)) < 0.05
    sparse_true[mask] = rng.standard_normal(int(mask.sum())) * 10.0

    observed = low_true + sparse_true
    low, sparse = rpca(observed)

    assert low.shape == observed.shape
    assert sparse.shape == observed.shape
    # Loose tolerance: the low-rank part is recovered approximately.
    assert reconstruction_error(low_true, low, kind="rel_l2") < 0.1
