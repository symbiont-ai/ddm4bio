"""Ground-truth tests for the statistical-learning wrappers.

ROC/AUC with a bootstrap CI on linearly separable data, Benjamini-Hochberg FDR
control on a mixture of tiny and large p-values, a permutation test that must
find genuinely predictive data significant, and k-fold cross-validation that
must return a finite mean score. All tests are deterministic and run offline.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from ddm4bio.methods.learning import bh_fdr, cross_validate, permutation_test, roc_with_ci


def _separable_classification(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Two well-separated Gaussian clouds as a (X, y) classification set."""
    rng = np.random.default_rng(seed)
    n_per = 50
    x_neg = rng.normal(-2.0, 1.0, size=(n_per, 4))
    x_pos = rng.normal(2.0, 1.0, size=(n_per, 4))
    x = np.vstack([x_neg, x_pos])
    y = np.array([0] * n_per + [1] * n_per)
    return x, y


def test_roc_with_ci_near_perfect_on_separable_scores():
    rng = np.random.default_rng(0)
    y_true = np.array([0] * 100 + [1] * 100)
    y_score = np.concatenate([rng.normal(0.0, 1.0, 100), rng.normal(5.0, 1.0, 100)])

    result = roc_with_ci(y_true, y_score, n_boot=500, seed=1)

    auc = result["auc"]
    lo, hi = result["auc_ci"]
    assert auc > 0.95
    # A valid CI brackets the point estimate.
    assert lo <= auc <= hi
    assert 0.0 <= lo <= 1.0
    assert 0.0 <= hi <= 1.0
    assert result["fpr"].shape == result["tpr"].shape


def test_bh_fdr_rejects_small_and_controls_large():
    small = np.array([1e-6, 1e-5, 1e-4, 2e-4])
    large = np.full(50, 0.8)
    pvalues = np.concatenate([small, large])

    result = bh_fdr(pvalues, alpha=0.05)
    reject = result["reject"]

    # All genuinely tiny p-values are rejected.
    assert reject[: small.size].all()
    # None of the large (null) p-values are rejected.
    assert not reject[small.size :].any()
    # q-values are valid probabilities.
    assert np.all(result["qvalues"] >= 0.0)
    assert np.all(result["qvalues"] <= 1.0)


def test_permutation_test_significant_on_predictive_data():
    x, y = _separable_classification(seed=0)
    result = permutation_test(LogisticRegression(max_iter=1000), x, y, n_perm=200, cv=5, seed=1)

    assert result["observed_score"] > 0.9
    assert result["p_value"] < 0.05
    assert result["permutation_scores"].shape == (200,)


def test_cross_validate_returns_finite_mean_in_range():
    x, y = _separable_classification(seed=2)
    result = cross_validate(LogisticRegression(max_iter=1000), x, y, cv=5, seed=1)

    assert np.isfinite(result["mean"])
    # Accuracy of a classifier lives in [0, 1]; separable data scores high.
    assert 0.0 <= result["mean"] <= 1.0
    assert result["mean"] > 0.8
    assert result["scores"].shape == (5,)
