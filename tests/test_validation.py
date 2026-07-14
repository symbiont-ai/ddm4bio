"""Unit tests for the recovery/validation metrics.

Checks the identity, permutation/sign invariance of ``source_recovery_score``,
the exact hand-computed values of ``reconstruction_error`` for each ``kind``,
and that the zero-norm guards do not raise.
"""

from __future__ import annotations

import numpy as np

from ddm4bio.methods.validation import reconstruction_error, source_recovery_score


def test_source_recovery_score_identity():
    rng = np.random.default_rng(0)
    sources = rng.standard_normal((3, 500))
    score = source_recovery_score(sources, sources)
    assert np.isclose(score, 1.0, atol=1e-9)


def test_source_recovery_score_permutation_and_sign_invariant():
    rng = np.random.default_rng(1)
    sources = rng.standard_normal((4, 400))

    # Permute rows and negate one row; the matched score should stay ~1.0.
    perm = np.array([2, 0, 3, 1])
    est = sources[perm].copy()
    est[1] = -est[1]

    score = source_recovery_score(sources, est)
    assert np.isclose(score, 1.0, atol=1e-9)


def test_reconstruction_error_matches_hand_computed():
    x_true = np.array([1.0, 2.0, 3.0, 4.0])
    x_hat = np.array([1.0, 2.0, 3.0, 0.0])
    diff = x_true - x_hat  # [0, 0, 0, 4]

    expected_l2 = float(np.sqrt(np.sum(diff**2)))  # 4.0
    expected_mse = float(np.mean(diff**2))  # 4.0
    expected_rel = expected_l2 / float(np.sqrt(np.sum(x_true**2)))

    assert np.isclose(reconstruction_error(x_true, x_hat, kind="l2"), expected_l2)
    assert np.isclose(reconstruction_error(x_true, x_hat, kind="mse"), expected_mse)
    assert np.isclose(reconstruction_error(x_true, x_hat, kind="rel_l2"), expected_rel)


def test_reconstruction_error_default_is_rel_l2():
    x_true = np.array([3.0, 4.0])
    x_hat = np.array([0.0, 0.0])
    # rel_l2 with a nonzero reference: ||diff|| / ||x_true|| = 5/5 = 1.0
    assert np.isclose(reconstruction_error(x_true, x_hat), 1.0)


def test_reconstruction_error_zero_norm_guard_does_not_raise():
    zeros = np.zeros(5)
    other = np.array([0.0, 1.0, 0.0, -1.0, 0.0])
    # Reference norm is zero: must fall back to the absolute error, not divide.
    value = reconstruction_error(zeros, other, kind="rel_l2")
    assert np.isfinite(value)
    assert np.isclose(value, float(np.sqrt(np.sum(other**2))))


def test_source_recovery_score_zero_variance_row_does_not_raise():
    rng = np.random.default_rng(2)
    sources = rng.standard_normal((2, 200))
    est = sources.copy()
    est[0] = 7.0  # constant row -> zero variance -> zero-denominator branch
    # Must return a finite score without raising.
    score = source_recovery_score(sources, est)
    assert np.isfinite(score)
    assert 0.0 <= score <= 1.0
