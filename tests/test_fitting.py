"""Ground-truth tests for the curve-fitting and regularized-estimation tools.

Hill-curve fitting must recover the true EC50 and Hill coefficient; the Lasso
must recover the exact nonzero support of a sparse linear model; and Tikhonov-
regularized differentiation must beat a plain finite difference on a noisy sine
whose analytic derivative is known. All tests are deterministic and offline.
"""

from __future__ import annotations

import numpy as np

from ddm4bio.methods.fitting import fit_hill, hill, lasso_select, regularized_derivative


def test_fit_hill_recovers_ec50_and_hill_coefficient():
    x = np.logspace(-2.0, 2.0, 40)
    true_ec50 = 5.0
    true_n = 2.0
    y_clean = hill(x, bottom=0.0, top=1.0, ec50=true_ec50, n=true_n)

    rng = np.random.default_rng(0)
    y = y_clean + 0.01 * rng.standard_normal(y_clean.shape)

    result = fit_hill(x, y, seed=0)

    assert result["success"]
    assert abs(result["ec50"] - true_ec50) < 0.5
    assert abs(result["hill_coeff"] - true_n) < 0.3


def test_lasso_select_recovers_true_support():
    rng = np.random.default_rng(0)
    n_samples, n_features = 200, 20
    x = rng.standard_normal((n_samples, n_features))

    true_support = np.array([1, 5, 10, 15])
    beta = np.zeros(n_features)
    beta[true_support] = np.array([3.0, -2.0, 4.0, -1.5])
    y = x @ beta + 0.05 * rng.standard_normal(n_samples)

    result = lasso_select(x, y, seed=0)

    assert np.array_equal(np.sort(result["selected"]), true_support)


def test_regularized_derivative_beats_finite_difference():
    n = 300
    t = np.linspace(0.0, 2.0 * np.pi, n)
    dx = t[1] - t[0]
    clean = np.sin(t)
    analytic_derivative = np.cos(t)

    rng = np.random.default_rng(0)
    noisy = clean + 0.05 * rng.standard_normal(n)

    reg = regularized_derivative(noisy, dx, lam=1e-1)
    finite_diff = np.gradient(noisy, dx)

    err_reg = np.linalg.norm(reg - analytic_derivative)
    err_fd = np.linalg.norm(finite_diff - analytic_derivative)
    assert err_reg < err_fd
