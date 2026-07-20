"""Curve-fitting and regularized-estimation tools for Week 2.

Dose-response models (4-parameter logistic and Hill), nonlinear least-squares
fitting with parameter standard errors, Tikhonov-regularized differentiation of
noisy signals, and sparse feature selection via the Lasso. Only numpy is
imported at module scope; scipy and scikit-learn are imported inside the
function bodies that need them so ``import ddm4bio`` works on numpy alone.

Array conventions
-----------------
The 1-D curve/model helpers operate on plain sample vectors of shape
``(n_samples,)``. ``lasso_select`` follows the scikit-learn feature-matrix
convention ``X`` of shape ``(n_samples, n_features)``.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def four_pl(
    x: np.ndarray,
    a: float,
    b: float,
    c: float,
    d: float,
) -> np.ndarray:
    """Four-parameter logistic (4PL) dose-response model.

    Evaluates ``d + (a - d) / (1 + (x / c) ** b)``, the classic sigmoid used
    for dose-response and immunoassay standard curves.

    Parameters
    ----------
    x : np.ndarray, shape (n_samples,)
        Independent variable (e.g. dose or concentration); expected positive.
    a : float
        Response at ``x -> 0`` (lower asymptote / minimum).
    b : float
        Hill slope controlling steepness of the transition.
    c : float
        Inflection point on the ``x`` axis (the EC50).
    d : float
        Response at ``x -> inf`` (upper asymptote / maximum).

    Returns
    -------
    np.ndarray, shape (n_samples,)
        Model response evaluated at each ``x``.
    """
    x = np.asarray(x, dtype=float)
    return d + (a - d) / (1.0 + (x / c) ** b)


def hill(
    x: np.ndarray,
    bottom: float,
    top: float,
    ec50: float,
    n: float,
) -> np.ndarray:
    """Hill dose-response model.

    Evaluates ``bottom + (top - bottom) * x ** n / (ec50 ** n + x ** n)``, a
    monotone sigmoid rising from ``bottom`` to ``top`` with half-maximal
    response at ``ec50``.

    Parameters
    ----------
    x : np.ndarray, shape (n_samples,)
        Independent variable (e.g. concentration); expected non-negative.
    bottom : float
        Lower asymptote (response at ``x = 0``).
    top : float
        Upper asymptote (response as ``x -> inf``).
    ec50 : float
        Concentration giving half-maximal response.
    n : float
        Hill coefficient: steepness of the transition (a binding-cooperativity index in
        receptor assays; an empirical steepness for phenotypic/viability readouts).

    Returns
    -------
    np.ndarray, shape (n_samples,)
        Model response evaluated at each ``x``.
    """
    x = np.asarray(x, dtype=float)
    xn = x**n
    return bottom + (top - bottom) * xn / (ec50**n + xn)


def fit_hill(
    x: np.ndarray,
    y: np.ndarray,
    p0: tuple[float, float, float, float] | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Fit the Hill model to dose-response data by nonlinear least squares.

    Uses :func:`scipy.optimize.curve_fit` (Levenberg-Marquardt / trust-region)
    to estimate the Hill parameters and their covariance. When ``p0`` is not
    supplied, robust initial guesses are derived from the data; ``seed`` only
    perturbs those guesses so the call is reproducible.

    Parameters
    ----------
    x : np.ndarray, shape (n_samples,)
        Independent variable (concentration/dose), non-negative.
    y : np.ndarray, shape (n_samples,)
        Observed response.
    p0 : tuple of float, optional
        Initial guess ``(bottom, top, ec50, n)``. Estimated from the data when
        None.
    seed : int, optional
        Seed for a local ``np.random.default_rng`` used to jitter the initial
        guess; does not touch global RNG state.

    Returns
    -------
    dict
        Keys: "bottom", "top", "ec50", "hill_coeff" (floats, the fitted
        parameters); "params" (np.ndarray of the four values in that order);
        "std_errors" (np.ndarray of 1-sigma standard errors); "covariance"
        (np.ndarray, shape (4, 4)); and "success" (bool).
    """
    from scipy.optimize import curve_fit

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if p0 is None:
        bottom0 = float(np.min(y))
        top0 = float(np.max(y))
        # EC50 guess: x nearest the half-maximal response level.
        half = bottom0 + 0.5 * (top0 - bottom0)
        ec50_0 = float(x[np.argmin(np.abs(y - half))])
        if ec50_0 <= 0.0:
            positive = x[x > 0.0]
            ec50_0 = float(np.median(positive)) if positive.size else 1.0
        n0 = 1.0
        rng = np.random.default_rng(seed)
        jitter = 1.0 + 0.01 * rng.standard_normal(4)
        p0 = (
            bottom0,
            top0 if top0 != bottom0 else bottom0 + 1.0,
            ec50_0 * jitter[2],
            n0 * jitter[3],
        )

    try:
        popt, pcov = curve_fit(hill, x, y, p0=p0, maxfev=10000)
        success = True
    except (RuntimeError, ValueError):
        popt = np.asarray(p0, dtype=float)
        pcov = np.full((4, 4), np.nan)
        success = False

    variances = np.diag(pcov)
    std_errors = np.sqrt(np.where(variances >= 0.0, variances, np.nan))

    bottom, top, ec50, hill_coeff = (float(v) for v in popt)
    return {
        "bottom": bottom,
        "top": top,
        "ec50": ec50,
        "hill_coeff": hill_coeff,
        "params": np.asarray(popt, dtype=float),
        "std_errors": std_errors,
        "covariance": pcov,
        "success": success,
    }


def regularized_derivative(
    y: np.ndarray,
    dx: float,
    lam: float = 1e-2,
) -> np.ndarray:
    """Estimate the derivative of a noisy 1-D signal by Tikhonov regularization.

    Rather than differencing the data directly (which amplifies noise), this
    finds the derivative ``u`` whose antiderivative best reproduces ``y`` while
    penalizing curvature of ``u``. Concretely it minimizes
    ``||A u - (y - y[0])||^2 + lam * ||D u||^2`` where ``A`` is the cumulative
    (trapezoidal) integration operator and ``D`` is the second-difference
    operator, then solves the resulting normal equations
    ``(A^T A + lam D^T D) u = A^T (y - y[0])``. Larger ``lam`` yields a
    smoother derivative estimate.

    Parameters
    ----------
    y : np.ndarray, shape (n_samples,)
        Noisy signal sampled on a uniform grid.
    dx : float
        Uniform spacing between samples.
    lam : float, default 1e-2
        Regularization strength; larger values smooth more aggressively.

    Returns
    -------
    np.ndarray, shape (n_samples,)
        Estimated derivative at each sample point.
    """
    from scipy.linalg import solve

    y = np.asarray(y, dtype=float)
    n = y.size
    if n < 2:
        return np.zeros(n)

    # Cumulative trapezoidal integration operator A: (A u)[i] approximates the
    # integral of u from x[0] to x[i], so A u reconstructs y - y[0].
    a_op = np.zeros((n, n))
    for i in range(1, n):
        a_op[i] = a_op[i - 1]
        a_op[i, i - 1] += 0.5 * dx
        a_op[i, i] += 0.5 * dx

    # Second-difference (curvature) penalty operator D, shape (n-2, n).
    d_op = np.zeros((max(n - 2, 0), n))
    for i in range(n - 2):
        d_op[i, i] = 1.0
        d_op[i, i + 1] = -2.0
        d_op[i, i + 2] = 1.0

    rhs_signal = y - y[0]
    lhs = a_op.T @ a_op + lam * (d_op.T @ d_op)
    rhs = a_op.T @ rhs_signal

    # Tiny ridge keeps the (rank-deficient) system well-posed for small lam.
    lhs = lhs + 1e-12 * np.eye(n)
    return solve(lhs, rhs, assume_a="sym")


def lasso_select(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Sparse feature selection via the Lasso.

    Fits an L1-penalized linear model to drive uninformative coefficients to
    exactly zero. When ``alpha`` is None the penalty is chosen automatically by
    cross-validation (:class:`sklearn.linear_model.LassoCV`); otherwise a plain
    :class:`~sklearn.linear_model.Lasso` with the given ``alpha`` is used.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Feature matrix (scikit-learn orientation).
    y : np.ndarray, shape (n_samples,)
        Target vector.
    alpha : float, optional
        L1 penalty strength. When None it is selected by cross-validation.
    seed : int, optional
        Seed for reproducibility; forwarded to the estimator's
        ``random_state`` (its coordinate-descent selection uses this).

    Returns
    -------
    dict
        Keys: "selected" (np.ndarray of int indices with nonzero coefficient);
        "coefficients" (np.ndarray, shape (n_features,)); "intercept" (float);
        "alpha" (float, the penalty actually used).
    """
    from sklearn.linear_model import Lasso, LassoCV

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    if alpha is None:
        model = LassoCV(cv=5, random_state=seed)
    else:
        model = Lasso(alpha=alpha, random_state=seed)
    model.fit(X, y)

    coefficients = np.asarray(model.coef_, dtype=float)
    selected = np.flatnonzero(coefficients != 0.0)
    return {
        "selected": selected,
        "coefficients": coefficients,
        "intercept": float(model.intercept_),
        "alpha": float(model.alpha_ if alpha is None else alpha),
    }
