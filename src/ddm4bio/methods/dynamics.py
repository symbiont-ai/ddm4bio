"""Data-driven dynamical-systems methods.

Educational, dependency-light implementations of three data-driven dynamics
tools: exact dynamic mode decomposition (DMD), sparse identification of
nonlinear dynamics (SINDy) via sequential thresholded least squares, and a
linear Kalman filter. Everything here runs on numpy alone, so heavier libraries
are not needed and are not imported.

Array orientation
-----------------
* :func:`dmd` uses the *snapshot* convention ``(n_features, n_time)`` with
  state variables in ROWS and successive time samples in COLUMNS.
* :func:`sindy_fit` uses the *trajectory* convention ``(n_time, n_state)`` with
  time samples in ROWS and state variables in COLUMNS.
* :func:`kalman_filter` uses ``(n_time, n_obs)`` observations and returns
  ``(n_time, n_state)`` filtered state estimates.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations_with_replacement

import numpy as np


@dataclass
class DMDResult:
    """Result of an exact dynamic mode decomposition.

    Attributes
    ----------
    eigenvalues: ``(r,)`` complex DMD eigenvalues (discrete-time growth/rotation
        factors of the reduced operator ``A_tilde``).
    modes: ``(n_features, r)`` complex exact-DMD spatial modes (columns).
    amplitudes: ``(r,)`` complex modal amplitudes fitting the first snapshot.
    """

    eigenvalues: np.ndarray
    modes: np.ndarray
    amplitudes: np.ndarray


def dmd(X: np.ndarray, r: int | None = None) -> DMDResult:
    """Exact dynamic mode decomposition of a snapshot matrix.

    Splits ``X`` into the two time-shifted snapshot matrices
    ``X1 = X[:, :-1]`` and ``X2 = X[:, 1:]``, forms the rank-``r`` reduced
    operator ``A_tilde = U* X2 V S^-1`` from the economy SVD ``X1 = U S V*``,
    eigendecomposes it, and lifts the eigenvectors back to the full space to
    obtain the exact-DMD modes.

    Parameters
    ----------
    X : np.ndarray, shape (n_features, n_time)
        Time-ordered snapshot matrix; state variables in rows, successive time
        samples in columns.
    r : int, optional
        SVD truncation rank. When ``None`` the full numerical rank (all
        available singular values) is used.

    Returns
    -------
    DMDResult
        Dataclass exposing ``eigenvalues`` ``(r,)``, ``modes``
        ``(n_features, r)``, and ``amplitudes`` ``(r,)``.
    """
    x_arr = np.asarray(X)
    if x_arr.ndim != 2:
        raise ValueError("X must be a 2-D array of shape (n_features, n_time)")
    if x_arr.shape[1] < 2:
        raise ValueError("X must have at least two time samples (columns)")

    x1 = x_arr[:, :-1]
    x2 = x_arr[:, 1:]

    # Economy SVD of the first snapshot matrix: X1 = U @ diag(s) @ Vh.
    u_mat, s_vals, vh_mat = np.linalg.svd(x1, full_matrices=False)

    max_rank = s_vals.size
    if r is None:
        rank = max_rank
    else:
        rank = int(r)
        if rank < 1:
            raise ValueError("r must be a positive integer")
        rank = min(rank, max_rank)

    u_r = u_mat[:, :rank]
    s_r = s_vals[:rank]
    v_r = vh_mat.conj().T[:, :rank]

    # Reduced operator A_tilde = U* @ X2 @ V @ diag(1/s).
    s_inv = np.diag(1.0 / s_r)
    a_tilde = u_r.conj().T @ x2 @ v_r @ s_inv

    eigenvalues, w_vecs = np.linalg.eig(a_tilde)

    # Exact DMD modes: Phi = X2 @ V @ diag(1/s) @ W.
    modes = x2 @ v_r @ s_inv @ w_vecs

    # Amplitudes fit the first snapshot: b = pinv(Phi) @ x[:, 0].
    amplitudes = np.linalg.pinv(modes) @ x_arr[:, 0].astype(modes.dtype)

    return DMDResult(eigenvalues=eigenvalues, modes=modes, amplitudes=amplitudes)


@dataclass
class SINDyResult:
    """Result of a sequential-thresholded-least-squares SINDy fit.

    Attributes
    ----------
    coefficients: ``(n_features, n_state)`` sparse coefficient matrix mapping
        library features (rows) to state derivatives (columns).
    feature_names: length-``n_features`` list of library feature-name strings
        (e.g. ``"1"``, ``"x0"``, ``"x0^2"``, ``"x0 x1"``).
    active_terms: flat ``set[str]`` of feature names active in *any* state
        equation. This is the attribute consumed by
        :func:`ddm4bio.methods.validation.term_recovery`.
    active_terms_by_state: length-``n_state`` list of ``set[str]``, the active
        feature names for each individual state equation.
    """

    coefficients: np.ndarray
    feature_names: list[str]
    active_terms: set[str]
    active_terms_by_state: list[set[str]]


def _poly_feature_names(n_state: int, poly_degree: int) -> list[tuple[str, tuple[int, ...]]]:
    """Return ``(name, index_combo)`` pairs for the monomial library.

    The library includes the constant term ``"1"`` and every monomial up to
    ``poly_degree`` with cross terms, e.g. for two states and degree 2:
    ``["1", "x0", "x1", "x0^2", "x0 x1", "x1^2"]``.
    """
    features: list[tuple[str, tuple[int, ...]]] = [("1", ())]
    for degree in range(1, poly_degree + 1):
        for combo in combinations_with_replacement(range(n_state), degree):
            counts: dict[int, int] = {}
            for idx in combo:
                counts[idx] = counts.get(idx, 0) + 1
            parts = []
            for idx in sorted(counts):
                power = counts[idx]
                parts.append(f"x{idx}" if power == 1 else f"x{idx}^{power}")
            features.append((" ".join(parts), combo))
    return features


def _build_library(x_arr: np.ndarray, features: list[tuple[str, tuple[int, ...]]]) -> np.ndarray:
    """Evaluate the monomial library on the trajectory ``x_arr``.

    ``x_arr`` is ``(n_time, n_state)``; the returned Theta is
    ``(n_time, n_features)``.
    """
    n_time = x_arr.shape[0]
    theta = np.empty((n_time, len(features)), dtype=float)
    for col, (_name, combo) in enumerate(features):
        term = np.ones(n_time, dtype=float)
        for idx in combo:
            term = term * x_arr[:, idx]
        theta[:, col] = term
    return theta


def sindy_fit(
    X: np.ndarray,
    t: np.ndarray,
    poly_degree: int = 2,
    threshold: float = 0.1,
    dxdt: np.ndarray | None = None,
    max_iter: int = 20,
) -> SINDyResult:
    """Fit sparse nonlinear dynamics with sequential thresholded least squares.

    Builds a polynomial candidate library (monomials up to ``poly_degree`` with
    cross terms, plus a constant), estimates state derivatives by finite
    difference, and runs STLSQ: solve the least-squares problem, zero out
    coefficients whose magnitude falls below ``threshold``, refit on the
    surviving terms, and repeat until the active set stops changing.

    Parameters
    ----------
    X : np.ndarray, shape (n_time, n_state)
        Measured state trajectories; time samples in rows, states in columns.
    t : np.ndarray
        Time stamps for the rows of ``X`` (1-D, length ``n_time``) or a scalar
        sample spacing ``dt``.
    poly_degree : int, default 2
        Maximum monomial degree in the candidate library.
    threshold : float, default 0.1
        Sparsity threshold for STLSQ (coefficients with ``abs < threshold`` are
        removed).
    dxdt : np.ndarray, optional, shape (n_time, n_state)
        Precomputed time derivatives. When ``None`` they are estimated by
        finite difference (``numpy.gradient``).
    max_iter : int, default 20
        Maximum number of thresholding iterations.

    Returns
    -------
    SINDyResult
        Dataclass exposing ``coefficients`` ``(n_features, n_state)``,
        ``feature_names``, flat ``active_terms``, and ``active_terms_by_state``.
    """
    x_arr = np.asarray(X, dtype=float)
    if x_arr.ndim != 2:
        raise ValueError("X must be a 2-D array of shape (n_time, n_state)")
    n_time, n_state = x_arr.shape

    if dxdt is None:
        t_arr = np.asarray(t, dtype=float)
        if t_arr.ndim == 0:
            derivatives = np.gradient(x_arr, float(t_arr), axis=0)
        else:
            derivatives = np.gradient(x_arr, t_arr, axis=0)
    else:
        derivatives = np.asarray(dxdt, dtype=float)
        if derivatives.shape != x_arr.shape:
            raise ValueError("dxdt must have the same shape as X")

    features = _poly_feature_names(n_state, poly_degree)
    feature_names = [name for name, _combo in features]
    theta = _build_library(x_arr, features)
    n_features = theta.shape[1]

    # Initial least-squares fit for all state equations at once.
    coefficients, *_ = np.linalg.lstsq(theta, derivatives, rcond=None)

    for _ in range(max_iter):
        small = np.abs(coefficients) < threshold
        new_coeffs = coefficients.copy()
        new_coeffs[small] = 0.0
        # Refit each state on its surviving (big-coefficient) library terms.
        for state in range(n_state):
            big = ~small[:, state]
            if not np.any(big):
                continue
            fit, *_ = np.linalg.lstsq(theta[:, big], derivatives[:, state], rcond=None)
            new_coeffs[big, state] = fit
        if np.array_equal(new_coeffs == 0.0, coefficients == 0.0):
            coefficients = new_coeffs
            break
        coefficients = new_coeffs

    active_terms_by_state: list[set[str]] = []
    for state in range(n_state):
        active = {
            feature_names[row] for row in range(n_features) if coefficients[row, state] != 0.0
        }
        active_terms_by_state.append(active)

    active_terms: set[str] = set().union(*active_terms_by_state) if active_terms_by_state else set()

    return SINDyResult(
        coefficients=coefficients,
        feature_names=feature_names,
        active_terms=active_terms,
        active_terms_by_state=active_terms_by_state,
    )


def kalman_filter(
    observations: np.ndarray,
    F: np.ndarray,
    H: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    x0: np.ndarray | None = None,
    P0: np.ndarray | None = None,
) -> np.ndarray:
    """Standard linear Kalman filter for state estimation.

    Runs the classic predict/update recursion. At each step the prior state is
    advanced through ``F`` (with process covariance ``Q``), then corrected by
    the current observation through ``H`` (with measurement covariance ``R``).

    Parameters
    ----------
    observations : np.ndarray, shape (n_time, n_obs)
        Sequence of observation vectors (time samples in rows).
    F : np.ndarray, shape (n_state, n_state)
        State-transition matrix.
    H : np.ndarray, shape (n_obs, n_state)
        Observation (measurement) matrix.
    Q : np.ndarray, shape (n_state, n_state)
        Process-noise covariance.
    R : np.ndarray, shape (n_obs, n_obs)
        Measurement-noise covariance.
    x0 : np.ndarray, optional, shape (n_state,)
        Initial prior state mean; zeros when ``None``.
    P0 : np.ndarray, optional, shape (n_state, n_state)
        Initial prior state covariance; identity when ``None``.

    Returns
    -------
    np.ndarray, shape (n_time, n_state)
        Filtered (posterior) state estimates, one per time step.
    """
    obs = np.asarray(observations, dtype=float)
    if obs.ndim == 1:
        obs = obs[:, np.newaxis]
    if obs.ndim != 2:
        raise ValueError("observations must be (n_time, n_obs)")

    f_mat = np.asarray(F, dtype=float)
    h_mat = np.asarray(H, dtype=float)
    q_mat = np.asarray(Q, dtype=float)
    r_mat = np.asarray(R, dtype=float)

    n_time = obs.shape[0]
    n_state = f_mat.shape[0]

    x_est = np.zeros(n_state, dtype=float) if x0 is None else np.asarray(x0, dtype=float).copy()
    p_est = np.eye(n_state) if P0 is None else np.asarray(P0, dtype=float).copy()

    identity = np.eye(n_state)
    states = np.empty((n_time, n_state), dtype=float)

    for k in range(n_time):
        # Predict.
        x_pred = f_mat @ x_est
        p_pred = f_mat @ p_est @ f_mat.T + q_mat

        # Update with the current observation.
        innovation = obs[k] - h_mat @ x_pred
        s_mat = h_mat @ p_pred @ h_mat.T + r_mat
        gain = p_pred @ h_mat.T @ np.linalg.inv(s_mat)
        x_est = x_pred + gain @ innovation
        p_est = (identity - gain @ h_mat) @ p_pred

        states[k] = x_est

    return states
