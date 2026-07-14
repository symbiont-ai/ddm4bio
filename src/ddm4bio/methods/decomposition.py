"""Matrix decomposition and dimensionality-reduction teaching wrappers.

Thin educational wrappers around common linear-algebra decompositions used in
data-driven modeling for the life sciences (PCA, SVD, robust PCA, ICA).

Array-shape note
----------------
The rest of the library follows a ``(n_components, n_samples)`` convention
(components in rows, samples in columns) for multi-source / multi-channel
arrays. The PCA-style helpers here are the deliberate exception: they take a
**feature matrix** shaped ``(n_samples, n_features)`` (samples in rows), which
is the standard convention for statistical decompositions and for scikit-learn.
Each function documents its own orientation explicitly.
"""

from __future__ import annotations

import numpy as np


def pca_reduce(X: np.ndarray, n_components: int, center: bool = True) -> np.ndarray:
    """Project data onto its top principal components.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Feature matrix with rows = samples, columns = features. NOTE: this is
        the transpose of the library-wide ``(n_components, n_samples)`` source
        convention -- PCA here operates on samples-in-rows data.
    n_components : int
        Number of principal components to retain.
    center : bool, default True
        If True, subtract the per-feature mean before decomposing.

    Returns
    -------
    np.ndarray, shape (n_samples, n_components)
        Data expressed in the reduced principal-component coordinates
        (the PCA scores / projections).
    """
    X = np.asarray(X, dtype=float)
    if center:
        X = X - X.mean(axis=0, keepdims=True)
    # Economy SVD: rows of Vt are the principal axes (right singular vectors).
    _, s, vt = np.linalg.svd(X, full_matrices=False)
    k = min(n_components, vt.shape[0])
    scores = X @ vt[:k].T
    return scores


def svd_lowrank(X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a truncated (rank-k) singular value decomposition.

    Parameters
    ----------
    X : np.ndarray, shape (m, n)
        Input matrix.
    k : int
        Target rank (number of singular triplets to keep).

    Returns
    -------
    tuple of np.ndarray
        (U, s, Vt) with shapes (m, k), (k,), (k, n) such that
        ``X ~= U @ diag(s) @ Vt``.
    """
    X = np.asarray(X, dtype=float)
    u, s, vt = np.linalg.svd(X, full_matrices=False)
    k = min(k, s.shape[0])
    return u[:, :k], s[:k], vt[:k]


def rpca(
    X: np.ndarray,
    mu: float | None = None,
    lam: float | None = None,
    max_iter: int = 200,
    tol: float = 1e-7,
) -> tuple[np.ndarray, np.ndarray]:
    """Robust PCA: decompose X into low-rank plus sparse components.

    Solves the Principal Component Pursuit problem via the inexact augmented
    Lagrange multiplier (ALM) method. The low-rank term ``L`` is updated by
    singular-value soft-thresholding and the sparse term ``S`` by elementwise
    soft-thresholding, with a scaled dual variable ``Y``.

    Parameters
    ----------
    X : np.ndarray, shape (m, n)
        Observed data matrix.
    mu : float or None, default None
        Augmented-Lagrangian penalty parameter. If None, uses the standard
        heuristic ``mu = m * n / (4 * ||X||_1)``.
    lam : float or None, default None
        Sparsity trade-off weight. If None, uses ``1 / sqrt(max(m, n))``.
    max_iter : int, default 200
        Maximum number of ALM iterations.
    tol : float, default 1e-7
        Convergence tolerance on the relative reconstruction residual
        ``||X - L - S||_F / ||X||_F``.

    Returns
    -------
    tuple of np.ndarray
        (L, S) with shapes (m, n) each, where ``X ~= L + S``; ``L`` is
        low-rank and ``S`` is sparse.
    """
    from numpy.linalg import norm, svd

    X = np.asarray(X, dtype=float)
    m, n = X.shape

    frob_x = norm(X, "fro")
    if frob_x == 0.0:
        return np.zeros_like(X), np.zeros_like(X)

    if lam is None:
        lam = 1.0 / np.sqrt(max(m, n))
    l1_x = np.abs(X).sum()
    if mu is None:
        mu = m * n / (4.0 * l1_x) if l1_x > 0 else 1.0
    inv_mu = 1.0 / mu

    def _soft_threshold(mat: np.ndarray, thresh: float) -> np.ndarray:
        return np.sign(mat) * np.maximum(np.abs(mat) - thresh, 0.0)

    low_rank = np.zeros_like(X)
    sparse = np.zeros_like(X)
    dual = np.zeros_like(X)

    for _ in range(max_iter):
        # Update L via singular-value soft-thresholding of (X - S + Y/mu).
        u, s, vt = svd(X - sparse + inv_mu * dual, full_matrices=False)
        s_thresh = np.maximum(s - inv_mu, 0.0)
        low_rank = (u * s_thresh) @ vt

        # Update S via elementwise soft-thresholding of (X - L + Y/mu).
        sparse = _soft_threshold(X - low_rank + inv_mu * dual, lam * inv_mu)

        residual = X - low_rank - sparse
        dual = dual + mu * residual

        if norm(residual, "fro") / frob_x < tol:
            break

    return low_rank, sparse


def ica_unmix(observations: np.ndarray, n_sources: int, seed: int | None = None) -> np.ndarray:
    """Independent component analysis to recover source signals.

    Parameters
    ----------
    observations : np.ndarray, shape (n_channels, n_samples)
        Observed (mixed) signals following the library-wide convention:
        channels in rows, samples in columns.
    n_sources : int
        Number of independent sources to estimate.
    seed : int or None, default None
        Seed forwarded to FastICA's ``random_state`` for reproducibility.

    Returns
    -------
    np.ndarray, shape (n_sources, n_samples)
        Estimated source signals, one source per row (matching the input
        channels-in-rows convention).
    """
    from sklearn.decomposition import FastICA

    observations = np.asarray(observations, dtype=float)
    # sklearn expects (n_samples, n_features); transpose from (channels, samples).
    mixed = observations.T
    ica = FastICA(n_components=n_sources, random_state=seed)
    sources = ica.fit_transform(mixed)  # (n_samples, n_sources)
    return sources.T


def explained_variance_ratio(X: np.ndarray, center: bool = True) -> np.ndarray:
    """Fraction of total variance explained by each principal component.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Feature matrix with rows = samples, columns = features.
    center : bool, default True
        If True, subtract the per-feature mean before decomposing.

    Returns
    -------
    np.ndarray, shape (min(n_samples, n_features),)
        Non-increasing vector of explained-variance ratios summing to 1.
    """
    X = np.asarray(X, dtype=float)
    if center:
        X = X - X.mean(axis=0, keepdims=True)
    s = np.linalg.svd(X, full_matrices=False, compute_uv=False)
    variances = s**2
    total = variances.sum()
    if total == 0.0:
        return np.zeros_like(variances)
    return variances / total
