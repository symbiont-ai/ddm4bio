"""Correctness/validation harness for method recovery metrics.

Dependency-light: numpy may be used at module top level. Everything else must
be imported inside the function body that uses it (added in Phase 1).
"""

import numpy as np


def source_recovery_score(true_sources, est_sources) -> float:
    """Max-matched absolute correlation between true and estimated sources.

    Intended for evaluating ICA-style source separation. Both arrays are
    expected to be shape (n_sources, n_samples). The metric greedily (or
    optimally) matches each true source to an estimated source, taking the
    absolute Pearson correlation of the best matching, and returns the mean
    absolute correlation over matched pairs (a float in [0, 1]).

    Parameters
    ----------
    true_sources : np.ndarray, shape (n_sources, n_samples)
    est_sources : np.ndarray, shape (n_sources, n_samples)

    Returns
    -------
    float
        Mean max-matched absolute correlation in [0, 1].
    """
    from scipy.optimize import linear_sum_assignment

    true = np.asarray(true_sources, dtype=float)
    est = np.asarray(est_sources, dtype=float)
    if true.ndim == 1:
        true = true[np.newaxis, :]
    if est.ndim == 1:
        est = est[np.newaxis, :]

    n_true = true.shape[0]
    n_est = est.shape[0]
    corr = np.zeros((n_true, n_est), dtype=float)
    for i in range(n_true):
        for j in range(n_est):
            row = true[i]
            col = est[j]
            row_c = row - row.mean()
            col_c = col - col.mean()
            denom = np.sqrt(np.sum(row_c**2) * np.sum(col_c**2))
            if denom == 0:
                corr[i, j] = 0.0
            else:
                corr[i, j] = abs(np.dot(row_c, col_c) / denom)

    row_ind, col_ind = linear_sum_assignment(-corr)
    matched = corr[row_ind, col_ind]
    if matched.size == 0:
        return 0.0
    return float(matched.mean())


def term_recovery(true_terms, sindy_model) -> dict:
    """Precision/recall over active library terms for a SINDy model.

    Compares the set of active (nonzero-coefficient) library terms recovered
    by ``sindy_model`` against the ground-truth active term set ``true_terms``.

    Parameters
    ----------
    true_terms : set or sequence of str
        Ground-truth active term identifiers.
    sindy_model : object
        A fitted SINDy model exposing its selected terms/coefficients.

    Returns
    -------
    dict
        Keys ``precision``, ``recall``, ``f1`` (floats), and optionally the
        matched term sets.
    """
    true_set = set(true_terms)

    active = None
    for attr in ("active_terms", "selected_terms", "active", "terms"):
        if hasattr(sindy_model, attr):
            active = getattr(sindy_model, attr)
            break
    if active is None:
        active = sindy_model
    est_set = set(active)

    true_positives = len(true_set & est_set)
    precision = true_positives / len(est_set) if est_set else 0.0
    recall = true_positives / len(true_set) if true_set else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def reconstruction_error(x_true, x_hat, kind: str = "rel_l2") -> float:
    """Reconstruction error for CS / denoising / DMD tasks.

    Parameters
    ----------
    x_true : np.ndarray
        Ground-truth signal/array.
    x_hat : np.ndarray
        Reconstructed estimate, same shape as ``x_true``.
    kind : str, default "rel_l2"
        Error type: "rel_l2" (relative L2 norm ||x_true - x_hat|| / ||x_true||),
        "l2" (absolute L2 norm), or "mse" (mean squared error).

    Returns
    -------
    float
        Non-negative scalar error.
    """
    true = np.asarray(x_true, dtype=float)
    hat = np.asarray(x_hat, dtype=float)
    diff = true - hat

    if kind == "mse":
        return float(np.mean(diff**2))
    if kind == "l2":
        return float(np.sqrt(np.sum(diff**2)))
    if kind == "rel_l2":
        num = np.sqrt(np.sum(diff**2))
        denom = np.sqrt(np.sum(true**2))
        if denom == 0:
            return float(num)
        return float(num / denom)
    raise ValueError(f"Unknown kind: {kind!r} (expected 'rel_l2', 'l2', or 'mse')")
