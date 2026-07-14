"""Statistical-learning and evaluation teaching wrappers.

Educational wrappers for cross-validation, learning curves, ROC analysis with
confidence intervals, permutation testing, and multiple-comparison correction.
scikit-learn is imported inside the function bodies. Phase 0 scaffold: bodies
raise NotImplementedError.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def cross_validate(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    scoring: str | None = None,
) -> dict[str, np.ndarray]:
    """Estimate generalization performance by k-fold cross-validation.

    Parameters
    ----------
    model : estimator
        Object with scikit-learn-compatible ``fit``/``predict`` methods.
    X : np.ndarray, shape (n_samples, n_features)
        Feature matrix.
    y : np.ndarray, shape (n_samples,)
        Target labels or values.
    cv : int, default 5
        Number of cross-validation folds.
    scoring : str, optional
        Scoring metric name; estimator default when None.

    Returns
    -------
    dict of str -> np.ndarray
        Mapping of metric name to per-fold score arrays (e.g. "test_score").
    """
    # sklearn imported inside body per scaffold rules.
    raise NotImplementedError


def learning_curve(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    train_sizes: np.ndarray | None = None,
    cv: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute training/validation scores versus training-set size.

    Parameters
    ----------
    model : estimator
        scikit-learn-compatible estimator.
    X : np.ndarray, shape (n_samples, n_features)
        Feature matrix.
    y : np.ndarray, shape (n_samples,)
        Targets.
    train_sizes : np.ndarray, optional
        Fractions or absolute counts of training examples to evaluate.
    cv : int, default 5
        Number of cross-validation folds.

    Returns
    -------
    tuple of np.ndarray
        (train_sizes_abs, train_scores, val_scores) where score arrays have
        shape (n_sizes, cv).
    """
    # sklearn imported inside body per scaffold rules.
    raise NotImplementedError


def roc_with_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """ROC curve and AUC with a bootstrap confidence interval.

    Parameters
    ----------
    y_true : np.ndarray, shape (n_samples,)
        Binary ground-truth labels.
    y_score : np.ndarray, shape (n_samples,)
        Predicted scores or probabilities for the positive class.
    n_boot : int, default 1000
        Number of bootstrap resamples for the AUC interval.
    alpha : float, default 0.05
        Significance level; the CI covers ``1 - alpha``.

    Returns
    -------
    dict
        Keys include "fpr", "tpr", "thresholds" (np.ndarray), "auc" (float),
        and "auc_ci" (tuple of float lower/upper bounds).
    """
    # sklearn imported inside body per scaffold rules.
    raise NotImplementedError


def permutation_test(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    n_permutations: int = 1000,
    scoring: str | None = None,
) -> tuple[float, np.ndarray, float]:
    """Permutation test for classifier/regressor significance.

    Parameters
    ----------
    model : estimator
        scikit-learn-compatible estimator.
    X : np.ndarray, shape (n_samples, n_features)
        Feature matrix.
    y : np.ndarray, shape (n_samples,)
        Targets.
    n_permutations : int, default 1000
        Number of label permutations.
    scoring : str, optional
        Scoring metric name.

    Returns
    -------
    tuple
        (score, permutation_scores, pvalue) where ``score`` is the observed
        score, ``permutation_scores`` has shape (n_permutations,), and
        ``pvalue`` is the empirical p-value.
    """
    # sklearn imported inside body per scaffold rules.
    raise NotImplementedError


def bh_fdr(pvals: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg false-discovery-rate correction.

    Parameters
    ----------
    pvals : np.ndarray, shape (n_tests,)
        Uncorrected p-values.
    alpha : float, default 0.05
        Target FDR level.

    Returns
    -------
    tuple of np.ndarray
        (reject, pvals_corrected) each of shape (n_tests,); ``reject`` is a
        boolean array of rejected null hypotheses and ``pvals_corrected`` holds
        the adjusted p-values.
    """
    raise NotImplementedError
