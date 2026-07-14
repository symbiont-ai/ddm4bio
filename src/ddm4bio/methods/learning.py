"""Statistical-learning and evaluation teaching wrappers.

Educational wrappers for cross-validation, learning curves, ROC analysis with
confidence intervals, permutation testing, and multiple-comparison correction.
scikit-learn is imported inside the function bodies so ``import ddm4bio`` works
on numpy alone. All feature matrices follow scikit-learn's convention of shape
(n_samples, n_features).
"""

from __future__ import annotations

from typing import Any

import numpy as np


def cross_validate(
    estimator: Any,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    scoring: str | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Estimate generalization performance by k-fold cross-validation.

    Parameters
    ----------
    estimator : estimator
        Object with scikit-learn-compatible ``fit``/``predict`` methods.
    X : np.ndarray, shape (n_samples, n_features)
        Feature matrix.
    y : np.ndarray, shape (n_samples,)
        Target labels or values.
    cv : int, default 5
        Number of cross-validation folds.
    scoring : str, optional
        Scoring metric name; estimator default when None.
    seed : int, optional
        Seed controlling the fold shuffling for reproducibility.

    Returns
    -------
    dict
        Keys "scores" (per-fold np.ndarray), "mean" (float), and "std" (float).
    """
    from sklearn.base import is_classifier
    from sklearn.model_selection import (
        KFold,
        StratifiedKFold,
        cross_val_score,
    )

    X = np.asarray(X)
    y = np.asarray(y)

    if is_classifier(estimator):
        splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    else:
        splitter = KFold(n_splits=cv, shuffle=True, random_state=seed)

    scores = cross_val_score(estimator, X, y, cv=splitter, scoring=scoring)
    scores = np.asarray(scores, dtype=float)
    return {
        "scores": scores,
        "mean": float(scores.mean()),
        "std": float(scores.std()),
    }


def learning_curve(
    estimator: Any,
    X: np.ndarray,
    y: np.ndarray,
    train_sizes: np.ndarray | None = None,
    cv: int = 5,
    seed: int | None = None,
) -> dict[str, np.ndarray]:
    """Compute training/validation scores versus training-set size.

    Parameters
    ----------
    estimator : estimator
        scikit-learn-compatible estimator.
    X : np.ndarray, shape (n_samples, n_features)
        Feature matrix.
    y : np.ndarray, shape (n_samples,)
        Targets.
    train_sizes : np.ndarray, optional
        Fractions or absolute counts of training examples to evaluate;
        defaults to ``np.linspace(0.1, 1.0, 5)``.
    cv : int, default 5
        Number of cross-validation folds.
    seed : int, optional
        Seed controlling the fold shuffling for reproducibility.

    Returns
    -------
    dict
        Keys "train_sizes" (absolute counts), "train_scores", and "val_scores";
        the score arrays have shape (n_sizes, cv).
    """
    from sklearn.base import is_classifier
    from sklearn.model_selection import (
        KFold,
        StratifiedKFold,
    )
    from sklearn.model_selection import (
        learning_curve as _sk_learning_curve,
    )

    X = np.asarray(X)
    y = np.asarray(y)

    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 5)
    train_sizes = np.asarray(train_sizes)

    if is_classifier(estimator):
        splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    else:
        splitter = KFold(n_splits=cv, shuffle=True, random_state=seed)

    sizes_abs, train_scores, val_scores = _sk_learning_curve(
        estimator, X, y, train_sizes=train_sizes, cv=splitter
    )
    return {
        "train_sizes": np.asarray(sizes_abs),
        "train_scores": np.asarray(train_scores, dtype=float),
        "val_scores": np.asarray(val_scores, dtype=float),
    }


def roc_with_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int | None = None,
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
    seed : int, optional
        Seed for the bootstrap resampling.

    Returns
    -------
    dict
        Keys "fpr", "tpr", "thresholds" (np.ndarray), "auc" (float), and
        "auc_ci" (tuple of float lower/upper bounds).
    """
    from sklearn.metrics import roc_auc_score, roc_curve

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc = float(roc_auc_score(y_true, y_score))

    rng = np.random.default_rng(seed)
    n = y_true.shape[0]
    boot_aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y_true_b = y_true[idx]
        # A resample must contain both classes for AUC to be defined.
        if np.unique(y_true_b).size < 2:
            continue
        boot_aucs.append(roc_auc_score(y_true_b, y_score[idx]))

    if boot_aucs:
        boot_aucs = np.asarray(boot_aucs, dtype=float)
        lo = float(np.quantile(boot_aucs, alpha / 2.0))
        hi = float(np.quantile(boot_aucs, 1.0 - alpha / 2.0))
    else:
        lo = hi = float("nan")

    return {
        "fpr": np.asarray(fpr),
        "tpr": np.asarray(tpr),
        "thresholds": np.asarray(thresholds),
        "auc": auc,
        "auc_ci": (lo, hi),
    }


def permutation_test(
    estimator: Any,
    X: np.ndarray,
    y: np.ndarray,
    n_perm: int = 1000,
    cv: int = 5,
    scoring: str | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Permutation test for classifier/regressor significance.

    Parameters
    ----------
    estimator : estimator
        scikit-learn-compatible estimator.
    X : np.ndarray, shape (n_samples, n_features)
        Feature matrix.
    y : np.ndarray, shape (n_samples,)
        Targets.
    n_perm : int, default 1000
        Number of label permutations.
    cv : int, default 5
        Number of cross-validation folds.
    scoring : str, optional
        Scoring metric name; estimator default when None.
    seed : int, optional
        Seed controlling both the CV splits and the label permutations.

    Returns
    -------
    dict
        Keys "observed_score" (float), "permutation_scores" (np.ndarray of
        shape (n_perm,)), and "p_value" (float).
    """
    from sklearn.base import is_classifier
    from sklearn.model_selection import (
        KFold,
        StratifiedKFold,
        permutation_test_score,
    )

    X = np.asarray(X)
    y = np.asarray(y)

    if is_classifier(estimator):
        splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    else:
        splitter = KFold(n_splits=cv, shuffle=True, random_state=seed)

    observed, perm_scores, p_value = permutation_test_score(
        estimator,
        X,
        y,
        cv=splitter,
        n_permutations=n_perm,
        scoring=scoring,
        random_state=seed,
    )
    return {
        "observed_score": float(observed),
        "permutation_scores": np.asarray(perm_scores, dtype=float),
        "p_value": float(p_value),
    }


def bh_fdr(pvalues: np.ndarray, alpha: float = 0.05) -> dict[str, np.ndarray]:
    """Benjamini-Hochberg false-discovery-rate correction (numpy only).

    Parameters
    ----------
    pvalues : np.ndarray, shape (n_tests,)
        Uncorrected p-values.
    alpha : float, default 0.05
        Target FDR level.

    Returns
    -------
    dict
        Keys "qvalues" (np.ndarray of adjusted q-values) and "reject" (boolean
        np.ndarray of rejected null hypotheses), each of shape (n_tests,).
    """
    pvalues = np.asarray(pvalues, dtype=float)
    n = pvalues.shape[0]
    if n == 0:
        return {
            "qvalues": np.array([], dtype=float),
            "reject": np.array([], dtype=bool),
        }

    order = np.argsort(pvalues)
    ranks = np.arange(1, n + 1)
    sorted_p = pvalues[order]

    # Adjusted q-values via the step-up BH procedure with monotonicity.
    qvals_sorted = sorted_p * n / ranks
    qvals_sorted = np.minimum.accumulate(qvals_sorted[::-1])[::-1]
    qvals_sorted = np.clip(qvals_sorted, 0.0, 1.0)

    qvalues = np.empty(n, dtype=float)
    qvalues[order] = qvals_sorted
    reject = qvalues <= alpha

    return {"qvalues": qvalues, "reject": reject}
