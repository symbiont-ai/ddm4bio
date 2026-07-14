"""Omics and tabular biomedical datasets.

Datasets:
- GDSC: Genomics of Drug Sensitivity in Cancer (drug response).
  Tier: open. License: GDSC terms (academic use); synthetic Hill-curve
  fallback provided when the download is unavailable.
- Breast Cancer Wisconsin (Diagnostic). Tier: open. License: CC BY-NC-SA
  (bundled with scikit-learn).
- Heart Disease (UCI, Cleveland). Tier: open. License: CC BY 4.0 (UCI).
Install: ``pip install scikit-learn pandas``.
"""

from __future__ import annotations

from typing import Any


def load_gdsc() -> Any:
    """Load GDSC drug-sensitivity data (with synthetic Hill-curve fallback).

    Returns
    -------
    object
        A container exposing dose-response measurements: cell lines x drugs
        with IC50/AUC values, or a synthetically generated Hill-curve dataset
        when the real download is unavailable.

    Notes
    -----
    Uses ``pandas`` (imported inside the body).
    """
    raise NotImplementedError(
        "load_gdsc is a Phase 0 stub; expected return exposes dose-response "
        "measurements (cell lines x drugs) or a synthetic Hill fallback."
    )


def load_breast_wisconsin() -> Any:
    """Load the Breast Cancer Wisconsin (Diagnostic) dataset.

    Returns
    -------
    object
        A container exposing the feature matrix ``X`` of shape
        ``(n_samples, n_features)`` and binary labels ``y`` of shape
        ``(n_samples,)``.

    Notes
    -----
    Uses ``sklearn.datasets`` (imported inside the body).
    """
    raise NotImplementedError(
        "load_breast_wisconsin is a Phase 0 stub; expected return exposes "
        "X[n,features] and y[n] binary labels."
    )


def load_heart_uci() -> Any:
    """Load the UCI Heart Disease (Cleveland) dataset.

    Returns
    -------
    object
        A container exposing the feature matrix ``X`` of shape
        ``(n_samples, n_features)`` and diagnosis labels ``y`` of shape
        ``(n_samples,)``.

    Notes
    -----
    Uses ``pandas`` (imported inside the body).
    """
    raise NotImplementedError(
        "load_heart_uci is a Phase 0 stub; expected return exposes X[n,features] and y[n] labels."
    )
