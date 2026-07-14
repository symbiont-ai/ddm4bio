"""Clinical tabular datasets.

Datasets:
- UCI Heart Disease (processed Cleveland). Tier: open. License: CC BY 4.0
  (UCI Machine Learning Repository). Real loader downloads the processed
  Cleveland CSV; ``sklearn.make_classification`` clinical-like fallback.
- Breast Cancer Wisconsin (Diagnostic). Tier: open. The "real" payload is the
  scikit-learn *bundled* Wisconsin Diagnostic Breast Cancer (WDBC) dataset, so
  no network is needed; a numpy two-Gaussian fallback covers a missing
  scikit-learn install.

Install: ``pip install scikit-learn pandas``. Heavy imports (pandas,
scikit-learn) live inside the function bodies so importing this module needs
only numpy. Fallbacks are deterministic given ``seed``.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np

from .registry import LoadedDataset

# UCI processed Cleveland heart-disease CSV (no header, "?" for missing).
_HEART_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)
_HEART_CACHE_NAME = "processed.cleveland.data"
_HEART_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target",
]
_HEART_FEATURES = _HEART_COLUMNS[:-1]


def _download_to_cache(url: str, dest: Path) -> Path:
    """Download ``url`` to ``dest`` once (idempotent). Return ``dest``.

    Existing cache files are returned untouched -- the real fetch is never
    repeated. The download is staged to a temp sibling then atomically
    replaced.
    """
    if dest.exists():
        return dest
    req = urllib.request.Request(url, headers={"User-Agent": "ddm4bio/0.1"})
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 (static https)
        data = resp.read()
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(dest)
    return dest


def _heart_fallback(seed: int | None) -> LoadedDataset:
    """Deterministic clinical-like classification table via scikit-learn."""
    import pandas as pd
    from sklearn.datasets import make_classification

    x_arr, y_arr = make_classification(
        n_samples=303,
        n_features=len(_HEART_FEATURES),
        n_informative=8,
        n_redundant=2,
        n_classes=2,
        class_sep=1.0,
        flip_y=0.01,
        random_state=seed,
    )
    payload = {
        "X": pd.DataFrame(x_arr, columns=_HEART_FEATURES),
        "y": pd.Series(y_arr.astype("int64"), name="target"),
    }
    provenance = (
        "synthetic/bundled fallback: scikit-learn make_classification "
        "clinical-like table (303 samples x 13 features, binary target), "
        f"seeded with seed={seed!r}; NOT the real UCI Cleveland data."
    )
    return LoadedDataset(payload=payload, source="fallback", provenance=provenance, key="heart_uci")


def load_heart_uci(
    *,
    cache_dir,
    download: bool = True,
    prefer_real: bool = True,
    seed: int | None = None,
    **opts,
) -> LoadedDataset:
    """Load the UCI Heart Disease (processed Cleveland) dataset.

    Parameters
    ----------
    cache_dir:
        Directory for the cached raw CSV download.
    download, prefer_real:
        When both True, fetch and cache the real UCI CSV; otherwise return the
        scikit-learn ``make_classification`` fallback.
    seed:
        Seed for the deterministic fallback table.
    **opts:
        Ignored extra loader options.

    Returns
    -------
    LoadedDataset
        ``payload`` is a dict with ``X`` (:class:`pandas.DataFrame`, 13 clinical
        features) and ``y`` (:class:`pandas.Series`, binary presence-of-disease
        label). ``source`` is ``"real"`` or ``"fallback"``.
    """
    if download and prefer_real:
        try:
            import pandas as pd

            cache_path = Path(cache_dir) / _HEART_CACHE_NAME
            _download_to_cache(_HEART_URL, cache_path)
            df = pd.read_csv(cache_path, header=None, names=_HEART_COLUMNS, na_values="?")
            payload = {
                "X": df[_HEART_FEATURES],
                # Original target is 0..4 severity; binarize to disease present.
                "y": (df["target"] > 0).astype("int64").rename("target"),
            }
            provenance = (
                "real: UCI Heart Disease processed Cleveland CSV "
                f"({_HEART_URL}); license CC BY 4.0 (UCI Machine Learning "
                "Repository). 13 clinical features; target binarized (>0 -> 1) "
                'to presence of heart disease; "?" treated as missing.'
            )
            return LoadedDataset(
                payload=payload, source="real", provenance=provenance, key="heart_uci"
            )
        except Exception:
            return _heart_fallback(seed)

    return _heart_fallback(seed)


def _breast_fallback(seed: int | None) -> LoadedDataset:
    """Deterministic two-Gaussian fallback table (used only if sklearn absent)."""
    import pandas as pd

    rng = np.random.default_rng(seed)
    n_benign, n_malignant, d = 357, 212, 30
    names = [f"feature_{i}" for i in range(d)]
    x_benign = rng.normal(0.0, 1.0, size=(n_benign, d))
    x_malignant = rng.normal(1.5, 1.0, size=(n_malignant, d))
    x_arr = np.vstack([x_benign, x_malignant])
    y_arr = np.concatenate([np.ones(n_benign, dtype="int64"), np.zeros(n_malignant, dtype="int64")])
    perm = rng.permutation(n_benign + n_malignant)
    x_arr, y_arr = x_arr[perm], y_arr[perm]
    payload = {
        "X": pd.DataFrame(x_arr, columns=names),
        "y": pd.Series(y_arr, name="target"),
        "feature_names": names,
    }
    provenance = (
        "synthetic/bundled fallback: two-Gaussian-blob table (569 samples x 30 "
        "features, binary target), seeded with "
        f"seed={seed!r}; scikit-learn was unavailable, so this is NOT the real "
        "WDBC data."
    )
    return LoadedDataset(
        payload=payload,
        source="fallback",
        provenance=provenance,
        key="breast_wisconsin",
    )


def load_breast_wisconsin(
    *,
    cache_dir,
    download: bool = True,
    prefer_real: bool = True,
    seed: int | None = None,
    **opts,
) -> LoadedDataset:
    """Load the Breast Cancer Wisconsin (Diagnostic) dataset.

    The real payload is the scikit-learn *bundled* WDBC dataset, which needs no
    network -- so it is returned as ``source="real"`` whenever scikit-learn is
    importable and ``prefer_real`` is set (regardless of ``download``). If
    scikit-learn is missing, or ``prefer_real`` is False, a deterministic numpy
    fallback is returned.

    Parameters
    ----------
    cache_dir:
        Unused (the dataset is bundled), accepted for contract uniformity.
    download, prefer_real:
        ``prefer_real`` selects the bundled real dataset; ``download`` is
        irrelevant since no network is used.
    seed:
        Seed for the deterministic fallback table.
    **opts:
        Ignored extra loader options.

    Returns
    -------
    LoadedDataset
        ``payload`` is a dict with ``X`` (:class:`pandas.DataFrame`), ``y``
        (:class:`pandas.Series`, 1 = benign, 0 = malignant per scikit-learn),
        and ``feature_names``. ``source`` is ``"real"`` or ``"fallback"``.
    """
    if prefer_real:
        try:
            from sklearn.datasets import load_breast_cancer

            data = load_breast_cancer(as_frame=True)
            payload = {
                "X": data.data,
                "y": data.target.rename("target"),
                "feature_names": list(data.feature_names),
            }
            provenance = (
                "real: scikit-learn bundled Wisconsin Diagnostic Breast Cancer "
                "(WDBC) dataset (569 samples x 30 features, binary "
                "benign/malignant); Wolberg, Street & Mangasarian (1995), UCI "
                "Machine Learning Repository. Open data, no network required."
            )
            return LoadedDataset(
                payload=payload,
                source="real",
                provenance=provenance,
                key="breast_wisconsin",
            )
        except Exception:
            return _breast_fallback(seed)

    return _breast_fallback(seed)
