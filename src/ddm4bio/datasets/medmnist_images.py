"""MedMNIST image datasets.

Dataset: MedMNIST v2 (e.g. BloodMNIST, PathMNIST, DermaMNIST).
Tier: open.
License: CC BY 4.0.
Real source: the per-collection ``.npz`` files published on Zenodo (record
10519652). We fetch the ``.npz`` DIRECTLY via :mod:`urllib` and read it with
:func:`numpy.load`, so this module does NOT depend on the ``medmnist`` pip
package (which would pull in ``torch``).
Fallback: ``sklearn.datasets.load_digits`` reshaped into a small labelled
image stack, so the module is usable with zero network access.
"""

from __future__ import annotations

import urllib.request

import numpy as np

from ddm4bio.datasets.registry import LoadedDataset

#: Zenodo record hosting the MedMNIST v2 ``.npz`` collections.
_ZENODO_RECORD = "10519652"
_ZENODO_BASE = f"https://zenodo.org/records/{_ZENODO_RECORD}/files"

#: Keys expected inside every MedMNIST 2D ``.npz`` archive.
_NPZ_KEYS = (
    "train_images",
    "train_labels",
    "val_images",
    "val_labels",
    "test_images",
    "test_labels",
)


def _download(url: str, dest, *, timeout: float = 60.0) -> None:
    """Fetch ``url`` to ``dest`` atomically (write to a temp then rename)."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "ddm4bio/0.1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = response.read()
    tmp.write_bytes(data)
    tmp.replace(dest)


def _fallback(*, name: str, seed: int | None, reason: str) -> LoadedDataset:
    """Build a deterministic bundled fallback from ``load_digits``."""
    from sklearn.datasets import load_digits

    digits = load_digits()
    # (n, 8, 8) uint8 grayscale images scaled from the 0..16 digit range.
    images = (digits.images / 16.0 * 255.0).astype(np.uint8)
    images = images[..., np.newaxis]  # (n, 8, 8, 1), C-last like MedMNIST.
    labels = digits.target.astype(np.int64)

    rng = np.random.default_rng(seed)
    order = rng.permutation(images.shape[0])
    images = images[order]
    labels = labels[order]

    n = images.shape[0]
    n_train = int(n * 0.7)
    n_val = int(n * 0.85)
    payload = {
        "train_images": images[:n_train],
        "train_labels": labels[:n_train, np.newaxis],
        "val_images": images[n_train:n_val],
        "val_labels": labels[n_train:n_val, np.newaxis],
        "test_images": images[n_val:],
        "test_labels": labels[n_val:, np.newaxis],
        "images": images,
        "labels": labels,
        "requested_name": name,
    }
    provenance = (
        "synthetic/bundled fallback: sklearn.datasets.load_digits reshaped to "
        f"an 8x8x1 uint8 image stack with a deterministic 70/15/15 split "
        f"(seed={seed}); stands in for MedMNIST {name!r}. Reason: {reason}"
    )
    return LoadedDataset(payload=payload, source="fallback", provenance=provenance, key=name)


def load_medmnist(
    *,
    cache_dir,
    download: bool = True,
    prefer_real: bool = True,
    seed: int | None = None,
    key: str = "bloodmnist",
    name: str | None = None,
    **opts,
) -> LoadedDataset:
    """Load a MedMNIST v2 2D image collection (real Zenodo ``.npz`` or fallback).

    Parameters
    ----------
    cache_dir:
        Directory under which the raw ``{name}.npz`` download is cached. The
        fetch is idempotent -- an existing cache file is reused, never
        re-downloaded.
    download:
        When True (and ``prefer_real``), attempt the real Zenodo download.
    prefer_real:
        When True (and ``download``), prefer the real dataset over the fallback.
    seed:
        Seed for the deterministic fallback split.
    key:
        Registry key forwarded by the dispatcher; it IS the MedMNIST subset to
        fetch (e.g. ``"bloodmnist"``, ``"pathmnist"``). Drives the download URL,
        the cache filename, and the stamped provenance/key.
    name:
        Optional explicit MedMNIST subset override for direct callers; when
        given it takes precedence over ``key``. Defaults to ``None`` so the
        registry key wins.
    **opts:
        Ignored; accepted for loader-contract compatibility.

    Returns
    -------
    LoadedDataset
        ``payload`` is a dict with ``train/val/test`` images and labels (real
        MedMNIST layout) or the bundled fallback stack. ``source`` is
        ``"real"`` or ``"fallback"``; provenance records the origin.

    Notes
    -----
    Heavy dependencies (``sklearn`` for the fallback) are imported inside the
    function body so importing this module needs only numpy.
    """
    from pathlib import Path

    cache_dir = Path(cache_dir)

    # The MedMNIST subset to fetch is the registry key (each MedMNIST key equals
    # its subset name); an explicit ``name`` overrides it for direct callers.
    name = name or key

    if not (download and prefer_real):
        return _fallback(
            name=name,
            seed=seed,
            reason="download disabled or prefer_real=False",
        )

    cache_path = cache_dir / f"{name}.npz"
    url = f"{_ZENODO_BASE}/{name}.npz?download=1"
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        if not cache_path.exists():
            _download(url, cache_path)
        with np.load(cache_path, allow_pickle=False) as archive:
            keys = set(archive.files)
            if not set(_NPZ_KEYS).issubset(keys):
                raise ValueError(
                    f"{cache_path.name} is missing MedMNIST keys; found {sorted(keys)}"
                )
            payload = {k: archive[k] for k in archive.files}
        provenance = (
            f"real MedMNIST v2 {name!r} from {url} (Zenodo record "
            f"{_ZENODO_RECORD}); CC BY 4.0; per-split 2D image arrays "
            "(N,H,W,C) uint8 with integer class labels."
        )
        return LoadedDataset(payload=payload, source="real", provenance=provenance, key=name)
    except Exception as exc:  # noqa: BLE001 - fall back on any real-fetch failure
        return _fallback(
            name=name,
            seed=seed,
            reason=f"real fetch failed ({type(exc).__name__}: {exc})",
        )
