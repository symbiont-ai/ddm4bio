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

#: Course-controlled durability mirror (a GitHub Release on this repo) holding the SAME
#: real ``.npz`` files, redistributed under CC BY 4.0. Tried only when the upstream Zenodo
#: fetch fails, so a Zenodo outage or rate-limit does not freeze the build. Not every subset
#: is mirrored -- an un-mirrored name simply 404s here and drops to the synthetic fallback.
_MIRROR_BASE = "https://github.com/symbiont-ai/ddm4bio/releases/download/data-mirror-v1"

#: Keys expected inside every MedMNIST 2D ``.npz`` archive.
_NPZ_KEYS = (
    "train_images",
    "train_labels",
    "val_images",
    "val_labels",
    "test_images",
    "test_labels",
)

#: Class index -> cell type for BloodMNIST, from the authoritative MedMNIST v2 spec
#: (MedMNIST ``info.py``; Yang et al., Scientific Data 2023, repackaging Acevedo et al.,
#: 2020). The ``.npz`` carries only integer labels, so these names are recorded here and
#: attached to the real dataset as :attr:`LoadedDataset.labels`. Index 3's full name is
#: "immature granulocytes (myelocytes, metamyelocytes and promyelocytes)", abbreviated here.
BLOODMNIST_LABELS: tuple[str, ...] = (
    "basophil",
    "eosinophil",
    "erythroblast",
    "immature granulocyte",
    "lymphocyte",
    "monocyte",
    "neutrophil",
    "platelet",
)

#: Per-subset class names for the MedMNIST collections used in this course.
_MEDMNIST_LABELS: dict[str, tuple[str, ...]] = {"bloodmnist": BLOODMNIST_LABELS}


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
    upstream_url = f"{_ZENODO_BASE}/{name}.npz?download=1"
    mirror_url = f"{_MIRROR_BASE}/{name}.npz"
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        origin = "cache"
        if not cache_path.exists():
            try:
                _download(upstream_url, cache_path)
                origin = "upstream"
            except Exception:
                # Upstream (Zenodo) unreachable: pull the SAME real .npz from the
                # course mirror (CC BY 4.0). Still real data, so the build stays real
                # and publishable. If the mirror is also gone, this raises and the
                # outer handler drops to the labelled synthetic fallback.
                _download(mirror_url, cache_path)
                origin = "mirror"
        with np.load(cache_path, allow_pickle=False) as archive:
            keys = set(archive.files)
            if not set(_NPZ_KEYS).issubset(keys):
                raise ValueError(
                    f"{cache_path.name} is missing MedMNIST keys; found {sorted(keys)}"
                )
            payload = {k: archive[k] for k in archive.files}
        origin_desc = {
            "cache": f"cached {name}.npz",
            "upstream": f"{upstream_url} (Zenodo record {_ZENODO_RECORD})",
            "mirror": f"course mirror {mirror_url} (Zenodo record {_ZENODO_RECORD}, CC BY 4.0)",
        }
        provenance = (
            f"real MedMNIST v2 {name!r} from {origin_desc[origin]}; CC BY 4.0; "
            "per-split 2D image arrays (N,H,W,C) uint8 with integer class labels."
        )
        return LoadedDataset(
            payload=payload,
            source="real",
            provenance=provenance,
            key=name,
            labels=_MEDMNIST_LABELS.get(name),
        )
    except Exception as exc:  # noqa: BLE001 - fall back on any real-fetch failure
        return _fallback(
            name=name,
            seed=seed,
            reason=f"real fetch failed ({type(exc).__name__}: {exc})",
        )
