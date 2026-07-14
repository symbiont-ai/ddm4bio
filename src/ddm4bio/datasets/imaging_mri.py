"""MRI imaging datasets.

Datasets:
- IXI: open brain MRI collection. Tier: open. License: CC BY-SA 3.0.
  Real path: attempt to fetch one small, openly-licensed NIfTI volume via
  :mod:`urllib` and read it with ``nibabel`` (imported inside the body).
  Fallback: a synthetic MRI-like Shepp-Logan phantom -- via
  ``skimage.data.shepp_logan_phantom`` when scikit-image is importable, else a
  numpy analytic elliptic phantom.
- fastMRI: knee/brain k-space. Tier: credentialed (requires an approved NYU
  Data Sharing Agreement). NEVER auto-downloaded; ``load_ixi`` is the open
  fallback stand-in.
Install: ``pip install nibabel`` (real IXI); ``scikit-image`` optional.
"""

from __future__ import annotations

import urllib.request

import numpy as np

from ddm4bio.datasets.registry import LoadedDataset

IXI_KEY = "ixi_mri"
FASTMRI_KEY = "fastmri"

#: Small, openly-licensed example NIfTI volume (nibabel test data, PDDL/MIT).
#: Overridable through the ``url`` loader option.
_DEFAULT_IXI_URL = (
    "https://raw.githubusercontent.com/nipy/nibabel/master/nibabel/tests/data/example4d.nii.gz"
)


def _download(url: str, dest, *, timeout: float = 60.0) -> None:
    """Fetch ``url`` to ``dest`` atomically (write to a temp then rename)."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "ddm4bio/0.1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = response.read()
    tmp.write_bytes(data)
    tmp.replace(dest)


def _numpy_phantom(size: int = 256) -> np.ndarray:
    """Analytic Shepp-Logan phantom on a ``size x size`` grid (numpy only).

    Deterministic: the phantom is a fixed sum of ellipses, so the output
    depends only on ``size`` (no RNG involved).
    """
    # (intensity, major axis, minor axis, x0, y0, rotation-degrees).
    ellipses = [
        (1.0, 0.69, 0.92, 0.0, 0.0, 0.0),
        (-0.8, 0.6624, 0.874, 0.0, -0.0184, 0.0),
        (-0.2, 0.11, 0.31, 0.22, 0.0, -18.0),
        (-0.2, 0.16, 0.41, -0.22, 0.0, 18.0),
        (0.1, 0.21, 0.25, 0.0, 0.35, 0.0),
        (0.1, 0.046, 0.046, 0.0, 0.1, 0.0),
        (0.1, 0.046, 0.046, 0.0, -0.1, 0.0),
        (0.1, 0.046, 0.023, -0.08, -0.605, 0.0),
        (0.1, 0.023, 0.023, 0.0, -0.606, 0.0),
        (0.1, 0.023, 0.046, 0.06, -0.605, 0.0),
    ]
    grid = np.linspace(-1.0, 1.0, size)
    x, y = np.meshgrid(grid, -grid)
    image = np.zeros((size, size), dtype=float)
    for intensity, a, b, x0, y0, phi in ellipses:
        rad = np.deg2rad(phi)
        cos, sin = np.cos(rad), np.sin(rad)
        xc, yc = x - x0, y - y0
        xr = xc * cos + yc * sin
        yr = -xc * sin + yc * cos
        inside = (xr / a) ** 2 + (yr / b) ** 2 <= 1.0
        image[inside] += intensity
    return image


def _ixi_fallback(*, seed: int | None, reason: str) -> LoadedDataset:
    """Build a deterministic MRI-like phantom fallback (2-D array)."""
    try:
        from skimage.data import shepp_logan_phantom

        phantom = np.asarray(shepp_logan_phantom(), dtype=float)
        origin = "skimage.data.shepp_logan_phantom"
    except Exception:  # noqa: BLE001 - scikit-image missing or import error
        phantom = _numpy_phantom()
        origin = "numpy analytic Shepp-Logan phantom"

    provenance = (
        f"synthetic/bundled fallback: {origin} as a 2-D MRI-like slice "
        f"(shape={phantom.shape}, deterministic, seed={seed}). Reason: {reason}"
    )
    return LoadedDataset(payload=phantom, source="fallback", provenance=provenance, key=IXI_KEY)


def load_ixi(
    *,
    cache_dir,
    download: bool = True,
    prefer_real: bool = True,
    seed: int | None = None,
    **opts,
) -> LoadedDataset:
    """Load an open MRI volume (real NIfTI via urllib+nibabel, or phantom).

    Parameters
    ----------
    cache_dir:
        Directory under which the raw ``.nii.gz`` download is cached. The fetch
        is idempotent -- an existing cache file is reused, never re-downloaded.
    download:
        When True (and ``prefer_real``), attempt the real NIfTI download.
    prefer_real:
        When True (and ``download``), prefer the real volume over the fallback.
    seed:
        Accepted for contract compatibility; the fallback phantom is fully
        deterministic.
    **opts:
        ``url`` overrides the source NIfTI URL.

    Returns
    -------
    LoadedDataset
        ``payload`` is a 2-D/3-D/4-D :class:`numpy.ndarray` (real volume) or a
        2-D phantom (fallback). ``source`` is ``"real"`` or ``"fallback"``.

    Notes
    -----
    ``nibabel`` (real read) and ``skimage`` (fallback) are imported inside the
    function body so importing this module needs only numpy.
    """
    from pathlib import Path

    cache_dir = Path(cache_dir)
    url = opts.get("url", _DEFAULT_IXI_URL)

    if not (download and prefer_real):
        return _ixi_fallback(seed=seed, reason="download disabled or prefer_real=False")

    cache_path = cache_dir / "ixi_sample.nii.gz"
    try:
        import nibabel as nib

        cache_dir.mkdir(parents=True, exist_ok=True)
        if not cache_path.exists():
            _download(url, cache_path)
        volume = np.asarray(nib.load(str(cache_path)).get_fdata(), dtype=float)
        provenance = (
            f"real MRI NIfTI volume from {url} (open small NIfTI, read with "
            f"nibabel); shape={volume.shape}. Open stand-in for the IXI "
            "collection (brain-development.org/ixi-dataset, CC BY-SA 3.0)."
        )
        return LoadedDataset(payload=volume, source="real", provenance=provenance, key=IXI_KEY)
    except Exception as exc:  # noqa: BLE001 - fall back on any real-fetch/read failure
        return _ixi_fallback(
            seed=seed,
            reason=f"real fetch/read failed ({type(exc).__name__}: {exc})",
        )


def load_fastmri(
    *,
    cache_dir,
    download: bool = True,
    prefer_real: bool = True,
    seed: int | None = None,
    **opts,
) -> LoadedDataset:
    """Load fastMRI (credentialed) -- raise for access, or return IXI fallback.

    fastMRI is a credentialed NYU Langone dataset governed by a Data Sharing
    Agreement (non-commercial research/education only, no redistribution), so it
    is never auto-downloaded.

    Parameters
    ----------
    cache_dir, download, prefer_real, seed, **opts:
        Standard loader-contract arguments; forwarded to :func:`load_ixi` for
        the open fallback.

    Returns
    -------
    LoadedDataset
        The open IXI fallback payload (keyed as ``"fastmri"``) when ``download``
        is False or ``prefer_real`` is False.

    Raises
    ------
    PermissionError
        When ``download`` and ``prefer_real`` -- fastMRI requires an approved
        application; instructions are in the message.
    """
    if download and prefer_real:
        raise PermissionError(
            "fastMRI is a credentialed dataset (NYU Langone / Meta AI) and is "
            "never auto-downloaded. Apply for access by signing the fastMRI "
            "Data Sharing Agreement at https://fastmri.med.nyu.edu/ (education "
            "and research use only; no redistribution), then stage the data "
            "locally. For an open alternative, ddm4bio uses load_ixi as the "
            "fallback (call get_dataset('fastmri', prefer_real=False) or "
            "download=False to receive that open phantom/volume payload)."
        )

    fallback = load_ixi(
        cache_dir=cache_dir,
        download=download,
        prefer_real=prefer_real,
        seed=seed,
        **opts,
    )
    fallback.key = FASTMRI_KEY
    fallback.provenance = (
        "credentialed fastMRI unavailable without a Data Sharing Agreement; "
        "returning the open load_ixi fallback -> " + fallback.provenance
    )
    return fallback
