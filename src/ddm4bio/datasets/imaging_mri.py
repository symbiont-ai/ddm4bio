"""MRI imaging datasets.

Datasets:
- IXI: open brain MRI collection. Tier: open. License: CC BY-SA 3.0.
- fastMRI: knee/brain k-space. Tier: credentialed (requires approved access).
  NEVER auto-download; users must apply at https://fastmri.med.nyu.edu/.
Install: ``pip install nibabel`` (IXI); fastMRI requires manual data staging.
"""

from __future__ import annotations

from typing import Any


def load_ixi_mri() -> Any:
    """Load an IXI brain MRI volume.

    Returns
    -------
    object
        A container exposing the image volume of shape ``(X, Y, Z)`` (or
        ``(X, Y, Z, T)``) and voxel spacing / affine metadata.

    Notes
    -----
    Uses ``nibabel`` (imported inside the body).
    """
    raise NotImplementedError(
        "load_ixi_mri is a Phase 0 stub; expected return exposes a 3D/4D "
        "volume and affine metadata."
    )


def load_fastmri() -> Any:
    """Load fastMRI k-space data (credentialed access only).

    Raises
    ------
    NotImplementedError
        Always. fastMRI is a credentialed dataset and is never auto-downloaded.
        Apply for access at https://fastmri.med.nyu.edu/ and stage the data
        locally, or use :func:`load_ixi_mri` as an open fallback.
    """
    raise NotImplementedError(
        "fastMRI is credentialed and is never auto-downloaded. Apply for "
        "access at https://fastmri.med.nyu.edu/ and stage data locally, or "
        "use load_ixi_mri() as an open fallback."
    )
