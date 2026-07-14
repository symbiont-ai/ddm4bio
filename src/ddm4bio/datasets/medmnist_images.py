"""MedMNIST image datasets.

Dataset: MedMNIST v2 (e.g. BloodMNIST, PathMNIST, DermaMNIST).
Tier: open.
License: CC BY 4.0.
Install: ``pip install medmnist``.
"""

from __future__ import annotations

import numpy as np


def load_medmnist(name: str = "bloodmnist") -> tuple[np.ndarray, np.ndarray]:
    """Load a MedMNIST 2D image collection.

    Parameters
    ----------
    name:
        MedMNIST subset name, e.g. ``"bloodmnist"``, ``"pathmnist"``.

    Returns
    -------
    tuple of (images, labels)
        ``images`` has shape ``(n_samples, H, W, C)`` (uint8, C in {1, 3}),
        ``labels`` has shape ``(n_samples,)`` (int class indices).

    Notes
    -----
    Heavy dependency ``medmnist`` is imported inside this function body.
    """
    raise NotImplementedError(
        "load_medmnist is a Phase 0 stub; expected return is "
        "(images[n,H,W,C] uint8, labels[n] int)."
    )
