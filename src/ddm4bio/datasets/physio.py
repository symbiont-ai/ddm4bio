"""Physiological signal datasets.

Dataset: MIT-BIH Arrhythmia Database (ECG).
Tier: open.
License: Open Data Commons / PhysioNet (freely available).
Install: ``pip install wfdb``.
"""

from __future__ import annotations

from typing import Any


def load_mitbih(record: str | None = None) -> Any:
    """Load an ECG record from the MIT-BIH Arrhythmia Database.

    Parameters
    ----------
    record:
        PhysioNet record id (e.g. ``"100"``). If ``None`` a default record
        is selected.

    Returns
    -------
    object
        A container exposing the signal array of shape ``(n_samples, n_leads)``,
        the sampling frequency, and beat annotations.

    Notes
    -----
    Uses ``wfdb`` (imported inside the body).
    """
    raise NotImplementedError(
        "load_mitbih is a Phase 0 stub; expected return exposes "
        "signal[n,leads], fs, and annotations."
    )
