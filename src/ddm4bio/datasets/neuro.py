"""Neurophysiology (EEG/MEG) datasets.

Dataset: MNE sample dataset (combined MEG/EEG auditory/visual evoked task).
Tier: open.
License: BSD-3-Clause (MNE-Python sample data).
Install: ``pip install mne``.
"""

from __future__ import annotations

from typing import Any


def load_mne_sample() -> Any:
    """Load the MNE sample EEG/MEG recording.

    Returns
    -------
    mne.io.Raw
        Raw MEG/EEG object with associated event annotations; channels x
        time samples at the recording sampling frequency.

    Notes
    -----
    Uses ``mne.datasets.sample`` (imported inside the body).
    """
    raise NotImplementedError(
        "load_mne_sample is a Phase 0 stub; expected return is an "
        "mne.io.Raw object (channels x samples)."
    )
