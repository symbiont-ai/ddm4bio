"""Neurophysiology (EEG/MEG) datasets.

Dataset (real): a lightweight open EEG recording -- a single PhysioNet EEG
Motor Movement/Imagery (``eegmmidb``) record fetched via ``wfdb``, or, if
``wfdb`` is unavailable, an ``mne`` bundled EEG dataset (both imported inside
the loader body). The multi-GB MNE ``sample`` dataset is deliberately NOT
downloaded by default.
Tier: open.
License: Open Data Commons Attribution License v1.0 (PhysioNet eegmmidb);
BSD-3-Clause for MNE-bundled data.
Install: ``pip install wfdb`` and/or ``pip install mne``.

When no network / neither optional dependency is present, the loader returns a
deterministic synthetic multichannel EEG-like signal so the course never
hard-fails.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ddm4bio.datasets.registry import LoadedDataset

_KEY = "mne_eeg"
_PN_DIR = "eegmmidb"
_EEGMMIDB_URL = "https://physionet.org/content/eegmmidb/1.0.0/"
_PN_LICENSE = "Open Data Commons Attribution License v1.0 (PhysioNet)"

# Canonical EEG band centers (Hz) used to build the synthetic oscillations.
_EEG_BANDS = {
    "delta": 2.0,
    "theta": 6.0,
    "alpha": 10.0,
    "beta": 20.0,
    "gamma": 40.0,
}


def _synthetic_eeg(
    *,
    seed: int | None,
    n_channels: int = 8,
    fs: float = 160.0,
    duration_s: float = 10.0,
) -> LoadedDataset:
    """Build a deterministic multichannel EEG-like fallback signal.

    Each channel is an independent mixture of band-limited oscillations (delta
    through gamma) with random per-channel amplitudes and phases, plus additive
    Gaussian noise. Sampling frequency defaults to 160 Hz (the eegmmidb rate).
    Signal shape is ``(n_channels, n_samples)``.
    """
    rng = np.random.default_rng(seed)
    n = int(round(fs * duration_s))
    t = np.arange(n) / fs

    signal = np.zeros((n_channels, n), dtype=float)
    band_freqs = list(_EEG_BANDS.values())
    for ch in range(n_channels):
        for f0 in band_freqs:
            amp = rng.uniform(0.3, 1.0)
            phase = rng.uniform(0.0, 2.0 * np.pi)
            signal[ch] += amp * np.sin(2.0 * np.pi * f0 * t + phase)
        signal[ch] += rng.normal(0.0, 0.3, size=n)

    ch_names = [f"EEG{i + 1:02d}" for i in range(n_channels)]
    payload = {
        "signal": signal,
        "fs": float(fs),
        "ch_names": ch_names,
    }
    return LoadedDataset(
        payload=payload,
        source="fallback",
        provenance=(
            "synthetic/bundled fallback: real open-EEG fetch unavailable "
            "(no network or wfdb/mne missing). Deterministic EEG-like signal: "
            "per-channel mixture of band-limited oscillations "
            "(delta/theta/alpha/beta/gamma) + noise, "
            f"{n_channels} channels, fs={fs} Hz, seed={seed}."
        ),
        key=_KEY,
    )


def _load_eegmmidb(cache_dir: Path, record: str, run: str) -> LoadedDataset:
    """Fetch a single PhysioNet eegmmidb EEG record via wfdb, caching it."""
    import wfdb

    rec_dir = cache_dir / "eegmmidb" / record
    rec_dir.mkdir(parents=True, exist_ok=True)
    base = f"{record}{run}"  # e.g. "S001R01"
    header_path = rec_dir / f"{base}.hea"

    if not header_path.exists():
        wfdb.dl_database(
            _PN_DIR,
            str(rec_dir),
            records=[f"{record}/{base}"],
        )

    rec = wfdb.rdrecord(str(rec_dir / base))
    # wfdb returns (n_samples, n_channels); EEG convention is channels x samples.
    signal = np.asarray(rec.p_signal, dtype=float).T
    fs = float(rec.fs)
    ch_names = (
        list(rec.sig_name)
        if rec.sig_name is not None
        else [f"ch{i}" for i in range(signal.shape[0])]
    )
    payload = {
        "signal": signal,
        "fs": fs,
        "ch_names": ch_names,
    }
    return LoadedDataset(
        payload=payload,
        source="real",
        provenance=(
            f"{_EEGMMIDB_URL} record {base!r} (pn_dir={_PN_DIR!r}); "
            f"license: {_PN_LICENSE}. PhysioNet EEG Motor Movement/Imagery "
            f"64-channel EEG at fs={fs} Hz; cached under {rec_dir}."
        ),
        key=_KEY,
    )


def load_eeg(
    *,
    cache_dir,
    download: bool = True,
    prefer_real: bool = True,
    seed: int | None = None,
    record: str = "S001",
    run: str = "R01",
    **opts,
) -> LoadedDataset:
    """Load a lightweight open EEG recording (channels x samples).

    Parameters
    ----------
    cache_dir:
        Directory for cached raw downloads.
    download:
        When True, attempt to fetch a real open EEG record.
    prefer_real:
        When True, prefer the real recording over the synthetic fallback.
    seed:
        RNG seed for the deterministic fallback signal.
    record, run:
        PhysioNet eegmmidb subject/run ids (e.g. ``"S001"``/``"R01"``).
    **opts:
        Ignored extra options (accepted for a uniform loader contract).

    Returns
    -------
    LoadedDataset
        ``payload`` is a dict with ``signal`` (``n_channels x n_samples``),
        ``fs``, and ``ch_names``. ``source`` is ``"real"`` when fetched,
        otherwise ``"fallback"``.

    Notes
    -----
    The real fetch first tries the small PhysioNet eegmmidb record via
    ``wfdb``; if that is unavailable it tries an ``mne`` bundled EEG dataset.
    The multi-GB MNE ``sample`` dataset is never downloaded here.
    """
    cache_dir = Path(cache_dir)

    if not (download and prefer_real):
        return _synthetic_eeg(seed=seed)

    errors: list[str] = []

    # Attempt 1: lightweight PhysioNet eegmmidb record via wfdb.
    try:
        return _load_eegmmidb(cache_dir, record=record, run=run)
    except Exception as exc:  # noqa: BLE001 -- fall through to next attempt
        errors.append(f"wfdb/eegmmidb: {type(exc).__name__}: {exc}")

    # Attempt 2: an mne bundled/testing EEG dataset (no multi-GB sample data).
    try:
        import mne
        from mne.datasets import ssvep

        # ssvep is a small BIDS EEG dataset; download honours MNE's cache.
        data_path = Path(ssvep.data_path(verbose=False))
        vhdr = sorted(data_path.rglob("*.vhdr"))
        if not vhdr:
            raise FileNotFoundError("no BrainVision .vhdr in mne ssvep dataset")
        raw = mne.io.read_raw_brainvision(str(vhdr[0]), preload=True, verbose=False)
        raw.pick("eeg")
        signal = np.asarray(raw.get_data(), dtype=float)  # channels x samples
        fs = float(raw.info["sfreq"])
        ch_names = list(raw.ch_names)
        payload = {"signal": signal, "fs": fs, "ch_names": ch_names}
        return LoadedDataset(
            payload=payload,
            source="real",
            provenance=(
                "mne.datasets.ssvep (BSD-3-Clause MNE-Python data); "
                f"BrainVision EEG at fs={fs} Hz, {signal.shape[0]} channels; "
                f"cached by MNE under {data_path}."
            ),
            key=_KEY,
        )
    except Exception as exc:  # noqa: BLE001 -- any failure -> deterministic fallback
        errors.append(f"mne: {type(exc).__name__}: {exc}")

    fallback = _synthetic_eeg(seed=seed)
    fallback.provenance += " (real fetch failed: " + " | ".join(errors) + ")"
    return fallback
