"""Physiological signal datasets.

Dataset: MIT-BIH Arrhythmia Database (ECG).
Tier: open.
License: Open Data Commons Attribution License v1.0 (PhysioNet, freely
available).
Install: ``pip install wfdb``.

The real fetch pulls a single PhysioNet MIT-BIH record via ``wfdb`` (imported
inside the loader body) and caches the raw ``.dat``/``.hea`` (and ``.atr`` when
available) files under ``cache_dir``. When the network or ``wfdb`` is
unavailable the loader returns a deterministic synthetic ECG-like signal so
that importing and running the course never hard-fails.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ddm4bio.datasets.registry import LoadedDataset

_KEY = "mitbih"
_PN_DIR = "mitdb"
_MITBIH_URL = "https://physionet.org/content/mitdb/1.0.0/"
_LICENSE = "Open Data Commons Attribution License v1.0 (PhysioNet)"

#: Course-controlled durability mirror (a GitHub Release on this repo) holding the SAME real
#: record files, redistributed under the PhysioNet ODC-BY license. Tried only when PhysioNet
#: itself is unreachable, so an outage does not freeze the build. Files are named
#: ``mitbih_{record}{ext}``; an un-mirrored record 404s and drops to the synthetic fallback.
_MIRROR_BASE = "https://github.com/symbiont-ai/ddm4bio/releases/download/data-mirror-v1"


def _download_record_from_mirror(rec_dir: Path, record: str) -> None:
    """Fetch a MIT-BIH record's files from the course mirror into ``rec_dir``.

    Serves the SAME real PhysioNet record (redistributed under ODC-BY) when PhysioNet is
    unreachable. ``.hea`` and ``.dat`` are required to read the signal; ``.atr`` (beat
    annotations) is best-effort. Raises if a required file is missing so the caller can
    drop to the synthetic fallback.
    """
    import urllib.error
    import urllib.request

    got: list[str] = []
    for ext, required in ((".hea", True), (".dat", True), (".atr", False)):
        url = f"{_MIRROR_BASE}/mitbih_{record}{ext}"
        dest = rec_dir / f"{record}{ext}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ddm4bio/0.1"})
            with urllib.request.urlopen(req, timeout=60.0) as resp:
                data = resp.read()
        except urllib.error.HTTPError as exc:
            if not required and exc.code == 404:
                continue
            raise
        tmp = dest.with_suffix(dest.suffix + ".part")
        tmp.write_bytes(data)
        tmp.replace(dest)
        got.append(ext)
    if ".hea" not in got or ".dat" not in got:
        raise RuntimeError(f"course mirror is missing required files for record {record!r}")


def _synthetic_ecg(
    *, seed: int | None, fs: float = 360.0, duration_s: float = 10.0
) -> LoadedDataset:
    """Build a deterministic ECG-like fallback signal.

    The signal is a periodic train of QRS-like Gaussian complexes riding on a
    slow sinusoidal baseline wander, with additive Gaussian measurement noise.
    Sampling frequency defaults to the MIT-BIH rate of 360 Hz.
    """
    rng = np.random.default_rng(seed)
    n = int(round(fs * duration_s))
    t = np.arange(n) / fs

    # Heart rate ~72 bpm -> one QRS complex every ~0.833 s.
    hr_bpm = 72.0
    rr = 60.0 / hr_bpm
    beat_times = np.arange(rr / 2.0, duration_s, rr)

    # QRS as a narrow Gaussian; add small Q and S deflections for shape.
    qrs = np.zeros(n, dtype=float)
    width = 0.010  # ~10 ms core width
    for bt in beat_times:
        qrs += 1.0 * np.exp(-0.5 * ((t - bt) / width) ** 2)
        qrs -= 0.15 * np.exp(-0.5 * ((t - (bt - 0.025)) / (width * 1.2)) ** 2)
        qrs -= 0.20 * np.exp(-0.5 * ((t - (bt + 0.030)) / (width * 1.5)) ** 2)
        # Broad, low T-wave after each beat.
        qrs += 0.25 * np.exp(-0.5 * ((t - (bt + 0.20)) / 0.040) ** 2)

    baseline = 0.10 * np.sin(2.0 * np.pi * 0.25 * t)  # slow wander (~0.25 Hz)
    noise = rng.normal(0.0, 0.02, size=n)
    signal = (qrs + baseline + noise).reshape(n, 1)

    payload = {
        "signal": signal.astype(float),
        "fs": float(fs),
        "sig_names": ["ECG_synth"],
    }
    return LoadedDataset(
        payload=payload,
        source="fallback",
        provenance=(
            "synthetic/bundled fallback: real MIT-BIH fetch unavailable "
            "(no network or wfdb missing). Deterministic ECG-like signal: "
            "periodic QRS-like Gaussian complexes + baseline wander + noise, "
            f"fs={fs} Hz, seed={seed}."
        ),
        key=_KEY,
    )


def load_mitbih(
    *,
    cache_dir,
    download: bool = True,
    prefer_real: bool = True,
    seed: int | None = None,
    record: str = "100",
    **opts,
) -> LoadedDataset:
    """Load an ECG record from the MIT-BIH Arrhythmia Database.

    Parameters
    ----------
    cache_dir:
        Directory for cached raw downloads.
    download:
        When True, attempt to fetch the real PhysioNet record.
    prefer_real:
        When True, prefer the real record over the synthetic fallback.
    seed:
        RNG seed for the deterministic fallback signal.
    record:
        PhysioNet MIT-BIH record id, e.g. ``"100"`` (default).
    **opts:
        Ignored extra options (accepted for a uniform loader contract).

    Returns
    -------
    LoadedDataset
        ``payload`` is a dict with ``signal`` (``n_samples x n_channels``),
        ``fs`` (sampling frequency), and ``sig_names``. ``source`` is
        ``"real"`` when fetched from PhysioNet, otherwise ``"fallback"``.
    """
    cache_dir = Path(cache_dir)

    if not (download and prefer_real):
        return _synthetic_ecg(seed=seed)

    try:
        import wfdb

        rec_dir = cache_dir / "mitbih"
        rec_dir.mkdir(parents=True, exist_ok=True)
        header_path = rec_dir / f"{record}.hea"

        origin = "cache"
        if not header_path.exists():
            try:
                # Real fetch: pull the record (.hea/.dat + annotations) from PhysioNet.
                wfdb.dl_database(
                    _PN_DIR,
                    str(rec_dir),
                    records=[record],
                    annotators="all",
                )
                origin = "upstream"
            except Exception:
                # PhysioNet unreachable: pull the SAME real record from the course
                # mirror (ODC-BY). Still real data, so the build stays real. If the
                # mirror is also gone this raises and drops to the synthetic fallback.
                _download_record_from_mirror(rec_dir, record)
                origin = "mirror"

        rec = wfdb.rdrecord(str(rec_dir / record))

        signal = np.asarray(rec.p_signal, dtype=float)
        if signal.ndim == 1:
            signal = signal.reshape(-1, 1)
        fs = float(rec.fs)
        sig_names = (
            list(rec.sig_name)
            if rec.sig_name is not None
            else [f"ch{i}" for i in range(signal.shape[1])]
        )

        payload = {
            "signal": signal,
            "fs": fs,
            "sig_names": sig_names,
        }
        origin_desc = {
            "cache": f"cached record {record!r}",
            "upstream": f"{_MITBIH_URL} record {record!r} (pn_dir={_PN_DIR!r})",
            "mirror": f"course mirror of {_MITBIH_URL} record {record!r} ({_LICENSE})",
        }
        return LoadedDataset(
            payload=payload,
            source="real",
            provenance=(
                f"real MIT-BIH Arrhythmia Database ECG from {origin_desc[origin]}; "
                f"license: {_LICENSE}; single-/2-lead at fs=360 Hz; cached under {rec_dir}."
            ),
            key=_KEY,
        )
    except Exception as exc:  # noqa: BLE001 -- any failure -> deterministic fallback
        fallback = _synthetic_ecg(seed=seed)
        fallback.provenance += f" (real fetch failed: {type(exc).__name__}: {exc})"
        return fallback
