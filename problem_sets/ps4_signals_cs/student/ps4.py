"""PS4 student template: signal processing and compressed sensing (Kutz Ch. 14).

Fill in every function marked ``# TODO``. The data loaders, the QC calls, and
the ``main`` driver are already wired for you -- you only implement the method
logic. Public function names and signatures MUST stay exactly as given; the
autograder imports this module and checks them.

Running ``python ps4.py`` will load data and run QC without error, then stop at
the first unimplemented function with ``NotImplementedError``. Implement the
functions top to bottom and rerun.

Everything is offline and seeded. numpy is imported at the top; import ``pywt``
inside ``wavelet_decompose`` only (keep the ddm4bio wrappers doing the rest).
"""

from __future__ import annotations

import numpy as np

from ddm4bio.config import GLOBAL_SEED, seed_everything
from ddm4bio.datasets.synthetic import make_sparse_signal
from ddm4bio.interpret import interpretation_block
from ddm4bio.methods.signals import (
    compressed_sensing_recon,  # noqa: F401  (use inside cs_reconstruct)
    stft,  # noqa: F401  (use inside compute_spectrogram)
    wavelet_denoise,  # noqa: F401  (use inside denoise_and_score)
)
from ddm4bio.methods.validation import reconstruction_error  # noqa: F401  (use for errors)
from ddm4bio.qc.signals import qc_signals

WAVELET = "db4"
WAVELET_LEVEL = 4


# --------------------------------------------------------------------------- #
# Data loaders (offline synthetic fixtures) -- provided, do not change.
# --------------------------------------------------------------------------- #
def load_nonstationary_signal(
    fs: float = 500.0,
    duration: float = 6.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Build a deterministic nonstationary signal (provided).

    A linear chirp (10 -> 80 Hz) with two time-gated tone bursts (40 Hz in
    [1, 2] s and 120 Hz in [3.5, 4.0] s).

    Returns
    -------
    tuple
        ``(t, x, fs)``.
    """
    n = int(round(fs * duration))
    t = np.arange(n) / fs
    f0, f1 = 10.0, 80.0
    chirp = np.sin(2.0 * np.pi * (f0 * t + 0.5 * (f1 - f0) / duration * t**2))
    gate1 = ((t >= 1.0) & (t < 2.0)).astype(float)
    gate2 = ((t >= 3.5) & (t < 4.0)).astype(float)
    x = chirp + 0.8 * gate1 * np.sin(2.0 * np.pi * 40.0 * t)
    x = x + 0.6 * gate2 * np.sin(2.0 * np.pi * 120.0 * t)
    return t, x, fs


def load_ecg_segment(
    fs: float = 360.0,
    duration: float = 4.0,
    noise: float = 0.35,
    seed: int = GLOBAL_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Load a real MIT-BIH ECG window as a clean reference and add noise (provided).

    Uses ``get_dataset("mitbih", download=False)`` so the fixture is fully
    offline and deterministic (a labeled synthetic ECG-like fallback at
    fs = 360 Hz is returned when the real PhysioNet record / ``wfdb`` is
    unavailable, with the same payload shape). The first ``duration`` seconds of
    channel 0 are the clean reference; zero-mean Gaussian noise of known standard
    deviation is added to build the noisy copy.

    Returns
    -------
    tuple
        ``(t, clean, noisy, fs)``.
    """
    from ddm4bio.datasets import get_dataset

    ds = get_dataset("mitbih", download=False, seed=seed)
    fs = float(ds.payload["fs"])
    signal = np.asarray(ds.payload["signal"], dtype=float)
    n = int(round(fs * duration))
    clean = signal[:n, 0]
    t = np.arange(clean.size) / fs
    rng = np.random.default_rng(seed)
    noisy = clean + noise * rng.standard_normal(clean.size)
    return t, clean, noisy, fs


def load_sparse_field(
    n: int = 128,
    k: int = 12,
    seed: int = GLOBAL_SEED,
) -> np.ndarray:
    """Return a k-sparse 1-D field (isolated point sources) as ground truth (provided)."""
    return make_sparse_signal(n, k, seed=seed).signal


# --------------------------------------------------------------------------- #
# Part A -- Method
# --------------------------------------------------------------------------- #
def compute_fft(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """One-sided amplitude spectrum of a real signal.

    Returns
    -------
    tuple
        ``(freqs, magnitude)`` -- nonnegative frequency bins (Hz) and the
        single-sided amplitude spectrum.
    """
    # TODO: Use np.fft.rfft to transform x and np.fft.rfftfreq for the bins.
    # Return the magnitude scaled to a single-sided amplitude (factor 2 / n).
    raise NotImplementedError


def compute_spectrogram(
    x: np.ndarray,
    fs: float,
    nperseg: int = 128,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Short-time Fourier power spectrogram via the ddm4bio STFT wrapper.

    Returns
    -------
    tuple
        ``(freqs, times, power)`` with ``power = |Zxx|**2`` of shape
        ``(freqs.size, times.size)``.
    """
    # TODO: Call stft(x, fs=fs, nperseg=nperseg) and square the magnitude of
    # the complex STFT coefficients to get power. Return (freqs, times, power).
    raise NotImplementedError


def wavelet_decompose(
    x: np.ndarray,
    wavelet: str = WAVELET,
    level: int = WAVELET_LEVEL,
) -> list[np.ndarray]:
    """Multilevel discrete wavelet decomposition.

    Return the PyWavelets coefficient list ``[cA_level, cD_level, ..., cD_1]``
    using ``mode="periodization"`` so the coefficient count matches ``len(x)``.
    """
    import pywt  # noqa: F401  (use pywt.wavedec below)

    # TODO: Return pywt.wavedec(x, wavelet, level=level, mode="periodization").
    raise NotImplementedError


def fourier_sensing_matrix(n: int, kept_indices: np.ndarray) -> np.ndarray:
    """Real-valued k-space sensing matrix for the retained frequencies.

    Take the rows of the ``n x n`` DFT matrix at ``kept_indices`` and stack their
    real and imaginary parts vertically.

    Returns
    -------
    np.ndarray, shape ``(2 * len(kept_indices), n)``.
    """
    # TODO: Build the DFT matrix (e.g. np.fft.fft(np.eye(n), axis=0)),
    # select the kept rows, and vstack [rows.real, rows.imag].
    raise NotImplementedError


def cs_reconstruct(
    y: np.ndarray,
    phi: np.ndarray,
    alpha: float = 1e-3,
    seed: int = 0,
) -> np.ndarray:
    """L1 (Lasso) compressed-sensing reconstruction via the ddm4bio wrapper."""
    # TODO: Return compressed_sensing_recon(y, phi, alpha=alpha, seed=seed).
    raise NotImplementedError


# --------------------------------------------------------------------------- #
# Part B -- Application
# --------------------------------------------------------------------------- #
def snr_db(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Signal-to-noise ratio in dB of ``estimate`` against a clean ``reference``."""
    # TODO: Compute 10 * log10( sum(ref**2) / sum((ref - est)**2) ).
    # Guard against a zero noise power (return math.inf / float("inf")).
    raise NotImplementedError


def denoise_and_score(
    noisy: np.ndarray,
    clean: np.ndarray,
    wavelet: str = WAVELET,
) -> dict:
    """Wavelet-denoise ``noisy`` and score the SNR gain against ``clean``.

    Returns
    -------
    dict
        ``denoised`` plus ``snr_before``, ``snr_after``, ``snr_gain`` (dB).
    """
    # TODO: Call wavelet_denoise(noisy, wavelet=wavelet); score snr_db before
    # and after; return the dict with the denoised signal and the three SNRs.
    raise NotImplementedError


def mri_undersample_reconstruct(
    field: np.ndarray,
    ratio: float,
    seed: int = GLOBAL_SEED,
    alpha: float = 1e-3,
) -> dict:
    """Simulate accelerated MRI: undersample k-space and reconstruct two ways.

    Retain a random fraction ``ratio`` of Fourier coefficients of ``field``.
    Reconstruct with (1) L1 compressed sensing and (2) zero-filling the missing
    k-space then inverse-transforming.

    Returns
    -------
    dict
        ``x_cs``, ``x_zf``, ``cs_error``, ``zf_error`` (relative L2), ``kept``,
        ``n_meas``, ``ratio``.
    """
    # TODO: n = field.size; rng = np.random.default_rng(seed);
    #   n_meas = max(1, int(round(ratio * n)));
    #   kept = np.sort(rng.choice(n, size=n_meas, replace=False)).
    # TODO: spectrum = np.fft.fft(field);
    #   phi = fourier_sensing_matrix(n, kept);
    #   y = np.concatenate([spectrum[kept].real, spectrum[kept].imag]);
    #   x_cs = cs_reconstruct(y, phi, alpha=alpha, seed=0).
    # TODO: Zero-filled: put spectrum[kept] into a length-n complex zero array
    #   and take the real part of np.fft.ifft(...).
    # TODO: Return the dict documented above using
    #   reconstruction_error(field, ..., kind="rel_l2") for the errors.
    raise NotImplementedError


def error_vs_sampling_ratio(
    field: np.ndarray,
    ratios: np.ndarray,
    seed: int = GLOBAL_SEED,
    alpha: float = 1e-3,
) -> dict:
    """Sweep the sampling ratio and record CS and zero-filled errors.

    Returns
    -------
    dict
        ``ratios`` and matching ``cs_errors`` / ``zf_errors`` (relative L2).
    """
    # TODO: For each ratio, call mri_undersample_reconstruct(field, ratio,
    # seed=seed, alpha=alpha) and collect cs_error / zf_error into arrays.
    raise NotImplementedError


# --------------------------------------------------------------------------- #
# Part C -- Quality control
# --------------------------------------------------------------------------- #
def incoherence_check(
    field: np.ndarray,
    ratio: float,
    seed: int = GLOBAL_SEED,
    alpha: float = 1e-3,
) -> dict:
    """Contrast incoherent vs. coherent sampling at a fixed measurement budget.

    Use the same number of measurements two ways: (1) random Fourier (k-space)
    rows -- incoherent with the spike basis -- and (2) direct spatial point
    samples (rows of the identity) -- maximally coherent. CS should succeed in
    the first case and fail in the second.

    Returns
    -------
    dict
        ``cs_error_incoherent``, ``cs_error_coherent`` (relative L2), ``kept``.
    """
    # TODO: n = field.size; rng = np.random.default_rng(seed);
    #   n_meas = max(1, int(round(ratio * n)));
    #   kept = np.sort(rng.choice(n, size=n_meas, replace=False)).
    # TODO: Incoherent case -- reconstruct from random Fourier rows using
    #   fourier_sensing_matrix + cs_reconstruct (as in mri_undersample_reconstruct).
    # TODO: Coherent case -- phi = np.eye(n)[kept, :], y = field[kept],
    #   reconstruct with cs_reconstruct.
    # TODO: Return the two relative-L2 errors and the kept index set.
    raise NotImplementedError


def minimum_acceptable_ratio(sweep: dict, tol: float = 0.05) -> float | None:
    """Smallest swept ratio whose CS error is at or below ``tol`` (provided).

    Returns ``None`` when no swept ratio meets the tolerance.
    """
    ratios = np.asarray(sweep["ratios"], dtype=float)
    cs_errors = np.asarray(sweep["cs_errors"], dtype=float)
    ok = np.flatnonzero(cs_errors <= tol)
    if ok.size == 0:
        return None
    return float(ratios[ok].min())


# --------------------------------------------------------------------------- #
# Driver: QC before results, then the honest interpretation block (provided).
# --------------------------------------------------------------------------- #
def main() -> None:
    """Run the full PS4 analysis with QC gates and an interpretation block."""
    seed_everything()

    # --- Part A: time-frequency + wavelet decomposition of a nonstationary signal.
    _t, x, fs = load_nonstationary_signal()
    print(qc_signals(x, fs=fs).render())
    print()
    freqs, magnitude = compute_fft(x, fs)
    f_spec, t_spec, power = compute_spectrogram(x, fs)
    coeffs = wavelet_decompose(x)
    dominant = float(freqs[int(np.argmax(magnitude))])
    print(
        f"[A] FFT bins={freqs.size} dominant~{dominant:.1f} Hz | "
        f"spectrogram={power.shape} | wavelet bands={len(coeffs)}"
    )
    print()

    # --- Part B1: wavelet denoising validated against a known SNR.
    _te, clean, noisy, fs_ecg = load_ecg_segment()
    print(qc_signals(noisy, fs=fs_ecg, reference=clean).render())
    print()
    den = denoise_and_score(noisy, clean)
    print(
        f"[B1] SNR before={den['snr_before']:.2f} dB after={den['snr_after']:.2f} dB "
        f"gain={den['snr_gain']:.2f} dB"
    )
    print()

    # --- Part B2: accelerated MRI CS reconstruction and error vs. sampling ratio.
    field = load_sparse_field()
    print(qc_signals(field, fs=1.0).render())
    print()
    ratios = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
    sweep = error_vs_sampling_ratio(field, ratios)
    for ratio, cs_err, zf_err in zip(
        sweep["ratios"], sweep["cs_errors"], sweep["zf_errors"], strict=True
    ):
        print(f"[B2] ratio={ratio:.2f} CS_err={cs_err:.4f} zero-filled_err={zf_err:.4f}")
    print()

    # --- Part C: incoherence check.
    inc = incoherence_check(field, 0.35)
    print(
        f"[C] incoherent CS_err={inc['cs_error_incoherent']:.4f} "
        f"coherent CS_err={inc['cs_error_coherent']:.4f}"
    )
    print()

    # --- Part D: interpretation with explicit confidence + limitations.
    min_ratio = minimum_acceptable_ratio(sweep, tol=0.05)
    ratio_txt = "n/a" if min_ratio is None else f"{min_ratio:.2f}"
    block = interpretation_block(
        claim=(
            "On seeded synthetic fixtures, L1 compressed sensing recovers the "
            f"sparse field from about {ratio_txt} of k-space (relative L2 error "
            "<= 0.05), wavelet thresholding yields a positive SNR gain on the "
            "ECG-like segment, and incoherent sampling is necessary for recovery."
        ),
        confidence="high",
        limitations_list=[
            "Synthetic field is exactly sparse; real ECG/MRI data is only "
            "approximately sparse in these bases.",
            "Denoising is scored on a single noise realization at one noise level.",
            "The Lasso regularization alpha is fixed, not cross-validated.",
            "A 1-D field is used as a proxy for a full 2-D MRI acquisition.",
        ],
        evidence=(
            "ground-truth relative-L2 reconstruction error and known-SNR gain "
            "measured on deterministic seeded fixtures"
        ),
    )
    print(block)


if __name__ == "__main__":
    main()
