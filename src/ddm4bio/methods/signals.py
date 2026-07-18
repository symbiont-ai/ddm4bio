"""Signal-processing teaching wrappers.

Educational wrappers for time-frequency analysis and wavelet denoising. Heavy
dependencies (scipy.signal, PyWavelets) are imported inside the function bodies
so that ``import ddm4bio`` succeeds on numpy alone.
"""

from __future__ import annotations

import numpy as np


def stft(
    x: np.ndarray,
    fs: float,
    window: str = "hann",
    nperseg: int = 256,
    noverlap: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Short-time Fourier transform of a 1-D signal.

    Parameters
    ----------
    x : np.ndarray, shape (n_samples,)
        Input time-domain signal (1-D).
    fs : float
        Sampling frequency in Hz.
    window : str, default "hann"
        Window function name passed to the underlying STFT routine.
    nperseg : int, default 256
        Length of each segment.
    noverlap : int, optional
        Number of points to overlap between segments; defaults to
        ``nperseg // 2`` when None.

    Returns
    -------
    tuple of np.ndarray
        (freqs, times, Zxx) where ``freqs`` has shape (n_freqs,), ``times``
        has shape (n_times,), and ``Zxx`` has shape (n_freqs, n_times) with
        complex STFT coefficients (frequency components in ROWS, time frames
        in columns).
    """
    from scipy.signal import stft as _scipy_stft

    x = np.asarray(x, dtype=float)
    if noverlap is None:
        noverlap = nperseg // 2
    freqs, times, zxx = _scipy_stft(
        x,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
    )
    return freqs, times, zxx


def wavelet_denoise(
    x: np.ndarray,
    wavelet: str = "db4",
    level: int | None = None,
    mode: str = "soft",
) -> np.ndarray:
    """Denoise a 1-D signal via multilevel wavelet thresholding.

    Uses a VisuShrink / universal threshold ``sqrt(2 * log(n)) * sigma`` where
    ``sigma`` is a robust noise estimate from the finest-level detail
    coefficients via the median absolute deviation (MAD). Detail coefficients
    are thresholded (soft or hard) and the signal is reconstructed.

    Parameters
    ----------
    x : np.ndarray, shape (n_samples,)
        Noisy input signal (1-D).
    wavelet : str, default "db4"
        Wavelet family name (PyWavelets convention).
    level : int, optional
        Decomposition level; chosen automatically via
        ``pywt.dwt_max_level`` when None.
    mode : str, default "soft"
        Thresholding mode, ``"soft"`` or ``"hard"``.

    Returns
    -------
    np.ndarray, shape (n_samples,)
        Denoised signal, same length as ``x``.
    """
    import pywt

    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    if n == 0:
        return x.copy()

    wav = pywt.Wavelet(wavelet)
    if level is None:
        level = pywt.dwt_max_level(n, wav.dec_len)
    if level < 1:
        return x.copy()

    coeffs = pywt.wavedec(x, wavelet, level=level, mode="periodization")
    # Robust noise estimate from finest-level detail coefficients (MAD).
    detail = coeffs[-1]
    sigma = np.median(np.abs(detail)) / 0.6745
    threshold = sigma * np.sqrt(2.0 * np.log(n)) if n > 1 else 0.0

    new_coeffs = [coeffs[0]]
    for detail_coeffs in coeffs[1:]:
        new_coeffs.append(pywt.threshold(detail_coeffs, threshold, mode=mode))

    denoised = pywt.waverec(new_coeffs, wavelet, mode="periodization")
    # waverec may return one extra sample for odd-length inputs.
    return denoised[:n]
