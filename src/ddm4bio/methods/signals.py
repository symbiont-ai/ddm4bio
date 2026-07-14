"""Signal-processing teaching wrappers.

Educational wrappers for time-frequency analysis, wavelet denoising, and
compressed-sensing reconstruction. Heavy dependencies (scipy.signal,
PyWavelets) are imported inside the function bodies. Phase 0 scaffold: bodies
raise NotImplementedError.
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
        Input time-domain signal.
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
        (f, t, Zxx) where ``f`` has shape (n_freqs,), ``t`` has shape
        (n_times,), and ``Zxx`` has shape (n_freqs, n_times) with complex STFT
        coefficients.
    """
    # scipy.signal imported inside body per scaffold rules.
    raise NotImplementedError


def wavelet_denoise(
    x: np.ndarray,
    wavelet: str = "db4",
    level: int | None = None,
    threshold: float | None = None,
) -> np.ndarray:
    """Denoise a 1-D signal via wavelet thresholding.

    Parameters
    ----------
    x : np.ndarray, shape (n_samples,)
        Noisy input signal.
    wavelet : str, default "db4"
        Wavelet family name (PyWavelets convention).
    level : int, optional
        Decomposition level; chosen automatically when None.
    threshold : float, optional
        Coefficient threshold; estimated from the data when None.

    Returns
    -------
    np.ndarray, shape (n_samples,)
        Denoised signal.
    """
    # PyWavelets (pywt) imported inside body per scaffold rules.
    raise NotImplementedError


def compressed_sensing_recon(
    y: np.ndarray,
    Phi: np.ndarray,
    n_iter: int = 100,
) -> np.ndarray:
    """Reconstruct a sparse signal from compressed measurements.

    Parameters
    ----------
    y : np.ndarray, shape (n_measurements,)
        Measurement vector.
    Phi : np.ndarray, shape (n_measurements, n_features)
        Sensing / measurement matrix such that ``y ~= Phi @ x``.
    n_iter : int, default 100
        Maximum number of solver iterations.

    Returns
    -------
    np.ndarray, shape (n_features,)
        Recovered sparse signal estimate ``x``.
    """
    # scipy / sklearn solvers imported inside body per scaffold rules.
    raise NotImplementedError
