"""Ground-truth tests for the signal-processing wrappers.

The STFT test checks the (freqs, times, Zxx) shapes are mutually consistent;
and wavelet denoising must move a noisy signal closer to its clean source. All
tests are deterministic and run offline.
"""

from __future__ import annotations

import numpy as np

from ddm4bio.methods.signals import stft, wavelet_denoise


def test_stft_shapes_are_consistent():
    fs = 1000.0
    n = 2048
    t = np.arange(n) / fs
    x = np.sin(2.0 * np.pi * 50.0 * t) + 0.5 * np.sin(2.0 * np.pi * 120.0 * t)

    nperseg = 128
    freqs, times, zxx = stft(x, fs=fs, nperseg=nperseg)

    assert freqs.ndim == 1
    assert times.ndim == 1
    # Zxx has frequency components in rows and time frames in columns.
    assert zxx.shape == (freqs.size, times.size)
    # A real STFT of a length-nperseg segment yields nperseg/2 + 1 frequencies.
    assert freqs.size == nperseg // 2 + 1
    assert np.iscomplexobj(zxx)


def test_wavelet_denoise_improves_snr():
    n = 1024
    t = np.linspace(0.0, 1.0, n, endpoint=False)
    clean = np.sin(2.0 * np.pi * 5.0 * t) + 0.5 * np.sin(2.0 * np.pi * 12.0 * t)

    rng = np.random.default_rng(3)
    noisy = clean + 0.4 * rng.standard_normal(n)

    denoised = wavelet_denoise(noisy, wavelet="db4")

    assert denoised.shape == clean.shape

    def snr_db(reference: np.ndarray, estimate: np.ndarray) -> float:
        signal_power = np.sum(reference**2)
        noise_power = np.sum((reference - estimate) ** 2)
        return 10.0 * np.log10(signal_power / noise_power)

    snr_noisy = snr_db(clean, noisy)
    snr_denoised = snr_db(clean, denoised)
    assert snr_denoised > snr_noisy
