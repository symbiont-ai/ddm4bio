"""Quality control for time-series / signal datasets."""

from .report import QCReport


def qc_signals(signals=None, fs=None, reference=None) -> QCReport:
    """Run quality-control checks on signal/time-series data.

    Computes and summarizes: sampling rate consistency, NaN/flatline
    detection, saturation/clipping detection, and SNR where a reference
    (clean) signal is available.

    Parameters
    ----------
    signals : array-like, optional
        Signal array, typically shape (n_channels, n_samples). A 1-D array is
        treated as a single channel. Following the library convention,
        components/channels are in ROWS and samples in COLUMNS.
    fs : float, optional
        Sampling frequency in Hz.
    reference : array-like, optional
        Clean reference signal for SNR estimation, matching ``signals`` shape.

    Returns
    -------
    QCReport
        Report with modality="signals", populated summary and warnings.
    """
    import numpy as np

    summary: dict = {}
    warnings: list = []

    if signals is None:
        summary["length"] = 0
        summary["sampling_rate"] = fs
        return QCReport(modality="signals", summary=summary, warnings=warnings)

    x = np.asarray(signals, dtype=float)
    if x.ndim == 1:
        x = x[np.newaxis, :]

    n_channels, n_samples = x.shape
    summary["n_channels"] = int(n_channels)
    summary["length"] = int(n_samples)
    summary["sampling_rate"] = fs
    if fs:
        summary["duration_s"] = float(n_samples) / float(fs)

    # NaN count.
    nan_count = int(np.isnan(x).sum())
    summary["nan_count"] = nan_count
    if nan_count:
        warnings.append(f"Signal contains {nan_count} NaN value(s).")

    # Flatline (constant) segments: runs where consecutive samples are equal.
    flatline_segments = 0
    for row in x:
        finite = row[np.isfinite(row)]
        if finite.size < 2:
            continue
        # Consecutive equal samples across the full (finite-aware) row.
        diffs = np.diff(row)
        is_flat = diffs == 0
        if not is_flat.any():
            continue
        # Count runs of True in is_flat (each run == one flatline segment).
        padded = np.concatenate(([False], is_flat, [False]))
        starts = np.flatnonzero((~padded[:-1]) & padded[1:])
        flatline_segments += int(starts.size)
    summary["flatline_segments"] = flatline_segments
    if flatline_segments:
        warnings.append(f"Detected {flatline_segments} flatline (constant) segment(s).")

    # Clipping / saturation fraction: values at the observed min or max.
    finite_x = x[np.isfinite(x)]
    if finite_x.size:
        obs_min = float(finite_x.min())
        obs_max = float(finite_x.max())
        if obs_max > obs_min:
            clipped = np.isclose(finite_x, obs_min) | np.isclose(finite_x, obs_max)
            clip_frac = float(clipped.sum()) / float(finite_x.size)
        else:
            # Entirely constant signal.
            clip_frac = 1.0
        summary["clipping_fraction"] = clip_frac
        if clip_frac > 0.01:
            warnings.append(
                f"Clipping/saturation: {clip_frac:.1%} of samples at the signal extremes."
            )
    else:
        summary["clipping_fraction"] = None

    # Rough SNR estimate (dB).
    snr_db = None
    if reference is not None:
        ref = np.asarray(reference, dtype=float)
        if ref.ndim == 1:
            ref = ref[np.newaxis, :]
        if ref.shape == x.shape:
            noise = x - ref
            sig_power = float(np.nanmean(ref**2))
            noise_power = float(np.nanmean(noise**2))
            if noise_power > 0 and sig_power > 0:
                snr_db = 10.0 * np.log10(sig_power / noise_power)
    if snr_db is None and finite_x.size:
        # Reference-free proxy: mean power vs. variance of first differences.
        sig_power = float(np.nanmean(x**2))
        diff_power = float(np.nanmean(np.diff(x, axis=1) ** 2)) if n_samples > 1 else 0.0
        if diff_power > 0 and sig_power > 0:
            snr_db = 10.0 * np.log10(sig_power / diff_power)
    summary["snr_db"] = float(snr_db) if snr_db is not None else None

    return QCReport(modality="signals", summary=summary, warnings=warnings)
