"""Reusable plotting helpers for ddm4bio.

Every helper returns a matplotlib ``Figure``/``Axes`` object and never calls
``plt.show()`` so figures compose cleanly in notebooks and tests.
``matplotlib`` (and any other heavy libs) are imported inside function bodies.

These helpers take *pre-computed* arrays (eigenvalues, spectrograms, ROC
coordinates, modes, time-frequency power) so the module depends on nothing
heavier than numpy and matplotlib. Compute the inputs with the analysis
modules, then hand the results here to be shown honestly.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def scree_plot(values: np.ndarray, ax: Any | None = None) -> Any:
    """Plot a scree (explained-variance) curve.

    Parameters
    ----------
    values: ``(n_components,)`` eigenvalues or explained variances, ordered
        from largest to smallest.
    ax: optional existing matplotlib Axes to draw into. If ``None`` a new
        Figure and Axes are created.

    Returns
    -------
    matplotlib.axes.Axes
        The Axes containing the scree plot. Does not call ``plt.show()``.
    """
    import matplotlib.pyplot as plt

    values = np.asarray(values, dtype=float).ravel()
    if ax is None:
        _, ax = plt.subplots()

    components = np.arange(1, values.size + 1)
    ax.plot(components, values, marker="o", linewidth=1.5)
    ax.set_xlabel("Component")
    ax.set_ylabel("Eigenvalue / explained variance")
    ax.set_title("Scree plot")
    ax.set_xticks(components)
    return ax


def spectrogram_plot(spectrogram: np.ndarray, ax: Any | None = None) -> Any:
    """Plot a pre-computed spectrogram as a heatmap.

    Parameters
    ----------
    spectrogram: ``(n_freqs, n_times)`` real-valued magnitude or power array,
        following the library convention of components (frequencies) in rows
        and samples (time frames) in columns.
    ax: optional existing matplotlib Axes to draw into. If ``None`` a new
        Figure and Axes are created.

    Returns
    -------
    matplotlib.axes.Axes
        The Axes containing the spectrogram. Does not call ``plt.show()``.
    """
    import matplotlib.pyplot as plt

    spectrogram = np.asarray(spectrogram, dtype=float)
    if ax is None:
        _, ax = plt.subplots()

    image = ax.imshow(
        spectrogram,
        aspect="auto",
        origin="lower",
        cmap="viridis",
    )
    ax.set_xlabel("Time frame")
    ax.set_ylabel("Frequency bin")
    ax.set_title("Spectrogram")
    ax.figure.colorbar(image, ax=ax, label="Power")
    return ax


def roc_with_ci(
    fpr: np.ndarray,
    tpr: np.ndarray,
    ci: tuple[np.ndarray, np.ndarray] | None = None,
    ax: Any | None = None,
) -> Any:
    """Plot an ROC curve with an optional confidence band.

    Parameters
    ----------
    fpr: ``(n_thresholds,)`` false-positive rates.
    tpr: ``(n_thresholds,)`` true-positive rates, aligned with ``fpr``.
    ci: optional ``(tpr_lower, tpr_upper)`` pair of ``(n_thresholds,)`` arrays
        giving a confidence band on the true-positive rate. When provided the
        band is shaded between the two curves.
    ax: optional existing matplotlib Axes to draw into. If ``None`` a new
        Figure and Axes are created.

    Returns
    -------
    matplotlib.axes.Axes
        The Axes containing the ROC curve and CI band. Does not call
        ``plt.show()``.
    """
    import matplotlib.pyplot as plt

    fpr = np.asarray(fpr, dtype=float).ravel()
    tpr = np.asarray(tpr, dtype=float).ravel()
    if ax is None:
        _, ax = plt.subplots()

    if ci is not None:
        lower = np.asarray(ci[0], dtype=float).ravel()
        upper = np.asarray(ci[1], dtype=float).ravel()
        ax.fill_between(fpr, lower, upper, alpha=0.25, label="Confidence band")

    ax.plot(fpr, tpr, linewidth=2.0, label="ROC")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0, color="0.5", label="Chance")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve")
    ax.legend(loc="lower right")
    return ax


def mode_grid(modes: np.ndarray, shape: tuple[int, int] | None = None, ncols: int = 4) -> Any:
    """Plot a grid of spatial modes / components.

    Parameters
    ----------
    modes: either a ``(n_modes, H, W)`` stack of 2D modes, or a
        ``(n_modes, H * W)`` array of flattened modes (components in rows,
        following the library convention). Flattened modes are reshaped using
        ``shape``.
    shape: ``(H, W)`` target shape used to reshape flattened modes. Required
        only when ``modes`` is 2D.
    ncols: number of columns in the grid.

    Returns
    -------
    matplotlib.figure.Figure
        The Figure containing the mode grid. Does not call ``plt.show()``.
    """
    import matplotlib.pyplot as plt

    modes = np.asarray(modes, dtype=float)
    if modes.ndim == 2:
        if shape is None:
            raise ValueError("shape is required to reshape 2D (n_modes, H*W) modes")
        modes = modes.reshape(modes.shape[0], shape[0], shape[1])
    elif modes.ndim != 3:
        raise ValueError("modes must be (n_modes, H, W) or (n_modes, H*W) with shape given")

    n_modes = modes.shape[0]
    ncols = max(1, min(ncols, n_modes))
    nrows = int(np.ceil(n_modes / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows), squeeze=False)
    for idx in range(nrows * ncols):
        ax = axes.flat[idx]
        if idx < n_modes:
            ax.imshow(modes[idx], cmap="RdBu_r", origin="lower")
            ax.set_title(f"Mode {idx + 1}")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Spatial modes")
    return fig


def time_freq_panel(
    times: np.ndarray,
    freqs: np.ndarray,
    power: np.ndarray,
    ax: Any | None = None,
) -> Any:
    """Plot a time-frequency power panel.

    Parameters
    ----------
    times: ``(n_times,)`` time-frame centers (seconds).
    freqs: ``(n_freqs,)`` frequency-bin centers (Hz).
    power: ``(n_freqs, n_times)`` real-valued power, with frequencies in rows
        and time frames in columns (library convention).
    ax: optional existing matplotlib Axes to draw into. If ``None`` a new
        Figure and Axes are created.

    Returns
    -------
    matplotlib.axes.Axes
        The Axes containing the time-frequency panel. Does not call
        ``plt.show()``.
    """
    import matplotlib.pyplot as plt

    times = np.asarray(times, dtype=float).ravel()
    freqs = np.asarray(freqs, dtype=float).ravel()
    power = np.asarray(power, dtype=float)
    if ax is None:
        _, ax = plt.subplots()

    mesh = ax.pcolormesh(times, freqs, power, cmap="magma", shading="auto")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("Time-frequency power")
    ax.figure.colorbar(mesh, ax=ax, label="Power")
    return ax
