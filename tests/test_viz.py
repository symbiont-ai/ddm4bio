"""Unit tests for the plotting helpers.

Uses the non-interactive Agg backend so no display is required. Checks that
``set_style`` runs and that every plot helper returns a matplotlib
``Axes``/``Figure`` on small inputs.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from ddm4bio.viz.plots import (  # noqa: E402
    mode_grid,
    roc_with_ci,
    scree_plot,
    spectrogram_plot,
    time_freq_panel,
)
from ddm4bio.viz.style import set_style  # noqa: E402


def teardown_function(_func):
    plt.close("all")


def test_set_style_runs():
    set_style()  # mutates rcParams; must not raise.


def test_scree_plot_returns_axes():
    values = np.array([5.0, 3.0, 1.0, 0.5])
    ax = scree_plot(values)
    assert isinstance(ax, Axes)


def test_spectrogram_plot_returns_axes():
    spec = np.abs(np.random.default_rng(0).standard_normal((8, 12)))
    ax = spectrogram_plot(spec)
    assert isinstance(ax, Axes)


def test_roc_with_ci_returns_axes():
    fpr = np.linspace(0.0, 1.0, 10)
    tpr = np.sqrt(fpr)
    ax = roc_with_ci(fpr, tpr)
    assert isinstance(ax, Axes)


def test_roc_with_ci_band_returns_axes():
    fpr = np.linspace(0.0, 1.0, 10)
    tpr = np.sqrt(fpr)
    lower = np.clip(tpr - 0.1, 0.0, 1.0)
    upper = np.clip(tpr + 0.1, 0.0, 1.0)
    ax = roc_with_ci(fpr, tpr, ci=(lower, upper))
    assert isinstance(ax, Axes)


def test_mode_grid_returns_figure():
    modes = np.random.default_rng(1).standard_normal((3, 5, 5))
    fig = mode_grid(modes)
    assert isinstance(fig, Figure)


def test_mode_grid_flattened_returns_figure():
    modes = np.random.default_rng(2).standard_normal((4, 16))
    fig = mode_grid(modes, shape=(4, 4))
    assert isinstance(fig, Figure)


def test_time_freq_panel_returns_axes():
    times = np.linspace(0.0, 1.0, 10)
    freqs = np.linspace(1.0, 20.0, 8)
    power = np.abs(np.random.default_rng(3).standard_normal((8, 10)))
    ax = time_freq_panel(times, freqs, power)
    assert isinstance(ax, Axes)
