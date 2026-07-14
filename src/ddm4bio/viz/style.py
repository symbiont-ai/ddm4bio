"""Plotting style configuration for ddm4bio figures."""

from __future__ import annotations

# Okabe-Ito colorblind-safe qualitative palette (hex strings, plain Python).
# Reference: Okabe & Ito, "Color Universal Design" (2008).
OKABE_ITO: list[str] = [
    "#000000",  # black
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
]


def set_style() -> None:
    """Apply the ddm4bio matplotlib style.

    Configures a colorblind-safe categorical palette and consistent font
    sizes for axes, ticks, and titles. Mutates the global matplotlib rcParams
    and returns ``None``.

    The palette is the Okabe-Ito qualitative palette (exposed at module level
    as :data:`OKABE_ITO`), which remains distinguishable under the common forms
    of color-vision deficiency.

    Notes
    -----
    ``matplotlib`` is imported inside the body so ``import ddm4bio`` does not
    require matplotlib to be installed.
    """
    import matplotlib as mpl
    from cycler import cycler

    mpl.rcParams.update(
        {
            # Colorblind-safe categorical color cycle.
            "axes.prop_cycle": cycler(color=OKABE_ITO),
            # Readable, consistent font sizes.
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 14,
            # Clean spines: drop the top and right borders.
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.6,
            # Sensible resolution for both screen and print.
            "figure.dpi": 120,
            "savefig.dpi": 150,
            "figure.autolayout": True,
        }
    )
