"""Epidemiology datasets.

Dataset: JHU CSSE COVID-19 (archived, frozen snapshot 2023-03-10).
Tier: archive.
License: CC BY 4.0 (JHU CSSE), for educational/academic use.
Install: ``pip install pandas``. Data are static CSVs from the frozen archive.
"""

from __future__ import annotations

from typing import Any


def load_jhu_covid() -> Any:
    """Load the frozen JHU CSSE COVID-19 time-series snapshot.

    Returns
    -------
    object
        A container exposing confirmed/deaths/recovered time series indexed
        by region and date, from the static 2023-03-10 archive CSVs.

    Notes
    -----
    Uses ``pandas`` (imported inside the body).
    """
    raise NotImplementedError(
        "load_jhu_covid is a Phase 0 stub; expected return exposes "
        "confirmed/deaths/recovered time series (region x date)."
    )
