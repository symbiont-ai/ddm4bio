"""Epidemiology datasets.

Dataset: JHU CSSE COVID-19 (archived, frozen snapshot 2023-03-10).
Tier: archive.
License: CC BY 4.0 (JHU CSSE), for educational/academic use.
Install: ``pip install pandas``. Data are static CSVs from the frozen archive.

The real loader fetches the archived JHU CSSE *global confirmed cases* time
series (a static CSV from the frozen 2023 GitHub archive), caches the raw
download under ``cache_dir`` (idempotently), and returns a tidy ``date`` /
``cases`` :class:`pandas.DataFrame` for the requested ``region``. When the
network, ``pandas``, or the parse is unavailable -- or when ``download`` /
``prefer_real`` are disabled -- it returns a deterministic, clearly-labelled
SIR-derived synthetic case-count series built from
:func:`ddm4bio.datasets.synthetic.make_sir`. Heavy imports (``pandas``) live in
the function bodies so importing this module needs only numpy.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np

from .registry import LoadedDataset

_KEY = "jhu_covid"

# Static raw CSV from the frozen JHU CSSE archive (confirmed cases, global).
_JHU_CONFIRMED_GLOBAL_URL = (
    "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/"
    "csse_covid_19_data/csse_covid_19_time_series/"
    "time_series_covid19_confirmed_global.csv"
)
_JHU_CACHE_NAME = "jhu_covid19_confirmed_global.csv"

_REAL_PROVENANCE = (
    "real: JHU CSSE COVID-19 archived global confirmed-cases time series "
    "(time_series_covid19_confirmed_global.csv) from the frozen GitHub archive "
    "https://github.com/CSSEGISandData/COVID-19 (raw URL "
    f"{_JHU_CONFIRMED_GLOBAL_URL}); license CC BY 4.0 (Johns Hopkins "
    "University), educational/academic use. Cumulative confirmed cases for "
    "region {region!r}, tidied to columns (date, cases)."
)


def _download_to_cache(url: str, dest: Path) -> Path:
    """Download ``url`` to ``dest`` once (idempotent). Return ``dest``.

    If ``dest`` already exists it is returned untouched -- the real fetch is
    never repeated. The download is written to a temporary sibling first and
    then atomically replaced so a crash mid-download cannot leave a corrupt
    cache file.
    """
    if dest.exists():
        return dest
    req = urllib.request.Request(url, headers={"User-Agent": "ddm4bio/0.1"})
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 (static https)
        data = resp.read()
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(dest)
    return dest


def _tidy_region(df, region: str):
    """Reduce the wide JHU frame to a tidy ``(date, cases)`` frame for a region."""
    import pandas as pd

    date_cols = list(df.columns[4:])  # first four cols: State, Country, Lat, Long
    sub = df[df["Country/Region"] == region]
    if sub.empty:
        available = ", ".join(sorted(df["Country/Region"].unique()))
        raise ValueError(f"region {region!r} not in JHU data. Available: {available}")
    series = sub[date_cols].sum(axis=0)
    dates = pd.to_datetime(series.index, format="%m/%d/%y")
    tidy = pd.DataFrame({"date": dates.to_numpy(), "cases": series.to_numpy(dtype="int64")})
    return tidy.reset_index(drop=True)


def _sir_fallback(region: str, seed: int | None) -> LoadedDataset:
    """Deterministic SIR-derived synthetic confirmed-case series."""
    import pandas as pd

    from ddm4bio.datasets.synthetic import make_sir

    rng = np.random.default_rng(seed)
    beta, gamma = 0.35, 0.12
    n_steps = 400
    population = 1_000_000

    traj = make_sir(beta=beta, gamma=gamma, t_max=200.0, n_steps=n_steps)
    # Cumulative confirmed ~ fraction ever infected (1 - S) x population.
    cumulative = (1.0 - traj.susceptible) * population
    noise = rng.normal(1.0, 0.01, size=n_steps)
    cases = np.maximum.accumulate(np.round(cumulative * noise)).astype("int64")
    dates = pd.date_range("2020-01-22", periods=n_steps, freq="D")

    payload = pd.DataFrame({"date": dates, "cases": cases})
    provenance = (
        "synthetic/bundled fallback: SIR-derived cumulative confirmed-case "
        f"series (beta={beta}, gamma={gamma}) from "
        "ddm4bio.datasets.synthetic.make_sir, scaled to a "
        f"{population:,}-person population and seeded with seed={seed!r}; "
        f"region={region!r} is a label only -- no real JHU data was fetched."
    )
    return LoadedDataset(payload=payload, source="fallback", provenance=provenance, key=_KEY)


def load_jhu_covid(
    *,
    cache_dir,
    download: bool = True,
    prefer_real: bool = True,
    seed: int | None = None,
    region: str = "US",
    **opts,
) -> LoadedDataset:
    """Load a tidy JHU CSSE COVID-19 confirmed-case series for a region.

    Parameters
    ----------
    cache_dir:
        Directory for the cached raw CSV download.
    download, prefer_real:
        When both True, fetch and cache the real archived JHU CSV; otherwise
        return the synthetic SIR-derived fallback.
    seed:
        Seed for the deterministic fallback series.
    region:
        JHU ``Country/Region`` value to extract (default ``"US"``).
    **opts:
        Ignored extra loader options.

    Returns
    -------
    LoadedDataset
        ``payload`` is a :class:`pandas.DataFrame` with columns ``date`` and
        ``cases``; ``source`` is ``"real"`` or ``"fallback"``.
    """
    if download and prefer_real:
        try:
            import pandas as pd

            cache_path = Path(cache_dir) / _JHU_CACHE_NAME
            _download_to_cache(_JHU_CONFIRMED_GLOBAL_URL, cache_path)
            df = pd.read_csv(cache_path)
            payload = _tidy_region(df, region)
            return LoadedDataset(
                payload=payload,
                source="real",
                provenance=_REAL_PROVENANCE.format(region=region),
                key=_KEY,
            )
        except Exception:
            return _sir_fallback(region, seed)

    return _sir_fallback(region, seed)
