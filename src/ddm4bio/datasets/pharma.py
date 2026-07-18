"""Pharmacometrics (PK/PD) datasets for the capstone.

Datasets:
- warfarin: warfarin pharmacokinetic/pharmacodynamic data -- a single oral dose
  in 32 subjects, with plasma warfarin concentration (``dvid == "cp"``) and the
  anticoagulation effect PCA, prothrombin-complex activity (``dvid == "pca"``),
  sampled over ~6 days, plus weight/age/sex covariates. Tier: open. License:
  GPL (>=3) as distributed in the nlmixr2data R package; the data originate with
  O'Reilly & Aggeler (1968) and were popularized by Holford (1986).
  ``load_warfarin`` fetches the real dataset (from nlmixr2data upstream, or the
  course mirror) and returns a tidy pandas DataFrame; on any failure it returns a
  deterministic synthetic PK/PD table with the identical schema (a
  one-compartment oral PK model plus a turnover/indirect-response PD model), so
  the capstone pipeline runs the same way offline.

Loader contract: see :mod:`ddm4bio.datasets.registry`. Heavy libraries (pandas,
pyreadr) are imported inside the loader body so importing this module needs only
numpy.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

from ddm4bio.datasets.registry import LoadedDataset

_WARFARIN_KEY = "warfarin"
_UPSTREAM_RDA = "https://raw.githubusercontent.com/nlmixr2/nlmixr2data/main/data/warfarin.rda"
_MIRROR_BASE = "https://github.com/symbiont-ai/ddm4bio/releases/download/data-mirror-v1"
_MIRROR_CSV = f"{_MIRROR_BASE}/warfarin.csv"
_LICENSE = (
    "GPL (>=3) as distributed in the nlmixr2data R package; data originally "
    "O'Reilly & Aggeler (1968), popularized by Holford (1986)"
)
_COLUMNS = ["id", "time", "amt", "dv", "dvid", "evid", "wt", "age", "sex"]


def load_warfarin(
    *,
    cache_dir: Path | str,
    download: bool = True,
    prefer_real: bool = True,
    seed: int | None = None,
    **opts: Any,
) -> LoadedDataset:
    """Load warfarin PK/PD data (real, with a synthetic PK/PD fallback).

    Parameters
    ----------
    cache_dir:
        Directory for the cached download (idempotent).
    download:
        When True, may fetch the real dataset (upstream, then the mirror).
    prefer_real:
        When True, prefer the real dataset over the synthetic fallback.
    seed:
        RNG seed making the synthetic fallback deterministic.
    **opts:
        ``url`` overrides the upstream download URL.

    Returns
    -------
    LoadedDataset
        ``payload`` is a tidy pandas ``DataFrame`` with columns
        ``id, time, amt, dv, dvid, evid, wt, age, sex`` -- ``dvid == "cp"`` rows
        are plasma concentrations, ``dvid == "pca"`` rows are the anticoagulation
        effect. ``source`` is ``"real"`` or ``"fallback"``.
    """
    cache_dir = Path(cache_dir)

    if download and prefer_real:
        errors = []
        for fetch in (_load_upstream, _load_mirror):
            try:
                return fetch(cache_dir, url=opts.get("url"))
            except Exception as exc:  # noqa: BLE001 -- always fall back on failure
                errors.append(f"{fetch.__name__}: {type(exc).__name__}")
        reason = "real warfarin fetch failed (" + " | ".join(errors) + ")"
    elif not download:
        reason = "download=False requested"
    else:
        reason = "prefer_real=False requested"

    return _warfarin_fallback(seed=seed, reason=reason)


def _tidy(frame: Any) -> Any:
    """Standardize column order/dtypes: dvid lowercase strings, id int."""
    import pandas as pd

    frame = frame.copy()
    frame.columns = [str(c) for c in frame.columns]
    frame["dvid"] = frame["dvid"].astype(str).str.lower()
    frame["id"] = pd.to_numeric(frame["id"]).astype(int)
    keep = [c for c in _COLUMNS if c in frame.columns]
    return frame[keep].reset_index(drop=True)


def _load_upstream(cache_dir: Path, url: str | None) -> LoadedDataset:
    """Download the nlmixr2data ``.rda`` and read it with pyreadr."""
    import pyreadr

    cache_dir.mkdir(parents=True, exist_ok=True)
    url = url or _UPSTREAM_RDA
    dest = cache_dir / "warfarin.rda"
    if not dest.exists():
        urllib.request.urlretrieve(url, dest)  # noqa: S310 -- https source
    frame = _tidy(next(iter(pyreadr.read_r(str(dest)).values())))
    provenance = (
        f"real: warfarin PK/PD from nlmixr2data upstream ({url}), cached at "
        f"{dest}; {_LICENSE}; {frame.shape[0]} rows, {frame['id'].nunique()} "
        "subjects (single oral dose; cp concentration + pca effect)."
    )
    return LoadedDataset(payload=frame, source="real", provenance=provenance, key=_WARFARIN_KEY)


def _load_mirror(cache_dir: Path, url: str | None) -> LoadedDataset:
    """Download the tidy CSV from the course data mirror."""
    import pandas as pd

    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / "warfarin.csv"
    if not dest.exists():
        urllib.request.urlretrieve(_MIRROR_CSV, dest)  # noqa: S310 -- https source
    frame = _tidy(pd.read_csv(dest))
    provenance = (
        f"real: warfarin PK/PD from the course mirror ({_MIRROR_CSV}), cached at "
        f"{dest}; {_LICENSE}; {frame.shape[0]} rows, {frame['id'].nunique()} subjects."
    )
    return LoadedDataset(payload=frame, source="real", provenance=provenance, key=_WARFARIN_KEY)


def _warfarin_fallback(*, seed: int | None, reason: str) -> LoadedDataset:
    """Deterministic synthetic PK/PD table with the warfarin schema.

    A one-compartment oral PK model per subject (with inter-subject variability
    in clearance) drives a turnover / indirect-response PD model for the
    anticoagulation effect, so the synthetic table has the same qualitative
    shape as the real one: concentrations rise then fall, and the effect (PCA)
    declines to a nadir and recovers.
    """
    import pandas as pd

    rng = np.random.default_rng(seed)
    n_subjects = 32
    cp_times = np.array([0.5, 1, 2, 3, 6, 9, 12, 24, 36, 48, 72, 96, 120, 144])
    pca_times = np.array([24, 36, 48, 72, 96, 120, 144])
    # Fine grid to integrate the turnover PD model (indirect response).
    grid = np.arange(0.0, 144.0 + 0.25, 0.25)

    kin, kout, imax, ic50 = 5.0, 0.05, 0.97, 1.5  # PD turnover; baseline PCA = kin/kout = 100

    records = []
    for sid in range(1, n_subjects + 1):
        weight = float(np.clip(rng.normal(70.0, 12.0), 45.0, 100.0))
        age = int(np.clip(rng.normal(45.0, 12.0), 21, 63))
        sex = "male" if rng.random() < 0.7 else "female"
        dose = float(round(1.5 * weight))  # weight-scaled oral dose (mg)

        ka = float(np.exp(rng.normal(np.log(0.5), 0.25)))
        ke = float(np.exp(rng.normal(np.log(0.018), 0.4)))  # ~1.6-day t1/2, CV ~40%
        vd = float(np.exp(rng.normal(np.log(8.0), 0.2)))

        def conc(t, ka=ka, ke=ke, vd=vd, dose=dose):
            return dose * ka / (vd * (ka - ke)) * (np.exp(-ke * t) - np.exp(-ka * t))

        # Integrate the turnover PD model over the fine grid.
        response = np.empty(grid.size)
        response[0] = kin / kout
        for i in range(1, grid.size):
            dt = grid[i] - grid[i - 1]
            c = max(conc(grid[i - 1]), 0.0)
            inhibition = imax * c / (ic50 + c)
            dr = kin * (1.0 - inhibition) - kout * response[i - 1]
            response[i] = max(response[i - 1] + dt * dr, 0.0)

        # Dosing record.
        records.append(
            dict(id=sid, time=0.0, amt=dose, dv=0.0, dvid="cp", evid=1, wt=weight, age=age, sex=sex)
        )
        # Concentration observations.
        for t in cp_times:
            c = max(conc(float(t)) + rng.normal(0.0, 0.4), 0.0)
            records.append(
                dict(
                    id=sid,
                    time=float(t),
                    amt=0.0,
                    dv=round(c, 2),
                    dvid="cp",
                    evid=0,
                    wt=weight,
                    age=age,
                    sex=sex,
                )
            )
        # Effect (PCA) observations.
        for t in pca_times:
            r = float(np.interp(t, grid, response)) + rng.normal(0.0, 2.0)
            records.append(
                dict(
                    id=sid,
                    time=float(t),
                    amt=0.0,
                    dv=round(max(r, 0.0), 1),
                    dvid="pca",
                    evid=0,
                    wt=weight,
                    age=age,
                    sex=sex,
                )
            )

    frame = pd.DataFrame.from_records(records)[_COLUMNS]
    provenance = (
        f"synthetic/bundled fallback: {reason}. One-compartment oral PK with "
        f"inter-subject clearance variability driving a turnover/indirect-response "
        f"PD model, for {n_subjects} subjects (schema matches the real warfarin "
        "dataset: cp concentration + pca effect); seeded, deterministic."
    )
    return LoadedDataset(payload=frame, source="fallback", provenance=provenance, key=_WARFARIN_KEY)
