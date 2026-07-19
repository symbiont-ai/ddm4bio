"""Pharmacometrics (PK/PD) and pharmacology datasets.

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
- ccle: CCLE (Cancer Cell Line Encyclopedia) pharmacological profiling -- REAL
  per-concentration dose-response. Each source row is one compound x cell-line
  experiment giving eight concentrations, eight median activity readings (0 =
  DMSO control, negative = growth inhibition), and eight per-dose replicate SDs,
  plus the CCLE authors' OWN fitted EC50/IC50/Amax/ActArea. ``load_ccle`` returns
  a *tidy long* table (one row per concentration), so a lesson can refit the Hill
  model to genuine raw measurements -- weighted by the per-dose SD -- and
  cross-check the recovered EC50/Amax against CCLE's published fit (non-circular:
  the fit and the reference come from independent estimations). Tier: open.
  License: CC BY 4.0 (Broad/DepMap; DOI-pinned at Zenodo 10.5281/zenodo.3905462);
  data from Barretina et al. (2012). On any failure it returns a deterministic
  synthetic table with the identical long schema and the same series identities,
  so the notebook's ``(compound, cell_line)`` selection runs the same offline.

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


# --------------------------------------------------------------------------- #
# CCLE pharmacological profiling -- real per-concentration dose-response.
# --------------------------------------------------------------------------- #

_CCLE_KEY = "ccle"
_CCLE_UPSTREAM = (
    "https://data.broadinstitute.org/ccle_legacy_data/pharmacological_profiling/"
    "CCLE_NP24.2009_Drug_data_2015.02.24.csv"
)
_CCLE_MIRROR = f"{_MIRROR_BASE}/ccle_drug_data.csv"
_CCLE_LICENSE = (
    "CC BY 4.0 (CCLE / Broad DepMap; DOI-pinned at Zenodo 10.5281/zenodo.3905462); "
    "data from Barretina et al. (2012), Nature 483:603-607"
)
# Tidy long schema: one row per concentration, carrying CCLE's own fitted summary
# columns (``*_pub``) alongside the raw measurement so a lesson can refit and then
# cross-check against the published estimate.
_CCLE_COLUMNS = [
    "cell_line",
    "compound",
    "target",
    "concentration",
    "activity",
    "activity_sd",
    "ec50_pub",
    "ic50_pub",
    "amax_pub",
    "actarea_pub",
    "fit_type",
    "n_doses",
]


def load_ccle(
    *,
    cache_dir: Path | str,
    download: bool = True,
    prefer_real: bool = True,
    seed: int | None = None,
    **opts: Any,
) -> LoadedDataset:
    """Load CCLE dose-response data (real raw per-concentration, synthetic fallback).

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
        ``payload`` is a tidy *long* pandas ``DataFrame`` with columns
        ``cell_line, compound, target, concentration, activity, activity_sd,
        ec50_pub, ic50_pub, amax_pub, actarea_pub, fit_type, n_doses`` -- one row
        per concentration. ``activity`` is CCLE's median activity (0 = DMSO
        control, negative = growth inhibition); the ``*_pub`` columns are the CCLE
        authors' own fitted summary for that series. ``source`` is ``"real"`` or
        ``"fallback"``.
    """
    cache_dir = Path(cache_dir)

    if download and prefer_real:
        errors = []
        for fetch in (_load_ccle_upstream, _load_ccle_mirror):
            try:
                return fetch(cache_dir, url=opts.get("url"))
            except Exception as exc:  # noqa: BLE001 -- always fall back on failure
                errors.append(f"{fetch.__name__}: {type(exc).__name__}")
        reason = "real CCLE fetch failed (" + " | ".join(errors) + ")"
    elif not download:
        reason = "download=False requested"
    else:
        reason = "prefer_real=False requested"

    return _ccle_fallback(seed=seed, reason=reason)


def _to_float(value: Any) -> float:
    """Parse a scalar to float, returning NaN on any failure."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _explode_ccle(frame: Any) -> Any:
    """Explode CCLE's wide table (comma-packed dose/activity/SD cells) to tidy long.

    Each source row packs its eight doses, eight activities, and eight SDs as
    comma-separated values inside single quoted cells; this splits and aligns
    them (dropping any row whose three lists disagree in length), producing one
    tidy row per concentration.
    """
    import pandas as pd

    columns = list(frame.columns)
    records: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False, name=None):
        cells = dict(zip(columns, row))
        try:
            doses = [float(v) for v in str(cells["Doses (uM)"]).split(",")]
            activity = [float(v) for v in str(cells["Activity Data (median)"]).split(",")]
            sds = [float(v) for v in str(cells["Activity SD"]).split(",")]
        except (KeyError, ValueError):
            continue
        if not len(doses) == len(activity) == len(sds) or not doses:
            continue
        for conc, act, sd in zip(doses, activity, sds):
            records.append(
                {
                    "cell_line": cells.get("Primary Cell Line Name"),
                    "compound": cells.get("Compound"),
                    "target": cells.get("Target"),
                    "concentration": conc,
                    "activity": act,
                    "activity_sd": sd,
                    "ec50_pub": _to_float(cells.get("EC50 (uM)")),
                    "ic50_pub": _to_float(cells.get("IC50 (uM)")),
                    "amax_pub": _to_float(cells.get("Amax")),
                    "actarea_pub": _to_float(cells.get("ActArea")),
                    "fit_type": cells.get("FitType"),
                    "n_doses": len(doses),
                }
            )
    return pd.DataFrame.from_records(records, columns=_CCLE_COLUMNS)


def _load_ccle_upstream(cache_dir: Path, url: str | None) -> LoadedDataset:
    """Download the CCLE pharmacological-profiling CSV and explode it to long form."""
    import pandas as pd

    cache_dir.mkdir(parents=True, exist_ok=True)
    url = url or _CCLE_UPSTREAM
    dest = cache_dir / "ccle_drug_data.csv"
    if not dest.exists():
        urllib.request.urlretrieve(url, dest)  # noqa: S310 -- https source
    frame = _explode_ccle(pd.read_csv(dest))
    n_series = frame[["compound", "cell_line"]].drop_duplicates().shape[0]
    provenance = (
        f"real: CCLE pharmacological profiling from the Broad legacy release "
        f"({url}), cached at {dest}; {_CCLE_LICENSE}; exploded to {frame.shape[0]} "
        f"per-concentration rows across {n_series} compound x cell-line series "
        "(raw activity + per-dose SD + CCLE's own fitted summary)."
    )
    return LoadedDataset(payload=frame, source="real", provenance=provenance, key=_CCLE_KEY)


def _load_ccle_mirror(cache_dir: Path, url: str | None) -> LoadedDataset:
    """Download the CCLE CSV from the course data mirror and explode it to long form."""
    import pandas as pd

    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / "ccle_drug_data.csv"
    if not dest.exists():
        urllib.request.urlretrieve(_CCLE_MIRROR, dest)  # noqa: S310 -- https source
    frame = _explode_ccle(pd.read_csv(dest))
    n_series = frame[["compound", "cell_line"]].drop_duplicates().shape[0]
    provenance = (
        f"real: CCLE pharmacological profiling from the course mirror ({_CCLE_MIRROR}), "
        f"cached at {dest}; {_CCLE_LICENSE}; exploded to {frame.shape[0]} "
        f"per-concentration rows across {n_series} compound x cell-line series."
    )
    return LoadedDataset(payload=frame, source="real", provenance=provenance, key=_CCLE_KEY)


def _ccle_fallback(*, seed: int | None, reason: str) -> LoadedDataset:
    """Deterministic synthetic dose-response table on the CCLE long schema.

    Builds a few named compound x cell-line series -- including the one the Week-2
    notebook selects -- as eight-point decreasing Hill activity curves (0 = DMSO
    control, negative = growth inhibition) with per-dose replicate SDs, plus a
    self-consistent published-fit summary (``ec50_pub`` etc. from the generating
    truth). The series identities match the real data so a ``(compound,
    cell_line)`` selection resolves the same way offline.
    """
    import pandas as pd

    from ddm4bio.methods.fitting import hill

    rng = np.random.default_rng(seed)
    doses = np.array([0.0025, 0.008, 0.025, 0.08, 0.25, 0.8, 2.53, 8.0])
    # (compound, cell_line, target, activity@0, activity@inf (Amax), ec50, hill_n)
    specs = [
        ("Panobinostat", "NCI-H2023", "HDAC", 15.0, -115.0, 0.084, 1.6),
        ("TAE684", "Hs 739.T", "ALK", -5.0, -123.0, 2.26, 3.4),
        ("Panobinostat", "SK-MEL-24", "HDAC", 10.0, -114.0, 0.105, 1.7),
    ]
    records: list[dict[str, Any]] = []
    for compound, cell_line, target, act0, amax, ec50, hill_n in specs:
        clean = hill(doses, act0, amax, ec50, hill_n)  # decreasing activity curve
        sd = np.clip(np.abs(rng.normal(0.0, 4.0, doses.size)), 0.05, None)
        activity = clean + rng.normal(0.0, sd)
        # IC50 = concentration where activity crosses -50 (CCLE's definition).
        frac = (-50.0 - act0) / (amax - act0)
        ic50 = (
            float(ec50 * (frac / (1.0 - frac)) ** (1.0 / hill_n))
            if 0.0 < frac < 1.0
            else float("nan")
        )
        # ActArea proxy: mean fractional inhibition over the (log-)dose ladder.
        actarea = float(np.mean(np.clip(-clean, 0.0, None)) / 100.0)
        for conc, act, s in zip(doses, activity, sd):
            records.append(
                {
                    "cell_line": cell_line,
                    "compound": compound,
                    "target": target,
                    "concentration": float(conc),
                    "activity": round(float(act), 3),
                    "activity_sd": round(float(s), 3),
                    "ec50_pub": round(ec50, 4),
                    "ic50_pub": round(ic50, 4),
                    "amax_pub": round(amax, 1),
                    "actarea_pub": round(actarea, 3),
                    "fit_type": "synthetic",
                    "n_doses": int(doses.size),
                }
            )
    frame = pd.DataFrame.from_records(records, columns=_CCLE_COLUMNS)
    provenance = (
        f"synthetic/bundled fallback: {reason}. {len(specs)} compound x cell-line "
        "dose-response series on the CCLE long schema (eight-point decreasing Hill "
        "activity + per-dose SD; 0 = DMSO control, negative = growth inhibition; "
        "the same series identities as the real data); seeded, deterministic."
    )
    return LoadedDataset(payload=frame, source="fallback", provenance=provenance, key=_CCLE_KEY)
