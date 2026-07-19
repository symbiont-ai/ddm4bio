"""Tests for the warfarin PK/PD loader.

The synthetic fallback must be deterministic, carry the real dataset's schema
(cp concentration + pca effect for 32 subjects), and show the qualitative PK/PD
shape (concentration rises then falls; the effect declines then recovers). The
real path is exercised best-effort: whichever source is returned, the loader
contract (tidy schema, both dvid channels) must hold.
"""

from __future__ import annotations

import numpy as np
import pytest

from ddm4bio.datasets.pharma import load_ccle, load_warfarin

_COLUMNS = ["id", "time", "amt", "dv", "dvid", "evid", "wt", "age", "sex"]
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


def test_fallback_schema_and_determinism(tmp_path):
    a = load_warfarin(cache_dir=tmp_path, prefer_real=False, seed=0)
    b = load_warfarin(cache_dir=tmp_path, prefer_real=False, seed=0)

    assert a.source == "fallback"
    assert a.key == "warfarin"
    assert list(a.payload.columns) == _COLUMNS
    assert a.payload["id"].nunique() == 32
    assert set(a.payload["dvid"].unique()) == {"cp", "pca"}
    # Deterministic under a fixed seed.
    assert a.payload.equals(b.payload)


def test_fallback_has_pkpd_shape(tmp_path):
    ds = load_warfarin(cache_dir=tmp_path, prefer_real=False, seed=0)
    df = ds.payload
    one = df[df["id"] == 1]

    conc = one[(one["dvid"] == "cp") & (one["evid"] == 0)].sort_values("time")
    peak_idx = int(np.argmax(conc["dv"].to_numpy()))
    # Concentration rises then falls: the peak is interior, not the first/last sample.
    assert 0 < peak_idx < len(conc) - 1

    pca = one[one["dvid"] == "pca"].sort_values("time")["dv"].to_numpy()
    # The anticoagulation effect declines below its ~100% baseline then recovers.
    assert pca.min() < 70.0
    assert pca[-1] > pca.min()


@pytest.mark.network
def test_loader_contract_holds_for_whichever_source(tmp_path):
    # Real if reachable (network + pyreadr), else the fallback -- either way the
    # tidy schema and both measurement channels must be present.
    ds = load_warfarin(cache_dir=tmp_path, prefer_real=True, seed=0)
    assert ds.source in {"real", "fallback"}
    assert list(ds.payload.columns) == _COLUMNS
    assert set(ds.payload["dvid"].unique()) == {"cp", "pca"}
    assert ds.payload["id"].nunique() >= 30


def _panobinostat(frame):
    """The compound x cell-line series the Week-2 notebook fits."""
    return frame[
        (frame["compound"] == "Panobinostat") & (frame["cell_line"] == "NCI-H2023")
    ].sort_values("concentration")


def test_ccle_fallback_schema_and_determinism(tmp_path):
    a = load_ccle(cache_dir=tmp_path, prefer_real=False, seed=0)
    b = load_ccle(cache_dir=tmp_path, prefer_real=False, seed=0)

    assert a.source == "fallback"
    assert a.key == "ccle"
    assert list(a.payload.columns) == _CCLE_COLUMNS
    # The notebook's selected series exists offline, with eight concentrations.
    assert len(_panobinostat(a.payload)) == 8
    # Deterministic under a fixed seed.
    assert a.payload.equals(b.payload)


def test_ccle_fallback_has_dose_response_shape(tmp_path):
    ds = load_ccle(cache_dir=tmp_path, prefer_real=False, seed=0)
    sel = _panobinostat(ds.payload)
    act = sel["activity"].to_numpy()

    # Activity falls from ~DMSO control to strong growth inhibition (negative).
    assert act[0] > act[-1]
    assert act.min() < -50.0
    # Per-dose replicate SDs are positive; the published summary is one value/series.
    assert (sel["activity_sd"].to_numpy() > 0).all()
    assert sel["ec50_pub"].nunique() == 1
    assert sel["amax_pub"].iloc[0] < 0.0


@pytest.mark.network
def test_ccle_loader_contract_holds_for_whichever_source(tmp_path):
    # Real if reachable, else the fallback -- either way the tidy long schema and
    # at least one full eight-point series must be present.
    ds = load_ccle(cache_dir=tmp_path, prefer_real=True, seed=0)
    assert ds.source in {"real", "fallback"}
    assert list(ds.payload.columns) == _CCLE_COLUMNS
    assert ds.payload.groupby(["compound", "cell_line"]).size().max() >= 8
