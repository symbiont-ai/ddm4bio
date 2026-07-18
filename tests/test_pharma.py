"""Tests for the warfarin PK/PD loader.

The synthetic fallback must be deterministic, carry the real dataset's schema
(cp concentration + pca effect for 32 subjects), and show the qualitative PK/PD
shape (concentration rises then falls; the effect declines then recovers). The
real path is exercised best-effort: whichever source is returned, the loader
contract (tidy schema, both dvid channels) must hold.
"""

from __future__ import annotations

import numpy as np

from ddm4bio.datasets.pharma import load_warfarin

_COLUMNS = ["id", "time", "amt", "dv", "dvid", "evid", "wt", "age", "sex"]


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


def test_loader_contract_holds_for_whichever_source(tmp_path):
    # Real if reachable (network + pyreadr), else the fallback -- either way the
    # tidy schema and both measurement channels must be present.
    ds = load_warfarin(cache_dir=tmp_path, prefer_real=True, seed=0)
    assert ds.source in {"real", "fallback"}
    assert list(ds.payload.columns) == _COLUMNS
    assert set(ds.payload["dvid"].unique()) == {"cp", "pca"}
    assert ds.payload["id"].nunique() >= 30
