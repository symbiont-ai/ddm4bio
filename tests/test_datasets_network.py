"""Network-only tests for the Phase-2 data layer (SKIPPED by default).

Every test here is marked ``@pytest.mark.network`` and is skipped unless pytest
is invoked with ``--run-network`` (see ``conftest.py``). When enabled, each test
fetches the REAL dataset from its source and asserts ``source == "real"`` with a
sane payload. These are the only tests in the suite that touch the network.
"""

from __future__ import annotations

import pytest

from ddm4bio.datasets import LoadedDataset, get_dataset

pytestmark = pytest.mark.network


def test_jhu_covid_real_download():
    """Real JHU CSSE COVID-19 series is a tidy (date, cases) frame."""
    loaded = get_dataset("jhu_covid")

    assert isinstance(loaded, LoadedDataset)
    assert loaded.source == "real"
    payload = loaded.payload
    assert set(("date", "cases")).issubset(set(payload.columns))
    assert len(payload) > 0
    assert int(payload["cases"].max()) > 0


def test_bloodmnist_real_download():
    """Real BloodMNIST payload carries the MedMNIST train/val/test arrays."""
    loaded = get_dataset("bloodmnist")

    assert isinstance(loaded, LoadedDataset)
    assert loaded.source == "real"
    payload = loaded.payload
    for field in ("train_images", "train_labels", "test_images", "test_labels"):
        assert field in payload
    images = payload["train_images"]
    assert images.ndim == 4  # (N, H, W, C)
    assert images.shape[0] > 0


def test_heart_uci_real_download():
    """Real UCI Cleveland heart data exposes an X frame and binary y."""
    loaded = get_dataset("heart_uci")

    assert isinstance(loaded, LoadedDataset)
    assert loaded.source == "real"
    payload = loaded.payload
    assert "X" in payload and "y" in payload
    assert len(payload["X"]) > 0
    assert len(payload["X"]) == len(payload["y"])
    assert set(payload["y"].unique()).issubset({0, 1})


def test_bloodmnist_mirror_when_upstream_down(tmp_path, monkeypatch):
    """With Zenodo unreachable, BloodMNIST still loads REAL data from the course mirror.

    Points the upstream base at a dead host and uses a fresh cache, so the loader must
    fall through to the GitHub-Release mirror. The result must be real (guard-safe: no
    ``fallback``/``synthetic`` in provenance) with the true BloodMNIST shape.
    """
    import ddm4bio.datasets.medmnist_images as mm

    monkeypatch.setattr(mm, "_ZENODO_BASE", "https://zenodo.invalid.nonexistent/files")
    loaded = mm.load_medmnist(cache_dir=tmp_path, key="bloodmnist", seed=0)

    assert isinstance(loaded, LoadedDataset)
    assert loaded.source == "real"
    assert "course mirror" in loaded.provenance
    assert "fallback" not in loaded.provenance and "synthetic" not in loaded.provenance
    images = loaded.payload["train_images"]
    assert images.ndim == 4 and images.shape[1:] == (28, 28, 3)


def test_mitbih_mirror_when_upstream_down(tmp_path, monkeypatch):
    """With PhysioNet unreachable, MIT-BIH still loads REAL data from the course mirror.

    Forces ``wfdb.dl_database`` to raise (as a PhysioNet outage would) and uses a fresh
    cache, so the loader must fetch the record files from the GitHub-Release mirror and
    read them with ``rdrecord``. The result must be the real record at fs=360 Hz.
    """
    wfdb = pytest.importorskip("wfdb")
    import ddm4bio.datasets.physio as ph

    def _boom(*args, **kwargs):
        raise RuntimeError("PhysioNet unreachable (simulated)")

    monkeypatch.setattr(wfdb, "dl_database", _boom)
    loaded = ph.load_mitbih(cache_dir=tmp_path, record="100", seed=0)

    assert isinstance(loaded, LoadedDataset)
    assert loaded.source == "real"
    assert "course mirror" in loaded.provenance
    assert "fallback" not in loaded.provenance and "synthetic" not in loaded.provenance
    assert loaded.payload["fs"] == 360.0
    assert loaded.payload["signal"].shape[0] > 100_000
