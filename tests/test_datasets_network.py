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
