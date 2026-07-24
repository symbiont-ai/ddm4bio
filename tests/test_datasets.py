"""Offline, deterministic tests for the Phase-2 data layer.

These tests exercise every registered loader's FALLBACK path with zero network
access. For each key in :data:`ddm4bio.datasets.DATASET_REGISTRY` we call
``get_dataset`` with ``download=False, prefer_real=False`` and assert a
clearly-labelled fallback comes back. Both flags are required: some loaders
(e.g. ``breast_wisconsin``) treat their "real" payload as a bundled, offline
scikit-learn dataset and honour only ``prefer_real``, so ``download=False``
alone would still yield ``source="real"``. Passing ``prefer_real=False`` too
forces the synthetic/bundled fallback uniformly across all loaders.

Also covered: unknown-key ``KeyError``, registry/spec integrity via
``list_datasets``, and fallback determinism under a fixed seed. All assertions
run fully offline.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from ddm4bio.datasets import (
    BLOODMNIST_LABELS,
    DATASET_REGISTRY,
    LoadedDataset,
    get_dataset,
    list_datasets,
)

_ALL_KEYS = sorted(DATASET_REGISTRY)
_TIERS = {"open", "archive", "credentialed"}


def test_bloodmnist_labels_are_authoritative_and_fallback_carries_none():
    """The 8 BloodMNIST class names live in the loader; the synthetic fallback has none."""
    assert BLOODMNIST_LABELS == (
        "basophil",
        "eosinophil",
        "erythroblast",
        "immature granulocyte",
        "lymphocyte",
        "monocyte",
        "neutrophil",
        "platelet",
    )
    # The offline fallback is synthetic digits, not the labelled cell dataset, so it
    # exposes no cell-type names (callers must gate on ``labels is not None``).
    fallback = get_dataset("bloodmnist", download=False, prefer_real=False)
    assert fallback.source == "fallback"
    assert fallback.labels is None


@pytest.mark.parametrize("key", _ALL_KEYS)
def test_every_loader_falls_back_offline(key):
    """Each loader returns a labelled fallback with download/prefer_real off."""
    loaded = get_dataset(key, download=False, prefer_real=False)

    assert isinstance(loaded, LoadedDataset)
    assert loaded.source == "fallback"
    assert loaded.payload is not None
    assert isinstance(loaded.provenance, str)
    assert loaded.provenance.strip() != ""
    # The loader always stamps a non-empty registry key on the result.
    assert isinstance(loaded.key, str) and loaded.key != ""


def test_unknown_key_raises_keyerror():
    """An unregistered key raises KeyError (message lists available keys)."""
    with pytest.raises(KeyError):
        get_dataset("no_such_key")


def test_list_datasets_matches_registry():
    """list_datasets returns exactly one spec per registry entry."""
    specs = list_datasets()
    assert len(specs) == len(DATASET_REGISTRY)
    assert {spec.key for spec in specs} == set(DATASET_REGISTRY)


@pytest.mark.parametrize("key", _ALL_KEYS)
def test_spec_tier_and_loader_are_well_formed(key):
    """Each spec has a valid tier and an importable ``module:function`` loader."""
    spec = DATASET_REGISTRY[key]
    assert spec.tier in _TIERS

    module_name, func_name = spec.loader.split(":")
    module = importlib.import_module(module_name)
    assert hasattr(module, func_name)
    assert callable(getattr(module, func_name))


def test_pbmc3k_fallback_is_deterministic():
    """The synthetic single-cell fallback is reproducible under a fixed seed."""
    a = get_dataset("pbmc3k", download=False, prefer_real=False, seed=123)
    b = get_dataset("pbmc3k", download=False, prefer_real=False, seed=123)

    assert a.source == b.source == "fallback"
    assert np.array_equal(a.payload["counts"], b.payload["counts"])
    assert np.array_equal(a.payload["labels"], b.payload["labels"])
