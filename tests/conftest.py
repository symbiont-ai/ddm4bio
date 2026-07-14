"""Pytest configuration for the data-layer network marker.

Tests marked ``@pytest.mark.network`` perform real downloads and are SKIPPED by
default; pass ``--run-network`` to opt in. The ``network`` marker is also
registered here so the suite is self-contained and warning-free even if the
project's ``pyproject.toml`` marker registration has not yet landed.
"""

from __future__ import annotations

import pytest


def pytest_addoption(parser):
    """Add ``--run-network`` to opt into the network-marked tests."""
    parser.addoption(
        "--run-network",
        action="store_true",
        default=False,
        help="run tests marked @pytest.mark.network (perform real downloads)",
    )


def pytest_configure(config):
    """Register the ``network`` marker (idempotent, no pyproject edit needed)."""
    config.addinivalue_line(
        "markers",
        "network: test performs a real network download; skipped unless --run-network is passed.",
    )


def pytest_collection_modifyitems(config, items):
    """Skip every ``network``-marked test unless ``--run-network`` was given."""
    if config.getoption("--run-network"):
        return
    skip_network = pytest.mark.skip(
        reason="needs --run-network to run (performs a real network download)"
    )
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip_network)
