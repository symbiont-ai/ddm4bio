"""Smoke test: package imports and basic runtime hooks work."""

import ddm4bio


def test_version_is_str():
    assert isinstance(ddm4bio.__version__, str)


def test_seed_everything_runs():
    ddm4bio.seed_everything()
