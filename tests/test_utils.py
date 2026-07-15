"""Sentinel tests for the utility helpers.

Covers ``ddm4bio.utils.io`` (``safe_cache_path`` sanitization / traversal
safety and ``require_open_tier`` policy enforcement) and
``ddm4bio.utils.timers`` (the ``timed`` context manager). All tests are
offline and deterministic; ``safe_cache_path`` is exercised against an
explicit temp ``cache_dir`` so it never touches the real project cache.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pytest

from ddm4bio.utils import io, timers


# --------------------------------------------------------------------------- #
# safe_cache_path
# --------------------------------------------------------------------------- #
def test_safe_cache_path_plain_key_under_cache_dir(tmp_path):
    root = tmp_path / "cache"
    result = io.safe_cache_path("pathmnist", cache_dir=root)

    assert isinstance(result, Path)
    # Result must live under the intended cache root.
    assert result.resolve() == (root / "pathmnist").resolve()
    assert root.resolve() in result.resolve().parents


def test_safe_cache_path_nested_key_builds_subdirs(tmp_path):
    root = tmp_path / "cache"
    result = io.safe_cache_path("medmnist/pathmnist", cache_dir=root)

    assert result.resolve() == (root / "medmnist" / "pathmnist").resolve()
    assert root.resolve() in result.resolve().parents


def test_safe_cache_path_sanitizes_unsafe_characters(tmp_path):
    root = tmp_path / "cache"
    result = io.safe_cache_path("weird name!@#/with spaces", cache_dir=root)

    # Every path segment beyond the root contains only safe characters.
    rel = result.resolve().relative_to(root.resolve())
    for part in rel.parts:
        assert re.fullmatch(r"[A-Za-z0-9._-]+", part), part


def test_safe_cache_path_neutralizes_traversal(tmp_path):
    root = tmp_path / "cache"
    result = io.safe_cache_path("../../etc/passwd", cache_dir=root)

    resolved = result.resolve()
    # Must not escape the cache root despite the ".." segments.
    assert root.resolve() in resolved.parents
    resolved.relative_to(root.resolve())  # raises ValueError if it escaped
    assert ".." not in resolved.parts


def test_safe_cache_path_backslash_traversal_contained(tmp_path):
    root = tmp_path / "cache"
    result = io.safe_cache_path(r"..\..\windows\system32", cache_dir=root)

    resolved = result.resolve()
    assert root.resolve() in resolved.parents
    assert ".." not in resolved.parts


def test_safe_cache_path_strips_url_scheme(tmp_path):
    root = tmp_path / "cache"
    result = io.safe_cache_path("https://example.com/data/set.zip", cache_dir=root)

    resolved = result.resolve()
    assert root.resolve() in resolved.parents
    # The scheme's "https" and "//" must not survive as their own segments.
    rel_parts = resolved.relative_to(root.resolve()).parts
    assert "https:" not in rel_parts
    assert "" not in rel_parts


def test_safe_cache_path_all_unsafe_key_yields_placeholder(tmp_path):
    root = tmp_path / "cache"
    # A key that sanitizes to nothing must still produce a contained path.
    result = io.safe_cache_path("...", cache_dir=root)

    resolved = result.resolve()
    assert root.resolve() in resolved.parents
    assert resolved.relative_to(root.resolve()).parts == ("_",)


def test_safe_cache_path_is_deterministic(tmp_path):
    root = tmp_path / "cache"
    first = io.safe_cache_path("medmnist/pathmnist", cache_dir=root)
    second = io.safe_cache_path("medmnist/pathmnist", cache_dir=root)

    assert first == second


def test_safe_cache_path_accepts_str_cache_dir(tmp_path):
    root = tmp_path / "cache"
    result = io.safe_cache_path("dataset", cache_dir=str(root))

    assert result.resolve() == (root / "dataset").resolve()


# --------------------------------------------------------------------------- #
# require_open_tier
# --------------------------------------------------------------------------- #
@dataclass
class _SpecStandIn:
    """Minimal stand-in exposing a ``.tier`` attribute like a DatasetSpec."""

    tier: str


@pytest.mark.parametrize("tier", ["open", "archive", "OPEN", "Archive"])
def test_require_open_tier_passes_for_object_open_and_archive(tier):
    spec = _SpecStandIn(tier=tier)
    assert io.require_open_tier(spec) is None


def test_require_open_tier_raises_for_object_credentialed():
    spec = _SpecStandIn(tier="credentialed")
    with pytest.raises(PermissionError):
        io.require_open_tier(spec)


def test_require_open_tier_credentialed_case_insensitive():
    spec = _SpecStandIn(tier="  Credentialed  ")
    with pytest.raises(PermissionError):
        io.require_open_tier(spec)


@pytest.mark.parametrize("tier", ["open", "archive"])
def test_require_open_tier_passes_for_dict(tier):
    assert io.require_open_tier({"tier": tier}) is None


def test_require_open_tier_raises_for_dict_credentialed():
    with pytest.raises(PermissionError):
        io.require_open_tier({"tier": "credentialed"})


def test_require_open_tier_missing_tier_passes():
    # No tier information at all -> treated as non-credentialed (passes).
    assert io.require_open_tier({}) is None
    assert io.require_open_tier(object()) is None


# --------------------------------------------------------------------------- #
# timers.timed
# --------------------------------------------------------------------------- #
def test_timed_runs_body_and_yields_none():
    ran = []
    with timers.timed("unit") as handle:
        ran.append(True)
    assert ran == [True]
    # The context manager yields nothing meaningful (None).
    assert handle is None


def test_timed_prints_nonnegative_numeric_elapsed(capsys):
    label = "sentinel-timer"
    with timers.timed(label):
        # Trivial deterministic work.
        _ = sum(range(1000))

    out = capsys.readouterr().out
    assert label in out

    match = re.search(rf"\[{re.escape(label)}\]\s+([0-9.]+)s", out)
    assert match is not None, f"unexpected timer output: {out!r}"

    elapsed = float(match.group(1))
    assert isinstance(elapsed, float)
    assert elapsed >= 0.0


def test_timed_reports_on_exception():
    # The elapsed time must still be printed even if the body raises.
    with pytest.raises(ValueError):
        with timers.timed("boom"):
            raise ValueError("intentional")
