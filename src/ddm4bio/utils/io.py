"""I/O and data-utility guards for ddm4bio.

This module provides small, dependency-light helpers for caching dataset
artifacts to a local path and for enforcing the project policy that
credentialed-tier datasets are never auto-downloaded.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


def safe_cache_path(key: str, cache_dir: str | Path | None = None) -> Path:
    """Return a filesystem-safe local cache path for a dataset ``key``.

    Parameters
    ----------
    key:
        An opaque dataset identifier (e.g. ``"medmnist/bloodmnist"`` or a URL).
        The key is sanitized into a safe relative path: any character outside
        ``[A-Za-z0-9._-]`` (including path separators) is replaced with ``_``,
        and ``..`` traversal segments are neutralized, so the result can never
        escape the cache root.
    cache_dir:
        Optional cache root. Defaults to ``CACHE_DIR`` from ``ddm4bio.config``.

    Returns
    -------
    pathlib.Path
        Absolute path to the cache location for ``key``. The parent directory
        is expected to be created lazily by the caller or implementation.
    """
    from ddm4bio.config import CACHE_DIR

    root = Path(cache_dir) if cache_dir is not None else Path(CACHE_DIR)

    # Strip any scheme (e.g. "https://") so it does not leak into the path.
    cleaned = re.sub(r"^[A-Za-z][A-Za-z0-9+.-]*://", "", str(key))

    # Split on any path-like separator and sanitize each segment.
    segments = re.split(r"[\\/]+", cleaned)
    safe_segments = []
    for seg in segments:
        seg = re.sub(r"[^A-Za-z0-9._-]", "_", seg)
        # Neutralize traversal and empty/dot-only segments.
        if seg in ("", ".", "..") or set(seg) == {"."}:
            continue
        safe_segments.append(seg)

    if not safe_segments:
        safe_segments = ["_"]

    return (root.joinpath(*safe_segments)).resolve()


def require_open_tier(spec: Any) -> None:
    """Guard that rejects credentialed-tier dataset specs.

    Enforces the policy: *never auto-download credentialed data*. Inspects the
    dataset ``spec`` for a tier indicator and raises if it is credentialed.

    Parameters
    ----------
    spec:
        A dataset spec object or mapping. The tier is read from a ``tier``
        attribute, or from a ``"tier"`` key if ``spec`` is a mapping. A tier of
        ``"credentialed"`` (case-insensitive) is disallowed; ``"open"`` and
        ``"archive"`` (or any non-credentialed value) pass and return ``None``.

    Returns
    -------
    None
        For open/archive (non-credentialed) tiers.

    Raises
    ------
    PermissionError
        If the resolved tier is ``"credentialed"``.
    """
    tier = getattr(spec, "tier", None)
    if tier is None and isinstance(spec, dict):
        tier = spec.get("tier")

    if isinstance(tier, str) and tier.strip().lower() == "credentialed":
        raise PermissionError(
            "Dataset tier is 'credentialed'; auto-download is prohibited. "
            "Use an open/synthetic fallback instead, or obtain the credentialed "
            "data manually as an authorized user."
        )
    return None
