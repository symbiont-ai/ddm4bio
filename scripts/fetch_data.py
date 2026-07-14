"""Warm the dataset cache for every non-credentialed dataset in the registry.

Iterates over :data:`ddm4bio.datasets.DATASET_REGISTRY` and calls
:func:`ddm4bio.datasets.get_dataset` for each ``open`` / ``archive`` key,
triggering a real fetch (cached under the configured cache directory) or a
graceful synthetic/bundled fallback. ``credentialed`` keys are skipped with a
note, since they require a data-use agreement and must never be auto-downloaded.

Run via ``make data`` or ``python scripts/fetch_data.py``.
"""

from __future__ import annotations

import ddm4bio  # noqa: F401 -- ensures the package imports cleanly first
from ddm4bio.datasets import DATASET_REGISTRY, get_dataset


def _warm_key(key: str) -> None:
    """Fetch one key and print its resolved source and provenance."""
    try:
        loaded = get_dataset(key)
    except Exception as exc:  # pragma: no cover - defensive, loaders should not raise
        print(f"[ERROR ] {key}: {type(exc).__name__}: {exc}")
        return
    print(f"[{loaded.source:>8}] {key}: {loaded.provenance}")


def main() -> None:
    """Warm the cache for all non-credentialed datasets; skip credentialed ones."""
    print(f"Warming dataset cache for {len(DATASET_REGISTRY)} registered datasets\n")
    for key, spec in DATASET_REGISTRY.items():
        if spec.tier == "credentialed":
            print(
                f"[  skip  ] {key}: credentialed tier "
                f"({spec.name}) -- apply for access at {spec.url}; "
                "not auto-downloaded"
            )
            continue
        _warm_key(key)
    print("\nDone. Real fetches are cached; fallbacks are synthetic/bundled.")


if __name__ == "__main__":
    main()
