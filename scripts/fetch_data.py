"""Warm the dataset cache for the registry (or a specific set of keys).

By default warms every non-credentialed dataset (``make data``). Pass explicit
KEYS to warm only those -- the docs-deploy workflow warms just the lesson
datasets this way. With ``--require-real`` the run FAILS (exit 1) if any
requested key resolves to a synthetic/bundled fallback after its retries, so a
transient source outage is caught at warm time -- before the expensive book
build -- instead of after it (where the post-build guard would trip). Real
fetches are cached, so a retry re-attempts the network only while the real source
is still unavailable.

Examples
--------
    python scripts/fetch_data.py                          # warm all; never fail
    python scripts/fetch_data.py mitbih pbmc3k            # warm just these
    python scripts/fetch_data.py --require-real --retries 3 --delay 15 mitbih
"""

from __future__ import annotations

import argparse
import sys
import time

import ddm4bio  # noqa: F401 -- ensures the package imports cleanly first
from ddm4bio.datasets import DATASET_REGISTRY, get_dataset


def warm_key(key: str, *, retries: int = 0, delay: float = 10.0, download: bool = True) -> str:
    """Fetch one key, retrying on fallback; return ``"real"``/``"fallback"``/``"error"``."""
    loaded = None
    for attempt in range(retries + 1):
        try:
            loaded = get_dataset(key, download=download)
        except Exception as exc:  # noqa: BLE001 -- report and keep the sweep going
            print(f"[ ERROR ] {key}: {type(exc).__name__}: {exc}")
            return "error"
        if loaded.source == "real":
            print(f"[  real  ] {key}: {loaded.provenance}")
            return "real"
        if attempt < retries:
            print(
                f"[fallback] {key}: attempt {attempt + 1}/{retries + 1} fell back; "
                f"retrying in {delay:.0f}s"
            )
            time.sleep(delay)
    print(f"[fallback] {key}: {loaded.provenance}")
    return "fallback"


def main(argv: list[str] | None = None) -> int:
    """Warm the requested keys; return non-zero on unmet ``--require-real``."""
    parser = argparse.ArgumentParser(description="Warm the ddm4bio dataset cache.")
    parser.add_argument("keys", nargs="*", help="keys to warm (default: all non-credentialed)")
    parser.add_argument(
        "--require-real", action="store_true", help="exit 1 if any key resolves to a fallback"
    )
    parser.add_argument("--retries", type=int, default=0, help="retry attempts on fallback")
    parser.add_argument("--delay", type=float, default=10.0, help="seconds between retries")
    parser.add_argument(
        "--offline", action="store_true", help="pass download=False (forces fallbacks; for testing)"
    )
    args = parser.parse_args(argv)

    keys = args.keys or list(DATASET_REGISTRY)
    download = not args.offline
    print(
        f"Warming {len(keys)} dataset(s); require_real={args.require_real}, "
        f"retries={args.retries}, download={download}\n"
    )

    failures = []
    for key in keys:
        if key not in DATASET_REGISTRY:
            print(f"[ ERROR ] {key}: unknown key")
            failures.append(key)
            continue
        spec = DATASET_REGISTRY[key]
        if spec.tier == "credentialed":
            print(f"[  skip  ] {key}: credentialed ({spec.name}); not auto-downloaded")
            continue
        source = warm_key(key, retries=args.retries, delay=args.delay, download=download)
        if args.require_real and source != "real":
            failures.append(key)

    if args.require_real and failures:
        print(f"\nFAILED: {len(failures)} dataset(s) did not fetch real: {', '.join(failures)}")
        print(
            "A real source was unavailable (most likely a transient outage). Re-run the job; "
            "if it persists, the source or a heavy dependency needs attention."
        )
        return 1
    print("\nDone. Real fetches are cached; fallbacks are synthetic/bundled.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
