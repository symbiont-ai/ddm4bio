"""Lightweight timing utilities."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def timed(label: str) -> Iterator[None]:
    """Context manager that times its body and prints the elapsed seconds.

    Parameters
    ----------
    label:
        Human-readable name printed alongside the elapsed wall-clock time.

    Yields
    ------
    None
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        print(f"[{label}] {elapsed:.6f}s")
