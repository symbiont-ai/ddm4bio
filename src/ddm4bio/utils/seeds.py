"""Determinism and seeding utilities.

Honesty-first helpers to make stochastic code reproducible and to verify
that a function actually behaves deterministically under a fixed seed.
"""

from __future__ import annotations

import contextlib
import random
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np


@contextlib.contextmanager
def with_seed(seed: int) -> Iterator[None]:
    """Temporarily set the numpy and stdlib ``random`` seeds.

    Saves the current ``random`` and ``numpy`` RNG states on entry, seeds both
    generators with ``seed``, yields, and restores the original states on exit
    so surrounding code is not affected.

    Parameters
    ----------
    seed:
        Integer seed applied to ``random.seed`` and ``numpy.random.seed``.

    Yields
    ------
    None
    """
    py_state = random.getstate()
    np_state = np.random.get_state()
    try:
        random.seed(seed)
        np.random.seed(seed)
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)


def _outputs_equal(first: Any, second: Any) -> bool:
    """Return ``True`` when two outputs are exactly equal.

    Handles scalars, numpy arrays, and (possibly nested) tuples/lists of
    array-likes by comparing elementwise with :func:`numpy.array_equal`.
    """
    if isinstance(first, (tuple, list)):
        if not isinstance(second, (tuple, list)) or len(first) != len(second):
            return False
        return all(_outputs_equal(a, b) for a, b in zip(first, second, strict=True))
    return bool(np.array_equal(first, second))


def check_determinism(
    fn: Callable[..., Any],
    seed: int | None = None,
    *args: Any,
    **kwargs: Any,
) -> bool:
    """Run ``fn`` twice and assert its outputs are identical.

    Invokes ``fn(*args, **kwargs)`` twice and compares the two results for
    exact equality. When ``seed`` is not ``None`` each call is wrapped in a
    fresh :func:`with_seed` context so both runs start from the same RNG
    state. Comparison uses :func:`numpy.array_equal` elementwise and handles
    scalars, numpy arrays, and (possibly nested) tuples/lists of arrays.

    Usable directly as a test assertion helper: it returns ``True`` when the
    two runs match and raises ``AssertionError`` when they differ.

    Parameters
    ----------
    fn:
        Callable whose output should be reproducible.
    seed:
        Seed used for both runs. If ``None``, ``fn`` is called without any
        seeding context (useful for functions that manage their own seeding).
    *args:
        Positional arguments forwarded to ``fn``.
    **kwargs:
        Keyword arguments forwarded to ``fn``.

    Returns
    -------
    bool
        ``True`` if the two runs produce identical output.

    Raises
    ------
    AssertionError
        If the two runs produce different output.
    """
    if seed is None:
        first = fn(*args, **kwargs)
        second = fn(*args, **kwargs)
    else:
        with with_seed(seed):
            first = fn(*args, **kwargs)
        with with_seed(seed):
            second = fn(*args, **kwargs)

    if not _outputs_equal(first, second):
        raise AssertionError(
            f"check_determinism: {getattr(fn, '__name__', repr(fn))} produced "
            f"different outputs across two runs (seed={seed}): "
            f"{first!r} != {second!r}"
        )
    return True
