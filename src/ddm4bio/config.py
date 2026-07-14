"""Global configuration: paths, cache directory, and the RNG seed registry.

This module must import cleanly with only numpy present. ``seed_everything``
seeds ``random`` and ``numpy`` unconditionally and ``torch`` only if it is
installed (lazy import), so determinism hooks never force a heavy dependency.
"""

import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

GLOBAL_SEED = 20260714

RAW_DIR = Path("data/raw")
CACHE_DIR = Path(os.environ.get("DDM4BIO_CACHE", "data/cache"))


@dataclass(frozen=True)
class Paths:
    """Canonical filesystem locations for the project.

    Notebooks and loaders should read paths from a ``Paths`` instance rather
    than hardcoding strings.
    """

    raw: Path = RAW_DIR
    cache: Path = CACHE_DIR


def seed_everything(seed: int = GLOBAL_SEED) -> int:
    """Seed all RNGs the course relies on for reproducibility.

    Seeds Python's ``random`` and ``numpy`` always, and ``torch`` if it is
    importable. Call this at the top of every notebook and stochastic test.

    Parameters
    ----------
    seed : int, default ``GLOBAL_SEED``
        The seed applied to every RNG.

    Returns
    -------
    int
        The seed that was applied (handy for logging).
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass
    return seed
