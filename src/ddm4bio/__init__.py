"""ddm4bio: Data-Driven Methods for the Life Sciences.

Shared teaching library for the course. Public entry points are re-exported
here; heavy optional dependencies (torch, scanpy, mne, ...) are imported lazily
inside the functions that use them, so importing this package requires only the
core numeric stack (numpy/scipy/pandas/scikit-learn).
"""

from .config import GLOBAL_SEED, Paths, seed_everything

__version__ = "0.1.0"

__all__ = ["GLOBAL_SEED", "Paths", "seed_everything", "__version__"]
