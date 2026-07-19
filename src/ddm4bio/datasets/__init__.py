"""Dataset registry and loader dispatch.

``DATASET_REGISTRY`` maps a short key to a dataset specification; ``get_dataset``
resolves a key to a cached, loaded dataset. Individual loaders live in the
sibling modules and are resolved lazily from the registry (they are not imported
here, to avoid import-order fragility and heavy top-level imports). The package
imports cleanly on numpy alone -- ``importlib`` and ``dataclasses`` are stdlib.
"""

from .medmnist_images import BLOODMNIST_LABELS
from .registry import (
    DATASET_REGISTRY,
    DatasetSpec,
    LoadedDataset,
    get_dataset,
    list_datasets,
)

__all__ = [
    "BLOODMNIST_LABELS",
    "DATASET_REGISTRY",
    "DatasetSpec",
    "LoadedDataset",
    "get_dataset",
    "list_datasets",
]
