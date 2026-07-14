"""Dataset registry and loader dispatch.

``DATASET_REGISTRY`` maps a short key to a dataset specification; ``get_dataset``
resolves a key to a cached, loaded dataset. Individual loaders live in the
sibling modules and are registered in Phase 2 (they are not imported here, to
avoid import-order fragility and heavy top-level imports).
"""

DATASET_REGISTRY: dict = {}


def get_dataset(key: str, **opts):
    """Return the dataset registered under ``key`` (download-to-cache as needed).

    Parameters
    ----------
    key : str
        Short registry key, e.g. ``"pbmc3k"``.
    **opts
        Loader-specific options.

    Returns
    -------
    object
        The loaded dataset object (AnnData / ndarray / DataFrame / mne / wfdb).
    """
    raise NotImplementedError
