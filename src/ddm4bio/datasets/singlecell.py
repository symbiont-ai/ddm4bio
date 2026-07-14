"""Single-cell transcriptomics datasets.

Dataset: PBMC 3k (10x Genomics peripheral blood mononuclear cells).
Tier: open.
License: 10x Genomics public data (freely redistributable via scanpy).
Install: ``pip install scanpy anndata``.
"""

from __future__ import annotations

from typing import Any


def load_pbmc3k() -> Any:
    """Load the PBMC 3k single-cell RNA-seq dataset.

    Returns
    -------
    anndata.AnnData
        AnnData object with shape ``(n_cells, n_genes)``; ``.X`` holds the
        raw count matrix, ``.obs`` and ``.var`` hold cell/gene annotations.

    Notes
    -----
    Uses ``scanpy.datasets.pbmc3k`` (imported inside the body).
    """
    raise NotImplementedError(
        "load_pbmc3k is a Phase 0 stub; expected return is an AnnData of shape (n_cells, n_genes)."
    )
