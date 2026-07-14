"""Single-cell transcriptomics datasets.

Dataset: PBMC 3k (10x Genomics peripheral blood mononuclear cells).
Tier: open.
License: 10x Genomics public data (freely redistributable via scanpy).
Install: ``pip install scanpy anndata`` (or ``scipy pandas`` for the manual
tarball path).

Loader contract (see :mod:`ddm4bio.datasets.registry`): ``load_pbmc3k`` tries the
real 10x PBMC3k matrix first (via :mod:`scanpy`, then via a direct tarball
download parsed with :mod:`scipy`/:mod:`pandas`) and caches the raw download
under ``cache_dir``; on any failure -- or when ``download``/``prefer_real`` is
False -- it returns a deterministic synthetic single-cell fallback with a few
latent cell-type clusters. Heavy libraries are imported inside the function
bodies so importing this module needs only numpy.
"""

from __future__ import annotations

import tarfile
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

from ddm4bio.datasets.registry import LoadedDataset

_KEY = "pbmc3k"

# 10x Genomics public PBMC3k filtered gene-barcode matrices tarball.
_PBMC3K_URL = (
    "https://cf.10xgenomics.com/samples/cell/pbmc3k/pbmc3k_filtered_gene_bc_matrices.tar.gz"
)
_LICENSE = "10x Genomics public data / Creative Commons (freely redistributable)"


def load_pbmc3k(
    *,
    cache_dir: Path | str,
    download: bool = True,
    prefer_real: bool = True,
    seed: int | None = None,
    **opts: Any,
) -> LoadedDataset:
    """Load the PBMC 3k single-cell RNA-seq dataset (real, with fallback).

    Parameters
    ----------
    cache_dir:
        Directory for cached raw downloads. Reused idempotently.
    download:
        When True, may fetch the real 10x matrix and cache it.
    prefer_real:
        When True, prefer the real dataset over the synthetic fallback.
    seed:
        RNG seed making the synthetic fallback deterministic.
    **opts:
        Ignored extra options (accepted for a uniform loader signature).

    Returns
    -------
    LoadedDataset
        ``payload`` is an ``anndata.AnnData`` of shape ``(n_cells, n_genes)``
        (real, or manual tarball when anndata is present), a ``dict`` with
        ``counts``/``gene_names``/``gene_ids``/``barcodes`` (manual tarball
        without anndata), or a synthetic ``dict`` with
        ``counts``/``labels``/``gene_names`` (fallback). ``source`` is
        ``"real"`` or ``"fallback"``; ``provenance`` records the origin.
    """
    cache_dir = Path(cache_dir)

    if download and prefer_real:
        # Attempt 1: scanpy's bundled downloader (returns a rich AnnData).
        try:
            return _load_pbmc3k_scanpy(cache_dir)
        except Exception:  # noqa: BLE001 -- any failure must fall through
            pass
        # Attempt 2: direct 10x tarball download parsed by scipy/pandas.
        try:
            return _load_pbmc3k_tarball(cache_dir)
        except Exception:  # noqa: BLE001 -- fall back to synthetic
            reason = "real 10x PBMC3k fetch failed (no network / missing deps)"
    elif not download:
        reason = "download=False requested"
    else:
        reason = "prefer_real=False requested"

    return _pbmc3k_fallback(seed=seed, reason=reason)


def _load_pbmc3k_scanpy(cache_dir: Path) -> LoadedDataset:
    """Load PBMC3k via ``scanpy.datasets.pbmc3k`` cached under ``cache_dir``."""
    import scanpy as sc

    cache_dir.mkdir(parents=True, exist_ok=True)
    # Point scanpy's downloader at our cache so the fetch is idempotent.
    sc.settings.datasetdir = str(cache_dir)
    adata = sc.datasets.pbmc3k()
    provenance = (
        f"real: scanpy.datasets.pbmc3k() cached under {cache_dir} from "
        f"{_PBMC3K_URL}; license: {_LICENSE}; 2,700 PBMCs x ~32,738 genes, "
        "raw UMI counts (10x Genomics 3k healthy-donor PBMCs)."
    )
    return LoadedDataset(payload=adata, source="real", provenance=provenance, key=_KEY)


def _load_pbmc3k_tarball(cache_dir: Path) -> LoadedDataset:
    """Download and parse the raw 10x PBMC3k tarball with scipy/pandas."""
    import pandas as pd
    from scipy.io import mmread

    cache_dir.mkdir(parents=True, exist_ok=True)
    tar_path = cache_dir / "pbmc3k_filtered_gene_bc_matrices.tar.gz"
    extract_dir = cache_dir / "pbmc3k_filtered_gene_bc_matrices"
    mtx_dir = extract_dir / "filtered_gene_bc_matrices" / "hg19"

    if not tar_path.exists():
        urllib.request.urlretrieve(_PBMC3K_URL, tar_path)  # noqa: S310 -- https
    if not mtx_dir.exists():
        with tarfile.open(tar_path) as tar:
            tar.extractall(extract_dir)  # noqa: S202 -- trusted 10x source

    # 10x market-matrix is genes x cells; transpose to cells x genes.
    matrix = mmread(str(mtx_dir / "matrix.mtx")).T.tocsr()
    genes = pd.read_csv(mtx_dir / "genes.tsv", sep="\t", header=None)
    barcodes = pd.read_csv(mtx_dir / "barcodes.tsv", sep="\t", header=None)
    gene_ids = genes[0].to_numpy()
    gene_names = genes[1].to_numpy() if genes.shape[1] > 1 else gene_ids
    cell_barcodes = barcodes[0].to_numpy()

    provenance = (
        f"real: 10x PBMC3k tarball downloaded from {_PBMC3K_URL} and cached at "
        f"{tar_path}; parsed with scipy.io.mmread + pandas; license: {_LICENSE}."
    )

    try:
        import anndata as ad

        adata = ad.AnnData(X=matrix)
        adata.obs_names = cell_barcodes.astype(str)
        adata.var_names = gene_ids.astype(str)
        adata.var["gene_symbols"] = gene_names
        payload: Any = adata
        provenance += " Parsed into anndata.AnnData (cells x genes)."
    except Exception:  # noqa: BLE001 -- anndata optional; fall to dict payload
        payload = {
            "counts": matrix,
            "gene_names": gene_names,
            "gene_ids": gene_ids,
            "barcodes": cell_barcodes,
        }
        provenance += " anndata unavailable; returned counts/gene/barcode dict."

    return LoadedDataset(payload=payload, source="real", provenance=provenance, key=_KEY)


def _pbmc3k_fallback(*, seed: int | None, reason: str) -> LoadedDataset:
    """Build a deterministic synthetic single-cell fallback."""
    counts, labels, gene_names = _synthetic_singlecell(seed=seed)
    provenance = (
        f"synthetic/bundled fallback: {reason}. Poisson-distributed counts for "
        f"{counts.shape[0]} cells x {counts.shape[1]} genes across "
        f"{int(labels.max()) + 1} latent cell-type clusters with "
        "cluster-specific marker genes (seeded, deterministic)."
    )
    payload = {"counts": counts, "labels": labels, "gene_names": gene_names}
    return LoadedDataset(payload=payload, source="fallback", provenance=provenance, key=_KEY)


def _synthetic_singlecell(
    *,
    seed: int | None,
    n_cells: int = 600,
    n_genes: int = 200,
    n_clusters: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a clustered synthetic UMI-count matrix.

    Each latent cell type over-expresses a disjoint block of marker genes on
    top of a shared lowly-expressed background; counts are Poisson draws so the
    matrix looks like sparse single-cell UMI data.

    Returns
    -------
    tuple of (counts, labels, gene_names)
        ``counts`` ``(n_cells, n_genes)`` int64, ``labels`` ``(n_cells,)`` int,
        ``gene_names`` ``(n_genes,)`` str.
    """
    rng = np.random.default_rng(seed)
    base_rate = rng.uniform(0.05, 0.4, size=n_genes)
    genes_per_cluster = max(1, n_genes // n_clusters)

    counts = np.empty((n_cells, n_genes), dtype=np.int64)
    labels = np.empty(n_cells, dtype=np.int64)
    cells_per = n_cells // n_clusters

    for c in range(n_clusters):
        start = c * cells_per
        end = n_cells if c == n_clusters - 1 else (c + 1) * cells_per
        rates = base_rate.copy()
        marker = slice(c * genes_per_cluster, (c + 1) * genes_per_cluster)
        rates[marker] += rng.uniform(3.0, 8.0, size=len(range(*marker.indices(n_genes))))
        counts[start:end] = rng.poisson(rates, size=(end - start, n_genes))
        labels[start:end] = c

    gene_names = np.array([f"gene_{i:04d}" for i in range(n_genes)])
    return counts, labels, gene_names
