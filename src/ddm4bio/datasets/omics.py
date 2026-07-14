"""Omics and tabular biomedical datasets.

Datasets:
- GDSC: Genomics of Drug Sensitivity in Cancer (drug dose-response).
  Tier: open. License: GDSC terms (academic use). ``load_gdsc`` downloads a
  dose-response CSV into ``cache_dir`` and returns a pandas DataFrame; on any
  failure it returns a deterministic synthetic Hill-curve dose-response table.
- TCGA: pan-cancer RNA-seq gene expression. Tier: open (GDC open-access).
  ``load_tcga`` attempts a SMALL public expression subset (size-guarded so it
  never forces a large download) and otherwise returns a synthetic
  tumor-vs-normal expression matrix.
- Breast Cancer Wisconsin (Diagnostic). Tier: open. License: CC BY-NC-SA
  (bundled with scikit-learn). [Phase 0 stub; served from ``tabular`` module.]
- Heart Disease (UCI, Cleveland). Tier: open. License: CC BY 4.0 (UCI).
  [Phase 0 stub; served from ``tabular`` module.]

Loader contract: see :mod:`ddm4bio.datasets.registry`. Heavy libraries
(pandas, scipy, scikit-learn) are imported inside the function bodies so that
importing this module needs only numpy.
Install: ``pip install scikit-learn pandas``.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

from ddm4bio.datasets.registry import LoadedDataset

_GDSC_KEY = "gdsc"
_TCGA_KEY = "tcga_expr"

# GDSC bulk dose-response CSV (overridable via opts["url"]).
_GDSC_URL = (
    "https://cog.sanger.ac.uk/cancerrxgene/GDSC_release8.5/GDSC2_fitted_dose_response_27Oct23.csv"
)
_GDSC_LICENSE = (
    "GDSC / Wellcome Sanger Institute & EMBL-EBI terms (free for academic / non-commercial use)"
)

# A small public TCGA expression subset (overridable via opts["url"]).
_TCGA_URL = (
    "https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/"
    "TCGA.LAML.sampleMap%2FHiSeqV2_PANCAN.gz"
)
_TCGA_LICENSE = "NIH Genomic Data Commons open-access data terms (UCSC Xena mirror)"
_TCGA_MAX_BYTES = 50_000_000  # never force a large download


def load_gdsc(
    *,
    cache_dir: Path | str,
    download: bool = True,
    prefer_real: bool = True,
    seed: int | None = None,
    **opts: Any,
) -> LoadedDataset:
    """Load GDSC drug dose-response data (real CSV, with synthetic fallback).

    Parameters
    ----------
    cache_dir:
        Directory for the cached CSV download (idempotent).
    download:
        When True, may download the real GDSC CSV.
    prefer_real:
        When True, prefer the real dataset over the synthetic fallback.
    seed:
        RNG seed making the synthetic Hill-curve fallback deterministic.
    **opts:
        ``url`` overrides the download URL.

    Returns
    -------
    LoadedDataset
        ``payload`` is a pandas ``DataFrame`` -- the real GDSC fitted
        dose-response table, or a synthetic long-format Hill-curve table with
        columns ``cell_line``/``drug``/``concentration``/``viability`` plus the
        true ``ic50``/``hill`` parameters. ``source`` is ``"real"`` or
        ``"fallback"``.
    """
    cache_dir = Path(cache_dir)

    if download and prefer_real:
        try:
            return _load_gdsc_real(cache_dir, url=opts.get("url"))
        except Exception:  # noqa: BLE001 -- always fall back on failure
            reason = "real GDSC CSV download/parse failed (no network / bad URL)"
    elif not download:
        reason = "download=False requested"
    else:
        reason = "prefer_real=False requested"

    return _gdsc_fallback(seed=seed, reason=reason)


def _load_gdsc_real(cache_dir: Path, url: str | None) -> LoadedDataset:
    """Download and parse the real GDSC dose-response CSV with pandas."""
    import pandas as pd

    cache_dir.mkdir(parents=True, exist_ok=True)
    url = url or _GDSC_URL
    dest = cache_dir / "gdsc_dose_response.csv"
    if not dest.exists():
        urllib.request.urlretrieve(url, dest)  # noqa: S310 -- https source

    frame = pd.read_csv(dest)
    provenance = (
        f"real: GDSC fitted dose-response CSV downloaded from {url} and cached "
        f"at {dest}; parsed with pandas.read_csv; license: {_GDSC_LICENSE}; "
        f"{frame.shape[0]} rows x {frame.shape[1]} columns."
    )
    return LoadedDataset(payload=frame, source="real", provenance=provenance, key=_GDSC_KEY)


def _gdsc_fallback(*, seed: int | None, reason: str) -> LoadedDataset:
    """Build a deterministic synthetic Hill-curve dose-response table."""
    import pandas as pd

    rng = np.random.default_rng(seed)
    n_cell_lines = 20
    n_drugs = 8
    doses = np.logspace(-3.0, 1.0, 9)  # micromolar, log-spaced

    records = []
    for line in range(n_cell_lines):
        for drug in range(n_drugs):
            # True per-(line, drug) Hill parameters.
            ic50 = float(10.0 ** rng.uniform(-2.0, 0.5))
            hill = float(rng.uniform(0.7, 2.5))
            for conc in doses:
                signal = 1.0 / (1.0 + (conc / ic50) ** hill)
                viability = float(np.clip(signal + rng.normal(0.0, 0.03), 0.0, 1.0))
                records.append(
                    {
                        "cell_line": f"CL_{line:03d}",
                        "drug": f"DRUG_{drug:02d}",
                        "concentration": float(conc),
                        "viability": viability,
                        "ic50": ic50,
                        "hill": hill,
                    }
                )

    frame = pd.DataFrame.from_records(records)
    provenance = (
        f"synthetic/bundled fallback: {reason}. Long-format Hill-curve "
        f"dose-response for {n_cell_lines} cell lines x {n_drugs} drugs at "
        f"{doses.size} log-spaced concentrations; viability = "
        "1/(1+(conc/ic50)^hill) with Gaussian noise; true ic50/hill included "
        "(seeded, deterministic)."
    )
    return LoadedDataset(payload=frame, source="fallback", provenance=provenance, key=_GDSC_KEY)


def load_tcga(
    *,
    cache_dir: Path | str,
    download: bool = True,
    prefer_real: bool = True,
    seed: int | None = None,
    **opts: Any,
) -> LoadedDataset:
    """Load a TCGA expression subset (size-guarded real, synthetic fallback).

    Parameters
    ----------
    cache_dir:
        Directory for the cached download (idempotent).
    download:
        When True, may attempt the small real TCGA subset.
    prefer_real:
        When True, prefer the real subset over the synthetic fallback.
    seed:
        RNG seed making the synthetic fallback deterministic.
    **opts:
        ``url`` overrides the source URL; ``max_bytes`` overrides the download
        size guard (default 50 MB) so a large file never downloads silently.

    Returns
    -------
    LoadedDataset
        ``payload`` is a pandas ``DataFrame`` (genes x samples) for the real
        subset, or a synthetic ``dict`` with ``expression`` (samples x genes),
        ``labels`` (0=normal, 1=tumor), and ``gene_names``. ``source`` is
        ``"real"`` or ``"fallback"``.
    """
    cache_dir = Path(cache_dir)

    if download and prefer_real:
        try:
            return _load_tcga_real(
                cache_dir,
                url=opts.get("url"),
                max_bytes=int(opts.get("max_bytes", _TCGA_MAX_BYTES)),
            )
        except Exception:  # noqa: BLE001 -- always fall back on failure
            reason = "real TCGA subset unavailable or exceeded the size guard"
    elif not download:
        reason = "download=False requested"
    else:
        reason = "prefer_real=False requested"

    return _tcga_fallback(seed=seed, reason=reason)


def _load_tcga_real(cache_dir: Path, url: str | None, max_bytes: int) -> LoadedDataset:
    """Fetch a small public TCGA expression subset, guarded against big files."""
    import pandas as pd

    cache_dir.mkdir(parents=True, exist_ok=True)
    url = url or _TCGA_URL
    dest = cache_dir / "tcga_expression_subset.tsv.gz"

    if not dest.exists():
        # Guard: refuse to force a large download.
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 -- https
            length = resp.headers.get("Content-Length")
        if length is None or int(length) > max_bytes:
            raise RuntimeError(
                f"TCGA subset size {length!r} unknown or exceeds "
                f"max_bytes={max_bytes}; refusing to download."
            )
        urllib.request.urlretrieve(url, dest)  # noqa: S310 -- https source

    # Xena expression matrices are gene-by-sample TSVs (gzip-compressed).
    frame = pd.read_csv(dest, sep="\t", index_col=0, compression="infer")
    provenance = (
        f"real: TCGA expression subset downloaded from {url} and cached at "
        f"{dest} (size-guarded at {max_bytes} bytes); parsed with pandas; "
        f"license: {_TCGA_LICENSE}; {frame.shape[0]} genes x "
        f"{frame.shape[1]} samples."
    )
    return LoadedDataset(payload=frame, source="real", provenance=provenance, key=_TCGA_KEY)


def _tcga_fallback(*, seed: int | None, reason: str) -> LoadedDataset:
    """Build a deterministic synthetic tumor-vs-normal expression matrix."""
    rng = np.random.default_rng(seed)
    n_samples = 200
    n_genes = 300
    n_de = 40  # differentially expressed genes between tumor and normal

    labels = np.zeros(n_samples, dtype=np.int64)
    labels[n_samples // 2 :] = 1  # second half = tumor

    # Per-gene log-baseline; expression is log-normal around it.
    base = rng.normal(6.0, 1.5, size=n_genes)
    expression = rng.normal(base, 0.5, size=(n_samples, n_genes))

    # Impose a reproducible tumor-vs-normal shift on the first n_de genes.
    de_shift = rng.uniform(1.0, 3.0, size=n_de) * rng.choice(np.array([-1.0, 1.0]), size=n_de)
    tumor = labels == 1
    expression[np.ix_(tumor, np.arange(n_de))] += de_shift

    gene_names = np.array([f"GENE_{i:04d}" for i in range(n_genes)])
    provenance = (
        f"synthetic/bundled fallback: {reason}. Log-normal expression for "
        f"{n_samples} samples x {n_genes} genes; balanced tumor/normal labels "
        f"with {n_de} differentially expressed marker genes (seeded, "
        "deterministic)."
    )
    payload = {
        "expression": expression,
        "labels": labels,
        "gene_names": gene_names,
    }
    return LoadedDataset(payload=payload, source="fallback", provenance=provenance, key=_TCGA_KEY)


def load_breast_wisconsin() -> Any:
    """Load the Breast Cancer Wisconsin (Diagnostic) dataset.

    Returns
    -------
    object
        A container exposing the feature matrix ``X`` of shape
        ``(n_samples, n_features)`` and binary labels ``y`` of shape
        ``(n_samples,)``.

    Notes
    -----
    Uses ``sklearn.datasets`` (imported inside the body). This Phase 0 stub is
    superseded by ``ddm4bio.datasets.tabular:load_breast_wisconsin``.
    """
    raise NotImplementedError(
        "load_breast_wisconsin is a Phase 0 stub; expected return exposes "
        "X[n,features] and y[n] binary labels."
    )


def load_heart_uci() -> Any:
    """Load the UCI Heart Disease (Cleveland) dataset.

    Returns
    -------
    object
        A container exposing the feature matrix ``X`` of shape
        ``(n_samples, n_features)`` and diagnosis labels ``y`` of shape
        ``(n_samples,)``.

    Notes
    -----
    Uses ``pandas`` (imported inside the body). This Phase 0 stub is superseded
    by ``ddm4bio.datasets.tabular:load_heart_uci``.
    """
    raise NotImplementedError(
        "load_heart_uci is a Phase 0 stub; expected return exposes X[n,features] and y[n] labels."
    )
