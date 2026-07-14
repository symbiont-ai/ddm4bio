"""Dataset registry, loader contract, and dispatch.

``DATASET_REGISTRY`` maps a short key to a :class:`DatasetSpec`; ``get_dataset``
resolves a key to a cached, loaded :class:`LoadedDataset`. Loader functions live
in the sibling modules and are resolved lazily via :mod:`importlib` from the
dotted ``"module:function"`` string in each spec, so importing this package
never triggers heavy dataset-specific imports.

LOADER CONTRACT
---------------
Every loader referenced by a :class:`DatasetSpec` MUST match this signature::

    def load_NAME(*, cache_dir, download=True, prefer_real=True,
                  seed=None, **opts) -> LoadedDataset

Behaviour:

* When ``download`` and ``prefer_real``: TRY to fetch the REAL dataset from its
  source and CACHE the raw download under ``cache_dir``. The fetch is idempotent
  -- if the cache file already exists, load from it and never re-download. On
  success return ``LoadedDataset(payload=REAL_OBJECT, source="real",
  provenance="source URL + license + what it is", key="KEY")``.
* On ANY failure (no network, missing optional dependency, HTTP/parse error),
  OR when ``download`` is False, OR when ``prefer_real`` is False: return a
  clearly-labelled FALLBACK built from :mod:`ddm4bio.datasets.synthetic` and/or
  scikit-learn bundled data, as ``LoadedDataset(payload=FALLBACK,
  source="fallback", provenance="synthetic/bundled fallback: why", key="KEY")``.
  NEVER raise merely because the real source is unavailable -- always fall back.
* EXCEPTION for the ``credentialed`` tier: when ``download`` and ``prefer_real``,
  raise :class:`PermissionError` with apply-for-access instructions; otherwise
  return the fallback.
* Do not print; carry provenance inside :class:`LoadedDataset`. Wrap the real
  fetch in ``try/except`` and fall back on ``Exception``.

Heavy libraries (pandas, scipy, scikit-learn, nibabel, wfdb, scanpy, anndata,
mne, ...) MUST be imported inside the loader body, not at module top, so that
importing this package works on numpy alone. Fallbacks must be deterministic
given ``seed``.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetSpec:
    """Static description of a dataset and how to load it.

    Parameters
    ----------
    key:
        Short registry key, e.g. ``"pbmc3k"``.
    name:
        Human-readable dataset name.
    modality:
        Data modality, e.g. ``"images"``, ``"singlecell"``, ``"signals"``,
        ``"imaging"``, ``"tabular"``, ``"timeseries"``.
    tier:
        Access tier: ``"open"``, ``"credentialed"``, or ``"archive"``.
    license:
        License / terms-of-use string.
    citation:
        Short citation for the dataset.
    url:
        Canonical source URL.
    loader:
        Dotted ``"module:function"`` path resolved lazily at load time.
    used_by:
        Tuple of course-unit keys (weeks / problem sets) that use the dataset.
    """

    key: str
    name: str
    modality: str
    tier: str
    license: str
    citation: str
    url: str
    loader: str
    used_by: tuple = field(default_factory=tuple)


@dataclass
class LoadedDataset:
    """A loaded dataset plus provenance metadata.

    Parameters
    ----------
    payload:
        The loaded object (AnnData / ndarray / DataFrame / mne / wfdb / ...).
    source:
        ``"real"`` if fetched from the true source, ``"fallback"`` if built
        from synthetic or bundled data.
    provenance:
        Free-text description of where the payload came from and why.
    key:
        The registry key this dataset was loaded under.
    """

    payload: Any
    source: str
    provenance: str
    key: str


DATASET_REGISTRY: dict[str, DatasetSpec] = {
    "bloodmnist": DatasetSpec(
        key="bloodmnist",
        name="BloodMNIST (MedMNIST v2)",
        modality="images",
        tier="open",
        license="CC BY 4.0",
        citation=(
            "Yang et al. (2023), MedMNIST v2: A large-scale lightweight "
            "benchmark for 2D and 3D biomedical image classification, "
            "Scientific Data 10:41."
        ),
        url="https://medmnist.com/",
        loader="ddm4bio.datasets.medmnist_images:load_medmnist",
        used_by=("wk1", "ps1"),
    ),
    "pathmnist": DatasetSpec(
        key="pathmnist",
        name="PathMNIST (MedMNIST v2)",
        modality="images",
        tier="open",
        license="CC BY 4.0",
        citation=(
            "Yang et al. (2023), MedMNIST v2: A large-scale lightweight "
            "benchmark for 2D and 3D biomedical image classification, "
            "Scientific Data 10:41. Source patches: Kather et al. (2019)."
        ),
        url="https://medmnist.com/",
        loader="ddm4bio.datasets.medmnist_images:load_medmnist",
        used_by=("wk6", "ps6"),
    ),
    "pbmc3k": DatasetSpec(
        key="pbmc3k",
        name="10x Genomics PBMC 3k (single-cell RNA-seq)",
        modality="singlecell",
        tier="open",
        license="Creative Commons / 10x Genomics terms of use",
        citation=(
            "10x Genomics (2016), 3k PBMCs from a Healthy Donor, "
            "Cell Ranger 1.1.0; popularized by the Scanpy PBMC3k tutorial."
        ),
        url=(
            "https://cf.10xgenomics.com/samples/cell/pbmc3k/pbmc3k_filtered_gene_bc_matrices.tar.gz"
        ),
        loader="ddm4bio.datasets.singlecell:load_pbmc3k",
        used_by=("wk5", "ps5"),
    ),
    "mne_eeg": DatasetSpec(
        key="mne_eeg",
        name="MNE sample EEG/MEG dataset",
        modality="signals",
        tier="open",
        license="BSD-3-Clause (MNE-Python sample data)",
        citation=(
            "Gramfort et al. (2013), MEG and EEG data analysis with "
            "MNE-Python, Frontiers in Neuroscience 7:267."
        ),
        url="https://mne.tools/stable/documentation/datasets.html",
        loader="ddm4bio.datasets.neuro:load_eeg",
        used_by=("wk4", "ps4"),
    ),
    "mitbih": DatasetSpec(
        key="mitbih",
        name="MIT-BIH Arrhythmia Database (ECG)",
        modality="signals",
        tier="open",
        license="Open Data Commons Attribution License v1.0 (PhysioNet)",
        citation=(
            "Moody & Mark (2001), The impact of the MIT-BIH Arrhythmia "
            "Database, IEEE Eng. Med. Biol. 20(3):45-50; Goldberger et al. "
            "(2000), PhysioNet, Circulation 101(23):e215-e220."
        ),
        url="https://physionet.org/content/mitdb/1.0.0/",
        loader="ddm4bio.datasets.physio:load_mitbih",
        used_by=("wk4", "ps4", "wk7", "ps7"),
    ),
    "ixi_mri": DatasetSpec(
        key="ixi_mri",
        name="IXI Brain MRI Dataset",
        modality="imaging",
        tier="open",
        license="CC BY-SA 3.0",
        citation=(
            "IXI - Information eXtraction from Images, Biomedical Image "
            "Analysis Group, Imperial College London."
        ),
        url="https://brain-development.org/ixi-dataset/",
        loader="ddm4bio.datasets.imaging_mri:load_ixi",
        used_by=("ps4",),
    ),
    "fastmri": DatasetSpec(
        key="fastmri",
        name="fastMRI (NYU Langone / Meta AI)",
        modality="imaging",
        tier="credentialed",
        license=("fastMRI Dataset Sharing Agreement (credentialed; non-commercial research only)"),
        citation=(
            "Zbontar et al. (2018), fastMRI: An Open Dataset and Benchmarks "
            "for Accelerated MRI, arXiv:1811.08839; Knoll et al. (2020), "
            "Radiology: Artificial Intelligence 2(1):e190007."
        ),
        url="https://fastmri.med.nyu.edu/",
        loader="ddm4bio.datasets.imaging_mri:load_fastmri",
        used_by=("ps4",),
    ),
    "gdsc": DatasetSpec(
        key="gdsc",
        name="GDSC - Genomics of Drug Sensitivity in Cancer",
        modality="tabular",
        tier="open",
        license="Free for academic/non-commercial use (Sanger/EMBL-EBI)",
        citation=(
            "Yang et al. (2013), Genomics of Drug Sensitivity in Cancer "
            "(GDSC), Nucleic Acids Research 41(D1):D955-D961; Iorio et al. "
            "(2016), Cell 166(3):740-754."
        ),
        url="https://www.cancerrxgene.org/",
        loader="ddm4bio.datasets.omics:load_gdsc",
        used_by=("wk2", "ps2"),
    ),
    "tcga_expr": DatasetSpec(
        key="tcga_expr",
        name="TCGA gene expression (pan-cancer RNA-seq)",
        modality="tabular",
        tier="open",
        license="NIH Genomic Data Commons open-access data terms",
        citation=(
            "The Cancer Genome Atlas Research Network (2013), The Cancer "
            "Genome Atlas Pan-Cancer analysis project, Nature Genetics "
            "45(10):1113-1120."
        ),
        url="https://portal.gdc.cancer.gov/",
        loader="ddm4bio.datasets.omics:load_tcga",
        used_by=("ps6", "capstone"),
    ),
    "jhu_covid": DatasetSpec(
        key="jhu_covid",
        name="JHU CSSE COVID-19 time series",
        modality="timeseries",
        tier="archive",
        license="CC BY 4.0 (Johns Hopkins University, non-commercial terms)",
        citation=(
            "Dong, Du & Gardner (2020), An interactive web-based dashboard "
            "to track COVID-19 in real time, Lancet Infectious Diseases "
            "20(5):533-534."
        ),
        url="https://github.com/CSSEGISandData/COVID-19",
        loader="ddm4bio.datasets.epi:load_jhu_covid",
        used_by=("wk7", "ps7"),
    ),
    "heart_uci": DatasetSpec(
        key="heart_uci",
        name="UCI Heart Disease (Cleveland)",
        modality="tabular",
        tier="open",
        license="CC BY 4.0 (UCI Machine Learning Repository)",
        citation=(
            "Janosi, Steinbrunn, Pfisterer & Detrano (1988), Heart Disease, "
            "UCI Machine Learning Repository."
        ),
        url="https://archive.ics.uci.edu/dataset/45/heart+disease",
        loader="ddm4bio.datasets.tabular:load_heart_uci",
        used_by=("ps3",),
    ),
    "breast_wisconsin": DatasetSpec(
        key="breast_wisconsin",
        name="Breast Cancer Wisconsin (Diagnostic)",
        modality="tabular",
        tier="open",
        license="CC BY 4.0 (UCI Machine Learning Repository)",
        citation=(
            "Wolberg, Street & Mangasarian (1995), Breast Cancer Wisconsin "
            "(Diagnostic), UCI Machine Learning Repository; also bundled in "
            "scikit-learn."
        ),
        url=("https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic"),
        loader="ddm4bio.datasets.tabular:load_breast_wisconsin",
        used_by=("wk2", "wk3", "wk6", "ps2", "ps3", "ps6"),
    ),
}


def list_datasets() -> list[DatasetSpec]:
    """Return every registered :class:`DatasetSpec`."""
    return list(DATASET_REGISTRY.values())


def get_dataset(
    key: str,
    *,
    download: bool = True,
    prefer_real: bool = True,
    cache_dir: Path | str | None = None,
    **opts: Any,
) -> LoadedDataset:
    """Resolve ``key`` to a cached, loaded :class:`LoadedDataset`.

    Parameters
    ----------
    key:
        Registry key; must be present in :data:`DATASET_REGISTRY`.
    download:
        When True, loaders may fetch the real source (and cache it).
    prefer_real:
        When True, loaders prefer the real dataset over the fallback.
    cache_dir:
        Directory for cached raw downloads. Defaults to
        :data:`ddm4bio.config.CACHE_DIR` and is created if missing.
    **opts:
        Loader-specific options, forwarded verbatim.

    Returns
    -------
    LoadedDataset
        The loaded dataset with provenance.

    Raises
    ------
    KeyError
        If ``key`` is not registered (message lists the available keys).
    """
    if key not in DATASET_REGISTRY:
        available = ", ".join(sorted(DATASET_REGISTRY))
        raise KeyError(f"Unknown dataset key {key!r}. Available keys: {available}")

    spec = DATASET_REGISTRY[key]

    if cache_dir is None:
        from ddm4bio.config import CACHE_DIR

        cache_dir = CACHE_DIR
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    module_name, func_name = spec.loader.split(":")
    module = importlib.import_module(module_name)
    loader = getattr(module, func_name)

    # Forward the registry key so a loader shared across several keys (e.g.
    # ``load_medmnist`` serving both ``bloodmnist`` and ``pathmnist``) can select
    # the correct subset instead of relying on a constant default. Loaders that
    # don't need it absorb ``key`` into their ``**opts`` and ignore it.
    return loader(
        cache_dir=cache_dir,
        download=download,
        prefer_real=prefer_real,
        key=key,
        **opts,
    )
