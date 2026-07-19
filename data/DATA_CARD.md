# DATA_CARD

The compliance record for every dataset used in the course. Kept in sync with
`ddm4bio.datasets.DATASET_REGISTRY` (this table is generated to mirror the
registry one-row-per-key). Each loader tries the real source when
`download=True, prefer_real=True` and otherwise returns a clearly-labelled
synthetic/bundled fallback (`source="fallback"`); it never raises just because
the real source is unavailable.

> **Why real *and* synthetic?** See
> [`docs/DATA_AND_GROUND_TRUTH.md`](../docs/DATA_AND_GROUND_TRUTH.md) for the
> policy: real data for the application; synthetic fixtures only where a method's
> correctness must be scored against a known ground truth; labeled fallbacks are
> an offline safety net, kept off the published site by a CI guard.

**Access tiers:** `open` (direct download, no login) · `archive` (frozen but
downloadable) · `credentialed` (requires DUA/registration — **never**
auto-downloaded; the loader raises `PermissionError` with apply-for-access
instructions, or returns the open/synthetic fallback).

| Key | Dataset | Modality | Tier | License | Access / how-to-fetch | Used by |
|---|---|---|---|---|---|---|
| `bloodmnist` | BloodMNIST (MedMNIST v2) | images | open | CC BY 4.0 ✓ | `.npz` over HTTPS from medmnist.com (`urlopen`, cached); fallback: synthetic images | wk1, ps1 |
| `pathmnist` | PathMNIST (MedMNIST v2) | images | open | CC BY 4.0 ✓ | `.npz` over HTTPS from medmnist.com (`urlopen`, cached); fallback: synthetic images | wk6, ps6 |
| `pbmc3k` | 10x PBMC 3k (single-cell RNA-seq) | singlecell | open | Creative Commons / 10x Genomics terms of use [VERIFY] | `scanpy` built-in or 10x `.tar.gz` (cached); fallback: synthetic counts | wk5, ps5 |
| `mne_eeg` | MNE sample EEG/MEG dataset | signals | open | BSD-3-Clause (MNE sample data) ✓ | `mne.datasets.sample` fetch; fallback: synthetic EEG | wk4, ps4 |
| `mitbih` | MIT-BIH Arrhythmia Database (ECG) | signals | open | ODC-BY v1.0 (PhysioNet) ✓ | `wfdb` from physionet.org (cached); fallback: synthetic ECG | wk4, ps4, wk7, ps7 |
| `ixi_mri` | IXI Brain MRI Dataset | imaging | open | CC BY-SA 3.0 ✓ | direct NIfTI download via `nibabel` (cached); fallback: Shepp–Logan phantom | ps4 |
| `fastmri` | fastMRI (NYU Langone / Meta AI) | imaging | **credentialed** | fastMRI Dataset Sharing Agreement (non-commercial research) ✓ | apply at fastmri.med.nyu.edu; loader raises `PermissionError` — do **not** auto-download; fallback: synthetic k-space / `ixi_mri` | ps4 |
| `ccle` | CCLE pharmacological profiling (dose–response) | tabular | open | CC BY 4.0 (CCLE; Broad/DepMap terms) ✓ | direct CSV from data.broadinstitute.org (cached), exploded to per-concentration long form; fallback: synthetic Hill dose–response | wk2 |
| `gdsc` | GDSC — Genomics of Drug Sensitivity in Cancer | tabular | open | Free for academic/non-commercial use (Sanger/EMBL-EBI) [VERIFY] | direct CSV from cancerrxgene.org (cached), fitted-summary table (published IC50, not re-fit); fallback: synthetic Hill dose–response | wk2, ps2 |
| `tcga_expr` | TCGA gene expression (pan-cancer RNA-seq) | tabular | open | NIH GDC open-access data terms [VERIFY] | GDC API / direct download (cached, heavy); fallback: synthetic expression | ps6, capstone |
| `jhu_covid` | JHU CSSE COVID-19 time series | timeseries | archive | CC BY 4.0 (JHU, non-commercial terms) [VERIFY] | static GitHub CSVs (frozen); fallback: synthetic epidemic curve | wk7, ps7 |
| `heart_uci` | UCI Heart Disease (Cleveland) | tabular | open | CC BY 4.0 (UCI ML Repository) ✓ | direct CSV from archive.ics.uci.edu (cached); fallback: synthetic clinical tabular | ps3 |
| `breast_wisconsin` | Breast Cancer Wisconsin (Diagnostic) | tabular | open | CC BY 4.0 (UCI) / bundled in scikit-learn ✓ | `sklearn.datasets.load_breast_cancer` (bundled, no network → `source="real"`); fallback: synthetic tabular | wk2, wk3, wk6, ps2, ps3, ps6 |

> **✓ = confirmed** against `DATASET_REGISTRY` (license/tier match the registry
> spec and the loader is implemented).
>
> **[VERIFY] items** must be confirmed at build time in Phase 2 (fetch the URL /
> check the license). If a `[VERIFY]` item fails, do **not** silently
> substitute — record the decision here and use the nearest open alternative.
> These remain flagged because the real source's terms still warrant a legal
> read even though the loader and its fallback are in place.
