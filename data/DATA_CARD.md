# DATA_CARD

The compliance record for every dataset used in the course. Kept in sync with
`ddm4bio.datasets.DATASET_REGISTRY`. Rows are populated in **Phase 2**; the
table below lists the planned datasets and their access tiers.

**Access tiers:** `open` (direct download, no login) · `archive` (frozen but
downloadable) · `credentialed` (requires DUA/registration — **never**
auto-downloaded; use the open/synthetic fallback).

| Key | Dataset | Primary use | Tier | License | Access |
|---|---|---|---|---|---|
| `bloodmnist` / `pathmnist` | MedMNIST v2 | PS1 eigen-cells; PS6 classification | open | CC BY 4.0 (code Apache-2.0) | `pip install medmnist` |
| `pbmc3k` | 10x PBMC 3k (scanpy) | PS5 PCA; PS6 clustering | open | [VERIFY] | scanpy built-in |
| `mne_sample` | MNE-Python sample (EEG/MEG) | PS4 time–freq; PS5 ICA | open | BSD-3 [VERIFY] | `mne.datasets.sample` |
| `mitbih` | PhysioNet MIT-BIH Arrhythmia | PS4 signals; PS7 Kalman | open | ODC-BY / PhysioNet [VERIFY] | `wfdb` |
| `ixi_mri` | IXI brain MRI | PS4 compressed-sensing MRI | open | CC BY-SA 3.0 [VERIFY] | direct download |
| `fastmri` | NYU fastMRI raw k-space | PS4 CS (optional) | credentialed | DUA, education-only | application required; do **not** auto-download; fall back to `ixi_mri` |
| `gdsc` | GDSC drug-sensitivity | PS2 dose–response | open | GDSC terms [VERIFY] | direct CSV; fallback synthetic Hill |
| `breast_wisconsin` | sklearn breast cancer | PS2 sparse features; PS3 baseline | open | public (sklearn) | bundled |
| `heart_uci` | UCI Heart Disease (Cleveland) | PS3 clinical tabular | open | UCI/CC BY 4.0 [VERIFY] | direct CSV |
| `tcga_expr` | TCGA expression (GDC/cBioPortal) | PS6 advanced; capstone | open | NIH GDC open [VERIFY] | API; heavy — optional |
| `jhu_covid` | JHU CSSE COVID-19 | PS7 DMD/SINDy epidemics | archive | CC BY 4.0 (JHU) [VERIFY] | static GitHub CSVs (frozen 2023-03-10) |

> **[VERIFY] items** must be confirmed at build time in Phase 2 (fetch the URL /
> check the license). If a `[VERIFY]` item fails, do **not** silently
> substitute — record the decision here and use the nearest open alternative.
