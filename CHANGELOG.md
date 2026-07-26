# Changelog

Work log for the ddm4bio course — public [`symbiont-ai/ddm4bio`](https://github.com/symbiont-ai/ddm4bio)
and private `symbiont-ai/ddm4bio-instructor`. Short SHAs refer to commits on `main`.

## 2026-07 — Course restructure

**State after this work:** 8 datasets in the registry · 8 modules (M1–M8), with Module 5
split into 5a/5b → 9 lesson notebooks · public site and instructor repo in sync.

> Note on labels: lessons are called **"Module N"** in all student-facing prose, but the
> notebook **filenames and URLs keep a `wk*` slug** (`lessons/wk1_…`) and the registry keys
> stay `wk*`/`ps*`. This split is intentional — it keeps deployed links stable. See
> `MEMORY`/course conventions.

### Module 2 — Dose-response science, done honestly
- Added a **real raw CCLE dose-response** dataset (Barretina 2012) with an SD-weighted Hill
  fit that recovers the published EC50/Amax (`2cab41a`, `99c0ac7`, `82c3c54`).
- **Removed GDSC entirely** — dropped the read-vs-fit detour once CCLE did the real fit;
  deleted the loader, registry entry, and warm-list course-wide (`50a335c`, `9876878`, `976e8cf`).
- Fixed the **mechanism-vs-assay science** — a Hill fit on a viability screen is
  *phenomenological*, not receptor-occupancy; added an IC50/EC50/Amax/Hill-slope vocabulary
  block (`0fc4ef5`).

### Module 8 — SHRED investigated, then cut
- Ran an honesty investigation of SHRED (leakage fix, FitzHugh–Nagumo field, "teach the
  boundary") and made the CNN baseline fair (`b2b3705`, `d1ff4a0`, `e2b2986`).
- **Cut SHRED entirely** — its only advantage was on a synthetic advection field, never real
  biomedical data, so it never earned its keep. Removed §4, renumbered, and rewrote the
  title / intro / interpretation / "two ways" narrative (`85b8c0f`); swept the last
  references (capstone preview + a workflow comment) (`1fa792e`).

### Module 1 redesign + Module 5 split
- **Module 1 is now "Solving Ax = b"** — determined systems, least squares, and conditioning
  — taught on landmark-based **retinal-image registration** (`97a6e89`).
- **Module 5 split into 5a** (SVD/PCA, eigen-cells, robust PCA, LDA) **and 5b** (ICA,
  t-SNE/UMAP); the **eigen-cells arc moved out of the old Module 1 into 5a** (`97a6e89`).
- Fixed a display-math rendering bug and added the missing **Chapter 2** reading (`73f6b53`);
  re-aligned cross-references course-wide (`97a6e89`).

### Dataset audit — pruned to a lean 8-key registry
- **Pruned 5 unused datasets** (`pathmnist`, `mne_eeg`, `ixi_mri`, `fastmri`, `tcga_expr`),
  deleting 3 now-dead loader modules; registry went from **13 → 8 keys** (`1fa792e`).
- Corrected stale `used_by` tuples — `pbmc3k`, `heart_uci`, and
  **`breast_wisconsin` → (wk2, wk6, ps2)**, verified against both repos (`5faa66d`).

### "Week" → "Module" relabel
- Relabeled all **student-facing prose** to "Module N" (the course is self-paced) while keeping
  `wk*` filenames, URLs, and registry keys as internal plumbing — so no links break — and
  preserving three genuine calendar "week" references in Module 7 (`05392f8` public,
  `0128dc2` instructor).

### Whole-course consistency audit
- A 6-dimension multi-agent audit (adversarially verified) found **4 real issues**, all fixed;
  5 candidate findings were correctly refuted (`e1fe1e7`, `d007e5e`):
  - `ps7` reading cited Chapter 7 (Visualization) → **Ch.15 + Ch.20–21**;
  - `wk5a` reading missing the LDA source → added **Ch.18 §2**;
  - "(Module 5)" → "(Module 5a)" for blood-cell classification;
  - `wk8` §3 had an orphan "### 3b" → added a paired "### 3a".

### Capstone — reference solution tested + honesty fix
- **Tested the warfarin PK/PD → dosing-policy reference pipeline end-to-end**; all checks pass:
  the real-data run (QC-first, **PK t½ = 1.80 d**, RL policy holds therapeutic range
  **87% vs 39%** for the best fixed dose, Q-learning reaches the model-based optimum), the
  report section-linter, `tests/test_control.py` (7/7), and the offline synthetic-fallback path.
- **Made `interpret()` source-aware** (`0429abd`): on the synthetic-fallback path it no longer
  claims the policy was "fit from real data" or that the fit "recovers warfarin's half-life";
  it now says "synthetic PK/PD stand-in … internal consistency, **not a validation against the
  real drug**" with an explicit offline-synthetic limitation. The real path is unchanged; the
  calibrate print `"real ke"` → `"the fitted ke"`.

### Instructor repo (`ddm4bio-instructor`) — kept in sync
- PS1 re-anchored **Module 1 → Module 5a** (`d5a8851`); Week→Module relabel (`0128dc2`); ps7
  chapter fix (`d007e5e`) — the full autograder suite stayed green (45/45) throughout.

### Supporting polish
- Sharpened the ground-truth philosophy (synthetic *verifies*, doesn't *license trust*)
  (`14b4469`); fixed the UMAP "no reusable function" wording across wk5/wk8 (`bc37586`);
  centralized BloodMNIST class labels in the loader (`0c5aa41`); refreshed the
  schedule / README / intro / how-it-works pages (`f1e2531`, `71b085a`, `057ef27`, `de71767`).
