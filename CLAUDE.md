# CLAUDE.md — conventions for ddm4bio

Repo conventions for future agent (and human) sessions. Follow these when
extending the repository so it stays reproducible and honest.

## Golden rules (enforced in review and CI)

1. **QC before results.** Every analysis path prints a `QCReport`
   (`ddm4bio.qc`) before producing any result. No result without QC.
2. **Honest method labeling.** Public function names state the actual method
   (`pca_reduce`, not `reduce`). If a notebook substitutes a simpler method
   than nominally expected, a markdown cell headed **"Method note"** says what
   was substituted and why. Never mislabel. See `docs/METHOD_LABELING.md`.
3. **Ground-truth validation.** Correctness-sensitive methods (ICA, SINDy,
   compressed sensing, DMD) ship a synthetic ground-truth fixture
   (`ddm4bio.datasets.synthetic`) and a test asserting recovery, in addition to
   the real-data path.
4. **Reproducibility.** Everything is seeded. `GLOBAL_SEED = 20260714`; call
   `ddm4bio.seed_everything()` first. A fresh clone + `make setup && make test`
   must pass offline (except tests explicitly marked `network`).

## Notebook contract (every lecture notebook)

1. Title + learning goal (cite Kutz chapter/section).
2. Setup — `seed_everything()`, imports, `viz.set_style()`.
3. Data load — via `get_dataset(...)`; print provenance.
4. QC — display the relevant `QCReport` **before any modeling cell**.
5. Method on ground truth (ICA/SINDy/CS/DMD) — recovery vs. known truth.
6. Method on real data — the biomedical application.
7. Interpretation — explicit **confidence statement** + named **limitations**.
8. Exercises pointer — link to the matching problem set.

## Dataset tiers

`open` (direct download), `archive` (frozen but downloadable), `credentialed`
(requires DUA/registration). **Never auto-download `credentialed` data** — raise
a clear "apply for access" error and provide an open/synthetic fallback (see
`ddm4bio.utils.io.require_open_tier` and `datasets/imaging_mri.py`).

## Dependency policy (Phase 0 decision)

Core deps (`numpy`, `scipy`, `pandas`, `scikit-learn`) are installed by default;
heavy, area-specific stacks are opt-in extras in `pyproject.toml`. Keep **heavy
imports inside function bodies**, never at module top level, so the package
imports on the core stack alone. Only `numpy` may be imported at module top.

## Make targets

`make setup` (venv + `pip install -e ".[dev]"`), `make lint` (ruff check +
format check), `make format`, `make test` (pytest), `make notebooks`
(nbmake, later phases), `make data` (fetch datasets, later phases), `make clean`.

## Build phases

Phase 0 scaffold (done) → 1 library core (done) → 2 data layer → 3 methods+
validation notebooks → 4 all 16 notebooks → 5 problem sets → 6 capstone+docs →
7 hardening.
See `DESIGN.md` in the course reference folder for the full spec.
