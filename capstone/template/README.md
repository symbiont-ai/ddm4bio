# Capstone submission — <your name / title>

Minimal, runnable skeleton for a ddm4bio capstone. Copy this directory, then
replace the demo fixture and the two methods with your own — **keep the
structure**: load → QC-before-results → one Group-A decomposition → one Group-B
learning/dynamics method → ground-truth validation → interpretation.

## Layout

```
.
├── Makefile          # `make data`, `make run`, `make report`
├── README.md         # this file
├── report.md         # fill from the mandated section headers
└── src/
    └── pipeline.py    # end-to-end demo; edit this
```

## Prerequisites

Install the course library (core stack only — the demo needs no heavy extras
and no network):

```bash
pip install -e ".[dev]"     # run from the ddm4bio repo root
```

## Run it

```bash
make data      # materialize inputs (demo: synthetic, generated in-process)
make run       # python src/pipeline.py
make report    # lint report.md for the mandated sections
```

`make run` prints the QC report, the SVD singular-value spectrum and chosen
rank, the DMD-recovered eigenvalues, the ground-truth validation error, and the
interpretation block with a calibrated confidence level.

## What the demo does

The demo (`src/pipeline.py`) uses a synthetic linear dynamical system with a
**known eigen-spectrum** as its ground truth:

1. **Load** — `make_linear_dynamics` generates a state trajectory with known
   eigenvalues.
2. **Quality control (before results)** — `qc_tabular` inspects the trajectory
   for missingness, duplicates, constant columns, and outliers.
3. **Group A — SVD** (`svd_lowrank`) — the singular-value spectrum picks the
   truncation rank.
4. **Group B — DMD** (`dmd`) — recovers the discrete-time eigenvalues at that
   rank.
5. **Ground-truth validation** — compares recovered eigenvalues to the known
   spectrum and reports the max error.
6. **Interpretation** — `interpretation_block` states the claim with a
   confidence level *derived from the validation error*, plus limitations.

## Make it yours

- Swap `load()` for a `ddm4bio` loader (or your own documented loader).
- Replace SVD/DMD with your chosen pair — **one from each group** (see the
  capstone `README.md` two-method table). Keep them *connected*.
- Keep QC first, keep a ground-truth control, keep the confidence calibrated to
  evidence.
- Fill in `report.md`. Every mandated section header must be present (run
  `make report` to check) and Quality Control must come before Results.
- Pin your seed via `seed_everything` and pin your dependencies so `make run`
  reproduces your headline numbers from a clean clone.
