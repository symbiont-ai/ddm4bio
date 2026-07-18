# Capstone submission — <your name / title>

Runnable skeleton for the ddm4bio warfarin capstone. Copy this directory, then
**deepen** each step and fill in `report.md` — keep the structure: load →
QC-before-results → fit a mechanistic PK/PD model → characterize variability →
calibrate a dosing environment → learn a policy with RL → validate → interpret.

## Layout

```
.
├── Makefile          # `make data`, `make run`, `make report`
├── README.md         # this file
├── report.md         # fill from the mandated section headers
└── src/
    └── pipeline.py    # end-to-end reference; deepen this
```

## Prerequisites

Install the course library. The warfarin loader reads the upstream R data file
with `pyreadr`; without it (or offline) the loader falls back to a synthetic
PK/PD table with the same schema, so the pipeline still runs.

```bash
pip install -e ".[dev]"     # from the ddm4bio repo root
pip install pyreadr         # to fetch the REAL warfarin data
```

## Run it

```bash
make data      # fetch + cache the warfarin dataset
make run       # python src/pipeline.py
make report    # lint report.md for the mandated sections
```

`make run` prints the data provenance (real vs fallback), the QC report, the
per-subject PK fit and recovered half-life, the exposure→effect check, the
clearance spread that calibrates the environment, the RL recovery of the optimal
policy, the time-in-therapeutic-range of the learned policy versus the best fixed
dose, and the interpretation block.

## What the reference does

`src/pipeline.py` runs the whole arc on the **real** warfarin PK/PD data:

1. **Load + QC** (`get_dataset("warfarin")`) — inspect subjects, sampling, doses,
   covariates, and both measurement channels *before* modeling.
2. **Fit PK** (`scipy.optimize`) — a one-compartment oral model per subject;
   validates that the median elimination half-life lands in warfarin's known
   range (~1–2.5 days).
3. **Characterize PD** — correlates drug exposure with the depth of the
   anticoagulation effect (the dose → exposure → effect chain).
4. **Calibrate** (`PKDosingEnv`) — turns the fitted per-patient clearance spread
   into the environment's stochastic clearance.
5. **Learn the policy** (`value_iteration` + `q_learning`) — value iteration is
   the model-based optimum; Q-learning recovers it model-free.
6. **Validate** (`policy_value`) — the learned policy's time-in-range versus the
   best fixed dose.
7. **Interpret** (`interpretation_block`) — a calibrated claim and named
   limitations.

## Make it yours

The reference is intentionally minimal. Deepen it — each is a place to earn marks:

- Replace the per-subject PK fit with a **population** fit, or add a **turnover /
  indirect-response PD** model and drive the therapeutic target from the *effect*
  (INR), not a normalized concentration.
- Characterize variability with **PCA/ICA + stability clustering** of the PK/PD
  parameters, and test whether **covariates** (weight, age) explain clearance.
- Validate on **held-out patients**, not just the fixture.
- Keep QC first, keep a ground-truth control (the known half-life; the
  value-iteration optimum), keep the confidence calibrated to the evidence.
- Fill in `report.md` (run `make report`; Quality Control must precede Results),
  and pin your seed so `make run` reproduces your headline numbers.
