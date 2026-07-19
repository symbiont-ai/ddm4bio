# Capstone — Warfarin PK/PD to a Dosing Policy

The capstone is where the course stops being a tour of methods and becomes a
single piece of scientific work. You carry **real warfarin pharmacokinetic /
pharmacodynamic (PK/PD) data** through the whole `ddm4bio` arc — from raw
measurements to a **defensible dosing policy** — and defend not just the answer
but the *process* that produced it.

This is a **designed** project: one dataset, one multi-step pipeline, one hybrid
at its core. Everyone works the same problem, so what is graded is depth and
rigor, not novelty of topic.

---

## Why warfarin

Warfarin is the ideal teaching drug: a narrow therapeutic index, real bleeding
toxicity, large patient-to-patient variability, and — crucially — a *measurable
effect*. The dataset is real: 32 subjects, a single oral dose, plasma warfarin
concentration (`dvid == "cp"`) **and** the anticoagulation effect PCA
(prothrombin-complex activity, `dvid == "pca"`, a stand-in for INR) over ~6 days,
with weight / age / sex covariates. Source: O'Reilly & Aggeler (1968),
popularized by Holford (1986), distributed in the `nlmixr2data` package (GPL ≥3).

```python
from ddm4bio.datasets import get_dataset
ds = get_dataset("warfarin")   # real data, cached; synthetic PK/PD fallback offline
df = ds.payload                # tidy: id, time, amt, dv, dvid, evid, wt, age, sex
```

## The hybrid at the core

The capstone braids the course's two paradigms into one pipeline:

- **Model-driven:** you *fit a mechanistic PK/PD model* to real concentration and
  effect data — a model whose parameters mean something and can be validated
  against known pharmacology.
- **Data-driven:** you then *learn a dosing policy* by reinforcement learning on
  a dosing environment **calibrated from that fit**.

A mechanistic model, grounded in real data, becomes the environment a learned
policy is optimized on. That is the "combine the paradigms" of the course's first
learning outcome, made concrete.

## What you build — the arc

| Step | Group | What you do | `ddm4bio` |
|------|-------|-------------|-----------|
| 1. QC + frame | — | Load warfarin; inspect sampling, doses, covariates, both channels, outliers **before** any modeling | `get_dataset` |
| 2. Fit PK | A (model) | Fit a one-compartment oral PK model per subject; **validate** the recovered elimination half-life against warfarin's known ~1–2.5 days | `scipy.optimize` (wk2) |
| 3. Fit PD | A (model) | Characterize the anticoagulation effect and its link to exposure (a turnover / indirect-response model, or an exposure→effect fit); this defines the therapeutic *effect* target | ODE / optimization (wk2, wk7) |
| 4. Characterize variability | B (data) | Reduce and cluster the per-patient PK/PD parameters; relate the spread to covariates (does weight explain clearance?) | `pca_reduce`, clustering, `select_k_silhouette`, `bh_fdr` (wk5–6) |
| 5. Calibrate the environment | — | Build a dosing environment whose clearance is drawn from the **fitted** patient-to-patient spread | `PKDosingEnv` |
| 6. Learn the policy | B (data) | Recover the optimal dosing policy: value iteration (model-based ground truth) + Q-learning (model-free) | `value_iteration`, `q_learning` (RL preview) |
| 7. Validate + interpret | — | Show the fit recovers known pharmacology, the learner reaches the optimum, and the policy holds the effect in-window **better than a fixed dose**; close with an interpretation block | `policy_value`, `interpretation_block` |

The two methods are *connected by construction*: the Group-A fit **is** what
calibrates the Group-B environment. A pipeline whose steps do not feed each other
does not score well on framing.

## Milestones

Three graded milestones. Dates are on the course schedule; the deliverables are
fixed.

| Milestone | Deliverable | Must contain |
|-----------|-------------|--------------|
| **1 — Proposal** | `proposal.md` (1–2 pages) | Your QC of the real data; your PK (and PD) model choice, named as the fit you will run; your **validation plan** (which known quantity — e.g. the half-life — anchors the fit); the risk you are most worried about. |
| **2 — Checkpoint** | Running pipeline through Step 6 + a progress note | PK fit that **recovers the known half-life**, the variability characterization, a calibrated environment, and Q-learning recovering the value-iteration optimum on a fixture. The "prove it before you trust it" gate. |
| **3 — Final** | Report + code + presentation | `report.md` filled from the template (all mandated sections), the reproducible pipeline (`make data && make run && make report` from a clean clone), and a short talk: fit → variability → policy → the in-range-vs-fixed-dose result → confidence → limitations. |

**Reproducibility bar:** a grader clones your repo, runs `make data && make run`,
and reproduces every headline number in your report (within a documented
tolerance), using pinned seeds and pinned dependencies.

## The reference pipeline

[`template/src/pipeline.py`](template/src/pipeline.py) runs the **entire arc**
end-to-end on the real data (and on a synthetic PK/PD fallback offline). It is
your **skeleton, not your deliverable** — each step is deliberately minimal (a
per-subject PK fit, a three-point clearance spread, a default RL environment).
Your job is to deepen it: a proper turnover PD fit, a covariate-aware variability
model, held-out-patient validation, a therapeutic window tied to an INR target.

```bash
cp -r capstone/template capstone/submissions/<your-name>
cd capstone/submissions/<your-name>
make data      # fetch + cache the warfarin dataset
make run       # load -> QC -> PK/PD fit -> variability -> calibrate -> RL -> validate
make report    # check the report has every mandated section, in order
```

## Read the rubric first

The [`rubric.md`](rubric.md) **non-negotiable standards** (QC before conclusions,
no silent method swaps, ground-truth validation before trusting results,
reproducibility) are pass/fail gates, not point deductions.

> **Not a clinical tool.** This is a pedagogical pipeline. Real warfarin dosing
> uses validated nomograms and INR monitoring; nothing here prescribes anything
> for real patients, and your interpretation must say so.
