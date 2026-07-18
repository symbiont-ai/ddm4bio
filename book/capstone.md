# Capstone — Warfarin PK/PD to a Dosing Policy

Your capstone is a single **designed** project: you carry real warfarin
pharmacokinetic / pharmacodynamic (PK/PD) data through the whole course arc — from
raw measurements to a defensible **dosing policy** — built around a genuine
model-driven ↔ data-driven hybrid. Everyone works the same problem, so what is
graded is depth and rigor, not novelty of topic.

You **fit** a mechanistic PK/PD model to real data (model-driven), **calibrate** a
dosing environment from the fitted patient-to-patient variability, and **learn** a
dosing policy by reinforcement learning (data-driven) — the value-iteration →
Q-learning loop previewed in [Week 8](lessons/wk8_deep_learning.ipynb), now grounded
in a real drug. The mechanistic fit *is* what calibrates the environment the
policy is optimized on: that is the course's "combine the two paradigms" made
concrete.

The full brief, rubric, and a runnable reference pipeline live in the course
repository:

- **[Capstone brief](https://github.com/symbiont-ai/ddm4bio/blob/main/capstone/README.md)** — the seven-step arc, the three milestones (Proposal → Checkpoint → Final), and how to start.
- **[Rubric](https://github.com/symbiont-ai/ddm4bio/blob/main/capstone/rubric.md)** — the six weighted criteria and the four pass/fail non-negotiables (QC before conclusions, no silent method swaps, ground-truth validation, reproducibility).
- **[Reference pipeline](https://github.com/symbiont-ai/ddm4bio/tree/main/capstone/template)** — a runnable skeleton (`make data && make run && make report`) that runs the entire arc on the real data (and a synthetic PK/PD fallback offline). It is deliberately minimal; your job is to deepen each step.

```{admonition} Not a clinical tool
:class: warning
This is a pedagogical pipeline. Real warfarin dosing uses validated nomograms and
INR monitoring; nothing here prescribes anything for real patients, and your
interpretation must say so.
```
