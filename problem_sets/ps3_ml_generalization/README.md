# PS3 — From Ranking to Deciding: Thresholds and the Base-Rate Trap

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/symbiont-ai/ddm4bio/blob/main/problem_sets/ps3_ml_generalization/ps3_colab.ipynb)

**Work in the browser:** click the badge to open this problem set in Google Colab — no local setup required. You can also work locally (see below).

**Reading:** Kutz, *Data-Driven Modeling & Scientific Computation*, Chapter 13, sections 1–2
(statistical methods — probability and Bayes' rule), plus any introduction to sensitivity, specificity,
and Bayes' rule for predictive values.

Module 3 measured a classifier's **generalization** (accuracy); its scores also **rank**
patients. But a deployed test does not just rank — it **decides**: it flags a patient or clears them,
at a threshold. AUC is silent on where that threshold goes, and on how the same test
behaves in a different population. This problem set is about the decision and its
traps, on the real UCI heart-disease cohort.

The trained classifier and its held-out scores (`load_heart_scores`) are
**provided** — this problem set is about what you *do* with the scores, not training
the model. Fill in the functions marked `# TODO` in `student/ps3.py`. The autograder
checks each on small hand-verified inputs, so keep the signatures exactly as given.

## Data

Real **UCI Heart Disease** (Cleveland) via `get_dataset("heart_uci")` — 303 patients
× 13 clinical features, binary disease label. `load_heart_scores` mean-imputes the
missing values, standardizes, trains a logistic model, and returns the held-out
predicted probabilities and labels. Seed everything through
`ddm4bio.seed_everything()` (called in `main`).

## Part A — Choosing the operating threshold

AUC is threshold-free; a clinic needs a cutoff. Implement:

- `sensitivity_specificity(scores, labels, threshold)` — flag `score >= threshold`;
  return sensitivity (of the diseased, the fraction caught) and specificity (of the
  healthy, the fraction cleared).
- `cost_optimal_threshold(scores, labels, candidate_thresholds, fn_cost, fp_cost)` —
  the threshold minimizing `fn_cost·FN + fp_cost·FP`, and the cost curve. A costlier
  missed case pushes the cutoff down toward higher sensitivity.
- `threshold_for_sensitivity(scores, labels, candidate_thresholds, target)` — the
  most specific threshold whose sensitivity meets a clinical floor.

## Part B — The base-rate trap

A threshold's sensitivity and specificity do not depend on how common the disease
is — but its **predictive value** does. Implement:

- `ppv_at_prevalence(sensitivity, specificity, prevalence)` — Bayes' positive
  predictive value.
- `npv_at_prevalence(sensitivity, specificity, prevalence)` — Bayes' negative
  predictive value.
- `ppv_curve(sensitivity, specificity, prevalences)` — PPV across prevalences,
  exposing how a great referral-clinic test can be near-useless for screening.

## Quality control & interpretation (required)

The provided `run_qc` reports the held-out prevalence and the ranking AUC, and notes
that AUC says nothing about the cutoff — printed before any result. The provided
`main` closes with a `ddm4bio.interpret.interpretation_block`: state how much the
operating-point analysis supports the claim, and name
the real limitations — sensitivity/specificity are small-sample estimates, the cost
ratio is stipulated, and PPV/NPV assume the operating characteristics transfer to the
new-prevalence population (distribution shift can violate this).

## Files

- `student/ps3.py` — your working file; fill in every `# TODO`.
- `rubric.md` — how this problem set is graded.
- `ps3_colab.ipynb` — one-click Google Colab launcher (badge at the top).
- The reference solution and the autograder are provided through the course and
  run automatically by GitHub Classroom.

## Running

```bash
python student/ps3.py          # runs until the first unimplemented function
```

To work in the browser instead, click the Colab badge at the top of this file.
The autograder runs automatically when you push to GitHub Classroom.
