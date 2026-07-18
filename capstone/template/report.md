# Capstone Report — Warfarin PK/PD to a Dosing Policy

**Author:** <your name>  ·  **Date:** <date>

> Keep every section header below. `make report` fails if any mandated section is
> missing, and Quality Control **must** appear before Results. Replace the italic
> prompts with your own content; delete none of the headings.

---

## Framing

*State the question in one or two sentences: can a dosing policy learned on a
warfarin PK/PD model fit from real data hold the anticoagulation effect in a
therapeutic window, and how does patient-to-patient variability limit it? Say why
it matters, and name your two connected pieces — the mechanistic PK/PD fit
(model-driven) and the RL dosing policy (data-driven) — and how the first
calibrates the second.*

## Methods

*Describe the pipeline step by step, each labeled with the `ddm4bio` function that
ran it.*

- **PK model (Group A)** (`scipy.optimize` / your fit): *the compartment model,
  how you fit it, and how you chose it.*
- **PD / effect** : *how you related exposure to the anticoagulation effect and
  set the therapeutic target.*
- **Variability** (`pca_reduce` / clustering): *how you summarized inter-patient
  spread and whether covariates explain it.*
- **Environment + policy (Group B)** (`PKDosingEnv`, `value_iteration`,
  `q_learning`): *how the fit calibrates the environment and how you learned and
  checked the policy.*

*If you changed methods since your proposal, say so here and why (no silent
method swaps).*

## Quality Control

*Reported BEFORE any result. Summarize the QC of the real warfarin data: number
of subjects, sampling per subject, dose and covariate ranges, missingness, both
measurement channels (cp / pca), and any outliers. State every drop/keep decision
and whether a QC finding could change the conclusion. Note the data provenance
(real vs synthetic fallback).*

## Results

*Present the outcome, with the **ground-truth validation** first: the PK fit
recovering warfarin's known ~1–2.5-day half-life, and Q-learning reaching the
value-iteration optimum. Then the headline result — the learned policy's time in
the therapeutic range versus the best fixed dose — with figures referenced in the
text.*

## Interpretation

*Interpret the results against the question. State an explicit, calibrated
confidence level (low / moderate / high) justified by the validation evidence,
and quantify uncertainty. Do not over- or under-claim.*

## Limitations

*List the honest limitations: the compartment-model and discretization
approximations, the size of the variability sample (32 subjects), the modeling
choices in the therapeutic window and toxicity penalty, and — explicitly — that
this is pedagogical, not clinical, dosing.*

## Reproducibility

*How to regenerate every headline number from a clean clone: the seed
(`seed_everything` / the pinned RL seed), pinned dependencies, the exact commands
(`make data && make run && make report`), and any documented tolerance. No hidden
manual steps.*
