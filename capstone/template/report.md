# Capstone Report — <title>

**Author:** <your name>  ·  **Track:** <single-cell / neuro / physiological / imaging / epidemic / sequence>  ·  **Date:** <date>

> Keep every section header below. `make report` fails if any mandated section
> is missing, and Quality Control **must** appear before Results. Replace the
> italic prompts with your own content; delete none of the headings.

---

## Framing

*State the biomedical question in one or two sentences. Why does it matter?
What would a good answer let someone do or decide? Name the two methods you use
(one from Group A: SVD/PCA, robust PCA, ICA, wavelets/CS; one from Group B:
supervised/unsupervised ML, DMD/SINDy, Kalman, autoencoder) and explain why the
question demands them and how they connect.*

## Methods

*Describe the pipeline step by step. Label each method explicitly with the
`ddm4bio` function that ran it.*

- **Group A — <name>** (`ddm4bio.methods...`): *what it does here, key
  parameters, and how you chose them (e.g. rank from the singular-value
  spectrum).*
- **Group B — <name>** (`ddm4bio.methods...`): *what it does here, its inputs
  from the Group-A step, key parameters and their justification.*

*If you changed methods since your proposal, say so here and say why (no silent
method swaps).*

## Quality Control

*Reported BEFORE any result. Summarize the QC checks and what they found:
shape, missingness, duplicates, constant columns, outliers, class balance,
leakage. State every drop/keep decision and its justification, and note whether
any QC finding could change the conclusion.*

## Results

*Present the outcome of the pipeline: figures, tables, metrics. Include the
**ground-truth validation** result — the known answer your pipeline recovered on
a synthetic fixture, held-out split, or control — and only then the result on
real data. Reference each figure in the text.*

## Interpretation

*Interpret the results against the question. State an explicit, calibrated
confidence level (low / moderate / high) and justify it from the validation
evidence. Quantify uncertainty. Do not over- or under-claim.*

## Limitations

*List the honest limitations: data caveats, assumptions your methods make,
parameter sensitivity, and what would change your conclusion. Be specific.*

## Reproducibility

*How to regenerate every headline number from a clean clone: the seed
(`seed_everything`), pinned dependencies, the exact commands
(`make data && make run && make report`), and any documented tolerance. No
hidden manual steps.*
