# Capstone Rubric

The capstone is graded on a 100-point scale across six weighted criteria, **on
top of** four non-negotiable standards. The weighted criteria decide your grade
*once* the non-negotiables are met. If a non-negotiable is violated, the work is
returned for revision before any weighted score is assigned — a brilliant method
built on unvalidated data or an irreproducible pipeline does not earn partial
credit for brilliance.

---

## Weighted criteria (100%)

### Scientific framing — 15%
Are the modeling choices justified, and does the pipeline actually answer the
warfarin dosing question?

- **Excellent** — The PK/PD model, the therapeutic target, and the calibration
  are justified by the data and the pharmacology, and the reader understands why
  the mechanistic fit *and* the learned policy — connected, the fit calibrating
  the policy — answer the dosing question on this data.
- **Adequate** — Reasonable modeling choices, but the link between the fit and
  the policy (or how the therapeutic target is defined) is asserted rather than
  argued.
- **Weak** — The reference pipeline run essentially unchanged, or modeling
  choices left unexplained and the fit and policy not clearly connected.

### Method rigor & correctness — 25%
Are the two methods used correctly, with the right assumptions, parameters, and
diagnostics?

- **Excellent** — Both methods (one per group) applied with correct array
  orientation, sensible hyperparameters justified by diagnostics (e.g. singular-
  value spectrum for rank choice), and correct handling of their assumptions.
  Every reported method maps to a named `ddm4bio` function.
- **Adequate** — Methods are correct but some choices are unexplained or
  defaults are used without justification.
- **Weak** — Misapplied method, wrong orientation, rank/parameter chosen
  arbitrarily, or an assumption silently violated.

### Data quality control — 20%
Was the data inspected and cleaned **before** any modeling, and is the QC
reported honestly?

- **Excellent** — QC runs before results; missingness, duplicates, constant
  columns, outliers, class balance, and leakage are checked and reported;
  every drop/keep decision is justified and its effect on the conclusion
  considered.
- **Adequate** — QC is present and precedes results but is shallow, or some
  decisions are unexplained.
- **Weak** — QC missing, performed after modeling, or conclusions drawn on data
  never inspected.

### Interpretation & calibration — 15%
Does the interpretation match the evidence, with a confidence level the evidence
actually supports?

- **Excellent** — Claims carry an explicit, *calibrated* confidence (low /
  moderate / high) that tracks the validation evidence; uncertainty is
  quantified; the conclusion neither over- nor under-claims.
- **Adequate** — Confidence is stated but only loosely justified.
- **Weak** — Overclaiming, no confidence level, or a confident conclusion
  unsupported by any validation.

### Reproducibility — 15%
Can someone else reproduce your numbers from a clean clone?

- **Excellent** — `make data && make run && make report` reproduces every
  headline number within a documented tolerance; seeds and dependencies pinned;
  environment documented; no hidden manual steps.
- **Adequate** — Reproduces with minor undocumented tweaks.
- **Weak** — Numbers cannot be regenerated; missing seeds; hard-coded local
  paths; hand-edited outputs.

### Communication — 10%
Is the report clear, correctly structured, and honest?

- **Excellent** — All mandated sections present and in order; figures labeled
  and referenced; writing is concise; limitations stated plainly.
- **Adequate** — Understandable but uneven, or a section is thin.
- **Weak** — Missing sections, unreadable figures, or buried/omitted
  limitations.

| Criterion | Weight |
|-----------|-------:|
| Scientific framing | 15% |
| Method rigor & correctness | 25% |
| Data quality control | 20% |
| Interpretation & calibration | 15% |
| Reproducibility | 15% |
| Communication | 10% |
| **Total** | **100%** |

---

## Non-negotiable standards (pass/fail gates)

These are not scored — they are required. Any violation sends the work back for
revision before a weighted grade is assigned. They encode the habits that
separate a data-driven *result* from a data-driven *mistake*.

1. **QC before conclusions.** Data quality control runs and is reported *before*
   any modeling result or claim. You may not present a conclusion drawn from
   data you have not inspected. The report's Quality Control section must appear
   before its Results section.

2. **No silent method swaps.** The methods you report must be the methods you
   actually ran, named as the `ddm4bio` functions that ran them. If you change a
   method after the proposal, disclose the change and the reason. Reporting one
   method while running another — or quietly substituting a different estimator
   — is a correctness violation, not a rounding error.

3. **Ground-truth validation before trusting real data.** Before you believe a
   number computed on real data, your pipeline must recover a *known* answer on
   a control: here, the PK fit recovering warfarin's **known ~1.5-day half-life** and Q-learning recovering the **value-iteration optimum**; more generally a synthetic fixture, a held-out split, or a published benchmark. "It ran without
   error" is not validation. Show the validation metric.

4. **Reproducibility.** Seeds are pinned (`seed_everything`), dependencies are
   pinned, paths are not hard-coded, and the headline numbers regenerate from a
   clean clone via the `Makefile` targets. If it does not reproduce, it is not
   finished.

---

## How the two axes combine

Think of the non-negotiables as the **gate** and the weighted criteria as the
**score**. Clear the gate, then earn your grade. The gate exists because in
biomedical modeling the cost of a confident, irreproducible, unvalidated
conclusion is not a lost point — it is a wrong answer that looks right.
