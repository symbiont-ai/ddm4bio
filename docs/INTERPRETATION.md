# Interpretation Convention

Every ddm4bio notebook must close with an **Interpretation** section that states,
explicitly and honestly, how much the results actually support the claim being
made. This is a hard convention, not a suggestion: reviewers reject notebooks
whose interpretation overreaches the evidence or hides its weaknesses.

An interpretation section has three mandatory parts:

1. **A claim.** One concrete statement about what the results show.
2. **A confidence level** — exactly one of `low`, `moderate`, or `high` — with
   the **evidence** that justifies it (effect size, sample size, held-out
   validation, agreement with an independent method, etc.). A confidence level
   without stated evidence is not acceptable.
3. **A named limitations list.** Specific, named limitations — not a generic
   "results may vary". Each limitation names the actual thing that constrains
   the interpretation (small n, no held-out set, substituted method, batch
   confound, unfiltered QC, etc.).

## Choosing a confidence level honestly

- **low** — suggestive only: small sample, no validation, exploratory, or a
  substituted/simplified method (see `METHOD_LABELING.md`, "Method note").
- **moderate** — supported but not definitive: reasonable sample, some
  validation, but known confounds or a single dataset.
- **high** — well supported: adequate power, held-out or cross-validated
  evidence, corrected for multiple comparisons, robust to the stated
  limitations.

When in doubt, choose the lower level. Overstating confidence is the failure
mode this convention exists to prevent.

## Formatting with `ddm4bio.interpret`

The `ddm4bio.interpret` module formats the block so every notebook looks the
same and nothing is left implicit. Use `interpretation_block` for the full
block:

```python
from ddm4bio.interpret import interpretation_block

print(interpretation_block(
    claim="The two treatment groups separate along the first component.",
    confidence="moderate",
    limitations_list=[
        "n = 24 samples; underpowered for subtle effects",
        "single batch — batch and treatment are confounded",
        "PCA substituted for diffusion map (see Method note)",
    ],
    evidence="5-fold CV AUC = 0.81; separation stable across folds",
))
```

renders:

```
Confidence: MODERATE -- The two treatment groups separate along the first component. (evidence: 5-fold CV AUC = 0.81; separation stable across folds)

Limitations:
- n = 24 samples; underpowered for subtle effects
- single batch — batch and treatment are confounded
- PCA substituted for diffusion map (see Method note)
```

The two components can also be used on their own:

- `confidence_statement(claim, confidence, evidence=None)` — the confidence
  line only. Passing a `confidence` other than `"low"`, `"moderate"`, or
  `"high"` raises `ValueError`, so a typo cannot silently produce an unlabeled
  claim.
- `limitations(items)` — the named-limitations block only. An empty list
  renders as `- none stated` rather than vanishing, so a missing limitations
  list is always visible.

## Relationship to the honesty policy

This convention is the notebook-facing counterpart of the repository honesty
policy in [`METHOD_LABELING.md`](METHOD_LABELING.md):

- If a notebook **substitutes a method** (e.g. PCA in place of a diffusion
  map), that substitution is documented in a `### Method note` cell *and*
  named again in the limitations list, and it should pull the confidence level
  down.
- The **statistical honesty norms** in `METHOD_LABELING.md` (FDR control, no
  leakage, QC-before-results) are the evidence you cite for a `high` confidence
  level — and any of them being unmet is a named limitation that caps
  confidence at `low` or `moderate`.

A claim is a promise about the evidence, exactly as a function name is a
promise about the algorithm. The interpretation block is where that promise is
made explicit and auditable.
