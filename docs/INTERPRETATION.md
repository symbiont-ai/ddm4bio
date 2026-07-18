# Interpretation Convention

Every ddm4bio notebook must close with an **Interpretation** section that states,
explicitly and honestly, what the results show and where they are weak. This is a
hard convention, not a suggestion: reviewers reject notebooks whose interpretation
overreaches the evidence or hides its weaknesses.

An interpretation section has two mandatory parts:

1. **A claim.** One concrete, plain-language statement of what the results show,
   stated *with whatever quantitative backing belongs in it* (effect size, sample
   size, held-out accuracy, agreement with an independent method, etc.). The claim
   carries its own evidence; it should not need a separate justification line.
2. **A named limitations list.** Specific, named limitations — not a generic
   "results may vary". Each limitation names the actual thing that constrains the
   interpretation (small n, no held-out set, substituted method, batch confound,
   unfiltered QC, possible leakage, etc.), including anything that would make the
   claim optimistic.

There is deliberately **no separate confidence rating**. A one-word label
(`low`/`moderate`/`high`) tends to confuse more than it helps and invites arguing
over the label instead of the substance. How much to trust the claim belongs *in
the claim* — precise numbers, honest comparisons — and in the *limitations* that
qualify it. If the evidence is weak, say so in the claim and name why in the
limitations, rather than stamping a level on it.

## Formatting with `ddm4bio.interpret`

The `ddm4bio.interpret` module formats the block as Markdown so every notebook
looks the same and it renders as wrapping prose, not an overflowing line of
monospaced `print` output. Build the block with `interpretation_block`, then
display it with `show_interpretation`:

```python
from ddm4bio.interpret import interpretation_block, show_interpretation

block = interpretation_block(
    claim=(
        "The two treatment groups separate along the first component "
        "(5-fold CV AUC = 0.81, stable across folds)."
    ),
    limitations_list=[
        "n = 24 samples; underpowered for subtle effects",
        "single batch -- batch and treatment are confounded",
        "PCA substituted for diffusion map (see Method note)",
    ],
)
show_interpretation(block)
```

renders (as Markdown):

> **Interpretation.** The two treatment groups separate along the first component
> (5-fold CV AUC = 0.81, stable across folds).
>
> **Limitations:**
> - n = 24 samples; underpowered for subtle effects
> - single batch -- batch and treatment are confounded
> - PCA substituted for diffusion map (see Method note)

The limitations block can be used on its own:

- `limitations(items)` — the named-limitations block only. An empty list renders
  as `- none stated` rather than vanishing, so a missing limitations list is
  always visible.

## Relationship to the honesty policy

This convention is the notebook-facing counterpart of the repository honesty
policy in [`METHOD_LABELING.md`](METHOD_LABELING.md):

- If a notebook **substitutes a method** (e.g. PCA in place of a diffusion map),
  that substitution is documented in a `### Method note` cell *and* named again in
  the limitations list.
- The **statistical honesty norms** in `METHOD_LABELING.md` (FDR control, no
  leakage, QC-before-results) are the backing you state in the claim — and any of
  them being unmet is a named limitation.

A claim is a promise about the evidence, exactly as a function name is a promise
about the algorithm. The interpretation block is where that promise is made
explicit and auditable.
