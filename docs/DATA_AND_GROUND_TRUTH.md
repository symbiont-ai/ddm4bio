# Real Data vs. Synthetic Ground Truth

This course runs on **real biomedical data**, and it also uses **synthetic
fixtures** in specific, deliberate places. This document states *when* and
*why*, so that "we use real data" is never read as more than it is — and so
that a synthetic fixture is never mistaken for a shortcut.

The whole policy in one sentence:

> **Score a claim on real data whenever the claim *can* be scored on real data.
> Use a synthetic fixture only when the claim is "the method recovered a known
> truth" — because real biological data has no known truth for what those
> methods recover.**

## The three roles data plays

### 1. Real application data (the default)

Every lesson and problem set loads a real dataset through the course data layer
(`get_dataset`; see [`data/DATA_CARD.md`](../data/DATA_CARD.md)) — BloodMNIST
cell crops, the Wisconsin breast-cytology panel, UCI heart-disease records,
MIT-BIH ECG, 10x pbmc3k single cells, the JHU COVID-19 series, and more.
Wherever that dataset carries the label or target the claim is actually about,
the **scored result is computed on the real data** — real inputs, real labels,
real held-out evaluation. For example:

- **Cell recognition** (wk1 / ps1): 1-nearest-neighbour accuracy measured on
  real, held-out BloodMNIST images.
- **Heart-disease classification** (ps3): 5-fold cross-validated AUC on real
  UCI patients.
- **Breast-cytology diagnosis** (ps6): classifier AUC on the real WDBC panel.

When the real source loads, there is nothing synthetic in these numbers. On the
**published site** that is guaranteed: the CI guard in §3 fails the deploy if any
lesson regressed to a fallback. A purely offline run instead uses the
clearly-labelled fallback (§3), as any dataset does — and it prints
`source=fallback`, so you always know which you got.

### 2. Synthetic ground-truth fixtures (for validating recovery)

Some methods do not classify a labelled target — they **recover a quantity you
cannot observe directly**: the independent sources behind a mixture (ICA), the
governing equations behind a trajectory (SINDy / DMD), a sparse signal behind
incomplete measurements (compressed sensing), or the parameters behind a noisy
curve. For these, "is the answer correct?" can only be asked against a **known**
truth — and **real biological data does not come with that truth.** You cannot
measure "did ICA recover the *true* sources" in real single-cell data, because
the true sources are unknown; you cannot score "did SINDy recover the *true*
equation" for a real epidemic, because no one knows the true equation.

So the course does what the research literature does: it **validates the method
on a synthetic problem where the answer is known** — injected sources, a known
ODE such as Lorenz, an exactly *k*-sparse signal — reports the recovery as a
hard number (a recovery correlation, term precision/recall, a reconstruction
error), and **then applies the validated method to the real data**, where it
makes a deliberately narrower, honest claim (a short-horizon forecast, a
qualitative embedding, a roughness reduction). For example:

- **Blind source separation** (wk5 / ps5): ICA recovery is scored on a synthetic
  recording with injected known sources; real pbmc3k drives a *qualitative* PCA
  read, because its true latent sources are unknown.
- **System identification** (wk7 / ps7): SINDy / DMD term recovery is scored on
  synthetic fixtures with known dynamics; real JHU-COVID and MIT-BIH drive the
  narrower *applied* claims (a forecast, a filtered signal).
- **Compressed sensing** (wk4 / ps4): exact recovery is scored on a known sparse
  field; the real MIT-BIH ECG is used for the denoising demonstration.

This is a **methodological necessity, not a convenience.** Swapping in real data
here would not make the recovery score *more real* — it would make it
*unmeasurable*, because there would be no ground truth to compare against.

### 3. Labeled fallbacks (a safety net, not a choice)

The data layer is offline-first. If a real source is unreachable — no network,
or a missing optional dependency — `get_dataset` returns a **clearly-labelled**
synthetic/bundled fallback (`source="fallback"`, provenance beginning
`"synthetic/bundled fallback: …"`) with the same payload shape, so a run never
crashes. This is a robustness fallback, **not** a pedagogical fixture, and it is
**not** meant to appear on the published site. Two safeguards keep it honest:

- Every load **prints its source and provenance**, so a fallback is always
  visible, never disguised as real.
- A **CI guard** in the deploy workflow greps the built lessons for the fallback
  markers and **fails the deploy** if any lesson regressed to a fallback (for
  example, a data dependency missing on the runner). The live site therefore
  shows real data, or it does not publish.

## How to tell which is which

You never have to guess:

1. **The source line.** Every dataset load prints `source=real` or
   `source=fallback` and a provenance string. Real reads name the origin
   (PhysioNet, 10x Genomics, UCI, Zenodo); fallbacks begin
   `synthetic/bundled fallback: …`.
2. **The interpretation block.** When a *scored* result comes from a synthetic
   fixture, the closing interpretation block says so in its evidence or
   limitations (see [`INTERPRETATION.md`](INTERPRETATION.md)). For instance, the
   compressed-sensing limitation reads: "Result is on a synthetic fixture that
   is exactly *k*-sparse…".
3. **Method notes.** A *method substitution* (a simpler method than a section
   implies) carries a `Method note` (see
   [`METHOD_LABELING.md`](METHOD_LABELING.md)) — a separate honesty mechanism
   from data provenance.

## The rule of thumb, restated

| When the claim is… | It is scored on… | Because… |
|---|---|---|
| "this classifies / predicts a labelled target" | **real data** | the label exists in the real data, so a real evaluation is possible |
| "this recovered an unobservable truth" | a **synthetic fixture** (then applied to real data) | scoring recovery needs a *known* truth, which real biological data lacks |
| (a real source was unreachable at build time) | a **labeled fallback** | offline safety only — always surfaced, and kept off the published site by the CI guard |

If you ever find a place where a claim *could* be scored on real data but isn't
— where a real, labelled benchmark exists for exactly that method — treat it as
a **bug in this policy, not an instance of it**, and open an issue. The synthetic
fixtures are there because recovery has no real ground truth, not because real
data was inconvenient.

## See also

- [`data/DATA_CARD.md`](../data/DATA_CARD.md) — the per-dataset registry, licenses, access tiers, and each loader's fallback.
- [`METHOD_LABELING.md`](METHOD_LABELING.md) — honest method naming and the `Method note` substitution policy.
- [`INTERPRETATION.md`](INTERPRETATION.md) — the mandatory claim / confidence / limitations block that closes every notebook.
