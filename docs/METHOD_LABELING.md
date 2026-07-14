# Method Labeling & Honesty Policy

This repository follows an honesty-first labeling policy so that names and
results never overclaim what the code actually does.

## 1. Names state the actual method

Public function names must state the concrete method they implement, not a
generic verb.

- `pca_reduce(...)` — not `reduce(...)`
- `umap_embed(...)` — not `embed(...)`
- `fdr_bh_correct(...)` — not `correct(...)`

If a function's implementation changes to a different method, its name must
change too. A name is a promise about the algorithm.

## 2. "Method note" cells for substitutions

If a notebook substitutes a simpler (or otherwise different) method than the
one its section title implies — for pedagogy, speed, or dependency reasons —
it must include a markdown cell headed exactly:

```
### Method note
```

The cell must state:

1. **What** was substituted (actual method used vs. the nominal method).
2. **Why** (teaching simplicity, runtime, missing dependency, etc.).
3. Any consequence for the interpretation of the results.

Example:

> ### Method note
> This section nominally covers diffusion-map embedding, but substitutes PCA
> for the embedding step to keep runtime under a few seconds on CPU. The
> qualitative cluster separation shown is therefore weaker than a full
> diffusion map would produce; do not read the axes as diffusion components.

## 3. Known-substitution marker + CI grep

When a notebook or module contains a known substitution, it carries a
machine-detectable marker (e.g. a `KNOWN_SUBSTITUTION` token in a cell or
comment). CI and reviewers grep for that marker; whenever it is present, they
also grep for a corresponding `Method note` cell. A marker without a matching
`Method note` fails review.

## 4. Statistical honesty norms

These norms are non-negotiable and are checked in review:

- **FDR control.** Any multiple-comparison result reports the correction used
  (e.g. Benjamini–Hochberg) and the target FDR. Raw per-test p-values are
  never presented as if significant without correction.
- **No leakage.** Feature selection, scaling, and any fitting that learns from
  data are fit on training folds only, then applied to held-out data. No step
  may see test labels or test-derived statistics before evaluation.
- **QC before results.** Quality-control filtering (cell/sample/feature QC)
  is performed and reported *before* any results, and the QC thresholds are
  stated. Results computed on unfiltered data must be labeled as such.

Reviewers may reject any output whose naming, notebook cells, or reporting
violate the above.
