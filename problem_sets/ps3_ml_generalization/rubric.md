# PS3 Grading Rubric — From Ranking to Deciding

Total: **100 points**. Grading combines the autograder (`tests/test_ps3.py`)
with a short read of the submitted code and interpretation. All work must run
offline and deterministically; a submission that only passes because it hard-codes
expected numbers or disables a check earns no credit for the affected part.

## Part A — Choosing the operating threshold — 40 points

- **`sensitivity_specificity` (16).** Correct TP/FN/TN/FP counting for
  `score >= threshold`; sensitivity = TP/(TP+FN), specificity = TN/(TN+FP), with
  empty denominators guarded. A threshold of 0 flags everyone (sensitivity 1,
  specificity 0).
- **`cost_optimal_threshold` (14).** Returns the cost curve and the cost-minimizing
  threshold; a larger `fn_cost` moves the chosen threshold no higher (toward more
  sensitivity), never the wrong way.
- **`threshold_for_sensitivity` (10).** Returns the *most specific* threshold that
  still meets the sensitivity floor (not merely any qualifying one), with a sane
  fallback when none qualify.

## Part B — The base-rate trap — 35 points

- **`ppv_at_prevalence` (14).** Exact Bayes PPV; matches hand-computed values
  (e.g. sens=spec=0.9 gives PPV 0.9 at prevalence 0.5 and 0.5 at prevalence 0.1) and
  falls monotonically as the disease gets rarer.
- **`npv_at_prevalence` (11).** Exact Bayes NPV; matches hand-computed values and
  rises as the disease gets rarer.
- **`ppv_curve` (10).** Agrees pointwise with `ppv_at_prevalence` across a list of
  prevalences and is monotone in prevalence.

## Quality control — 10 points

- **QC before results (6).** The provided `run_qc` (prevalence, score range, ranking
  AUC, and the note that AUC is threshold-free) is printed before any operating-point
  result, per the course "QC before results" rule; the real dataset source is
  reported.
- **Honest use of the scores (4).** The threshold is chosen on the held-out scores,
  not refit to the labels being scored; PPV/NPV are computed from the operating
  characteristics, not peeked from the new-prevalence data.

## Interpretation & honesty — 10 points

- A clear `interpretation_block` -- a claim stated with the evidence
  actually generated (threshold-swept sensitivity/specificity, a cost-minimizing
  operating point, Bayes' PPV/NPV across prevalences).
- At least two honest, specific limitations — small-sample sensitivity/specificity,
  a stipulated cost ratio, and the assumption that operating characteristics transfer
  across prevalence (which distribution shift can break). Overclaiming ("this test is
  accurate") is penalized; the base-rate caveat is rewarded.

## Reproducibility — 5 points

- Runs top-to-bottom with a fixed seed; imports follow the repository policy (only
  `numpy` at module top level, heavier libraries inside function bodies); the file is
  `ruff`-clean under `E`, `F`, `I` at line length 100; and no public function
  signature was changed.
