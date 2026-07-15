"""Student template for PS3: from ranking to deciding.

Week 3 measured how well a classifier *ranks* patients (AUC, generalization). But a
deployed test does not rank -- it *decides*: it flags a patient or clears them, at a
threshold. This problem set is about that decision and its traps, on the real UCI
heart-disease cohort.

- Part A -- **choosing the operating threshold**: pick a cutoff for a stated cost
  trade-off and for a target sensitivity, and read off the sensitivity/specificity
  you actually get.
- Part B -- **the base-rate trap**: a threshold's sensitivity/specificity do not
  depend on how common the disease is, but its predictive value does -- recompute
  PPV / NPV as prevalence varies.

Fill in every function body marked ``# TODO``. The classifier and its held-out
scores (`load_heart_scores`), the QC driver, and `main` are provided -- this problem
set is about what you *do* with the scores, not training the model. The autograder
imports these functions by name, so keep the signatures exactly as given. Run with
``python ps3.py``; it stops at the first unimplemented function.
"""

from __future__ import annotations

import numpy as np

from ddm4bio.config import GLOBAL_SEED, seed_everything
from ddm4bio.interpret import interpretation_block

# --------------------------------------------------------------------------- #
# Provided: real held-out scores + QC (do not edit)                            #
# --------------------------------------------------------------------------- #


def load_heart_scores(
    test_size: float = 0.4, seed: int = GLOBAL_SEED
) -> tuple[np.ndarray, np.ndarray, str]:
    """Train a logistic model on real UCI heart data; return held-out scores.

    Returns ``(scores, labels, source)`` -- predicted disease probabilities on the
    held-out patients and their 0/1 labels. Missing values are mean-imputed.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    from ddm4bio.datasets import get_dataset

    ds = get_dataset("heart_uci", seed=seed)
    x = np.asarray(ds.payload["X"], dtype=float)
    x = np.where(np.isnan(x), np.nanmean(x, axis=0), x)
    y = np.asarray(ds.payload["y"], dtype=int)
    x_tr, x_te, y_tr, y_te = train_test_split(
        x, y, test_size=test_size, random_state=seed, stratify=y
    )
    mean, std = x_tr.mean(axis=0), x_tr.std(axis=0) + 1e-12
    clf = LogisticRegression(max_iter=2000).fit((x_tr - mean) / std, y_tr)
    scores = clf.predict_proba((x_te - mean) / std)[:, 1]
    return scores, y_te, ds.source


def run_qc(scores: np.ndarray, labels: np.ndarray) -> None:
    """Print a QC block before any results (provided)."""
    from sklearn.metrics import roc_auc_score

    in_range = bool(np.all((scores >= 0.0) & (scores <= 1.0)))
    prevalence = float(np.mean(labels))
    auc = float(roc_auc_score(labels, scores))
    print(f"QC: {len(labels)} held-out patients, disease prevalence = {prevalence:.2f}.")
    print(f"    scores are probabilities in [0, 1]: {in_range}; ranking AUC = {auc:.3f}.")
    print("    AUC is threshold-free -- it says nothing about where to set the cutoff.")


# --------------------------------------------------------------------------- #
# Part A -- Choosing the operating threshold  (you implement)                  #
# --------------------------------------------------------------------------- #


def sensitivity_specificity(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    """Sensitivity and specificity when patients with ``score >= threshold`` are flagged.

    Sensitivity = TP / (TP + FN) (of the diseased, the fraction caught);
    specificity = TN / (TN + FP) (of the healthy, the fraction cleared).
    """
    # TODO: flag rows where score >= threshold; count TP/FN/TN/FP against labels
    # (1 = disease, 0 = healthy); return {"sensitivity": TP/(TP+FN),
    # "specificity": TN/(TN+FP)}, guarding empty denominators.
    raise NotImplementedError("Implement sensitivity_specificity.")


def cost_optimal_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    candidate_thresholds: list[float],
    fn_cost: float,
    fp_cost: float,
) -> tuple[float, np.ndarray]:
    """Threshold minimizing expected cost ``fn_cost * FN + fp_cost * FP``.

    Returns ``(best_threshold, cost_by_threshold)``. A high ``fn_cost`` (missing
    disease is expensive) pushes the threshold down toward higher sensitivity.
    """
    # TODO: for each candidate threshold, count FN and FP and form the weighted
    # cost; best_threshold is the candidate with the smallest cost. Return
    # (best_threshold, cost_by_threshold as an array).
    raise NotImplementedError("Implement cost_optimal_threshold.")


def threshold_for_sensitivity(
    scores: np.ndarray,
    labels: np.ndarray,
    candidate_thresholds: list[float],
    target_sensitivity: float,
) -> float:
    """Most specific threshold whose sensitivity is at least ``target_sensitivity``.

    Among thresholds that catch enough of the diseased, pick the highest (most
    specific). Fall back to the lowest candidate if none qualify.
    """
    # TODO: keep the candidate thresholds whose sensitivity_specificity(...)
    # sensitivity >= target_sensitivity; return the largest such threshold (or the
    # smallest candidate if none qualify).
    raise NotImplementedError("Implement threshold_for_sensitivity.")


# --------------------------------------------------------------------------- #
# Part B -- The base-rate trap  (you implement)                                #
# --------------------------------------------------------------------------- #


def ppv_at_prevalence(sensitivity: float, specificity: float, prevalence: float) -> float:
    """Positive predictive value at a given disease ``prevalence`` (Bayes' rule).

    PPV = (sens * prev) / (sens * prev + (1 - spec) * (1 - prev)).
    """
    # TODO: apply the Bayes formula above; guard a zero denominator.
    raise NotImplementedError("Implement ppv_at_prevalence.")


def npv_at_prevalence(sensitivity: float, specificity: float, prevalence: float) -> float:
    """Negative predictive value at a given disease ``prevalence`` (Bayes' rule).

    NPV = (spec * (1 - prev)) / (spec * (1 - prev) + (1 - sens) * prev).
    """
    # TODO: apply the Bayes formula above; guard a zero denominator.
    raise NotImplementedError("Implement npv_at_prevalence.")


def ppv_curve(sensitivity: float, specificity: float, prevalences: list[float]) -> np.ndarray:
    """PPV across a range of prevalences at fixed sensitivity/specificity."""
    # TODO: return np.array([ppv_at_prevalence(sensitivity, specificity, p)
    # for p in prevalences]).
    raise NotImplementedError("Implement ppv_curve.")


# --------------------------------------------------------------------------- #
# Provided: driver                                                             #
# --------------------------------------------------------------------------- #


def main() -> None:
    """Choose an operating point on real heart data and expose the base-rate trap."""
    seed_everything()
    scores, labels, source = load_heart_scores()
    print(f"Application data: UCI Heart Disease via get_dataset -> source={source}\n")

    print("== Quality control (before results) ==")
    run_qc(scores, labels)

    grid = [round(float(t), 2) for t in np.linspace(0.05, 0.95, 19)]

    print("\n== Part A: choosing the operating threshold ==")
    for t in (0.3, 0.5, 0.7):
        ss = sensitivity_specificity(scores, labels, t)
        print(
            f"    threshold={t}: sensitivity={ss['sensitivity']:.2f} "
            f"specificity={ss['specificity']:.2f}"
        )
    best_cost, _curve = cost_optimal_threshold(scores, labels, grid, fn_cost=5.0, fp_cost=1.0)
    ss_cost = sensitivity_specificity(scores, labels, best_cost)
    print(
        f"  cost-optimal threshold (a missed case costs 5x a false alarm) = {best_cost}: "
        f"sensitivity={ss_cost['sensitivity']:.2f} specificity={ss_cost['specificity']:.2f}"
    )
    t90 = threshold_for_sensitivity(scores, labels, grid, target_sensitivity=0.9)
    ss90 = sensitivity_specificity(scores, labels, t90)
    print(
        f"  threshold for >=90% sensitivity = {t90}: "
        f"sensitivity={ss90['sensitivity']:.2f} specificity={ss90['specificity']:.2f}"
    )

    print("\n== Part B: the base-rate trap ==")
    sens, spec = ss_cost["sensitivity"], ss_cost["specificity"]
    prevalences = [0.5, 0.2, 0.1, 0.05, 0.02]
    ppvs = ppv_curve(sens, spec, prevalences)
    print(f"  at the cost-optimal operating point (sens={sens:.2f}, spec={spec:.2f}):")
    for p, v in zip(prevalences, ppvs):
        print(f"    prevalence={p:>4.2f}  PPV={v:.2f}")
    npv_screen = npv_at_prevalence(sens, spec, 0.02)
    print(f"  NPV at screening prevalence 0.02 = {npv_screen:.3f}")

    print("\n== Interpretation ==")
    block = interpretation_block(
        claim=(
            f"The same heart-disease classifier decides differently depending on the "
            f"question: a cost-optimal cutoff reaches {sens:.0%} sensitivity, but its "
            f"positive predictive value falls from {ppvs[0]:.2f} in a high-prevalence clinic "
            f"to {ppvs[-1]:.2f} at screening prevalence -- the same test, a very different tool."
        ),
        confidence="high",
        limitations_list=[
            "Sensitivity and specificity are single held-out estimates (n small); a "
            "different split shifts the chosen threshold.",
            "The cost ratio and target sensitivity are stipulated, not derived from a real "
            "clinical utility model.",
            "PPV/NPV assume the held-out sensitivity/specificity transfer unchanged to the "
            "new-prevalence population, which distribution shift can violate.",
        ],
        evidence=(
            "threshold-swept sensitivity/specificity on real data, a cost-minimizing "
            "operating point, and Bayes' PPV/NPV across prevalences"
        ),
    )
    print(block)


if __name__ == "__main__":
    main()
