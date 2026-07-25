"""Student template for PS2: letting the data choose the model.

Module 2 fit a model at a complexity you picked. Here you face the real question:
when you do NOT know the truth, how complex a model does the data support? The
answer is **cross-validation** — score each candidate on data it was not fit on,
and let held-out error choose.

- Part A — **model-complexity selection**: fit polynomials of growing degree to a
  noisy response curve and pick the degree that generalizes (cross-validated error
  is U-shaped: underfit high, overfit high).
- Part B — **sparse feature selection**: cross-validate the Lasso penalty to pick
  the biomarker set that generalizes. On synthetic data with a known driver set you
  measure recovery; on real breast-cytology data you read off a panel.

Fill in every function body marked ``# TODO``. You fit polynomials with
`np.polyfit`/`np.polyval` and a Lasso with scikit-learn's `Lasso` directly (import
it inside the function bodies that use it). The fold splitter (`kfold_indices`), the
data generators, the QC driver, and `main` are provided — this problem set is about
*selecting* with them, not re-deriving them. The autograder imports these functions
by name, so keep the signatures exactly as given. Run with ``python ps2.py``; it
stops at the first unimplemented function.
"""

from __future__ import annotations

import numpy as np

from ddm4bio.config import GLOBAL_SEED, seed_everything
from ddm4bio.interpret import interpretation_block

# --------------------------------------------------------------------------- #
# Provided: data generators + fold splitter (do not edit)                      #
# --------------------------------------------------------------------------- #


def make_response_curve(
    degree: int = 3, n: int = 90, noise: float = 1.2, seed: int = GLOBAL_SEED
) -> tuple[np.ndarray, np.ndarray, int]:
    """A noisy 1-D response whose true trend is a polynomial of ``degree``."""
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(-2.0, 2.0, n))
    coefs = rng.uniform(0.8, 1.5, degree + 1) * rng.choice([-1.0, 1.0], degree + 1)
    y_true = np.polyval(coefs, x)
    y = y_true + rng.normal(0.0, noise * float(y_true.std()), n)
    return x, y, degree


def make_sparse_regression(
    n: int = 120, p: int = 60, k: int = 6, noise: float = 1.0, seed: int = GLOBAL_SEED
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Linear regression ``y = X beta + noise`` with only ``k`` nonzero drivers."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, p))
    beta = np.zeros(p)
    support = np.sort(rng.choice(p, k, replace=False))
    beta[support] = rng.uniform(1.5, 3.0, k) * rng.choice([-1.0, 1.0], k)
    y = x @ beta + rng.normal(0.0, noise, n)
    return x, y, support


def load_biomarkers(seed: int = GLOBAL_SEED) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Real Breast-Cancer-Wisconsin measurements, standardized (offline)."""
    from ddm4bio.datasets import get_dataset

    ds = get_dataset("breast_wisconsin", seed=seed)
    x = np.asarray(ds.payload["X"], dtype=float)
    x = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-12)
    y = np.asarray(ds.payload["y"], dtype=float)
    names = list(ds.payload["feature_names"])
    return x, y, names


def kfold_indices(n: int, folds: int = 5, seed: int = GLOBAL_SEED) -> list[np.ndarray]:
    """(provided) Deterministic k-fold: a list of held-out index arrays."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    return [np.sort(part) for part in np.array_split(order, folds)]


# --------------------------------------------------------------------------- #
# Part A -- Model-complexity selection by cross-validation  (you implement)    #
# --------------------------------------------------------------------------- #


def poly_cv_mse(x: np.ndarray, y: np.ndarray, degree: int, folds: list[np.ndarray]) -> float:
    """Cross-validated mean squared error of a degree-``degree`` polynomial fit.

    For each held-out fold: fit the polynomial on the OTHER folds (`np.polyfit`),
    predict the held-out points (`np.polyval`), and average the MSE over folds.
    """
    # TODO: loop over `folds`; for each, train on the complement (np.setdiff1d
    # against np.arange(len(x))), fit np.polyfit(x_train, y_train, degree), predict
    # np.polyval(coefs, x_test), and accumulate the mean squared test error. Return
    # the average over folds.
    raise NotImplementedError("Implement poly_cv_mse.")


def select_degree(
    x: np.ndarray, y: np.ndarray, candidate_degrees: list[int], folds: list[np.ndarray]
) -> tuple[int, np.ndarray]:
    """Degree from ``candidate_degrees`` with the lowest cross-validated MSE.

    Returns ``(best_degree, mse_by_degree)``; the curve falls (underfitting) then
    rises (overfitting).
    """
    # TODO: build mse_by_degree = [poly_cv_mse(x, y, d, folds) for d in
    # candidate_degrees] as an array; best_degree is the candidate with the
    # smallest CV-MSE. Return (best_degree, mse_by_degree).
    raise NotImplementedError("Implement select_degree.")


# --------------------------------------------------------------------------- #
# Part B -- Sparse feature selection by cross-validation  (you implement)      #
# --------------------------------------------------------------------------- #


def lasso_cv_mse(x: np.ndarray, y: np.ndarray, alpha: float, folds: list[np.ndarray]) -> float:
    """Cross-validated MSE of a Lasso fit at penalty ``alpha``.

    For each held-out fold: fit a Lasso on the OTHER folds, predict the held-out
    rows (`model.predict`), and average the MSE.
    """
    # TODO: like poly_cv_mse, but fit sklearn.linear_model.Lasso(alpha=alpha,
    # max_iter=10000).fit(x_train, y_train) (import Lasso inside this function) and
    # predict the held-out rows with model.predict(x_test).
    raise NotImplementedError("Implement lasso_cv_mse.")


def select_alpha(
    x: np.ndarray, y: np.ndarray, candidate_alphas: list[float], folds: list[np.ndarray]
) -> tuple[float, np.ndarray]:
    """Penalty from ``candidate_alphas`` with the lowest cross-validated MSE.

    Returns ``(best_alpha, mse_by_alpha)``.
    """
    # TODO: build mse_by_alpha over candidate_alphas with lasso_cv_mse; best_alpha
    # is the one with the smallest CV-MSE. Return (best_alpha, mse_by_alpha).
    raise NotImplementedError("Implement select_alpha.")


def selected_features(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Indices of the features with a nonzero Lasso coefficient at ``alpha``."""
    # TODO: fit sklearn.linear_model.Lasso(alpha=alpha, max_iter=10000).fit(x, y)
    # (import Lasso inside this function) and return np.flatnonzero(model.coef_).
    raise NotImplementedError("Implement selected_features.")


def support_scores(selected: np.ndarray, true_support: np.ndarray) -> dict:
    """Precision and recall of a selected feature set against the true drivers.

    Returns ``{"precision": ..., "recall": ...}``. Precision is the fraction of
    selected features that are real drivers; recall is the fraction of real drivers
    that were selected. Return zeros (not a crash) if a set is empty.
    """
    # TODO: intersect the selected indices with the true support; precision =
    # hits / #selected, recall = hits / #true. Guard empty sets.
    raise NotImplementedError("Implement support_scores.")


# --------------------------------------------------------------------------- #
# Provided: QC + driver                                                        #
# --------------------------------------------------------------------------- #


def run_qc(folds: list[np.ndarray], n: int) -> None:
    """Print a QC block before any results (provided)."""
    covered = np.concatenate(folds)
    disjoint = len(covered) == n and len(np.unique(covered)) == n
    print(
        f"Cross-validation QC: {len(folds)} folds partition all {n} rows "
        f"exactly once (disjoint & complete = {disjoint})."
    )
    print("No fold is fit on its own held-out rows, so each CV score is out-of-sample.")


def main() -> None:
    """Select model complexity and a sparse feature set by cross-validation."""
    seed_everything()

    # ---- Part A: model-complexity selection (synthetic, known truth) --------
    print("== Part A: choosing polynomial complexity by cross-validation ==")
    x, y, true_degree = make_response_curve(degree=3)
    folds_a = kfold_indices(len(x), folds=5)
    run_qc(folds_a, len(x))
    degrees = [1, 2, 3, 4, 5, 6, 8, 10]
    best_degree, mse_by_degree = select_degree(x, y, degrees, folds_a)
    for d, m in zip(degrees, mse_by_degree):
        print(f"    degree={d:>2d}  CV-MSE={m:8.3f}{'  <-- selected' if d == best_degree else ''}")
    print(f"  true degree = {true_degree}; cross-validation selected degree {best_degree}")

    # ---- Part B: sparse feature selection -----------------------------------
    print("\n== Part B: choosing a sparse feature set by cross-validation ==")
    xs, ys, true_support = make_sparse_regression(k=6)
    folds_b = kfold_indices(len(xs), folds=5)
    alphas = [round(a, 4) for a in np.geomspace(0.01, 2.0, 10)]
    best_alpha, _mse_by_alpha = select_alpha(xs, ys, alphas, folds_b)
    sel = selected_features(xs, ys, best_alpha)
    scores = support_scores(sel, true_support)
    print(f"  synthetic recovery (true drivers = {len(true_support)}):")
    print(
        f"    CV-best alpha = {best_alpha}; selected {len(sel)} features; "
        f"precision={scores['precision']:.2f} recall={scores['recall']:.2f}"
    )

    # Real application: a biomarker panel on breast-cytology data.
    xb, yb, names = load_biomarkers()
    folds_r = kfold_indices(len(xb), folds=5)
    best_alpha_r, _ = select_alpha(xb, yb, alphas, folds_r)
    panel = selected_features(xb, yb, best_alpha_r)
    print(f"  real WDBC panel: CV-best alpha = {best_alpha_r}; {len(panel)} biomarkers:")
    print("    " + ", ".join(names[i] for i in panel[:8]) + (" ..." if len(panel) > 8 else ""))

    print("\n== Interpretation ==")
    block = interpretation_block(
        claim=(
            f"Cross-validation recovers the true model complexity (degree {best_degree} of a "
            f"degree-{true_degree} signal) and the true drivers (recall {scores['recall']:.2f}) "
            f"without ever seeing the ground truth, and yields a {len(panel)}-marker panel on "
            f"real breast-cytology data."
        ),
        limitations_list=[
            "CV-tuned Lasso recovers the real drivers but over-selects (precision "
            f"{scores['precision']:.2f}) -- held-out error favors keeping a few spurious features.",
            "Each CV score is a single held-out estimate; a different fold seed shifts it.",
            "The real panel has no ground-truth driver set, so it is read qualitatively, "
            "not scored for recovery.",
        ],
    )
    print(block)


if __name__ == "__main__":
    main()
