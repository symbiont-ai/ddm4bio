"""Student template for PS3: machine learning and generalization.

Fill in every function marked ``# TODO``. The public signatures must stay
exactly as given -- the autograder imports these names and checks shapes,
return keys, and performance thresholds.

What is already wired for you (do not rewrite):
* :func:`true_function` and :func:`generate_synthetic_curve` -- the Part A data.
* :func:`load_clinical_data` -- the offline breast-cancer loader (Part B).
* :func:`run_tabular_qc` -- the quality-control call (Part C).
* :func:`main` -- the end-to-end orchestration that calls your functions in
  order and prints the interpretation block.

What you implement: the model-fitting and evaluation logic (Parts A/B/C).

Only numpy is imported at module top level; import scikit-learn, scipy, and
pandas *inside* the function bodies. Keep everything offline, deterministic, and
seeded.

Reading: Kutz, *Data-Driven Modeling & Scientific Computation*, Ch. 6 (neural
networks) and Ch. 13 (regression, model selection, cross-validation).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ddm4bio import seed_everything
from ddm4bio.interpret import interpretation_block
from ddm4bio.methods.learning import (  # noqa: F401  (used by the functions you implement)
    cross_validate,
    learning_curve,
    permutation_test,
)
from ddm4bio.qc.report import QCReport, assert_no_leakage  # noqa: F401  (assert_no_leakage: yours)
from ddm4bio.qc.tabular import qc_tabular

SEED = 20260714


# --------------------------------------------------------------------------- #
# Part A -- method on a known synthetic function
# --------------------------------------------------------------------------- #
def true_function(x: np.ndarray) -> np.ndarray:
    """Noise-free target function the models must learn (GIVEN -- do not edit).

    ``sin(1.5 x) + 0.3 x``: smooth and easy to fit inside a bounded window, but
    it diverges from any polynomial fit once evaluated outside that window.
    """
    x = np.asarray(x, dtype=float)
    return np.sin(1.5 * x) + 0.3 * x


def generate_synthetic_curve(
    n_samples: int,
    x_low: float,
    x_high: float,
    noise: float = 0.15,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample noisy observations of :func:`true_function` (GIVEN -- do not edit).

    Returns ``X`` of shape ``(n_samples, 1)`` and ``y`` of shape ``(n_samples,)``.
    """
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(x_low, x_high, size=n_samples))
    y = true_function(x) + rng.normal(0.0, noise, size=n_samples)
    return x.reshape(-1, 1), y


def fit_polynomial(X: np.ndarray, y: np.ndarray, degree: int) -> Any:
    """Fit an ordinary least-squares polynomial of the given ``degree``.

    Return a *fitted* scikit-learn estimator with a ``predict`` method. The
    standard recipe is a pipeline of ``PolynomialFeatures(degree)`` followed by
    ``LinearRegression``.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, 1)
    y : np.ndarray, shape (n_samples,)
    degree : int

    Returns
    -------
    estimator
        Fitted polynomial regression pipeline.
    """
    # TODO: import LinearRegression, PolynomialFeatures, make_pipeline from
    # sklearn; build the pipeline; call .fit(X, y); return the fitted model.
    raise NotImplementedError


def fit_small_nn(
    X: np.ndarray,
    y: np.ndarray,
    hidden_layer_sizes: tuple[int, ...] = (32, 32),
    seed: int | None = None,
) -> Any:
    """Fit a small multilayer-perceptron regressor.

    Return a *fitted* estimator. Scale the inputs first (a ``StandardScaler`` in
    a pipeline with ``MLPRegressor``); pass ``random_state=seed`` so results are
    deterministic, and allow enough iterations to converge.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, 1)
    y : np.ndarray, shape (n_samples,)
    hidden_layer_sizes : tuple of int
    seed : int, optional

    Returns
    -------
    estimator
        Fitted MLP regression pipeline.
    """
    # TODO: build StandardScaler + MLPRegressor(hidden_layer_sizes, random_state=
    # seed, max_iter large enough); fit and return it.
    raise NotImplementedError


def interp_vs_extrap_error(
    model: Any,
    x_low: float,
    x_high: float,
    pad: float | None = None,
    n_eval: int = 400,
    seed: int | None = None,
) -> dict[str, float]:
    """Compare a fitted model's error inside vs. just outside its training domain.

    Draw fresh evaluation points: interpolation points inside ``[x_low, x_high]``
    and extrapolation points inside ``[x_high, x_high + pad]`` (``pad`` defaults
    to half the training width). Score each against :func:`true_function` with
    mean squared error.

    Returns
    -------
    dict
        Keys ``"interp_mse"`` and ``"extrap_mse"`` (floats).
    """
    # TODO: draw x_in and x_out with a seeded np.random.default_rng; predict with
    # the model on each region; return the two MSEs against true_function(...).
    raise NotImplementedError


def kfold_cv(
    estimator: Any,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    scoring: str | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """k-fold cross-validation via ``ddm4bio.methods.learning.cross_validate``.

    Returns
    -------
    dict
        Keys ``"scores"``, ``"mean"``, ``"std"``.
    """
    # TODO: call cross_validate(estimator, X, y, cv=cv, scoring=scoring, seed=seed)
    # and return its result.
    raise NotImplementedError


def model_learning_curve(
    estimator: Any,
    X: np.ndarray,
    y: np.ndarray,
    train_sizes: np.ndarray | None = None,
    cv: int = 5,
    seed: int | None = None,
) -> dict[str, np.ndarray]:
    """Learning curve via ``ddm4bio.methods.learning.learning_curve``.

    Returns
    -------
    dict
        Keys ``"train_sizes"``, ``"train_scores"``, ``"val_scores"``.
    """
    # TODO: call learning_curve(estimator, X, y, train_sizes=train_sizes, cv=cv,
    # seed=seed) and return its result.
    raise NotImplementedError


# --------------------------------------------------------------------------- #
# Part B -- clinical application on load_breast_cancer
# --------------------------------------------------------------------------- #
def load_clinical_data() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load the bundled breast-cancer diagnostic dataset (GIVEN -- do not edit).

    Returns ``X`` (569, 30), binary ``y`` (569,), and the 30 feature names.
    """
    from sklearn.datasets import load_breast_cancer

    data = load_breast_cancer()
    X = np.asarray(data.data, dtype=float)
    y = np.asarray(data.target, dtype=int)
    feature_names = list(data.feature_names)
    return X, y, feature_names


def build_models(seed: int | None = None) -> dict[str, Any]:
    """Construct the three classifiers compared in Part B.

    Each should be a pipeline that standardizes features first, then applies:
    * ``"linear"``      -- a nearly *unregularized* logistic regression
      (use a very large ``C``);
    * ``"nn"``          -- a shallow ``MLPClassifier`` (one small hidden layer,
      ``random_state=seed``);
    * ``"regularized"`` -- a *strongly* penalized logistic regression
      (small ``C``, e.g. 0.05).

    Return the three *unfitted* estimators.

    Returns
    -------
    dict
        Keys ``"linear"``, ``"nn"``, ``"regularized"``.
    """
    # TODO: build the three StandardScaler-prefixed pipelines described above and
    # return them in a dict with exactly these three keys.
    raise NotImplementedError


def performance_vs_n(
    estimator: Any,
    X: np.ndarray,
    y: np.ndarray,
    train_sizes: np.ndarray | None = None,
    cv: int = 5,
    seed: int | None = None,
) -> dict[str, np.ndarray]:
    """Trace validation performance as the training-set size ``n`` grows.

    Use :func:`model_learning_curve`, then reduce the per-fold score matrices to
    their per-size means.

    Returns
    -------
    dict
        Keys ``"train_sizes"``, ``"train_mean"``, ``"val_mean"`` (1-D arrays).
    """
    # TODO: default train_sizes to np.linspace(0.1, 1.0, 6) if None; call
    # model_learning_curve; return train_sizes plus the mean over axis=1 of the
    # train and val score matrices.
    raise NotImplementedError


# --------------------------------------------------------------------------- #
# Part C -- quality control
# --------------------------------------------------------------------------- #
def run_tabular_qc(X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> QCReport:
    """Assemble a DataFrame and run ``qc_tabular`` (GIVEN -- do not edit)."""
    import pandas as pd

    df = pd.DataFrame(X, columns=list(feature_names))
    df["target"] = y
    return qc_tabular(df)


def make_splits(
    X: np.ndarray,
    y: np.ndarray,
    val_size: float = 0.2,
    test_size: float = 0.2,
    seed: int | None = None,
) -> dict[str, Any]:
    """Stratified train/validation/test split with an explicit leakage guard.

    Split the *indices* ``np.arange(n)`` (so you can verify no overlap): first
    hold out a test set, then split the remainder into train and validation.
    Call :func:`ddm4bio.qc.report.assert_no_leakage` on each pair of index sets.

    Returns
    -------
    dict
        ``X_train/y_train/X_val/y_val/X_test/y_test`` and
        ``train_idx/val_idx/test_idx``.
    """
    # TODO: use sklearn.model_selection.train_test_split on the index array with
    # stratify=y; rescale val_size relative to the remaining pool; call
    # assert_no_leakage on every pair; return the split arrays and indices.
    raise NotImplementedError


def _auc(estimator: Any, X: np.ndarray, y: np.ndarray) -> float:
    """Area under the ROC curve using probabilities or decision scores."""
    # TODO: use predict_proba(X)[:, 1] when available, else decision_function(X);
    # return roc_auc_score(y, scores) as a float.
    raise NotImplementedError


def generalization_gap(estimator: Any, splits: dict[str, Any]) -> dict[str, float]:
    """Fit on train and report the train/validation/test ROC-AUC gap.

    Returns
    -------
    dict
        Keys ``"train_auc"``, ``"val_auc"``, ``"test_auc"``, and ``"gap"``
        (train minus validation AUC).
    """
    # TODO: fit the estimator on the train split only; compute AUC on each split
    # via _auc; return the three AUCs and gap = train_auc - val_auc.
    raise NotImplementedError


def calibration_report(
    estimator: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Assess how well predicted probabilities match observed frequencies.

    Fit on train, predict probabilities on test, then build a reliability curve
    and a Brier score.

    Returns
    -------
    dict
        Keys ``"prob_true"``, ``"prob_pred"`` (reliability arrays) and
        ``"brier"`` (Brier score; lower is better).
    """
    # TODO: fit; prob = predict_proba(X_test)[:, 1]; use
    # sklearn.calibration.calibration_curve and sklearn.metrics.brier_score_loss.
    raise NotImplementedError


def permutation_beats_chance(
    estimator: Any,
    X: np.ndarray,
    y: np.ndarray,
    n_perm: int = 200,
    cv: int = 5,
    scoring: str | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Permutation test via ``ddm4bio.methods.learning.permutation_test``.

    Returns
    -------
    dict
        Keys ``"observed_score"``, ``"permutation_scores"``, ``"p_value"``.
    """
    # TODO: call permutation_test(estimator, X, y, n_perm=n_perm, cv=cv,
    # scoring=scoring, seed=seed) and return its result.
    raise NotImplementedError


# --------------------------------------------------------------------------- #
# Orchestration / demonstration (GIVEN -- do not edit)
# --------------------------------------------------------------------------- #
def main() -> None:
    """Run the full PS3 analysis end to end and print an interpretation block."""
    seed_everything(SEED)

    # ----- Part A: interpolation vs. extrapolation on a known function ----- #
    x_low, x_high = 0.0, 2.0 * np.pi
    X_tr, y_tr = generate_synthetic_curve(80, x_low, x_high, noise=0.15, seed=SEED)

    poly = fit_polynomial(X_tr, y_tr, degree=9)
    nn_reg = fit_small_nn(X_tr, y_tr, hidden_layer_sizes=(32, 32), seed=SEED)

    poly_err = interp_vs_extrap_error(poly, x_low, x_high, seed=SEED)
    nn_err = interp_vs_extrap_error(nn_reg, x_low, x_high, seed=SEED)

    print("== Part A: interpolation vs. extrapolation ==")
    print(
        f"polynomial(deg=9): interp MSE={poly_err['interp_mse']:.4f}  "
        f"extrap MSE={poly_err['extrap_mse']:.4f}"
    )
    print(
        f"small NN         : interp MSE={nn_err['interp_mse']:.4f}  "
        f"extrap MSE={nn_err['extrap_mse']:.4f}"
    )

    # ----- Part B: clinical application ----- #
    X, y, feature_names = load_clinical_data()

    print("\n== Quality control (run BEFORE modeling) ==")
    report = run_tabular_qc(X, y, feature_names)
    print(report.render())

    models = build_models(seed=SEED)
    print("\n== Part B: cross-validated ROC-AUC on breast cancer ==")
    for name, est in models.items():
        cv_res = kfold_cv(est, X, y, cv=5, scoring="roc_auc", seed=SEED)
        print(f"{name:>12}: AUC = {cv_res['mean']:.3f} +/- {cv_res['std']:.3f}")

    pvn = performance_vs_n(models["regularized"], X, y, cv=5, seed=SEED)
    print("\nvalidation accuracy vs n (regularized):")
    for n_i, v_i in zip(pvn["train_sizes"], pvn["val_mean"]):
        print(f"  n={int(n_i):>4}: val acc = {v_i:.3f}")

    # ----- Part C: generalization gap, calibration, permutation test ----- #
    splits = make_splits(X, y, seed=SEED)
    gap = generalization_gap(build_models(seed=SEED)["regularized"], splits)
    print("\n== Part C: generalization gap (regularized) ==")
    print(
        f"train AUC={gap['train_auc']:.3f}  val AUC={gap['val_auc']:.3f}  "
        f"test AUC={gap['test_auc']:.3f}  gap={gap['gap']:.3f}"
    )

    calib = calibration_report(
        build_models(seed=SEED)["regularized"],
        splits["X_train"],
        splits["y_train"],
        splits["X_test"],
        splits["y_test"],
    )
    print(f"Brier score (test) = {calib['brier']:.4f}")

    perm = permutation_beats_chance(
        build_models(seed=SEED)["regularized"], X, y, n_perm=200, scoring="roc_auc", seed=SEED
    )
    print(
        f"permutation test: observed AUC={perm['observed_score']:.3f}  "
        f"p-value={perm['p_value']:.4f}"
    )

    # ----- Part D: interpretation ----- #
    print("\n== Part D: interpretation ==")
    print(
        interpretation_block(
            claim=("State here whether the classifier generalizes, and to what."),
            confidence="moderate",
            limitations_list=[
                "Name the real limitations you observed (dataset size, no external "
                "cohort, extrapolation failure, threshold/calibration, ...).",
            ],
            evidence=(
                f"5-fold CV AUC {perm['observed_score']:.3f} with permutation "
                f"p={perm['p_value']:.4f}; train-val AUC gap {gap['gap']:.3f}; "
                f"Brier {calib['brier']:.3f}."
            ),
        )
    )


if __name__ == "__main__":
    try:
        main()
    except NotImplementedError:
        print(
            "Reached an unimplemented function. Fill in the TODOs in this file, "
            "then re-run. Start with fit_polynomial and work top to bottom."
        )
