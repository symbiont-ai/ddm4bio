"""PS2 student template: curve fitting, regularized differentiation, sparsity.

Fill in each function body where you see ``# TODO``. The imports, the offline
data loaders, and the QC/interpretation plumbing are already wired for you, so
you only implement the method logic. Every function must keep the signature
given here -- the autograder imports these names directly.

Reading: Kutz, "Data-Driven Modeling & Scientific Computation," Ch. 4
(curve fitting and regression) and Ch. 5 (sparsity and compressed sensing).

Run this file (``python ps2.py``) at any point: it should import cleanly and
stop at the first ``NotImplementedError`` you have not yet replaced.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ddm4bio.config import GLOBAL_SEED
from ddm4bio.interpret import interpretation_block

# hill is used by the provided data generator; fit_hill / regularized_derivative /
# lasso_select are the library calls you will use inside the TODO bodies below.
from ddm4bio.methods.fitting import (  # noqa: F401
    fit_hill,
    hill,
    lasso_select,
    regularized_derivative,
)


# --------------------------------------------------------------------------- #
# Data generation (offline ground truth) -- provided, do not change           #
# --------------------------------------------------------------------------- #
def simulate_dose_response(
    ec50: float = 5.0,
    hill_coeff: float = 1.8,
    bottom: float = 0.05,
    top: float = 1.0,
    n_doses: int = 12,
    n_replicates: int = 6,
    noise: float = 0.03,
    seed: int = GLOBAL_SEED,
) -> dict[str, Any]:
    """Generate a synthetic dose-response assay with replicate noise.

    Provided for you. Returns a dict with keys "dose" (shape ``(n_doses,)``),
    "response" (shape ``(n_replicates, n_doses)``), and "true" (the ground-truth
    Hill parameters).
    """
    rng = np.random.default_rng(seed)
    dose = np.logspace(-2.0, 2.0, n_doses)
    clean = hill(dose, bottom, top, ec50, hill_coeff)
    response = clean[None, :] + noise * rng.standard_normal((n_replicates, n_doses))
    return {
        "dose": dose,
        "response": response,
        "true": {
            "bottom": bottom,
            "top": top,
            "ec50": ec50,
            "hill_coeff": hill_coeff,
        },
    }


def _standardize_columns(x: np.ndarray) -> np.ndarray:
    """Return ``x`` with each column shifted to zero mean and unit variance.

    Provided helper -- use it before Lasso when features are on different
    scales.
    """
    x = np.asarray(x, dtype=float)
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    sd = np.where(sd > 0.0, sd, 1.0)
    return (x - mu) / sd


# --------------------------------------------------------------------------- #
# Part A / B: nonlinear dose-response fit                                      #
# --------------------------------------------------------------------------- #
def fit_dose_response(
    dose: np.ndarray,
    response: np.ndarray,
    seed: int = GLOBAL_SEED,
) -> dict[str, Any]:
    """Fit a Hill dose-response curve and score the goodness of fit.

    Accept a 1-D response vector or a 2-D ``(n_replicates, n_doses)`` array.
    Flatten replicates against their doses, call
    :func:`ddm4bio.methods.fitting.fit_hill`, then extend its result dict with:

    * "r_squared": coefficient of determination on the flattened points.
    * "residuals": flattened observed-minus-fit residuals.
    * "mean_residuals": per-dose residual of the replicate mean, ordered by
      dose (used by the QC residual-structure check).
    * "dose": the dose grid.

    Returns
    -------
    dict
        The ``fit_hill`` dict plus the four keys above.
    """
    dose = np.asarray(dose, dtype=float)
    response = np.asarray(response, dtype=float)

    # TODO: flatten replicates (if 2-D) against their doses, call fit_hill,
    # compute R^2 and residuals, and return the extended dict described above.
    raise NotImplementedError("implement fit_dose_response")


def bootstrap_ec50(
    dose: np.ndarray,
    response: np.ndarray,
    n_boot: int = 500,
    ci: float = 0.95,
    seed: int = GLOBAL_SEED,
) -> dict[str, Any]:
    """Bootstrap the EC50 and Hill coefficient over replicate noise.

    For each of ``n_boot`` iterations: resample the replicates at every dose
    with replacement, refit the replicate mean with ``fit_hill``, and record the
    EC50 and Hill coefficient. Report the medians as point estimates and the
    central ``ci`` percentile band as the confidence interval. Drop fits that
    did not converge or gave a non-positive EC50.

    Returns
    -------
    dict
        Keys: "ec50", "hill_coeff" (medians); "ec50_ci", "hill_ci" (``(lo, hi)``
        tuples); "ec50_samples", "hill_samples" (valid draws); "n_valid".
    """
    dose = np.asarray(dose, dtype=float)
    response = np.asarray(response, dtype=float)
    if response.ndim != 2:
        raise ValueError("response must be 2-D (n_replicates, n_doses) for the bootstrap")

    # TODO: build a np.random.default_rng(seed), run the replicate bootstrap,
    # and summarize the EC50/Hill distributions into point estimates and
    # percentile intervals. Hint: np.take_along_axis lets you resample replicate
    # rows per dose column.
    raise NotImplementedError("implement bootstrap_ec50")


# --------------------------------------------------------------------------- #
# Part A: regularized vs finite-difference differentiation                    #
# --------------------------------------------------------------------------- #
def compare_derivative_methods(
    y: np.ndarray,
    dx: float,
    deriv_true: np.ndarray,
    lam: float = 0.1,
) -> dict[str, Any]:
    """Compare regularized differentiation to a finite difference.

    Estimate the derivative of ``y`` two ways -- with
    :func:`ddm4bio.methods.fitting.regularized_derivative` and with
    ``np.gradient`` -- and score each by its L2 distance to the known analytic
    derivative ``deriv_true``.

    Returns
    -------
    dict
        Keys: "reg_derivative", "fd_derivative"; "error_regularized",
        "error_finite_difference"; "improvement" (fd error minus reg error).
    """
    y = np.asarray(y, dtype=float)
    deriv_true = np.asarray(deriv_true, dtype=float)

    # TODO: compute both derivative estimates, their L2 errors against
    # deriv_true, and return the dict described above.
    raise NotImplementedError("implement compare_derivative_methods")


# --------------------------------------------------------------------------- #
# Part A / B: L1 sparse feature selection                                      #
# --------------------------------------------------------------------------- #
def sparse_biomarkers(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float | None = None,
    standardize: bool = True,
    seed: int = GLOBAL_SEED,
) -> dict[str, Any]:
    """Select a sparse biomarker set with the Lasso.

    Optionally z-score the columns (use ``_standardize_columns``), then delegate
    to :func:`ddm4bio.methods.fitting.lasso_select`.

    Returns
    -------
    dict
        The ``lasso_select`` dict: "selected", "coefficients", "intercept",
        "alpha".
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # TODO: standardize (if requested) and call lasso_select; return its dict.
    raise NotImplementedError("implement sparse_biomarkers")


def stability_selection(
    x: np.ndarray,
    y: np.ndarray,
    n_boot: int = 100,
    alpha: float | None = None,
    subsample: float = 0.75,
    threshold: float = 0.6,
    standardize: bool = True,
    seed: int = GLOBAL_SEED,
) -> dict[str, Any]:
    """Rank features by how often the Lasso selects them across subsamples.

    Each iteration: draw a random ``subsample`` fraction of the rows without
    replacement, standardize that subsample independently (if requested), run
    ``lasso_select``, and tally which features it keeps. Features selected in at
    least ``threshold`` fraction of iterations are "stable".

    Returns
    -------
    dict
        Keys: "frequency" (per-feature selection frequency, shape
        ``(n_features,)``); "stable" (indices with frequency >= threshold);
        "threshold"; "n_boot".
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # TODO: build a np.random.default_rng(seed), run the subsampling loop,
    # accumulate per-feature selection counts, convert to frequencies, and
    # threshold to get the stable set.
    raise NotImplementedError("implement stability_selection")


# --------------------------------------------------------------------------- #
# Part C: quality control                                                      #
# --------------------------------------------------------------------------- #
def residual_structure(residuals: np.ndarray) -> dict[str, Any]:
    """Summarize residual structure for a goodness-of-fit QC check.

    Compute the lag-1 autocorrelation of the (dose-ordered) residuals and the
    number of same-sign runs. A good fit leaves near-zero autocorrelation and
    many sign runs; systematic misfit leaves strong positive autocorrelation and
    few runs.

    Returns
    -------
    dict
        Keys: "lag1_autocorr" and "n_sign_runs".
    """
    # TODO: coerce residuals to a float array, then compute the lag-1
    # autocorrelation and the count of same-sign runs.
    raise NotImplementedError("implement residual_structure")


def load_breast_cancer_frame():
    """Load the scikit-learn breast-cancer panel as a labelled DataFrame.

    Provided for you. Returns ``(x, y, frame)`` where ``frame`` carries the
    features plus a ``target`` column so ``ddm4bio.qc.qc_tabular`` reports class
    balance.
    """
    import pandas as pd
    from sklearn.datasets import load_breast_cancer

    data = load_breast_cancer()
    x = np.asarray(data.data, dtype=float)
    y = np.asarray(data.target, dtype=int)
    frame = pd.DataFrame(x, columns=list(data.feature_names))
    frame["target"] = y
    return x, y, frame


# --------------------------------------------------------------------------- #
# Driver -- provided, wires QC and the interpretation block                    #
# --------------------------------------------------------------------------- #
def main() -> None:
    """Run the full PS2 pipeline and print QC + an interpretation block."""
    from ddm4bio.qc.tabular import qc_tabular

    assay = simulate_dose_response(seed=GLOBAL_SEED)
    dose, response, truth = assay["dose"], assay["response"], assay["true"]

    fit = fit_dose_response(dose, response, seed=GLOBAL_SEED)
    boot = bootstrap_ec50(dose, response, n_boot=400, seed=GLOBAL_SEED)

    n = 300
    t = np.linspace(0.0, 2.0 * np.pi, n)
    dx = t[1] - t[0]
    rng = np.random.default_rng(GLOBAL_SEED)
    noisy = np.sin(t) + 0.05 * rng.standard_normal(n)
    deriv = compare_derivative_methods(noisy, dx, np.cos(t), lam=0.1)

    x_bc, y_bc, frame = load_breast_cancer_frame()

    # QC-before-results (golden rule): inspect the tabular panel first.
    print(qc_tabular(frame).render())
    print()

    sparse = sparse_biomarkers(x_bc, y_bc, seed=GLOBAL_SEED)
    stab = stability_selection(
        x_bc, y_bc, n_boot=100, alpha=0.02, threshold=0.6, seed=GLOBAL_SEED
    )
    feature_names = list(frame.columns[:-1])
    struct = residual_structure(fit["mean_residuals"])

    print(f"true EC50={truth['ec50']:.3f}  fitted EC50={fit['ec50']:.3f}")
    print(f"R^2 = {fit['r_squared']:.4f}")
    print(
        f"bootstrap EC50 = {boot['ec50']:.3f} "
        f"95% CI [{boot['ec50_ci'][0]:.3f}, {boot['ec50_ci'][1]:.3f}]"
    )
    print(f"error(reg)={deriv['error_regularized']:.4f}")
    print(f"error(fd)={deriv['error_finite_difference']:.4f}")
    print(f"residual lag-1 autocorr={struct['lag1_autocorr']:.3f}")
    print(f"residual sign runs={struct['n_sign_runs']}")
    print(f"Lasso selected {sparse['selected'].size} features")

    lo, hi = boot["ec50_ci"]
    stable_names = [feature_names[i] for i in stab["stable"]]
    print(
        interpretation_block(
            claim=(
                f"The assay's half-maximal dose is EC50 = {boot['ec50']:.2f} "
                f"(95% CI [{lo:.2f}, {hi:.2f}]); {len(stable_names)} breast-cancer "
                "features are selected stably across subsamples."
            ),
            confidence="moderate",
            limitations_list=[
                "synthetic assay with a known Hill curve",
                "bootstrap over replicate noise only, not the dose design",
                "Lasso is scale- and alpha-sensitive; no held-out test set",
            ],
            evidence=f"R^2 = {fit['r_squared']:.3f}; stability selection across 100 subsamples",
        )
    )


if __name__ == "__main__":
    main()
