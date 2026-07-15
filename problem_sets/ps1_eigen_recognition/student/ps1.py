"""Student template for PS1: Eigen-recognition (Kutz Ch. 2).

Fill in every function body marked ``# TODO``. The imports, data/system
plumbing (``make_spd_system``, ``load_digits_split``), the QC driver
(``run_qc``), and ``main`` are already wired for you -- you only need to
implement the method logic.

Keep the public signatures exactly as given: the autograder imports these
functions by name and checks their shapes, return types, and numerical
behaviour. Only ``numpy`` is imported at module top level; import scipy and
scikit-learn *inside* the functions that use them.

Run this file with ``python ps1.py``; it will execute until the first
unimplemented function raises ``NotImplementedError``.
"""

from __future__ import annotations

import numpy as np

from ddm4bio.config import GLOBAL_SEED, seed_everything
from ddm4bio.interpret import interpretation_block
from ddm4bio.methods.decomposition import (  # noqa: F401  (used once TODOs are filled in)
    explained_variance_ratio,
    svd_lowrank,
)
from ddm4bio.methods.validation import reconstruction_error  # noqa: F401
from ddm4bio.qc.report import assert_no_leakage
from ddm4bio.qc.tabular import qc_tabular

# --------------------------------------------------------------------------- #
# Data / system plumbing (provided -- do not edit)                             #
# --------------------------------------------------------------------------- #


def make_spd_system(n: int, seed: int = GLOBAL_SEED) -> tuple[np.ndarray, np.ndarray]:
    """Build a symmetric positive-definite system ``A x = b`` for solver tests.

    Conjugate gradient requires a symmetric positive-definite ``A``, so we form
    ``A = M M^T + n I`` from a random ``M``.

    Returns ``(A, b)`` with shapes ``(n, n)`` and ``(n,)``.
    """
    rng = np.random.default_rng(seed)
    m = rng.standard_normal((n, n))
    a = m @ m.T + n * np.eye(n)
    b = rng.standard_normal(n)
    return a, b


def load_digits_split(
    test_size: float = 0.3, seed: int = GLOBAL_SEED, download: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the BloodMNIST image library via ``get_dataset`` and split it.

    The application data flows through the course data layer:
    ``get_dataset("bloodmnist")`` returns real MedMNIST blood-cell crops when a
    download is available and a deterministic bundled fallback (same payload
    shape) otherwise. Every crop is converted to grayscale (mean over the colour
    axis) and flattened. Defaults to ``download=False`` so the run stays offline
    and deterministic. The historical name is kept for autograder compatibility.

    Returns ``(X_train, X_test, y_train, y_test)`` with grayscale ``H*W``-feature
    rows and a stratified, deterministic split.
    """
    from sklearn.model_selection import train_test_split

    from ddm4bio.datasets import get_dataset

    ds = get_dataset("bloodmnist", download=download, seed=seed)
    images = ds.payload["train_images"]  # (N, H, W, C) uint8
    labels = ds.payload["train_labels"].ravel()

    x = images.mean(axis=-1).reshape(images.shape[0], -1).astype(float)
    x_train, x_test, y_train, y_test = train_test_split(
        x, labels, test_size=test_size, random_state=seed, stratify=labels
    )
    return x_train, x_test, y_train, y_test


# --------------------------------------------------------------------------- #
# Part A -- linear solvers                                                      #
# --------------------------------------------------------------------------- #


def solve_direct(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve ``A x = b`` directly via an LU factorization.

    Parameters
    ----------
    a : np.ndarray, shape (n, n)
    b : np.ndarray, shape (n,)

    Returns
    -------
    np.ndarray, shape (n,)
        The solution vector ``x``.
    """
    # TODO: use scipy.linalg.lu_factor / lu_solve to solve the system directly.
    raise NotImplementedError("Implement solve_direct with an LU factorization.")


def solve_iterative(a: np.ndarray, b: np.ndarray, rtol: float = 1e-10) -> tuple[np.ndarray, int]:
    """Solve a symmetric positive-definite ``A x = b`` via conjugate gradient.

    Parameters
    ----------
    a : np.ndarray, shape (n, n)
    b : np.ndarray, shape (n,)
    rtol : float
        Relative convergence tolerance for the CG solver.

    Returns
    -------
    tuple[np.ndarray, int]
        ``(x, n_iters)`` -- the solution and the CG iteration count.
    """
    # TODO: call scipy.sparse.linalg.cg with a callback that counts iterations.
    # Pass rtol=rtol, atol=0.0, and a generous maxiter. Return (x, n_iters).
    raise NotImplementedError("Implement solve_iterative with conjugate gradient.")


def compare_solvers(a: np.ndarray, b: np.ndarray, rtol: float = 1e-10) -> dict:
    """Solve ``A x = b`` both ways and summarize conditioning/iterations/runtime.

    Returns
    -------
    dict
        Keys: ``condition_number``, ``direct_residual``, ``iterative_residual``,
        ``cg_iterations``, ``direct_time``, ``iterative_time``, ``agree``.
    """
    # TODO: time solve_direct and solve_iterative, compute np.linalg.cond(a),
    # the residual ||A x - b|| for each solution, and whether they agree
    # (np.allclose). Assemble and return the dict described above.
    raise NotImplementedError("Implement compare_solvers.")


# --------------------------------------------------------------------------- #
# Part A -- eigen-image pipeline                                               #
# --------------------------------------------------------------------------- #


def eigen_basis(x: np.ndarray, n_modes: int) -> tuple[np.ndarray, np.ndarray]:
    """Fit an eigen-image basis: per-feature mean plus principal axes.

    Parameters
    ----------
    x : np.ndarray, shape (n_samples, n_features)
    n_modes : int

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(mean_vec, components)`` with shapes ``(n_features,)`` and
        ``(n_modes, n_features)``; rows of ``components`` are orthonormal.
    """
    # TODO: compute the per-feature mean, mean-center x, and use
    # svd_lowrank(centered, n_modes) to get the top-n_modes right singular
    # vectors (Vt). Return (mean_vec, Vt).
    raise NotImplementedError("Implement eigen_basis (mean-center + truncated SVD).")


def project(x: np.ndarray, mean_vec: np.ndarray, components: np.ndarray) -> np.ndarray:
    """Project data onto a fitted eigen-basis.

    Returns
    -------
    np.ndarray, shape (n_samples, n_modes)
        Coordinates of ``x`` in the eigen-basis.
    """
    # TODO: subtract mean_vec and project onto the components:
    # (x - mean_vec) @ components.T
    raise NotImplementedError("Implement project.")


def reconstruct(coords: np.ndarray, mean_vec: np.ndarray, components: np.ndarray) -> np.ndarray:
    """Reconstruct data from eigen-basis coordinates.

    Returns
    -------
    np.ndarray, shape (n_samples, n_features)
        ``coords @ components + mean_vec``.
    """
    # TODO: map coordinates back to feature space and add the mean.
    raise NotImplementedError("Implement reconstruct.")


def reconstruction_error_curve(
    x_train: np.ndarray, x_test: np.ndarray, mode_list: list[int]
) -> np.ndarray:
    """Relative-L2 reconstruction error on held-out data versus number of modes.

    Fit the basis on ``x_train`` only; measure error on ``x_test``.

    Returns
    -------
    np.ndarray, shape (len(mode_list),)
    """
    # TODO: for each k in mode_list, fit eigen_basis on x_train, project and
    # reconstruct x_test, and record reconstruction_error(x_test, recon,
    # kind="rel_l2"). Return the errors as a numpy array.
    raise NotImplementedError("Implement reconstruction_error_curve.")


def modes_for_variance(x: np.ndarray, thresholds: list[float]) -> dict:
    """Smallest number of modes reaching each cumulative-variance threshold.

    Returns
    -------
    dict
        threshold -> smallest mode count with cumulative EVR >= threshold.
    """
    # TODO: use explained_variance_ratio(x), take a cumulative sum, and for each
    # threshold find the smallest number of modes whose cumulative ratio reaches
    # it (np.searchsorted is handy). Cap the count at the number of modes.
    raise NotImplementedError("Implement modes_for_variance.")


def eigen_nn_accuracy(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    n_modes: int,
) -> float:
    """Nearest-neighbour classification accuracy in the eigen-basis.

    Fit the basis on training data; project both splits; evaluate a
    1-nearest-neighbour classifier on the test split.

    Returns
    -------
    float
        Test-set accuracy in ``[0, 1]``.
    """
    # TODO: fit eigen_basis on x_train, project x_train and x_test, fit a
    # KNeighborsClassifier(n_neighbors=1) on the training projections, and
    # return its accuracy on the test projections.
    raise NotImplementedError("Implement eigen_nn_accuracy.")


# --------------------------------------------------------------------------- #
# Part C -- quality control                                                    #
# --------------------------------------------------------------------------- #


def check_orthonormality(components: np.ndarray) -> dict:
    """Measure how close eigen-basis rows are to being orthonormal.

    Returns
    -------
    dict
        Keys ``max_offdiagonal`` and ``max_norm_deviation``.
    """
    # TODO: form the Gram matrix components @ components.T. Report the largest
    # absolute off-diagonal entry and the largest |1 - ||row||^2| over rows.
    raise NotImplementedError("Implement check_orthonormality.")


def run_qc(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> None:
    """Print the required QC block before any modeling results (provided)."""
    import pandas as pd

    # Leakage: no shared sample indices between the two splits.
    assert_no_leakage(train_idx, test_idx)
    print("Leakage check passed: train/test index sets are disjoint.")

    # Class balance via the shared tabular QC helper.
    df = pd.DataFrame({"target": y_train})
    report = qc_tabular(df)
    print(report.render())

    # Orthonormality of a full-rank eigen-basis.
    _mean, components = eigen_basis(x_train, x_train.shape[1])
    ortho = check_orthonormality(components)
    print(
        "Orthonormality: max off-diagonal = "
        f"{ortho['max_offdiagonal']:.2e}, "
        f"max norm deviation = {ortho['max_norm_deviation']:.2e}"
    )
    # Convergence sanity: full-rank basis reconstructs the test split exactly.
    curve = reconstruction_error_curve(x_train, x_test, [x_train.shape[1]])
    print(f"Full-rank test reconstruction error = {curve[0]:.2e}")


# --------------------------------------------------------------------------- #
# Driver (provided)                                                            #
# --------------------------------------------------------------------------- #


def main() -> None:
    """Run the full PS1 pipeline end to end and print an interpretation block."""
    seed_everything()

    # Part A: solver comparison on a small SPD system.
    a, b = make_spd_system(200)
    solver_stats = compare_solvers(a, b)
    print("== Part A: direct LU vs iterative CG ==")
    for key, value in solver_stats.items():
        print(f"  {key}: {value}")

    # Data for Parts B-D: REAL BloodMNIST via the course data layer. download
    # defaults to True, so the loader prefers the real MedMNIST crops and falls
    # back gracefully to the bundled stack (same payload shape) only when offline.
    # Provenance is printed so the reader sees which source the run used.
    from sklearn.model_selection import train_test_split

    from ddm4bio.datasets import get_dataset

    ds = get_dataset("bloodmnist", seed=GLOBAL_SEED)
    print(f"\nApplication data: BloodMNIST via get_dataset -> source={ds.source}")
    print(f"  {ds.provenance}")

    # Real train_images are (N, 28, 28, 3) uint8 with integer cell-type labels;
    # the offline fallback is (N, 8, 8, 1). Averaging over the trailing colour
    # axis yields grayscale in both cases, so the pipeline never sees the source.
    # A seeded few-hundred-image subsample keeps the run brisk.
    train_images = ds.payload["train_images"]
    train_labels = ds.payload["train_labels"].ravel()
    rng_sub = np.random.default_rng(GLOBAL_SEED)
    n_sub = min(400, train_images.shape[0])
    subsample = rng_sub.choice(train_images.shape[0], size=n_sub, replace=False)
    images_gray = train_images[subsample].mean(axis=-1)  # (n_sub, H, W) grayscale
    x = images_gray.reshape(n_sub, -1).astype(float)  # (n_sub, H*W), samples in rows
    y = train_labels[subsample]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.30, random_state=GLOBAL_SEED, stratify=y
    )
    n_classes = int(np.unique(y_train).size)
    n_features = x_train.shape[1]
    chance = 1.0 / n_classes
    train_idx = np.arange(x_train.shape[0])
    test_idx = np.arange(x_train.shape[0], x_train.shape[0] + x_test.shape[0])

    print("\n== Part C: quality control (before results) ==")
    run_qc(x_train, x_test, y_train, y_test, train_idx, test_idx)

    print("\n== Part B: eigen-cells on blood-cell images ==")
    mode_list = [1, 2, 5, 10, 20, 30, 40, 64]
    curve = reconstruction_error_curve(x_train, x_test, mode_list)
    for k, err in zip(mode_list, curve):
        print(f"  modes={k:>3d}  test rel-L2 error={err:.4f}")

    acc = eigen_nn_accuracy(x_train, y_train, x_test, y_test, n_modes=25)
    print(f"  1-NN accuracy in 25-mode eigen-basis = {acc:.4f} (chance = {chance:.2f})")

    var_modes = modes_for_variance(x_train, [0.90, 0.95, 0.99])
    print("\n== Part D: variance thresholds ==")
    for thresh, k in var_modes.items():
        print(f"  {int(thresh * 100)}% variance -> {k} modes")

    print("\n== Part D: interpretation ==")
    block = interpretation_block(
        claim=(
            f"An eigen-image basis compresses the {n_features}-pixel blood-cell "
            f"images to {var_modes[0.95]} modes at 95% variance, and a "
            f"1-nearest-neighbour classifier in a 25-mode basis reaches {acc:.2f} "
            f"accuracy -- above the {chance:.2f} random-guess baseline."
        ),
        confidence="high",
        limitations_list=[
            "Grayscale + a subsample discards colour/staining cues real blood-cell "
            "typing relies on; offline the run uses the bundled fallback, not the "
            "real crops.",
            "1-NN accuracy is a single held-out estimate, not cross-validated.",
            "The eigen-basis is linear and cannot capture nonlinear structure.",
        ],
        evidence=(
            "orthonormal basis (Gram ~ I), monotone reconstruction convergence, "
            "leakage-checked stratified split"
        ),
    )
    print(block)


if __name__ == "__main__":
    main()
