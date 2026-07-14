"""End-to-end capstone demo pipeline.

This is the minimal, runnable skeleton for a ddm4bio capstone. It runs offline
on a bundled synthetic fixture with only the core ``ddm4bio`` stack installed
(numpy / scipy / pandas / scikit-learn) -- no network, no heavy extras.

It demonstrates the mandated shape of a capstone analysis:

    load -> quality control (BEFORE results)
         -> ONE Group-A decomposition   (SVD / PCA family)
         -> ONE Group-B dynamics method (DMD)
         -> ground-truth validation      (recover the known spectrum)
         -> interpretation with explicit confidence

The two methods are connected: the SVD singular-value spectrum chooses the
truncation rank that DMD then uses. Replace the fixture with your dataset and
the two methods with your chosen pair -- keep the structure and the QC-first,
validate-before-you-trust discipline.

Run with:  ``python src/pipeline.py``  (or ``make run``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ddm4bio import seed_everything
from ddm4bio.datasets.synthetic import make_linear_dynamics
from ddm4bio.interpret import interpretation_block
from ddm4bio.methods.decomposition import svd_lowrank
from ddm4bio.methods.dynamics import dmd
from ddm4bio.qc.tabular import qc_tabular


def load() -> tuple[np.ndarray, np.ndarray]:
    """Load the demo data: a linear system with a known eigen-spectrum.

    Returns the ``(n_steps, d)`` state trajectory (the "observations") and the
    ``(d,)`` ground-truth eigenvalues used later for validation. A real
    capstone would replace this with a ``ddm4bio`` loader for its dataset.
    """
    true_eigs = np.array([0.95, 0.9 + 0.1j, 0.9 - 0.1j])
    fixture = make_linear_dynamics(eigs=true_eigs, n_steps=200, seed=0)
    return fixture.trajectory, fixture.eigs


def quality_control(trajectory: np.ndarray) -> None:
    """Inspect the data BEFORE any modeling and print the QC report.

    Non-negotiable standard #1: QC precedes conclusions. We wrap the trajectory
    in a DataFrame so the shared tabular QC checks (missingness, duplicates,
    constant columns, outliers) run and are reported up front.
    """
    df = pd.DataFrame(
        trajectory,
        columns=[f"x{i}" for i in range(trajectory.shape[1])],
    )
    report = qc_tabular(df)
    print(report.render())
    if report.warnings:
        print(
            "\n[QC] Warnings were raised above -- resolve or justify each "
            "before trusting any result.\n"
        )
    else:
        print("\n[QC] No blocking issues; proceeding to modeling.\n")


def decompose(trajectory: np.ndarray, energy: float = 0.99) -> int:
    """Group-A method (SVD): pick the truncation rank from the spectrum.

    Runs a truncated SVD of the snapshot matrix and chooses the smallest rank
    whose singular values capture ``energy`` of the total spectral energy. This
    rank feeds the Group-B (DMD) step -- the two methods are connected, not
    independent.

    Returns the chosen rank ``r``.
    """
    # Snapshot convention for dynamics: (n_features, n_time).
    snapshots = trajectory.T
    full_rank = min(snapshots.shape)
    _, s, _ = svd_lowrank(snapshots, k=full_rank)

    energy_frac = np.cumsum(s**2) / np.sum(s**2)
    r = int(np.searchsorted(energy_frac, energy) + 1)
    r = max(1, min(r, full_rank))

    print("[Group A: SVD] singular values:", np.array2string(s, precision=3))
    print(f"[Group A: SVD] rank capturing {energy:.0%} energy: r = {r}\n")
    return r


def run_dynamics(trajectory: np.ndarray, r: int) -> np.ndarray:
    """Group-B method (DMD): recover discrete-time eigenvalues at rank ``r``.

    Returns the estimated eigenvalues, sorted by magnitude (descending).
    """
    snapshots = trajectory.T
    result = dmd(snapshots, r=r)
    eigs = result.eigenvalues[np.argsort(-np.abs(result.eigenvalues))]
    print("[Group B: DMD] recovered eigenvalues:", np.array2string(eigs, precision=3))
    return eigs


def validate(true_eigs: np.ndarray, est_eigs: np.ndarray) -> float:
    """Ground-truth validation: how well DMD recovered the known spectrum.

    Non-negotiable standard #3: before trusting a number on real data, recover a
    known answer on a control. We match each true eigenvalue to its nearest
    estimate and report the maximum absolute error.
    """
    remaining = list(est_eigs)
    errors = []
    for lam in true_eigs:
        j = int(np.argmin([abs(lam - e) for e in remaining]))
        errors.append(abs(lam - remaining.pop(j)))
    max_err = float(np.max(errors))
    print(f"[Validation] max |true - est| eigenvalue error: {max_err:.2e}\n")
    return max_err


def interpret(max_err: float) -> str:
    """Interpretation with an explicit, calibrated confidence level.

    Confidence is chosen from the validation evidence, not asserted: tiny
    recovery error -> high confidence; large error -> low confidence.
    """
    if max_err < 1e-6:
        confidence = "high"
    elif max_err < 1e-2:
        confidence = "moderate"
    else:
        confidence = "low"

    block = interpretation_block(
        claim=(
            "The SVD-truncated DMD pipeline recovers the linear system's "
            "eigen-spectrum from its state trajectory."
        ),
        confidence=confidence,
        limitations_list=[
            "Demo runs on a noise-free synthetic fixture; real signals add "
            "measurement noise that degrades eigenvalue recovery.",
            "Rank was chosen by an energy threshold; a different threshold "
            "changes the truncation and the recovered modes.",
            "Ground truth here is exact by construction -- real data has no "
            "such oracle, so treat this as a correctness check of the pipeline.",
        ],
        evidence=f"max eigenvalue recovery error = {max_err:.2e}",
    )
    return block


def main() -> None:
    """Run the full pipeline end to end."""
    seed_everything()  # non-negotiable #4: pinned seed for reproducibility.

    print("=" * 64)
    print("ddm4bio capstone template -- demo pipeline")
    print("=" * 64 + "\n")

    trajectory, true_eigs = load()
    quality_control(trajectory)  # QC BEFORE results.
    r = decompose(trajectory)  # Group A: SVD.
    est_eigs = run_dynamics(trajectory, r)  # Group B: DMD.
    max_err = validate(true_eigs, est_eigs)  # ground-truth validation.

    print("-" * 64)
    print("INTERPRETATION")
    print("-" * 64)
    print(interpret(max_err))


if __name__ == "__main__":
    main()
