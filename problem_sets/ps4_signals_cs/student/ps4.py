"""PS4: how undersampled can you go? (the compressed-sensing limit).

Module 4 reconstructs one sparse signal from one set of compressed measurements. This
problem set asks the question underneath that demo: **how few measurements can you
get away with?** Compressed sensing does not degrade gracefully -- below a sharp
threshold recovery fails completely, above it recovery succeeds. You will map that
limit.

- Part A -- **the recovery cliff**: for a fixed sparsity, sweep the number of random
  measurements and find the sharp transition from failure to success -- the minimum
  measurements the signal demands.
- Part B -- **the phase transition**: sweep sparsity too, and watch the minimum
  measurements grow with it (m* ~ a few per nonzero) -- the boundary that governs
  every compressed acquisition.

Fill in every function body marked ``# TODO``. The sparse-signal generator
(`make_sparse`), the QC driver, and `main` are provided -- this problem set is about
mapping the limit, not re-deriving the sparse recovery (`recover` is a one-call
L1/Lasso fit). The autograder imports these functions by name, so keep the
signatures exactly as given. Run with ``python ps4.py``; it stops at the first
unimplemented function.
"""

from __future__ import annotations

import numpy as np

from ddm4bio.config import GLOBAL_SEED, seed_everything
from ddm4bio.interpret import interpretation_block

# --------------------------------------------------------------------------- #
# Provided: sparse signals + a real ECG (do not edit)                          #
# --------------------------------------------------------------------------- #


def make_sparse(n: int, k: int, seed: int = GLOBAL_SEED) -> np.ndarray:
    """(provided) A length-``n`` signal with exactly ``k`` nonzero entries."""
    from ddm4bio.datasets.synthetic import make_sparse_signal

    return np.asarray(make_sparse_signal(n, k, seed=seed).signal, dtype=float)


def real_ecg_effective_sparsity(n: int = 256, energy: float = 0.95) -> tuple[int, str]:
    """(provided) How few Fourier coefficients hold ``energy`` of a real ECG segment.

    Real biosignals are compressible -- their energy concentrates in a few
    coefficients -- which is exactly what makes compressed sensing apply to them.
    """
    from ddm4bio.datasets import get_dataset

    ds = get_dataset("mitbih", seed=GLOBAL_SEED)
    segment = np.asarray(ds.payload["signal"])[:n, 0].astype(float)
    power = np.abs(np.fft.rfft(segment)) ** 2
    order = np.argsort(power)[::-1]
    cumulative = np.cumsum(power[order]) / power.sum()
    n_coeff = int(np.searchsorted(cumulative, energy) + 1)
    return n_coeff, ds.source


# --------------------------------------------------------------------------- #
# Part A -- The recovery cliff  (you implement)                                #
# --------------------------------------------------------------------------- #


def measurement_matrix(m: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """A random Gaussian sensing matrix of shape ``(m, n)``, scaled by ``1/sqrt(m)``."""
    # TODO: return rng.standard_normal((m, n)) / sqrt(m) -- m rows (measurements),
    # n columns (signal length), scaled so each measurement has unit-ish energy.
    raise NotImplementedError("Implement measurement_matrix.")


def recover(signal: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Take compressed measurements ``y = matrix @ signal`` and reconstruct the signal.

    CS recovery IS an L1/Lasso fit -- call sklearn Lasso directly with
    ``fit_intercept=False``.
    """
    # TODO: CS recovery IS an L1/Lasso fit -- call sklearn Lasso directly with
    # fit_intercept=False. Form the measurements y = matrix @ signal, then fit
    # sklearn.linear_model.Lasso(alpha=1e-3, fit_intercept=False, selection="random",
    # random_state=None, max_iter=5000) on (matrix, y) and return its .coef_ as a
    # length-len(signal) 1-D array. Import Lasso inside the function body.
    raise NotImplementedError("Implement recover.")


def recovery_error(recovered: np.ndarray, true_signal: np.ndarray) -> float:
    """Relative L2 error ``||recovered - true|| / ||true||``."""
    # TODO: return the L2 norm of (recovered - true_signal) divided by the L2 norm
    # of true_signal (guard a zero denominator with a small epsilon).
    raise NotImplementedError("Implement recovery_error.")


def recovery_error_curve(
    signal: np.ndarray, m_values: list[int], seed: int = GLOBAL_SEED, n_trials: int = 5
) -> np.ndarray:
    """Recovery error vs. number of measurements ``m``, averaged over random matrices.

    For each ``m``: draw ``n_trials`` measurement matrices, recover from each, and
    average the relative error. Because which random matrix you draw matters, the
    average over trials is what makes the cliff sharp: the curve stays high (failure)
    below the sampling limit, then drops once ``m`` is large enough for the sparsity.
    """
    # TODO: rng = np.random.default_rng(seed). For each m in m_values, average
    # recovery_error(recover(signal, measurement_matrix(m, len(signal), rng)), signal)
    # over n_trials draws. Return the per-m mean errors as an array.
    raise NotImplementedError("Implement recovery_error_curve.")


def min_measurements_for_recovery(
    signal: np.ndarray, m_values: list[int], tol: float = 0.3, seed: int = GLOBAL_SEED
) -> int:
    """Smallest ``m`` (scanning ``m_values`` in order) whose recovery error is below ``tol``.

    This is the sampling limit -- the fewest measurements the signal demands. Return
    the largest candidate if none succeed.
    """
    # TODO: compute recovery_error_curve(signal, m_values, seed=seed); return the
    # first m_values entry whose error < tol, or the last candidate if none do.
    raise NotImplementedError("Implement min_measurements_for_recovery.")


# --------------------------------------------------------------------------- #
# Part B -- The phase transition  (you implement)                              #
# --------------------------------------------------------------------------- #


def phase_transition(
    n: int, sparsities: list[int], m_values: list[int], tol: float = 0.3, seed: int = GLOBAL_SEED
) -> np.ndarray:
    """Minimum measurements to recover, as a function of sparsity.

    For each ``k`` in ``sparsities``: build a ``k``-sparse length-``n`` signal and find
    its `min_measurements_for_recovery`. Return one ``m*`` per sparsity; ``m*`` grows
    with ``k`` (a few measurements per nonzero).
    """
    # TODO: for each k, build make_sparse(n, k, seed=seed + k) and take its
    # min_measurements_for_recovery(..., tol=tol, seed=seed). Return the m* array.
    raise NotImplementedError("Implement phase_transition.")


# --------------------------------------------------------------------------- #
# Provided: QC + driver                                                        #
# --------------------------------------------------------------------------- #


def run_qc(signal: np.ndarray, m_values: list[int]) -> None:
    """Print a QC block before any results (provided)."""
    n = len(signal)
    k = int(np.count_nonzero(signal))
    undersampled = [m for m in m_values if m < n]
    print(f"QC: length-{n} signal with {k} nonzeros (sparsity {k / n:.1%}).")
    print(
        f"    {len(undersampled)}/{len(m_values)} candidate measurement counts are "
        f"undersampled (m < n) -- the compressed-sensing regime."
    )


def main() -> None:
    """Map the compressed-sensing sampling limit; motivate it on a real ECG."""
    seed_everything()
    n = 128
    m_values = [8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80]

    n_coeff, source = real_ecg_effective_sparsity()
    print(
        f"Real ECG (MIT-BIH via get_dataset -> source={source}): "
        f"{n_coeff} Fourier coefficients hold 95% of a 256-sample segment's energy"
    )
    print("  -> real biosignals are compressible, which is what lets compressed sensing apply.\n")

    print("== Quality control (before results) ==")
    signal = make_sparse(n, k=8)
    run_qc(signal, m_values)

    print("\n== Part A: the recovery cliff (sparsity k = 8) ==")
    curve = recovery_error_curve(signal, m_values)
    for m, e in zip(m_values, curve):
        print(f"    m={m:>3d}  rel-error={e:.3f}{'  <-- recovers' if e < 0.3 else ''}")
    m_star = min_measurements_for_recovery(signal, m_values)
    print(f"  minimum measurements to recover = {m_star}  ({m_star / 8:.1f} per nonzero)")

    print("\n== Part B: the phase transition (m* vs sparsity) ==")
    sparsities = [2, 4, 8, 12, 16]
    m_star_curve = phase_transition(n, sparsities, m_values)
    for k, ms in zip(sparsities, m_star_curve):
        print(f"    k={k:>2d}  ->  m* = {ms:>2d}   ({ms / k:.1f} measurements per nonzero)")
    grows = bool(np.all(np.diff(m_star_curve) >= 0))
    print(f"  m* is non-decreasing in sparsity: {grows}")

    print("\n== Interpretation ==")
    block = interpretation_block(
        claim=(
            f"Compressed sensing has a sharp sampling limit: a {8}-sparse length-{n} signal is "
            f"unrecoverable below {m_star} random measurements and recovers cleanly above it, and "
            f"that limit grows with sparsity (from {m_star_curve[0]} to {m_star_curve[-1]} "
            f"measurements as k goes {sparsities[0]} to {sparsities[-1]})."
        ),
        limitations_list=[
            "Signals are exactly k-sparse in the canonical basis; real signals are only "
            "approximately sparse (in a wavelet/Fourier basis), which softens the cliff.",
            "The transition is mapped at one tolerance and one measurement grid; the exact m* "
            "shifts with both.",
            "The L1 solver uses a fixed regularization, so recovery above the cliff is good but "
            "not machine-exact.",
        ],
    )
    print(block)


if __name__ == "__main__":
    main()
