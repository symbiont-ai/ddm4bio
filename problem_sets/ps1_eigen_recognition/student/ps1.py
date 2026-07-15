"""Student template for PS1: the eigen-subspace as a model of normal cells.

Week 1 built an eigen-image basis and used it to *recognize* cells. Here you turn
the same basis into a **model of what a normal cell looks like** and put it to two
new uses on real BloodMNIST:

- Part A -- **denoising**: the clean signal lives in a few principal axes while
  noise spreads across all of them, so projecting a noisy image onto the top-k
  eigen-subspace and reconstructing removes noise. Find the rank that maximizes
  the signal-to-noise ratio (SNR).
- Part B -- **out-of-QC detection**: an image that does not fit the normal
  subspace reconstructs badly, so its reconstruction error is a novelty score.
  Flag corrupted acquisitions and report detection AUC + a false-alarm-bounded
  threshold.

Fill in every function body marked ``# TODO``. The eigen-basis primitives
(``eigen_basis``, ``project``, ``reconstruct``), the data loader, the noise
helper, the QC driver, and ``main`` are provided -- this problem set is about
what you *do* with the subspace, not rebuilding it. The autograder imports these
functions by name and checks their behaviour on its own fixtures, so keep the
signatures exactly as given. Run with ``python ps1.py``; it stops at the first
unimplemented function.
"""

from __future__ import annotations

import numpy as np

from ddm4bio.config import GLOBAL_SEED, seed_everything
from ddm4bio.interpret import interpretation_block
from ddm4bio.methods.decomposition import svd_lowrank
from ddm4bio.qc.report import assert_no_leakage

# --------------------------------------------------------------------------- #
# Provided: data + eigen-basis primitives (do not edit; these are Week 1's)    #
# --------------------------------------------------------------------------- #


def load_blood_cells(
    n: int = 800, test_size: float = 0.4, seed: int = GLOBAL_SEED
) -> tuple[np.ndarray, np.ndarray, str, str]:
    """Load real BloodMNIST as flattened grayscale vectors and split it.

    Returns ``(X_train, X_test, source, provenance)``. ``X_train`` is the library
    of *clean, normal* cells we model; ``X_test`` is held out.
    """
    from sklearn.model_selection import train_test_split

    from ddm4bio.datasets import get_dataset

    ds = get_dataset("bloodmnist", seed=seed)
    images = ds.payload["train_images"]  # (N, H, W, C) uint8
    gray = images.mean(axis=-1).reshape(images.shape[0], -1).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.choice(gray.shape[0], size=min(n, gray.shape[0]), replace=False)
    x_train, x_test = train_test_split(gray[idx], test_size=test_size, random_state=seed)
    return x_train, x_test, ds.source, ds.provenance


def eigen_basis(x: np.ndarray, n_modes: int) -> tuple[np.ndarray, np.ndarray]:
    """(provided, from Week 1) Per-feature mean + top-``n_modes`` principal axes."""
    mean_vec = x.mean(axis=0)
    _u, _s, vt = svd_lowrank(x - mean_vec, n_modes)
    return mean_vec, vt


def project(x: np.ndarray, mean_vec: np.ndarray, components: np.ndarray) -> np.ndarray:
    """(provided, from Week 1) Coordinates of ``x`` in the eigen-basis."""
    return (x - mean_vec) @ components.T


def reconstruct(coords: np.ndarray, mean_vec: np.ndarray, components: np.ndarray) -> np.ndarray:
    """(provided, from Week 1) Map basis coordinates back to pixel space."""
    return coords @ components + mean_vec


def add_noise(x: np.ndarray, rng: np.random.Generator, level: float) -> np.ndarray:
    """(provided) Additive Gaussian noise scaled to the data's spread."""
    return x + rng.normal(0.0, level * float(x.std()), size=x.shape)


# --------------------------------------------------------------------------- #
# Part A -- Denoising with the eigen-subspace  (you implement)                 #
# --------------------------------------------------------------------------- #


def snr_db(estimate: np.ndarray, reference: np.ndarray) -> float:
    """Signal-to-noise ratio, in decibels, of ``estimate`` against clean ``reference``.

    ``SNR = 10 * log10( ||reference||^2 / ||estimate - reference||^2 )``.
    """
    # TODO: return 10 * log10( sum(reference**2) / sum((estimate - reference)**2) ).
    # Add a tiny constant to the denominator so a perfect estimate does not divide
    # by zero.
    raise NotImplementedError("Implement snr_db.")


def denoise(x_noisy: np.ndarray, mean_vec: np.ndarray, components: np.ndarray) -> np.ndarray:
    """Denoise images by projecting onto the eigen-subspace and reconstructing.

    The clean signal is (approximately) low-rank in this basis while the noise is
    not, so a rank-limited round-trip keeps the signal and drops most of the noise.
    """
    # TODO: project x_noisy onto the basis and reconstruct it (use the provided
    # project() and reconstruct()).
    raise NotImplementedError("Implement denoise.")


def best_rank_for_denoising(
    x_train: np.ndarray, x_noisy: np.ndarray, x_clean: np.ndarray, candidate_ks: list[int]
) -> tuple[int, np.ndarray]:
    """Rank from ``candidate_ks`` whose denoising maximizes SNR against ``x_clean``.

    For each ``k``: fit ``eigen_basis`` on ``x_train``, ``denoise`` ``x_noisy``, and
    score the result with ``snr_db`` against ``x_clean``.

    Returns ``(best_k, snr_by_k)`` where ``snr_by_k[i]`` is the output SNR (dB) at
    ``candidate_ks[i]``. Too few modes underfit the signal; too many re-admit
    noise, so the curve peaks at an intermediate rank.
    """
    # TODO: build snr_by_k by looping over candidate_ks (fit eigen_basis on
    # x_train, denoise x_noisy, snr_db vs x_clean); best_k is the k with the
    # highest SNR. Return (best_k, snr_by_k as a numpy array).
    raise NotImplementedError("Implement best_rank_for_denoising.")


# --------------------------------------------------------------------------- #
# Part B -- Out-of-QC detection with reconstruction error  (you implement)     #
# --------------------------------------------------------------------------- #


def reconstruction_anomaly_score(
    x: np.ndarray, mean_vec: np.ndarray, components: np.ndarray
) -> np.ndarray:
    """Per-image relative reconstruction error against the normal subspace.

    A normal cell lies close to the subspace and reconstructs well (small score);
    an out-of-QC image does not (large score). Returns one score per row of ``x``.
    """
    # TODO: reconstruct x through the basis, then return the per-row relative
    # error ||x - recon|| / ||x|| (norm along axis=1; guard the denominator).
    raise NotImplementedError("Implement reconstruction_anomaly_score.")


def detection_auc(scores: np.ndarray, is_anomaly: np.ndarray) -> float:
    """ROC-AUC of the anomaly ``scores`` against binary ground truth ``is_anomaly``.

    AUC = probability a random anomaly scores higher than a random normal image;
    0.5 is chance, 1.0 is perfect separation.
    """
    # TODO: return sklearn.metrics.roc_auc_score(is_anomaly, scores) as a float.
    raise NotImplementedError("Implement detection_auc.")


def flag_threshold(scores_normal: np.ndarray, max_false_alarm: float) -> float:
    """Score threshold that flags at most ``max_false_alarm`` of NORMAL images.

    Setting the cutoff at the ``(1 - max_false_alarm)`` quantile of the normal
    scores bounds the false-alarm (false-positive) rate at ``max_false_alarm``.
    """
    # TODO: return the (1 - max_false_alarm) quantile of scores_normal
    # (np.quantile is handy).
    raise NotImplementedError("Implement flag_threshold.")


# --------------------------------------------------------------------------- #
# Provided: QC + driver                                                        #
# --------------------------------------------------------------------------- #


def run_qc(x_train: np.ndarray, x_test: np.ndarray) -> None:
    """Print the required QC block before any results (provided)."""
    n_tr, n_te = x_train.shape[0], x_test.shape[0]
    train_idx = np.arange(n_tr)
    test_idx = np.arange(n_tr, n_tr + n_te)
    assert_no_leakage(train_idx, test_idx)
    print("Leakage check passed: the normal-cell library and the held-out set are disjoint.")
    print(f"Library (normal) cells: {n_tr}   held-out cells: {n_te}   pixels: {x_train.shape[1]}")
    # A full-rank basis must reconstruct held-out cells almost exactly (sanity),
    # using only the provided primitives so QC runs before any graded function.
    sample = x_test[:50]
    mean_vec, components = eigen_basis(x_train, min(n_tr, x_train.shape[1]))
    recon = reconstruct(project(sample, mean_vec, components), mean_vec, components)
    err = float(np.linalg.norm(sample - recon) / (np.linalg.norm(sample) + 1e-12))
    print(f"Full-rank reconstruction error on held-out cells = {err:.2e} (near 0 expected)")


def main() -> None:
    """Run denoising + out-of-QC detection end to end and interpret the results."""
    seed_everything()
    x_train, x_test, source, provenance = load_blood_cells()
    print(f"Application data: BloodMNIST via get_dataset -> source={source}")
    print(f"  {provenance}\n")

    print("== Quality control (before results) ==")
    run_qc(x_train, x_test)

    rng = np.random.default_rng(GLOBAL_SEED)

    # ---- Part A: denoising --------------------------------------------------
    print("\n== Part A: denoising by low-rank eigen-projection ==")
    x_noisy = add_noise(x_test, rng, level=0.6)
    input_snr = snr_db(x_noisy, x_test)
    candidate_ks = [2, 5, 10, 20, 40, 80, 160]
    best_k, snr_by_k = best_rank_for_denoising(x_train, x_noisy, x_test, candidate_ks)
    print(f"  noisy input SNR = {input_snr:.2f} dB")
    for k, s in zip(candidate_ks, snr_by_k):
        print(f"    k={k:>3d}  output SNR = {s:5.2f} dB   gain = {s - input_snr:+.2f} dB")
    peak_gain = float(np.asarray(snr_by_k).max()) - input_snr
    print(f"  best rank k = {best_k}  ->  peak SNR gain = {peak_gain:+.2f} dB")

    # ---- Part B: out-of-QC detection ----------------------------------------
    print("\n== Part B: out-of-QC detection by reconstruction error ==")
    x_corrupt = add_noise(x_test, rng, level=2.0)  # heavily corrupted acquisitions
    pool = np.vstack([x_test, x_corrupt])
    is_anomaly = np.concatenate([np.zeros(len(x_test)), np.ones(len(x_corrupt))])
    mean_vec, components = eigen_basis(x_train, best_k)
    scores = reconstruction_anomaly_score(pool, mean_vec, components)
    auc = detection_auc(scores, is_anomaly)
    thr = flag_threshold(scores[is_anomaly == 0], max_false_alarm=0.05)
    detected = float(np.mean(scores[is_anomaly == 1] > thr))
    false_alarm = float(np.mean(scores[is_anomaly == 0] > thr))
    print(f"  detection AUC = {auc:.3f}")
    print(
        f"  threshold at 5% target false-alarm: flags {detected:.0%} of corrupt "
        f"cells (actual false-alarm {false_alarm:.0%})"
    )

    print("\n== Interpretation ==")
    block = interpretation_block(
        claim=(
            f"The same eigen-basis that recognizes cells also models a normal cell: a "
            f"rank-{best_k} projection denoises held-out microscopy by {peak_gain:.1f} dB, "
            f"and reconstruction error separates corrupted acquisitions from clean ones "
            f"at AUC {auc:.2f}."
        ),
        confidence="high",
        limitations_list=[
            "Noise is synthetic additive Gaussian; real acquisition artifacts (blur, "
            "saturation, debris) are not identically distributed.",
            "SNR gain and detection AUC are single held-out estimates, not cross-validated.",
            "The subspace is linear; genuinely nonlinear novelties may still project onto it.",
        ],
        evidence=(
            "single-peaked SNR-vs-rank curve, leakage-checked split, corrupted-vs-clean "
            "separation measured by ROC-AUC with a false-alarm-bounded threshold"
        ),
    )
    print(block)


if __name__ == "__main__":
    main()
