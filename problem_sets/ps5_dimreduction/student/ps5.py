"""PS5: signal or noise? A significance test for the rank of an expression matrix.

Week 5 computes a PCA/SVD spectrum and eyeballs a scree plot to guess how many
components matter. This problem set replaces the guess with a principled stopping
rule -- **Horn's parallel analysis**: permute each gene column independently to build
a rank-matched *noise* null (all cross-feature structure destroyed, marginals kept),
then keep only the leading components whose real eigenvalue beats the null. You will
recover the true latent dimensionality with a statistical test the lesson never
covers, and expose why the popular analytic shortcut (the Marchenko-Pastur edge) is
untrustworthy on real, non-Gaussian expression data.

- Part A -- **validate on synthetic matrices of KNOWN planted rank**: your permutation
  null + stopping rule must recover the injected dimensionality exactly, and degrade
  honestly only at extreme noise.
- Part B -- **apply to real PBMC3k**: report how many principal components carry real
  structure, and contrast the parallel-analysis count against the naive MP-edge count.

Fill in every function body marked ``# TODO``. The SVD/PCA mechanics
(``pca_eigenvalues``), the null draw (``permute_columns``), the MP edge
(``marchenko_pastur_edge``), the data (``make_planted_rank``, ``load_pbmc3k_topvar``),
the QC driver, and ``main`` are provided -- this problem set is about the significance
*test*, not the decomposition. The autograder imports these functions by name, so keep
the signatures exactly as given. Run with ``python ps5.py``; it stops at the first
unimplemented function.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ddm4bio.config import GLOBAL_SEED, seed_everything
from ddm4bio.interpret import interpretation_block

# --------------------------------------------------------------------------- #
# Provided: PCA mechanics, the null draw, the MP edge, and the fixtures        #
# (do not edit)                                                                #
# --------------------------------------------------------------------------- #


def pca_eigenvalues(X: np.ndarray, center: bool = True) -> np.ndarray:
    """(provided) Descending covariance eigenvalues of a feature matrix.

    The variances captured by each principal component: squares of the singular
    values of the (optionally centered) matrix divided by ``n - 1``.
    """
    X = np.asarray(X, dtype=float)
    if center:
        X = X - X.mean(axis=0, keepdims=True)
    s = np.linalg.svd(X, full_matrices=False, compute_uv=False)
    return (s**2) / max(X.shape[0] - 1, 1)


def permute_columns(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """(provided) One parallel-analysis null draw: shuffle each feature independently.

    Independently permuting every column destroys all cross-feature correlation while
    preserving each feature's marginal distribution -- the null of "no shared structure".
    """
    X = np.asarray(X, dtype=float)
    out = np.empty_like(X)
    n = X.shape[0]
    for j in range(X.shape[1]):
        out[:, j] = X[rng.permutation(n), j]
    return out


def marchenko_pastur_edge(X: np.ndarray) -> tuple[float, float, float]:
    """(provided) Analytic Marchenko-Pastur upper bulk edge for a feature matrix.

    Returns ``(lam_plus, sigma2, gamma)`` where ``gamma = p / n``, ``sigma2`` is a
    bulk noise-variance estimate (the mean covariance eigenvalue), and
    ``lam_plus = sigma2 * (1 + sqrt(gamma))**2`` is the edge above which, *for pure
    i.i.d. Gaussian noise*, no eigenvalue should fall. Real expression data is not
    Gaussian, so this edge is systematically wrong -- which is the point of the contrast.
    """
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    gamma = p / n
    sigma2 = float(np.mean(pca_eigenvalues(X)))
    lam_plus = sigma2 * (1.0 + np.sqrt(gamma)) ** 2
    return float(lam_plus), sigma2, float(gamma)


def make_planted_rank(
    n: int = 200, p: int = 80, k: int = 3, noise_sd: float = 1.0, seed: int = GLOBAL_SEED
) -> np.ndarray:
    """(provided) A feature matrix with exactly ``k`` planted directions plus noise.

    ``X = scores @ loadings + noise_sd * N(0, 1)`` with ``k`` random score/loading
    factors, so the true number of signal components is exactly ``k`` (0 = pure noise).
    """
    rng = np.random.default_rng(seed)
    if k > 0:
        signal = rng.standard_normal((n, k)) @ rng.standard_normal((k, p))
    else:
        signal = np.zeros((n, p))
    return signal + noise_sd * rng.standard_normal((n, p))


def load_pbmc3k_topvar(
    n_genes: int = 1000, n_cells: int | None = None, seed: int = GLOBAL_SEED
) -> tuple[np.ndarray, str]:
    """(provided) Real PBMC3k, library-normalized + log1p + top-variance genes.

    Pulls 10x PBMC3k through the course data layer (offline-cached; a structurally
    identical synthetic matrix offline). Returns ``(X, source)`` with ``X`` of shape
    ``(n_cells, n_genes)``. Optionally subsamples cells for runtime.
    """
    from ddm4bio.datasets import get_dataset

    ds = get_dataset("pbmc3k")
    payload = ds.payload
    counts = payload.X if hasattr(payload, "X") else payload["counts"]
    counts = np.asarray(counts.toarray() if hasattr(counts, "toarray") else counts, dtype=float)
    library = counts.sum(1, keepdims=True)
    library[library == 0] = 1.0
    target = float(np.median(counts.sum(1)))
    log_counts = np.log1p(counts / library * target)
    top = np.argsort(log_counts.var(0))[::-1][: min(n_genes, log_counts.shape[1])]
    x = log_counts[:, top]
    if n_cells is not None and n_cells < x.shape[0]:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(x.shape[0], size=n_cells, replace=False))
        x = x[idx]
    return x, ds.source


# --------------------------------------------------------------------------- #
# Part A -- The permutation null and the stopping rule  (you implement)        #
# --------------------------------------------------------------------------- #


def null_eigenvalue_spectrum(
    X: np.ndarray, n_perm: int = 50, percentile: float = 95.0, seed: int = GLOBAL_SEED
) -> dict:
    """Build the rank-matched permutation null of the eigenvalue spectrum.

    Draw ``n_perm`` column-permuted copies of ``X``, take each one's descending
    eigenvalue spectrum, and summarize the null at every rank. Returns a dict with
    parallel arrays (length ``min(n, p)``): ``threshold`` (the ``percentile`` of the
    null eigenvalue at that rank), ``null_mean``, and ``null_std``.
    """
    # TODO: rng = np.random.default_rng(seed). For each of n_perm draws, take
    # pca_eigenvalues(permute_columns(X, rng)); stack into an (n_perm, min(n,p)) array.
    # Return {"threshold": np.percentile(draws, percentile, axis=0),
    #         "null_mean": draws.mean(0), "null_std": draws.std(0)}.
    raise NotImplementedError("Implement null_eigenvalue_spectrum.")


def count_significant_pcs(
    X: np.ndarray, n_perm: int = 50, percentile: float = 95.0, seed: int = GLOBAL_SEED
) -> int:
    """Number of leading PCs whose real eigenvalue beats the null (contiguous rule).

    Compare each real eigenvalue against the null ``threshold`` at the same rank and
    count the leading run of PCs that exceed it, stopping at the first that does not.
    This is Horn's parallel-analysis stopping rule; it recovers the planted rank on
    synthetic fixtures.
    """
    # TODO: real = pca_eigenvalues(X); threshold = null_eigenvalue_spectrum(...)["threshold"].
    # Count the leading PCs where real > threshold, STOPPING at the first that does not
    # (a contiguous run from the top). Return that count as an int.
    raise NotImplementedError("Implement count_significant_pcs.")


def significance_ratios(
    X: np.ndarray, n_perm: int = 50, percentile: float = 95.0, seed: int = GLOBAL_SEED
) -> np.ndarray:
    """Per-PC ratio of the real eigenvalue to its null threshold.

    A diagnostic vector: ratio > 1 marks a significant PC, and the index where it
    first dips below 1 is the signal/noise boundary (equals ``count_significant_pcs``).
    """
    # TODO: return pca_eigenvalues(X) divided elementwise by the null threshold
    # (guard against a zero threshold).
    raise NotImplementedError("Implement significance_ratios.")


def recover_rank_vs_noise(
    noise_levels: Sequence[float],
    k_true: int = 3,
    n: int = 200,
    p: int = 80,
    n_datasets: int = 5,
    n_perm: int = 50,
    seed: int = GLOBAL_SEED,
) -> dict:
    """Known-truth validation harness: mean recovered rank vs. noise level.

    For each noise level, build ``n_datasets`` planted-rank-``k_true`` matrices and
    average ``count_significant_pcs``. Returns ``{noise_level: mean_recovered_k}``; the
    mean equals ``k_true`` across moderate SNR and degrades gracefully at extreme noise.
    """
    # TODO: for each noise level (index level_index), build n_datasets matrices with
    # make_planted_rank(n, p, k=k_true, noise_sd=level, seed=seed + 1000*level_index + d)
    # and average count_significant_pcs(..., n_perm=n_perm, seed=seed). Return the
    # {float(level): mean_recovered_k} dict.
    raise NotImplementedError("Implement recover_rank_vs_noise.")


# --------------------------------------------------------------------------- #
# Part B -- The naive analytic contrast  (you implement)                       #
# --------------------------------------------------------------------------- #


def marchenko_pastur_count(X: np.ndarray) -> int:
    """Count PCs whose eigenvalue exceeds the analytic Marchenko-Pastur edge.

    The naive analytic rule: real eigenvalues above ``marchenko_pastur_edge(X)``'s
    ``lam_plus`` are called signal. It assumes pure Gaussian noise, so on real data it
    badly OVER-counts -- the losing side of the contrast with parallel analysis.
    """
    # TODO: lam_plus = marchenko_pastur_edge(X)[0]; return the number of
    # pca_eigenvalues(X) strictly greater than lam_plus, as an int.
    raise NotImplementedError("Implement marchenko_pastur_count.")


def compare_selection_rules(
    X: np.ndarray, n_perm: int = 50, percentile: float = 95.0, seed: int = GLOBAL_SEED
) -> dict:
    """Contrast the two rank-selection rules and state which to trust.

    Returns ``parallel_analysis_k``, ``mp_edge_k``, whether they ``agree``, the
    ``trusted_rule`` (always the data-adaptive permutation null), and a ``verdict``.
    """
    # TODO: pa_k = count_significant_pcs(X, n_perm=n_perm, percentile=percentile, seed=seed);
    # mp_k = marchenko_pastur_count(X). Return the dict with keys parallel_analysis_k,
    # mp_edge_k, agree (pa_k == mp_k), trusted_rule="parallel_analysis", and a verdict
    # string explaining why the permutation null is trusted over the MP edge.
    raise NotImplementedError("Implement compare_selection_rules.")


# --------------------------------------------------------------------------- #
# Provided: QC + driver                                                        #
# --------------------------------------------------------------------------- #


def run_qc(X: np.ndarray, label: str) -> None:
    """Print a QC block before any results (provided)."""
    n, p = X.shape
    evr = pca_eigenvalues(X)
    evr = evr / evr.sum()
    print(
        f"QC [{label}]: matrix {n} x {p} (samples x features); "
        f"top-3 variance explained {float(evr[:3].sum()):.1%}, "
        f"aspect ratio gamma=p/n={p / n:.3f}."
    )


def main() -> None:
    """Validate parallel analysis on known-rank fixtures, then apply it to real PBMC3k."""
    seed_everything()

    print("== Part A: recover a KNOWN planted rank (synthetic ground truth) ==")
    x1 = make_planted_rank(n=200, p=80, k=3, noise_sd=1.0, seed=GLOBAL_SEED)
    run_qc(x1, "planted k=3")
    k_hat = count_significant_pcs(x1)
    ratios = significance_ratios(x1)
    print(f"    planted k=3  ->  recovered k_hat={k_hat}")
    print(
        f"    significance ratios (real/null) PC1..PC5: "
        f"{np.array2string(ratios[:5], precision=2, floatmode='fixed')}"
    )
    cmp1 = compare_selection_rules(x1)
    print(
        f"    Marchenko-Pastur edge keeps {cmp1['mp_edge_k']} (on clean Gaussian noise it "
        f"agrees; on real data it will not)"
    )

    print("\n    noise sweep (mean recovered rank over 5 datasets each):")
    sweep = recover_rank_vs_noise([0.3, 0.5, 1.0, 2.0, 4.0, 8.0], k_true=3)
    for noise, mean_k in sweep.items():
        print(f"      noise_sd={noise:>4.1f}  ->  mean recovered k={mean_k:.1f}")

    x0 = make_planted_rank(n=200, p=80, k=0, noise_sd=1.0, seed=GLOBAL_SEED)
    print(f"    pure-noise sanity (planted k=0)  ->  recovered k_hat={count_significant_pcs(x0)}")

    print("\n== Part B: how many PCs are real in PBMC3k? (real data) ==")
    x_real, source = load_pbmc3k_topvar(n_genes=1000, n_cells=1200)
    print(f"[pbmc3k] source={source}")
    run_qc(x_real, "pbmc3k")
    cmp_real = compare_selection_rules(x_real)
    ratios_real = significance_ratios(x_real)
    mask = pca_eigenvalues(x_real) > null_eigenvalue_spectrum(x_real)["threshold"]
    print(
        f"    parallel analysis keeps k={cmp_real['parallel_analysis_k']} PCs "
        f"(contiguous block: {''.join('1' if m else '0' for m in mask[:14])}...)"
    )
    print(
        f"    significance ratios PC1/PC{cmp_real['parallel_analysis_k']}/"
        f"PC{cmp_real['parallel_analysis_k'] + 1}: "
        f"{ratios_real[0]:.2f} / {ratios_real[cmp_real['parallel_analysis_k'] - 1]:.2f} / "
        f"{ratios_real[cmp_real['parallel_analysis_k']]:.2f}"
    )
    print(
        f"    Marchenko-Pastur edge keeps k={cmp_real['mp_edge_k']} "
        f"(over-counts on non-Gaussian data -- the failing contrast)"
    )

    print("\n== Interpretation ==")
    block = interpretation_block(
        claim=(
            f"Parallel analysis recovers the true latent dimensionality of a single-cell "
            f"expression matrix: it returns exactly the planted rank on synthetic data across "
            f"a wide SNR range, and keeps {cmp_real['parallel_analysis_k']} significant "
            f"principal components on real PBMC3k -- while the Marchenko-Pastur edge "
            f"over-counts ({cmp_real['mp_edge_k']} PCs) because real expression noise "
            f"is not Gaussian."
        ),
        limitations_list=[
            "The permutation null has a per-rank false-positive rate equal to (100 - "
            "percentile)%, so on pure noise it can occasionally admit one spurious component.",
            "The Marchenko-Pastur edge assumes i.i.d. Gaussian entries and a bulk noise "
            "variance; real expression data violates both, which is exactly why it fails here.",
            "The real 'true rank' is not a single objective integer; the PBMC3k count is a "
            "defensible cutoff, not a universal constant, and shifts with gene selection.",
        ],
    )
    print(block)


if __name__ == "__main__":
    main()
