"""Student template for PS5: dimensionality reduction and source separation.

Fill in the FIVE method functions marked with ``# TODO``. The offline data
plumbing, the quality-control driver, and the interpretation block are already
wired for you -- you only implement the method logic.

Array conventions follow the library:
- PCA / SVD helpers take a **feature matrix** ``(n_samples, n_features)``.
- ICA sources / multichannel recordings are ``(n_sources, n_samples)``
  (components in rows, samples in columns).

Reading: Kutz Ch. 15 (SVD / PCA) and Ch. 16 (ICA, robust PCA).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ddm4bio.datasets.synthetic import make_mixed_sources
from ddm4bio.interpret import interpretation_block
from ddm4bio.methods.decomposition import (  # noqa: F401  (wired for your TODOs)
    explained_variance_ratio,
    ica_unmix,
    pca_reduce,
    rpca,
    svd_lowrank,
)
from ddm4bio.methods.validation import reconstruction_error, source_recovery_score

SEED = 0


# --------------------------------------------------------------------------- #
# Data plumbing (provided; do not modify).
# --------------------------------------------------------------------------- #
@dataclass
class ExpressionMatrix:
    """Synthetic expression-style dataset with known injected structure.

    Attributes
    ----------
    X: ``(n_samples, n_genes)`` expression matrix (samples in rows).
    group: ``(n_samples,)`` biological condition label in ``{-1, +1}``.
    batch: ``(n_samples,)`` nuisance batch label in ``{-1, +1}``.
    bio_direction: ``(n_genes,)`` unit gene program driven by ``group``.
    batch_direction: ``(n_genes,)`` unit gene program driven by ``batch``.
    """

    X: np.ndarray
    group: np.ndarray
    batch: np.ndarray
    bio_direction: np.ndarray
    batch_direction: np.ndarray


def make_expression_matrix(
    n_samples: int = 120,
    n_genes: int = 50,
    bio_amp: float = 3.0,
    batch_amp: float = 1.5,
    noise: float = 0.5,
    seed: int = SEED,
) -> ExpressionMatrix:
    """Generate a synthetic expression matrix with a known biological axis.

    Two orthogonal gene programs are injected: a *biological* program tied to a
    binary condition label and a weaker *batch* program tied to a nuisance
    label. Because ``bio_amp > batch_amp`` the leading principal component
    should track biology and the second should track batch.
    """
    rng = np.random.default_rng(seed)

    bio_dir = rng.standard_normal(n_genes)
    bio_dir /= np.linalg.norm(bio_dir)
    batch_dir = rng.standard_normal(n_genes)
    batch_dir -= (batch_dir @ bio_dir) * bio_dir  # orthogonalize against bio
    batch_dir /= np.linalg.norm(batch_dir)

    group = np.array([-1.0, 1.0] * (n_samples // 2))[:n_samples]
    rng.shuffle(group)
    batch = np.array([-1.0, 1.0] * (n_samples // 2))[:n_samples]
    rng.shuffle(batch)

    x = (
        bio_amp * np.outer(group, bio_dir)
        + batch_amp * np.outer(batch, batch_dir)
        + noise * rng.standard_normal((n_samples, n_genes))
    )
    return ExpressionMatrix(
        X=x,
        group=group,
        batch=batch,
        bio_direction=bio_dir,
        batch_direction=batch_dir,
    )


def load_single_cell_expression() -> tuple[np.ndarray, np.ndarray | None, str, str]:
    """Load the real PBMC3k single-cell matrix (or its offline fallback).

    Pulls 10x PBMC3k through ``ddm4bio.datasets.get_dataset`` so the same code
    path returns the genuine matrix when available and a structurally identical
    synthetic single-cell matrix offline. The counts are library-size normalized
    and ``log1p``-compressed (the standard single-cell transform), then reduced
    to the top-variance genes.

    Returns
    -------
    tuple
        ``(expr, labels, source, provenance)`` where ``expr`` is a
        ``(n_cells, n_genes)`` log-normalized matrix, ``labels`` is a
        ``(n_cells,)`` array of cell labels (present only in the fallback; the
        real payload is unlabelled and yields ``None``), and ``source`` /
        ``provenance`` are the data-layer strings.
    """
    from ddm4bio.datasets import get_dataset

    ds = get_dataset("pbmc3k")
    payload = ds.payload
    if hasattr(payload, "X"):  # real AnnData
        counts = payload.X
        labels = None
    else:  # labelled synthetic fallback dict
        counts = payload["counts"]
        labels = np.asarray(payload["labels"])
    counts = np.asarray(counts.toarray() if hasattr(counts, "toarray") else counts, dtype=float)

    library = counts.sum(axis=1, keepdims=True)
    library[library == 0] = 1.0
    target = float(np.median(counts.sum(axis=1)))
    log_counts = np.log1p(counts / library * target)

    n_keep = min(1000, log_counts.shape[1])
    top_var = np.argsort(log_counts.var(axis=0))[::-1][:n_keep]
    return log_counts[:, top_var], labels, ds.source, ds.provenance


def _label_separation(values: np.ndarray, group_labels: np.ndarray) -> float:
    """Correlation ratio (eta) of a 1-D score against categorical labels."""
    values = np.asarray(values, dtype=float)
    grand = values.mean()
    total = ((values - grand) ** 2).sum()
    if total == 0.0:
        return 0.0
    between = sum(
        int((group_labels == g).sum()) * (values[group_labels == g].mean() - grand) ** 2
        for g in np.unique(group_labels)
    )
    return float(np.sqrt(between / total))


# --------------------------------------------------------------------------- #
# Part A / B method logic -- IMPLEMENT THESE FIVE FUNCTIONS.
# --------------------------------------------------------------------------- #
def svd_decompose(X: np.ndarray, center: bool = True) -> dict:
    """Full economy SVD of a feature matrix with explained-variance ratios.

    Parameters
    ----------
    X: ``(n_samples, n_features)`` feature matrix.
    center: subtract the per-feature mean before decomposing.

    Returns
    -------
    dict
        Keys ``U`` ``(n_samples, r)``, ``singular_values`` ``(r,)``,
        ``Vt`` ``(r, n_features)``, and ``explained_variance_ratio`` ``(r,)``,
        where ``r = min(n_samples, n_features)``.
    """
    # TODO: center X if requested, take the full economy SVD with svd_lowrank
    # (rank r = min(X.shape)), compute explained_variance_ratio, and pack the
    # four arrays into the documented dict.
    raise NotImplementedError("Implement svd_decompose")


def pca_scores(X: np.ndarray, n_components: int, center: bool = True) -> np.ndarray:
    """Project a feature matrix onto its top principal components.

    Returns
    -------
    np.ndarray, shape ``(n_samples, n_components)``
        PCA scores (projections onto the leading components).
    """
    # TODO: return the top-`n_components` PCA projection of X (use pca_reduce).
    raise NotImplementedError("Implement pca_scores")


def robust_pca(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a matrix into low-rank plus sparse parts (robust PCA / PCP).

    Returns
    -------
    tuple of np.ndarray
        ``(L, S)`` with ``X ~= L + S``; ``L`` low-rank, ``S`` sparse.
    """
    # TODO: return the low-rank + sparse decomposition of X (use rpca).
    raise NotImplementedError("Implement robust_pca")


def run_ica(observations: np.ndarray, n_sources: int, seed: int = SEED) -> np.ndarray:
    """Recover independent sources from mixed multichannel observations.

    Parameters
    ----------
    observations: ``(n_channels, n_samples)`` observed mixtures.
    n_sources: number of independent sources to estimate.
    seed: forwarded to FastICA for reproducibility.

    Returns
    -------
    np.ndarray, shape ``(n_sources, n_samples)``
        Estimated source signals (one per row).
    """
    # TODO: run ICA (use ica_unmix) and return the estimated sources.
    raise NotImplementedError("Implement run_ica")


def scaling_sensitivity(X: np.ndarray, n_components: int = 1) -> dict:
    """Compare explained variance on raw vs per-feature standardized data.

    Returns
    -------
    dict
        Keys ``raw_evr`` and ``scaled_evr`` (both ``(min(shape),)`` arrays),
        ``raw_top`` and ``scaled_top`` (floats: summed top-``n_components``
        ratios), and ``sensitive`` (bool: does standardization move the top
        concentration by more than 0.1?).
    """
    # TODO: compute explained_variance_ratio on raw X and on z-scored X
    # (guard zero-variance columns), sum the top-`n_components` ratios of each,
    # and report whether they differ by more than 0.1.
    raise NotImplementedError("Implement scaling_sensitivity")


# --------------------------------------------------------------------------- #
# Quality control + application driver (provided; do not modify).
# --------------------------------------------------------------------------- #
def _abs_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Absolute Pearson correlation between two 1D arrays."""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(abs(np.corrcoef(a, b)[0, 1]))


def quality_control(expr: ExpressionMatrix, mixed, est_sources: np.ndarray) -> dict:
    """Run the required QC checks and return a metrics dictionary."""
    decomp = svd_decompose(expr.X)
    evr = decomp["explained_variance_ratio"]

    u, s, vt = svd_lowrank(expr.X - expr.X.mean(axis=0, keepdims=True), 2)
    recon = (u * s) @ vt + expr.X.mean(axis=0, keepdims=True)
    recon_err = reconstruction_error(expr.X, recon, kind="rel_l2")

    pc1_bio = _abs_corr(vt[0], expr.bio_direction)
    pc2_batch = _abs_corr(vt[1], expr.batch_direction)

    inflated = expr.X.copy()
    inflated[:, 0] *= 50.0
    sens = scaling_sensitivity(inflated, n_components=1)

    recovery = source_recovery_score(mixed.sources, est_sources)

    return {
        "evr_top2": float(evr[:2].sum()),
        "rank2_recon_rel_l2": float(recon_err),
        "pc1_vs_bio": pc1_bio,
        "pc2_vs_batch": pc2_batch,
        "scaling_sensitive": sens["sensitive"],
        "scaling_raw_top": sens["raw_top"],
        "scaling_scaled_top": sens["scaled_top"],
        "ica_recovery": float(recovery),
        "ica_trustworthy": bool(recovery > 0.9),
    }


def main() -> None:
    """Run the full PS5 application (B), QC (C), and interpretation (D)."""
    expr = make_expression_matrix(seed=SEED)
    mixed = make_mixed_sources(3, 2000, seed=SEED)

    est_sources = run_ica(mixed.observations, 3, seed=SEED)
    scores = pca_scores(expr.X, 2)

    qc = quality_control(expr, mixed, est_sources)
    pc1_vs_group = _abs_corr(scores[:, 0], expr.group)
    pc2_vs_batch_scores = _abs_corr(scores[:, 1], expr.batch)

    print("=== PS5: dimensionality reduction & source separation ===")
    print(f"PC1 leading loading vs biological program : {qc['pc1_vs_bio']:.3f}")
    print(f"PC2 leading loading vs batch program      : {qc['pc2_vs_batch']:.3f}")
    print(f"PC1 scores vs condition label             : {pc1_vs_group:.3f}")
    print(f"PC2 scores vs batch label                 : {pc2_vs_batch_scores:.3f}")
    print(f"Top-2 variance explained                  : {qc['evr_top2']:.3f}")
    print(f"Rank-2 reconstruction (rel L2)            : {qc['rank2_recon_rel_l2']:.3f}")
    print(
        f"PCA scale-sensitive (raw {qc['scaling_raw_top']:.2f} vs "
        f"scaled {qc['scaling_scaled_top']:.2f}): {qc['scaling_sensitive']}"
    )
    print(f"ICA source recovery vs ground truth       : {qc['ica_recovery']:.3f}")
    print(f"ICA trustworthy (recovery > 0.9)          : {qc['ica_trustworthy']}")

    rng = np.random.default_rng(SEED)
    low_true = rng.standard_normal((50, 3)) @ rng.standard_normal((3, 50))
    sparse_true = np.zeros((50, 50))
    mask = rng.random((50, 50)) < 0.05
    sparse_true[mask] = rng.standard_normal(int(mask.sum())) * 10.0
    low, sparse = robust_pca(low_true + sparse_true)
    print(
        f"Robust PCA low-rank recovery (rel L2)     : "
        f"{reconstruction_error(low_true, low, kind='rel_l2'):.3f}"
    )

    # ---- Part B (real data): PCA of a single-cell expression matrix ----
    # The synthetic matrix above is the *validation* fixture (known injected
    # directions, checked by the autograder). Here we apply the same validated
    # PCA to real PBMC3k single-cell data pulled through the course data layer,
    # which falls back to a structurally identical synthetic matrix offline.
    sc_expr, sc_labels, sc_source, sc_prov = load_single_cell_expression()
    sc_scores = pca_scores(sc_expr, 2)
    sc_evr = svd_decompose(sc_expr)["explained_variance_ratio"]
    print(f"\n[pbmc3k] source={sc_source}: {sc_prov}")
    print(f"Single-cell matrix (cells x genes)        : {sc_expr.shape}")
    print(f"Single-cell top-2 variance explained      : {float(sc_evr[:2].sum()):.3f}")
    if sc_labels is not None:
        print(
            f"PC1 vs provided cell labels (eta)         : "
            f"{_label_separation(sc_scores[:, 0], sc_labels):.3f}"
        )
    else:
        print(
            "PC1 vs cell labels                        : "
            "real payload is unlabelled; read PCs from gene loadings"
        )

    confidence = "high" if qc["ica_trustworthy"] and qc["pc1_vs_bio"] > 0.9 else "moderate"
    block = interpretation_block(
        claim=(
            "The dominant principal component reflects the injected biological "
            "condition, the second reflects batch, and ICA recovers the true "
            "independent sources on the synthetic recording."
        ),
        confidence=confidence,
        evidence=(
            f"PC1-biology |r|={qc['pc1_vs_bio']:.2f}, PC2-batch "
            f"|r|={qc['pc2_vs_batch']:.2f}, ICA recovery={qc['ica_recovery']:.2f} "
            f"(>0.9 gate passed)"
        ),
        limitations_list=[
            "Ground truth is synthetic; real expression/recordings add unmodeled noise.",
            "PCA is scale-dependent, so normalization choices change which axis leads.",
            "ICA sign/permutation are unidentifiable; only matched |correlation| is meaningful.",
            "Batch and biology were injected orthogonal; real confounds are often entangled.",
        ],
    )
    print("\n--- Interpretation ---")
    print(block)


if __name__ == "__main__":
    main()
