"""PS6 -- Unsupervised discovery and supervised inference (student version).

Fill in every function marked ``# TODO``. The data-loading and quality-control
plumbing is already wired for you: run this file and it will load the fixtures,
print a QC report, and then stop at the first method you still need to write.

Read the companion README.md for the assignment brief (parts A-D) and the
required reading in Kutz chapters 17-18 and 13. You will lean on the course
library:

    ddm4bio.methods.clustering -- kmeans_cluster, gmm_cluster,
        hierarchical_cluster, select_k_silhouette, select_k_bic,
        consensus_cluster
    ddm4bio.methods.learning   -- cross_validate, roc_with_ci, bh_fdr
    ddm4bio.interpret          -- interpretation_block

Keep every public signature exactly as given: the autograder imports these
names. numpy is imported at module level; import scipy/scikit-learn inside the
function bodies, as the solution does.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ddm4bio.config import GLOBAL_SEED
from ddm4bio.interpret import interpretation_block

# These course-library helpers are pre-wired for you to use in the TODOs below.
# The lint-ignore markers just silence "imported but unused" until you use them.
from ddm4bio.methods.clustering import (  # noqa: F401
    consensus_cluster,
    gmm_cluster,
    hierarchical_cluster,
    kmeans_cluster,
    select_k_bic,
    select_k_silhouette,
)
from ddm4bio.methods.learning import (  # noqa: F401
    bh_fdr,
    cross_validate,
    roc_with_ci,
)
from ddm4bio.qc.report import QCReport, assert_no_leakage  # noqa: F401

# ---------------------------------------------------------------------------
# Data loading (offline fixtures) -- provided; do not edit.
# ---------------------------------------------------------------------------


def load_subtype_data(
    n_samples: int = 300,
    n_features: int = 20,
    centers: int = 3,
    cluster_std: float = 1.0,
    seed: int = GLOBAL_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic expression matrix with known latent subtypes (provided)."""
    from sklearn.datasets import make_blobs

    x, y = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=centers,
        cluster_std=cluster_std,
        random_state=seed,
    )
    return np.asarray(x, dtype=float), np.asarray(y, dtype=int)


def load_diagnostic_data() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Breast-cancer diagnostic dataset via the course data layer (provided).

    Resolves through ``get_dataset("breast_wisconsin", download=False)``, which
    returns the offline scikit-learn-bundled WDBC data deterministically.
    """
    from ddm4bio.datasets import get_dataset

    payload = get_dataset("breast_wisconsin", download=False).payload
    x = np.asarray(payload["X"], dtype=float)
    y = np.asarray(payload["y"], dtype=int)
    names = [str(n) for n in payload["feature_names"]]
    return x, y, names


def _unpack_singlecell(payload: Any) -> tuple[Any, np.ndarray | None]:
    """Return ``(counts, labels)`` from an AnnData or a plain-dict payload.

    The real 10x payload is an ``anndata.AnnData`` (raw counts in ``.X``, no
    ground-truth cell types); the synthetic fallback is a dict with ``counts``
    and planted ``labels``.
    """
    if hasattr(payload, "X"):  # AnnData: counts in .X, no ground-truth labels
        return payload.X, None
    return payload["counts"], payload.get("labels")


def load_singlecell_data(
    n_keep: int = 600,
    n_top: int = 1000,
    n_pcs: int = 30,
    seed: int = GLOBAL_SEED,
) -> tuple[np.ndarray, np.ndarray | None, str]:
    """Real PBMC3k single-cell RNA-seq reduced to a clustering-ready embedding.

    Loads the 10x Genomics PBMC3k matrix through ``get_dataset("pbmc3k")``
    (``download=True`` by default, with a graceful synthetic fallback when the
    source is unreachable), unpacks the AnnData-or-dict payload to raw counts
    (plus planted labels when the fallback provides them), then normalizes each
    cell to a common library size, takes ``log1p``, keeps the most variable
    genes, and reduces to principal components.

    Returns ``(embedding, labels, source)`` where ``embedding`` has shape
    ``(n_cells, n_pcs)``, ``labels`` are the planted cell types when the payload
    carries them (synthetic fallback) else ``None``, and ``source`` is the
    provenance tag (``"real"`` or ``"fallback"``).
    """
    from sklearn.decomposition import PCA

    from ddm4bio.datasets import get_dataset

    ds = get_dataset("pbmc3k", seed=seed)
    counts_raw, raw_labels = _unpack_singlecell(ds.payload)

    rng = np.random.default_rng(seed)
    n_keep = min(n_keep, counts_raw.shape[0])
    cells = np.sort(rng.choice(counts_raw.shape[0], size=n_keep, replace=False))

    sub = counts_raw[cells]
    counts = sub.toarray() if hasattr(sub, "toarray") else np.asarray(sub)
    counts = np.asarray(counts, dtype=float)
    labels = None if raw_labels is None else np.asarray(raw_labels)[cells]

    library = counts.sum(axis=1, keepdims=True)
    library[library == 0] = 1.0
    logn = np.log1p(counts / library * 1e4)

    n_top = min(n_top, logn.shape[1])
    top = np.argsort(logn.var(axis=0))[::-1][:n_top]
    n_comp = min(n_pcs, n_keep - 1, n_top)
    embed = PCA(n_components=n_comp, random_state=seed).fit_transform(logn[:, top])
    return np.asarray(embed, dtype=float), labels, ds.source


# ---------------------------------------------------------------------------
# (A) Method + (B) Application
# ---------------------------------------------------------------------------


def select_number_of_subtypes(
    X: np.ndarray, k_range: object, seed: int = GLOBAL_SEED
) -> dict[str, Any]:
    """Pick the number of clusters via silhouette AND mixture BIC, then compare.

    Return a dict with keys ``best_k_silhouette``, ``best_k_bic`` (ints),
    ``silhouette``, ``bic`` (the raw result dicts from the helpers), and
    ``agree`` (bool: do the two criteria pick the same k?).

    Hint: silhouette needs k >= 2; BIC accepts k >= 1. Use
    ``select_k_silhouette`` and ``select_k_bic``.
    """
    # TODO: run both model-selection criteria and report whether they agree.
    raise NotImplementedError


def cluster_all_methods(X: np.ndarray, k: int, seed: int = GLOBAL_SEED) -> dict[str, np.ndarray]:
    """Partition ``X`` into ``k`` clusters with k-means, GMM, and hierarchical.

    Return a dict with keys ``kmeans``, ``gmm``, ``hierarchical`` mapping to
    integer label arrays of shape ``(n_samples,)``. Use the clustering helpers
    (Ward linkage for the hierarchical one).
    """
    # TODO: call the three clustering helpers and collect their labels.
    raise NotImplementedError


def cluster_agreement(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    """Adjusted Rand index between two labelings (permutation-invariant).

    Return a float in ``[-0.5, 1.0]``. Use sklearn's ``adjusted_rand_score``.
    """
    # TODO: import adjusted_rand_score and return the ARI as a float.
    raise NotImplementedError


def build_classifier(name: str, seed: int = GLOBAL_SEED) -> Any:
    """Return a StandardScaler-wrapped classifier by short name.

    Support ``"lda"``, ``"svm"``, ``"tree"``, ``"nn"``. Wrap each estimator in a
    ``Pipeline([("scale", StandardScaler()), ("clf", ...)])`` so the scaler is
    refit inside every CV fold (no leakage). Raise ``ValueError`` for an unknown
    name. Pass ``random_state=seed`` to the stochastic estimators.
    """
    # TODO: build and return the requested pipeline (raise ValueError otherwise).
    raise NotImplementedError


def evaluate_classifiers(
    X: np.ndarray,
    y: np.ndarray,
    names: object = ("lda", "svm", "tree", "nn"),
    cv: int = 5,
    seed: int = GLOBAL_SEED,
) -> dict[str, dict[str, Any]]:
    """Cross-validate each named classifier (stratified k-fold accuracy).

    Return ``name -> {"mean", "std", "scores"}``. Use ``build_classifier`` and
    the shared ``cross_validate`` helper with ``scoring="accuracy"``.
    """
    # TODO: loop over names, build each classifier, and cross-validate it.
    raise NotImplementedError


def _positive_scores(clf: Any, x: np.ndarray) -> np.ndarray:
    """Positive-class score from a fitted classifier (provided helper).

    Prefers ``predict_proba``; falls back to ``decision_function`` for margin
    classifiers. Both are monotone in the positive-class evidence.
    """
    if hasattr(clf, "predict_proba"):
        return np.asarray(clf.predict_proba(x))[:, 1]
    return np.asarray(clf.decision_function(x), dtype=float)


def diagnostic_auc(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.3,
    clf_name: str = "lda",
    n_boot: int = 1000,
    seed: int = GLOBAL_SEED,
) -> dict[str, Any]:
    """Train a diagnostic classifier and report held-out ROC-AUC with a CI.

    Steps: make a stratified train/test split of the row indices; guard it with
    ``assert_no_leakage``; fit ``build_classifier(clf_name)`` on the train rows;
    score the test rows with ``_positive_scores``; then call ``roc_with_ci``.

    Return a dict with keys ``auc`` (float), ``auc_ci`` (tuple), ``fpr``/``tpr``
    (arrays), ``n_test`` (int), and ``clf_name`` (str).
    """
    # TODO: split (stratified, leakage-checked), fit, score, and call roc_with_ci.
    raise NotImplementedError


# ---------------------------------------------------------------------------
# (C) Quality control
# ---------------------------------------------------------------------------


def assess_cluster_stability(
    X: np.ndarray,
    k: int,
    n_boot: int = 50,
    subsample: float = 0.8,
    seed: int = GLOBAL_SEED,
    min_stability: float = 0.8,
) -> dict[str, Any]:
    """Consensus-resampling stability of a ``k``-cluster solution.

    Run ``consensus_cluster``; summarize the upper-triangular off-diagonal
    entries of the consensus matrix by the Proportion of Ambiguously Clustered
    pairs (PAC) -- the fraction of pair values in the indecisive band
    ``(0.1, 0.9)``. Define ``stability = 1 - PAC`` and
    ``is_reproducible = stability >= min_stability``. When not reproducible,
    append a human-readable string to a ``warnings`` list.

    Return a dict with keys ``consensus_matrix``, ``labels``, ``pac`` (float),
    ``stability`` (float), ``is_reproducible`` (bool), ``warnings`` (list).
    """
    # TODO: run consensus clustering, compute PAC/stability, and flag instability.
    raise NotImplementedError


def per_feature_tests(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Two-sample Welch t-test p-value for every feature between two classes.

    Return an array of shape ``(n_features,)`` of uncorrected p-values. Raise
    ``ValueError`` if ``y`` does not have exactly two classes. Hint:
    ``scipy.stats.ttest_ind(..., axis=0, equal_var=False)``.
    """
    # TODO: split rows by class and run a per-column two-sample t-test.
    raise NotImplementedError


def fdr_correct(pvalues: np.ndarray, alpha: float = 0.05) -> dict[str, np.ndarray]:
    """Benjamini-Hochberg FDR control over a family of p-values.

    Return a dict with keys ``qvalues`` and ``reject``. Use the ``bh_fdr``
    helper.
    """
    # TODO: apply Benjamini-Hochberg correction via bh_fdr.
    raise NotImplementedError


def run_qc(X: np.ndarray, y: np.ndarray, feature_names: list[str] | None = None) -> QCReport:
    """Tabular QC on the feature matrix and its labels (provided plumbing)."""
    import pandas as pd

    from ddm4bio.qc.tabular import qc_tabular

    x = np.asarray(X, dtype=float)
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(x.shape[1])]
    df = pd.DataFrame(x, columns=feature_names)
    df["target"] = np.asarray(y)
    return qc_tabular(df)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the full PS6 workflow end to end and print an interpretation block."""
    seed = GLOBAL_SEED

    # (B) Discover candidate cell subtypes in REAL single-cell PBMC3k data.
    embed, sc_labels, sc_source = load_singlecell_data(seed=seed)
    print(f"pbmc3k source={sc_source} embedding={embed.shape[0]} cells x {embed.shape[1]} PCs")

    # QC plumbing runs before any modeling (provided). With no cell types the
    # target is a placeholder just for the tabular shape/missingness check.
    qc_target = sc_labels if sc_labels is not None else np.zeros(embed.shape[0], dtype=int)
    print(run_qc(embed, qc_target).render())
    print()

    # --- From here down you must implement the method functions above. ---
    sel = select_number_of_subtypes(embed, range(1, 7), seed=seed)
    k = sel["best_k_silhouette"]
    print(
        f"Selected k: silhouette={sel['best_k_silhouette']} "
        f"BIC={sel['best_k_bic']} (agree={sel['agree']})"
    )

    labels = cluster_all_methods(embed, k=k, seed=seed)
    ari_km_gmm = cluster_agreement(labels["kmeans"], labels["gmm"])
    if sc_labels is not None:
        ari_ref = cluster_agreement(sc_labels, labels["kmeans"])
        print(f"ARI vs planted labels (k-means)={ari_ref:.3f} | k-means~GMM={ari_km_gmm:.3f}")
    else:
        print(f"No ground-truth labels (real cells) | k-means~GMM={ari_km_gmm:.3f}")

    stab = assess_cluster_stability(embed, k=k, seed=seed)
    print(f"Cluster stability={stab['stability']:.3f} reproducible={stab['is_reproducible']}")
    for w in stab["warnings"]:
        print(f"  WARNING: {w}")

    x_dx, y_dx, names = load_diagnostic_data()
    cv_results = evaluate_classifiers(x_dx, y_dx, seed=seed)
    for name, r in cv_results.items():
        print(f"CV accuracy [{name}]: {r['mean']:.3f} +/- {r['std']:.3f}")

    dx = diagnostic_auc(x_dx, y_dx, clf_name="lda", seed=seed)
    lo, hi = dx["auc_ci"]
    print(f"Diagnostic ROC-AUC={dx['auc']:.3f} (95% CI {lo:.3f}-{hi:.3f})")

    pvals = per_feature_tests(x_dx, y_dx)
    fdr = fdr_correct(pvals, alpha=0.05)
    n_sig = int(fdr["reject"].sum())
    print(f"FDR-significant features: {n_sig}/{pvals.size} at alpha=0.05")

    # (D) Interpretation & confidence.
    print()
    print(
        interpretation_block(
            claim=(
                f"Real PBMC3k cells ({sc_source}) split into {k} candidate subtypes "
                f"and the breast-cancer classifier is diagnostic (AUC {dx['auc']:.2f})."
            ),
            confidence="high",
            limitations_list=[
                "Real single-cell subtypes have no ground-truth labels; the evidence "
                "is cross-method agreement and consensus stability, not an ARI.",
                "A single train/test split gives one AUC point estimate.",
                "FDR bounds the expected false-discovery rate, not any single feature.",
            ],
            evidence=(
                f"k-means~GMM ARI={ari_km_gmm:.2f}, consensus stability "
                f"{stab['stability']:.2f}, AUC CI {lo:.2f}-{hi:.2f}, BH-FDR at alpha=0.05"
            ),
        )
    )


if __name__ == "__main__":
    main()
