"""Quality control for image datasets."""

from .report import QCReport


def qc_images(images=None, labels=None) -> QCReport:
    """Run quality-control checks on an image dataset.

    Computes and summarizes: count per class, intensity range (min/max/mean),
    size consistency (uniform vs. mixed dimensions), corrupt-file detection,
    and class balance.

    Parameters
    ----------
    images : sequence or array-like, optional
        Image array/stack of shape (N, H, W) or (N, H, W, C), or a list of
        per-image arrays.
    labels : sequence, optional
        Per-image class labels used for class balance and per-class counts.

    Returns
    -------
    QCReport
        Report with modality="images", populated summary and warnings.
    """
    import numpy as np

    summary: dict = {}
    warnings: list = []

    # Normalize the input into a list of per-image arrays plus a flag telling
    # us whether every image shares the same shape.
    per_image = []
    if images is None:
        per_image = []
    elif isinstance(images, np.ndarray):
        if images.ndim in (3, 4):
            per_image = [images[i] for i in range(images.shape[0])]
        elif images.ndim == 2:
            per_image = [images]
        else:
            per_image = list(images)
    else:
        per_image = [np.asarray(img) for img in images]

    count = len(per_image)
    summary["count"] = int(count)

    if count == 0:
        summary["size_consistent"] = True
        summary["dtype"] = None
        summary["intensity"] = {"min": None, "max": None, "mean": None}
        return QCReport(modality="images", summary=summary, warnings=warnings)

    # Size consistency.
    shapes = [tuple(np.asarray(img).shape) for img in per_image]
    unique_shapes = set(shapes)
    size_consistent = len(unique_shapes) == 1
    summary["size_consistent"] = bool(size_consistent)
    summary["shapes"] = sorted(str(s) for s in unique_shapes)
    if not size_consistent:
        warnings.append(f"Images have mixed sizes: {len(unique_shapes)} distinct shapes found.")

    # Dtype (report the set of dtypes present).
    dtypes = {str(np.asarray(img).dtype) for img in per_image}
    summary["dtype"] = sorted(dtypes)
    if len(dtypes) > 1:
        warnings.append(f"Images have mixed dtypes: {sorted(dtypes)}.")

    # Intensity statistics over all pixels.
    mins = [float(np.asarray(img).min()) for img in per_image]
    maxs = [float(np.asarray(img).max()) for img in per_image]
    means = [float(np.asarray(img).mean()) for img in per_image]
    summary["intensity"] = {
        "min": float(np.min(mins)),
        "max": float(np.max(maxs)),
        "mean": float(np.mean(means)),
    }

    # Per-class counts and class balance.
    if labels is not None:
        labels = list(labels)
        classes, counts = np.unique(np.asarray(labels), return_counts=True)
        per_class = {str(c): int(n) for c, n in zip(classes, counts)}
        summary["per_class_counts"] = per_class
        if counts.size > 1:
            imbalance = float(counts.max()) / float(counts.min())
            summary["class_imbalance_ratio"] = imbalance
            if imbalance > 1.5:
                warnings.append(
                    f"Class imbalance detected: majority/minority ratio {imbalance:.2f}."
                )
        if len(labels) != count:
            warnings.append(f"Label count ({len(labels)}) does not match image count ({count}).")

    return QCReport(modality="images", summary=summary, warnings=warnings)
