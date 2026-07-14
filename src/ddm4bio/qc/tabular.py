"""Quality control for tabular datasets."""

from .report import QCReport


def qc_tabular(df) -> QCReport:
    """Run quality-control checks on a tabular dataset.

    Computes and summarizes: shape, per-column dtype summary, missingness
    (per-column and overall), duplicate rows, constant columns, class balance
    (for a target column if present), and simple outlier flags (e.g. z-score
    or IQR based) for numeric columns.

    Parameters
    ----------
    df : pandas.DataFrame
        The tabular data to inspect.

    Returns
    -------
    QCReport
        Report with modality="tabular", populated summary and warnings.
    """
    n_rows, n_cols = df.shape
    summary: dict = {}
    warnings: list = []

    summary["shape"] = (int(n_rows), int(n_cols))
    summary["dtypes"] = {col: str(dtype) for col, dtype in df.dtypes.items()}

    # Missingness per column and overall.
    null_counts = df.isna().sum()
    per_col_missing = {}
    for col in df.columns:
        frac = float(null_counts[col]) / n_rows if n_rows else 0.0
        per_col_missing[col] = frac
        if frac > 0.20:
            warnings.append(
                f"Column '{col}' has high missingness: {frac:.1%} of values are missing."
            )
    summary["missingness"] = per_col_missing
    total_cells = n_rows * n_cols
    summary["missingness_overall"] = float(null_counts.sum()) / total_cells if total_cells else 0.0

    # Duplicate rows.
    summary["duplicate_rows"] = int(df.duplicated().sum())

    # Constant columns (a single unique non-null value, ignoring NaNs).
    constant_columns = []
    for col in df.columns:
        if df[col].nunique(dropna=True) <= 1:
            constant_columns.append(col)
            warnings.append(f"Column '{col}' is constant (<=1 unique value).")
    summary["constant_columns"] = constant_columns

    # Class balance for a target-like column.
    target_col = None
    for candidate in ("target", "label", "class"):
        if candidate in df.columns:
            target_col = candidate
            break
    if target_col is not None:
        counts = df[target_col].value_counts(dropna=False)
        summary["class_balance"] = {str(k): int(v) for k, v in counts.items()}

    # IQR-rule outlier count per numeric column.
    outlier_counts = {}
    numeric_cols = df.select_dtypes(include="number").columns
    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            outlier_counts[col] = 0
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = (series < lower) | (series > upper)
        outlier_counts[col] = int(mask.sum())
    summary["outliers_iqr"] = outlier_counts

    return QCReport(modality="tabular", summary=summary, warnings=warnings)
