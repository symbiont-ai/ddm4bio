"""QC report container and leakage guard."""

from dataclasses import dataclass, field


@dataclass
class QCReport:
    """Structured quality-control report for a dataset/modality.

    Attributes
    ----------
    modality : str
        Data modality, e.g. "tabular", "images", "signals".
    summary : dict
        Metric name -> value mapping summarizing the QC checks.
    warnings : list
        Human-readable warning strings for flagged issues.
    """

    modality: str
    summary: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)

    def render(self) -> str:
        """Render the report as a human-readable multi-line string.

        Returns
        -------
        str
            Formatted report text.
        """
        lines = [f"QC Report: {self.modality}"]
        lines.append("-" * len(lines[0]))

        lines.append("Summary:")
        if self.summary:
            for key, value in self.summary.items():
                lines.append(f"  {key}: {value}")
        else:
            lines.append("  (no summary metrics)")

        lines.append("Warnings:")
        if self.warnings:
            for warning in self.warnings:
                lines.append(f"  - {warning}")
        else:
            lines.append("  (no warnings)")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize the report to a plain dictionary.

        Returns
        -------
        dict
            Keys: ``modality``, ``summary``, ``warnings``.
        """
        return {
            "modality": self.modality,
            "summary": dict(self.summary),
            "warnings": list(self.warnings),
        }


def assert_no_leakage(train_idx, test_idx, groups=None) -> None:
    """Assert there is no index (or group) overlap between train and test.

    Parameters
    ----------
    train_idx : sequence
        Indices assigned to the training split.
    test_idx : sequence
        Indices assigned to the test split.
    groups : sequence, optional
        Per-sample group labels. If provided, overlap is checked at the group
        level: no group may appear in both splits.

    Raises
    ------
    ValueError
        If any index (or group) appears in both splits.
    """
    if groups is None:
        overlap = set(train_idx) & set(test_idx)
        if overlap:
            raise ValueError(f"Index leakage between train and test: {sorted(overlap)}")
        return

    groups = list(groups)
    train_groups = {groups[i] for i in train_idx}
    test_groups = {groups[i] for i in test_idx}
    group_overlap = train_groups & test_groups
    if group_overlap:
        raise ValueError(f"Group leakage between train and test: {sorted(group_overlap)}")
