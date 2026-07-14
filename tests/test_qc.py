"""Unit tests for tabular quality control.

Builds a small DataFrame with a missing value, a duplicate row, a constant
column, and a target column, then checks that ``qc_tabular`` captures those
facts and that ``QCReport`` renders / round-trips correctly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ddm4bio.qc.report import QCReport
from ddm4bio.qc.tabular import qc_tabular


def _make_df() -> pd.DataFrame:
    # Rows 1 and 2 are identical -> a duplicate row.
    # Column 'a' has one missing value. Column 'const' is constant.
    # Column 'target' is a target-like column for class balance.
    return pd.DataFrame(
        {
            "a": [1.0, 2.0, 2.0, 3.0, np.nan],
            "const": [5, 5, 5, 5, 5],
            "target": [0, 1, 1, 0, 1],
        }
    )


def test_qc_tabular_captures_facts():
    df = _make_df()
    report = qc_tabular(df)

    assert isinstance(report, QCReport)
    assert report.modality == "tabular"

    summary = report.summary
    assert summary["shape"] == (5, 3)

    # Missing value recorded for column 'a'.
    assert summary["missingness"]["a"] > 0.0
    assert summary["missingness_overall"] > 0.0

    # Duplicate row detected.
    assert summary["duplicate_rows"] >= 1

    # Constant column detected and warned about.
    assert "const" in summary["constant_columns"]
    assert any("const" in w for w in report.warnings)

    # Target column drives a class-balance summary.
    assert "class_balance" in summary
    assert set(summary["class_balance"].keys()) == {"0", "1"}


def test_qc_report_render_mentions_modality():
    report = qc_tabular(_make_df())
    text = report.render()
    assert isinstance(text, str)
    assert "tabular" in text


def test_qc_report_to_dict_round_trips():
    report = qc_tabular(_make_df())
    d = report.to_dict()

    assert d["modality"] == report.modality
    assert d["summary"] == report.summary
    assert d["warnings"] == report.warnings

    # Reconstructing a QCReport from the dict reproduces the fields.
    rebuilt = QCReport(**d)
    assert rebuilt.modality == report.modality
    assert rebuilt.summary == report.summary
    assert rebuilt.warnings == report.warnings
