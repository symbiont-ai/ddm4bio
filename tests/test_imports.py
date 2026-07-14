"""Import tests: subpackages import and key callables exist.

These are hasattr checks only; the NotImplementedError stubs are never called.
"""

from ddm4bio import datasets, methods, qc, utils, viz
from ddm4bio.methods import validation
from ddm4bio.qc import images, report, signals, tabular


def test_subpackages_importable():
    for pkg in (qc, methods, datasets, viz, utils):
        assert pkg is not None


def test_validation_callables_exist():
    assert hasattr(validation, "source_recovery_score")
    assert hasattr(validation, "term_recovery")
    assert hasattr(validation, "reconstruction_error")


def test_qc_callables_exist():
    assert hasattr(report, "QCReport")
    assert hasattr(report, "assert_no_leakage")
    assert hasattr(tabular, "qc_tabular")
    assert hasattr(images, "qc_images")
    assert hasattr(signals, "qc_signals")
