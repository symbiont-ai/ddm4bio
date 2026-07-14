"""Tests for the assert_no_leakage guard."""

import pytest

from ddm4bio.qc.report import assert_no_leakage


def test_disjoint_indices_do_not_raise():
    assert_no_leakage([0, 1, 2], [3, 4])


def test_overlapping_indices_raise():
    with pytest.raises(ValueError):
        assert_no_leakage([0, 1, 2], [2, 3, 4])
