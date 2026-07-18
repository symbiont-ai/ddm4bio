"""Unit tests for the honest-interpretation helpers.

Checks that the formatted strings carry the provided content and that
``confidence_statement`` rejects invalid confidence levels.
"""

from __future__ import annotations

import pytest

from ddm4bio.interpret import (
    confidence_statement,
    interpretation_block,
    limitations,
)


def test_confidence_statement_contains_content():
    claim = "the mode explains the dominant oscillation"
    evidence = "cross-validated on 5 folds"
    text = confidence_statement(claim, "moderate", evidence=evidence)

    assert isinstance(text, str)
    assert claim in text
    assert evidence in text
    assert "MODERATE" in text
    # the claim must LEAD -- confidence is a separate labelled run, not a prefix on the claim
    assert text.startswith("**Interpretation.**")
    assert text.index(claim) < text.index("Confidence:")


def test_confidence_statement_accepts_all_valid_levels():
    for level in ("low", "moderate", "high"):
        text = confidence_statement("a claim", level)
        assert level.upper() in text


def test_confidence_statement_rejects_invalid_level():
    with pytest.raises(ValueError):
        confidence_statement("a claim", "certain")


def test_limitations_lists_items():
    items = ["small sample size", "no external validation"]
    text = limitations(items)
    assert isinstance(text, str)
    for item in items:
        assert item in text


def test_limitations_empty_is_explicit():
    text = limitations([])
    assert isinstance(text, str)
    assert "none stated" in text


def test_interpretation_block_combines_content():
    claim = "signal X drives the observed dynamics"
    lims = ["confounded by noise", "limited time horizon"]
    text = interpretation_block(claim, "high", lims, evidence="R^2 = 0.95")

    assert isinstance(text, str)
    assert claim in text
    assert "HIGH" in text
    assert "R^2 = 0.95" in text
    for lim in lims:
        assert lim in text


def test_interpretation_block_rejects_invalid_level():
    with pytest.raises(ValueError):
        interpretation_block("a claim", "definitely", ["a limitation"])
