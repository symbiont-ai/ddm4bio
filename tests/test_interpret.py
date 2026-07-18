"""Unit tests for the honest-interpretation helpers.

Checks that the formatted Markdown leads with the claim and carries the named
limitations, and that there is no confidence rating in the output.
"""

from __future__ import annotations

from ddm4bio.interpret import interpretation_block, limitations


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


def test_interpretation_block_leads_with_claim_and_lists_limits():
    claim = "signal X drives the observed dynamics"
    lims = ["confounded by noise", "limited time horizon"]
    text = interpretation_block(claim, lims)

    assert isinstance(text, str)
    assert text.startswith("**Interpretation.**")
    assert claim in text
    for lim in lims:
        assert lim in text


def test_interpretation_block_has_no_confidence_rating():
    text = interpretation_block("a claim", ["a limitation"])
    assert "Confidence" not in text


def test_interpretation_block_empty_limitations_is_explicit():
    text = interpretation_block("a claim", [])
    assert "none stated" in text
