"""Honest-interpretation helpers for ddm4bio notebooks.

Every ddm4bio notebook must end its analysis with an explicit *interpretation
block*: a claim, a stated confidence level backed by evidence, and a named
list of limitations. These helpers format that block consistently so the
honesty norm is easy to follow and easy to review.

See ``docs/INTERPRETATION.md`` for the convention and ``docs/METHOD_LABELING.md``
for the surrounding honesty policy.
"""

from __future__ import annotations

# Allowed confidence levels, from weakest to strongest evidence.
_VALID_CONFIDENCE = ("low", "moderate", "high")


def confidence_statement(claim: str, confidence: str, evidence: str | None = None) -> str:
    """Format an honest confidence statement about a single claim.

    Parameters
    ----------
    claim: the substantive claim being made about the results.
    confidence: one of ``"low"``, ``"moderate"``, or ``"high"``. Any other
        value raises :class:`ValueError`.
    evidence: optional description of the evidence that justifies the stated
        confidence level (e.g. effect size, sample size, cross-validation).

    Returns
    -------
    str
        A formatted block leading with the claim, then a separate
        ``Confidence: <LEVEL>`` line and (when given) an ``Evidence:`` line, e.g.::

            Claim: <claim>

            Confidence: MODERATE
            Evidence: <evidence>

    Raises
    ------
    ValueError
        If ``confidence`` is not one of the allowed levels.
    """
    level = confidence.strip().lower()
    if level not in _VALID_CONFIDENCE:
        raise ValueError(f"confidence must be one of {_VALID_CONFIDENCE!r}, got {confidence!r}")

    parts = [f"Claim: {claim.strip()}", "", f"Confidence: {level.upper()}"]
    if evidence:
        parts.append(f"Evidence: {evidence.strip()}")
    return "\n".join(parts)


def limitations(items: list[str]) -> str:
    """Format a named-limitations block from a list of strings.

    Parameters
    ----------
    items: individual, named limitations of the analysis. An empty list is
        rendered explicitly as "none stated" so a missing block is never
        silently mistaken for an analysis without limitations.

    Returns
    -------
    str
        A formatted block headed ``"Limitations:"`` with one dash-prefixed
        line per item.
    """
    cleaned = [item.strip() for item in items if item and item.strip()]
    if not cleaned:
        return "Limitations:\n- none stated"
    lines = "\n".join(f"- {item}" for item in cleaned)
    return f"Limitations:\n{lines}"


def interpretation_block(
    claim: str,
    confidence: str,
    limitations_list: list[str],
    evidence: str | None = None,
) -> str:
    """Build the standard interpretation block for a notebook.

    Combines :func:`confidence_statement` and :func:`limitations` into the
    single block every ddm4bio notebook must include in its Interpretation
    section.

    Parameters
    ----------
    claim: the substantive claim being made about the results.
    confidence: one of ``"low"``, ``"moderate"``, or ``"high"``.
    limitations_list: named limitations of the analysis.
    evidence: optional description of the supporting evidence.

    Returns
    -------
    str
        The confidence statement followed by a blank line and the named
        limitations block.

    Raises
    ------
    ValueError
        If ``confidence`` is not one of the allowed levels.
    """
    header = confidence_statement(claim, confidence, evidence)
    body = limitations(limitations_list)
    return f"{header}\n\n{body}"
