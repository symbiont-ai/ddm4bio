"""Honest-interpretation helpers for ddm4bio notebooks.

Every ddm4bio notebook must end its analysis with an explicit *interpretation
block*: a claim, a stated confidence level backed by evidence, and a named list
of limitations. These helpers format that block as **Markdown** and render it
with :func:`show_interpretation`, so the claim, evidence, and limitations wrap
and read as prose -- rather than an over-long, horizontally-scrolling line of
monospaced ``print`` output that hides the claim off the right edge.

See ``docs/INTERPRETATION.md`` for the convention and ``docs/METHOD_LABELING.md``
for the surrounding honesty policy.
"""

from __future__ import annotations

# Allowed confidence levels, from weakest to strongest evidence.
_VALID_CONFIDENCE = ("low", "moderate", "high")


def confidence_statement(claim: str, confidence: str | None, evidence: str | None = None) -> str:
    """Format the claim plus its confidence and optional evidence as Markdown.

    Leads with the claim as a sentence, then a separate bold
    ``**Confidence: <LEVEL>.**`` run, so the confidence *qualifies* the claim
    instead of being read as the claim itself.

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
        Markdown of the form::

            **Interpretation.** <claim>

            **Confidence: MODERATE.** <evidence>

    Raises
    ------
    ValueError
        If ``confidence`` is not one of the allowed levels.
    """
    if confidence is None:                       # opt out of a confidence line entirely
        lead = f"**Interpretation.** {claim.strip()}"
        return f"{lead}\n\n{evidence.strip()}" if evidence else lead

    level = confidence.strip().lower()
    if level not in _VALID_CONFIDENCE:
        raise ValueError(f"confidence must be one of {_VALID_CONFIDENCE!r}, got {confidence!r}")

    parts = [f"**Interpretation.** {claim.strip()}", ""]
    conf = f"**Confidence: {level.upper()}.**"
    if evidence:
        conf += f" {evidence.strip()}"
    parts.append(conf)
    return "\n".join(parts)


def limitations(items: list[str]) -> str:
    """Format a named-limitations block as a Markdown bulleted list.

    Parameters
    ----------
    items: individual, named limitations of the analysis. An empty list is
        rendered explicitly as "none stated" so a missing block is never
        silently mistaken for an analysis without limitations.

    Returns
    -------
    str
        Markdown headed ``**Limitations:**`` with one bullet per item.
    """
    cleaned = [item.strip() for item in items if item and item.strip()]
    body = "\n".join(f"- {item}" for item in cleaned) if cleaned else "- none stated"
    return f"**Limitations:**\n\n{body}"


def interpretation_block(
    claim: str,
    confidence: str | None,
    limitations_list: list[str],
    evidence: str | None = None,
) -> str:
    """Build the standard interpretation block for a notebook, as Markdown.

    Combines :func:`confidence_statement` and :func:`limitations`. Render it with
    :func:`show_interpretation` (not ``print``) so it wraps as prose.

    Parameters
    ----------
    claim: the substantive claim being made about the results.
    confidence: one of ``"low"``, ``"moderate"``, or ``"high"``.
    limitations_list: named limitations of the analysis.
    evidence: optional description of the supporting evidence.

    Returns
    -------
    str
        Markdown: the confidence statement, a blank line, then the limitations.

    Raises
    ------
    ValueError
        If ``confidence`` is not one of the allowed levels.
    """
    header = confidence_statement(claim, confidence, evidence)
    body = limitations(limitations_list)
    return f"{header}\n\n{body}"


def show_interpretation(block: str) -> None:
    """Render a Markdown interpretation block in a notebook.

    ``block`` is the Markdown string returned by :func:`interpretation_block`.
    Rendering it as Markdown -- rather than ``print``-ing it -- lets the claim,
    evidence, and each limitation wrap to the page width, instead of overflowing
    a single monospaced line that scrolls the claim out of view.
    """
    from IPython.display import Markdown, display

    display(Markdown(block))
