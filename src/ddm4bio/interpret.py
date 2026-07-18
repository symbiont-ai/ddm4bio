"""Honest-interpretation helpers for ddm4bio notebooks.

Every ddm4bio notebook ends its analysis with an explicit *interpretation block*:
a plain-language claim plus a named list of limitations. These helpers format that
block as **Markdown** and render it with :func:`show_interpretation`, so the claim
and limitations wrap and read as prose -- rather than an over-long, horizontally
scrolling line of monospaced ``print`` output.

See ``docs/INTERPRETATION.md`` for the convention and ``docs/METHOD_LABELING.md``
for the surrounding honesty policy. There is deliberately no separate "confidence"
rating: how much to trust the claim belongs in the claim itself and in the named
limitations, not in a one-word label.
"""

from __future__ import annotations


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


def interpretation_block(claim: str, limitations_list: list[str]) -> str:
    """Build the standard interpretation block as Markdown.

    A claim, a blank line, then the named limitations. Render it with
    :func:`show_interpretation` (not ``print``) so it wraps as prose.

    Parameters
    ----------
    claim: the substantive, plain-language claim about the results, stated with
        whatever quantitative backing belongs in it.
    limitations_list: named limitations of the analysis.

    Returns
    -------
    str
        Markdown: ``**Interpretation.** <claim>`` then the limitations block.
    """
    return f"**Interpretation.** {claim.strip()}\n\n{limitations(limitations_list)}"


def show_interpretation(block: str) -> None:
    """Render a Markdown interpretation block in a notebook.

    ``block`` is the Markdown string from :func:`interpretation_block`. Rendering it
    as Markdown -- rather than ``print``-ing it -- lets the claim and each limitation
    wrap to the page width instead of overflowing a single monospaced line.
    """
    from IPython.display import Markdown, display

    display(Markdown(block))
