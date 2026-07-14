"""Lint a capstone report for the mandated section headers.

Usage:
    python check_capstone.py [PATH_TO_REPORT]

Exits nonzero if any mandated section header is missing, or if the Quality
Control section does not appear before the Results section (the "QC before
conclusions" non-negotiable). Exits zero when the report is well-formed.

Pure standard library; no third-party dependencies.
"""

from __future__ import annotations

import sys

# Each mandated section is keyed by a canonical name and matched against the
# report's Markdown headings by a lowercase substring, so authors may append
# parentheticals (e.g. "## Interpretation (with confidence)").
REQUIRED_SECTIONS = (
    "Framing",
    "Methods",
    "Quality Control",
    "Results",
    "Interpretation",
    "Limitations",
    "Reproducibility",
)


def extract_headings(text: str) -> list[str]:
    """Return the lowercased text of every Markdown ATX heading line."""
    headings = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("#"):
            heading = line.lstrip("#").strip().lower()
            if heading:
                headings.append(heading)
    return headings


def find_section_index(headings: list[str], name: str) -> int:
    """Return the index of the first heading containing ``name``, else -1."""
    needle = name.lower()
    for index, heading in enumerate(headings):
        if needle in heading:
            return index
    return -1


def check_report(text: str) -> list[str]:
    """Return a list of problem strings; empty means the report passed."""
    headings = extract_headings(text)
    problems = []

    positions = {}
    for section in REQUIRED_SECTIONS:
        index = find_section_index(headings, section)
        if index < 0:
            problems.append(f"missing mandated section: '{section}'")
        else:
            positions[section] = index

    # Non-negotiable: Quality Control must precede Results.
    if "Quality Control" in positions and "Results" in positions:
        if positions["Quality Control"] >= positions["Results"]:
            problems.append(
                "'Quality Control' must appear before 'Results' (QC before conclusions)"
            )

    return problems


def main(argv: list[str]) -> int:
    """CLI entry point. Returns a process exit code."""
    path = argv[1] if len(argv) > 1 else "report.md"

    try:
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
    except OSError as error:
        print(f"FAIL: cannot read report '{path}': {error}", file=sys.stderr)
        return 2

    problems = check_report(text)
    if problems:
        print(f"FAIL: {path} is missing required structure:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    print(f"OK: {path} has all mandated sections in a valid order.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
