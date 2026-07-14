# ddm4bio

**Data-Driven Modeling & Scientific Computation for the Life Sciences** — a
reproducible graduate-course repository: a pip-installable teaching library
(`ddm4bio`) plus lecture-demo notebooks, problem sets, and a capstone.

Content follows J. N. Kutz, *Data-Driven Modeling & Scientific Computation*
(2nd ed.). This repo is built in phases; **Phase 0 is the scaffold** — the
package tree, interfaces, config, tests, and CI are in place, while method
bodies are stubs to be filled in later phases.

## Quickstart

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
ruff check src tests
pytest
```

With `make` installed you can instead run `make setup`, `make lint`, `make test`.

### No `make`? Raw equivalents

| Target       | Command                                             |
|--------------|-----------------------------------------------------|
| `make setup` | `python -m venv .venv && pip install -e ".[dev]"`   |
| `make lint`  | `ruff check src tests && ruff format --check src tests` |
| `make test`  | `pytest`                                            |
| `make format`| `ruff format src tests`                             |

## Dependencies

The core install is intentionally light (`numpy`, `scipy`, `pandas`,
`scikit-learn`, `matplotlib`) so setup is fast and CI runs offline. Heavier, area-specific
stacks are opt-in extras:

```bash
pip install -e ".[signals]"     # PyWavelets, mne, wfdb
pip install -e ".[singlecell]"  # scanpy, anndata
pip install -e ".[dynamics]"    # pysindy, pydmd
pip install -e ".[dl]"          # torch
pip install -e ".[imaging]"     # medmnist, scikit-image, nibabel
```

## Layout

```
src/ddm4bio/   shared library (config, qc, viz, methods, datasets, utils)
notebooks/     16 lecture demos (jupytext-paired) — later phases
problem_sets/  7 problem sets (student scaffold + solution + tests) — later phases
capstone/      capstone spec + template — later phases
data/          DATA_CARD.md + gitignored raw/ and cache/
tests/         library unit tests
docs/          conventions (see METHOD_LABELING.md)
```

## Conventions

See [`CLAUDE.md`](CLAUDE.md) for the golden rules (QC before results, honest
method labeling, ground-truth validation, reproducible seeds) that every
contribution — human or agent — must follow.

## License

MIT (code). Dataset licenses are tracked separately in
[`data/DATA_CARD.md`](data/DATA_CARD.md).
