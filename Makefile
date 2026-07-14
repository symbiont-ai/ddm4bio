.PHONY: setup lint format test notebooks data clean

VENV ?= .venv

setup:
	python -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -e ".[dev]"

lint:
	ruff check src tests
	ruff format --check src tests

format:
	ruff format src tests

test:
	pytest

notebooks:
	@echo "nbmake notebook smoke tests land in a later phase"

data:
	$(VENV)/bin/python scripts/fetch_data.py

clean:
	rm -rf .ruff_cache .pytest_cache .mypy_cache build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
