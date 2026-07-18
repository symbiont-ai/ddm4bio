# Capstone template Makefile. Targets: data, run, report.
# The pipeline runs on the real warfarin PK/PD data (synthetic fallback offline).

PYTHON ?= python

.PHONY: all data run report clean

all: data run report

## data: fetch + cache the warfarin dataset (falls back to synthetic offline).
data:
	$(PYTHON) -c "from ddm4bio.datasets import get_dataset; d = get_dataset('warfarin'); print('[data] warfarin:', d.source, '-', d.payload['id'].nunique(), 'subjects cached')"

## run: execute the end-to-end pipeline (load -> QC -> PK/PD -> RL -> validate).
run:
	$(PYTHON) src/pipeline.py

## report: fail if report.md is missing any mandated section header.
report:
	$(PYTHON) ../../scripts/check_capstone.py report.md

clean:
	@echo "[clean] nothing to remove; cached data lives under the ddm4bio cache dir."
