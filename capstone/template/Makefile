# Capstone template Makefile. Targets: data, run, report.
# The demo runs offline on synthetic data with only the core ddm4bio stack.

PYTHON ?= python

.PHONY: all data run report clean

all: data run report

## data: materialize inputs. The demo fixture is synthetic and generated
## in-process by the pipeline, so there is nothing to download here.
data:
	@echo "[data] demo uses an in-process synthetic fixture; nothing to fetch."

## run: execute the end-to-end pipeline (load -> QC -> SVD + DMD -> interpret).
run:
	$(PYTHON) src/pipeline.py

## report: fail if report.md is missing any mandated section header.
report:
	$(PYTHON) ../../scripts/check_capstone.py report.md

clean:
	@echo "[clean] nothing to remove for the demo."
