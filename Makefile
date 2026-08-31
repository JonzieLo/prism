.PHONY: all test integration snapshot figure forwards forward-curve segmentation

PYTHON ?= python3
# PYTHON := .venv/bin/python
DB ?= snapshots.db
OUTPUT ?= figs/delta_vs_strike.png
FORWARD_OUTPUT ?= figs/forward_curve.png

all: test integration snapshot forwards segmentation

test:
	$(PYTHON) -m pytest -q

integration:
	$(PYTHON) -m pytest -q -m integration

snapshot:
	$(PYTHON) -m benchmarks.delta_strike_plot --db $(DB) --fetch --output $(OUTPUT)

forwards:
	$(PYTHON) -m pytest -q -m forwards

forward_curve:
	$(PYTHON) -m benchmarks.forward_curve_plot --db $(DB) --fetch --output $(FORWARD_OUTPUT)

segmentation:
	$(PYTHON) -m pytest -q -m segmentation