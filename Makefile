.PHONY: all test integration snapshot figure forwards forward_curve segmentation raw_surface

PYTHON ?= python3
# PYTHON := .venv/bin/python
DB ?= snapshots.db
OUTPUT ?= figs/delta_vs_strike.png
FORWARD_OUTPUT ?= figs/forward_curve.png
RAW_SURFACE_OUTPUT ?= figs/raw_vol_smiles.png
SURFACE_MODEL ?= black76

all: test integration snapshot forwards forward_curve segmentation

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

raw_surface:
	$(PYTHON) -m benchmarks.raw_vol_surface_plot --db $(DB) --model $(SURFACE_MODEL) --output $(RAW_SURFACE_OUTPUT)