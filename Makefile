.PHONY: all test integration snapshot figure forwards

PYTHON ?= python3
DB ?= snapshots.db
OUTPUT ?= figs/delta_vs_strike.png

all: test integration snapshot

test:
	$(PYTHON) -m pytest -q

integration:
	$(PYTHON) -m pytest -q -m integration

snapshot:
	$(PYTHON) benchmarks/delta_strike_plot.py --db $(DB) --fetch --output $(OUTPUT)

forwards:
	$(PYTHON) -m pytest -q -m forwards