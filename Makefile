.PHONY: all test integration snapshot figure

PYTHON ?= python3
DB ?= snapshots.db
OUTPUT ?= delta_vs_strike.png
SNAPSHOT ?= data/snapshots/btc_20260825_snapshot.json.gz

all: test integration snapshot

test:
	$(PYTHON) -m pytest -q

integration:
	$(PYTHON) -m pytest -q -m integration

snapshot:
	$(PYTHON) benchmarks/delta_strike_plot.py --db $(DB) --fetch --output $(OUTPUT)
