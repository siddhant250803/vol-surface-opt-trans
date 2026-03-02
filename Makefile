.PHONY: run dry-run test install clean-cache

CONFIG ?= configs/base.yaml
VENV ?= .venv
PYTHON := $(VENV)/bin/python

run:
	$(PYTHON) -m pipeline.run --config $(CONFIG)

dry-run:
	$(PYTHON) -m pipeline.run --config $(CONFIG) --dry-run

test:
	$(PYTHON) -m pytest tests/ -v

install:
	python3 -m venv $(VENV) 2>/dev/null || true
	$(VENV)/bin/pip install -e ".[dev]" -q
	@echo "Run: source $(VENV)/bin/activate"

clean-cache:
	rm -rf outputs/cache/* data/processed/*
