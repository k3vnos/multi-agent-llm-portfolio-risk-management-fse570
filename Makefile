PY := .venv/bin
.PHONY: install test lint verify-data

install:
	python3.12 -m venv .venv && $(PY)/pip install -e ".[dev,news]"

test:
	$(PY)/pytest

lint:
	$(PY)/ruff check src tests

verify-data:
	$(PY)/riskcausal verify-data --profile full
