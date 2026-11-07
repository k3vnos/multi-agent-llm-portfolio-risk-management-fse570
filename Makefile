PY := .venv/bin
.PHONY: install install-locked check test lint verify-data dev app app-stop app-test

install:
	python3.12 -m venv .venv && $(PY)/pip install -e ".[dev,news,agents,api,dashboard]"

install-locked:   # exact versions that were tested (requirements-lock.txt)
	python3.12 -m venv .venv && $(PY)/pip install -r requirements-lock.txt && $(PY)/pip install -e . --no-deps

check:            # read-only: what is installed / missing after a fresh clone
	./scripts/check_setup.sh

test:
	$(PY)/pytest

lint:
	$(PY)/ruff check src tests

verify-data:
	$(PY)/riskcausal verify-data --profile full

dev:
	$(PY)/riskcausal run --profile dev

app:
	./run.sh start

app-stop:
	./run.sh stop

app-test:
	./run.sh test
