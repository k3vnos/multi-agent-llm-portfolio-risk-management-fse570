PY := .venv/bin
.PHONY: install test lint verify-data dev app app-stop app-test

install:
	python3.12 -m venv .venv && $(PY)/pip install -e ".[dev,news]"

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
