.PHONY: install dev test lint run

install:
	python3 -m pip install -e .

dev:
	python3 -m pip install -e ".[dev]"

test:
	pytest tests/ -v --cov=context_tracker --cov-report=term-missing

lint:
	ruff check src/ tests/
	mypy src/context_tracker

run:
	python3 -m context_tracker.server

hook-install:
	python3 -m context_tracker.installer install

hook-uninstall:
	python3 -m context_tracker.installer uninstall
