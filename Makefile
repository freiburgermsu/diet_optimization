.PHONY: install test lint optimize validate clean

install:
	pip install -e ".[dev]"

test:
	pytest -v

lint:
	ruff check diet_opt tests

validate:
	python -m diet_opt validate

optimize:
	python -m diet_opt optimize

clean:
	rm -rf build dist *.egg-info .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
