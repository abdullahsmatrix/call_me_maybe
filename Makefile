UV := uv
.PHONY: install run debug clean lint lint-strict

install:
	$(UV) sync --group dev

run:
	$(UV) run python3 -m src

test-scripts:
	$(UV) run python3 pytest test

debug:
	$(UV) run python3 -m pdb src

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -prune -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

cache_clean:
	$(UV) cache clean
lint:
	$(UV) run flake8 .
	$(UV) run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	$(UV) run flake8 .
	$(UV) run mypy . --strict
