.PHONY: lint format-check type-check test quality

lint:
	uv run ruff check .

format-check:
	uv run ruff format --check .

type-check:
	uv run mypy .

test:
	uv run pytest --cov=. --cov-report=term-missing

quality:
	$(MAKE) lint
	$(MAKE) format-check
	$(MAKE) type-check
	$(MAKE) test