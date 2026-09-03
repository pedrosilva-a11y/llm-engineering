.PHONY: lint format-check type-check test quality report predictions

MODEL ?= qwen2.5-1.5b
HARDWARE ?= L4

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

report:
	uv run python -m serving.cost_models.main --model $(MODEL) --hardware $(HARDWARE)

predictions: report
	@{ \
		cat serving/cost_models/outputs/analytical_report.md; \
		printf '\n\n'; \
		cat serving/cost_models/outputs/prediction_notes.md; \
	} > serving/cost_models/outputs/predictions.md
