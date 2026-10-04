.PHONY: \
	lint \
	format-check \
	type-check \
	test quality \
	report \
	predictions \
	frontend-lint \
	frontend-format-check \
	frontend-type-check \
	frontend-build \
	frontend-quality \
	frontend-dev \
	frontend-dev-checked

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

frontend-lint:
	cd frontend && npm run lint

frontend-format-check:
	cd frontend && npm run format-check

frontend-type-check:
	cd frontend && npm run type-check

frontend-build:
	cd frontend && npm run build

frontend-quality:
	$(MAKE) frontend-lint
	$(MAKE) frontend-format-check
	$(MAKE) frontend-type-check
	$(MAKE) frontend-build

frontend-dev:
	cd frontend && npm run dev

frontend-dev-checked:
	$(MAKE) frontend-quality
	$(MAKE) frontend-dev

report:
	uv run python -m serving.cost_models.main --model $(MODEL) --hardware $(HARDWARE)

predictions: report
	@{ \
		cat serving/cost_models/outputs/analytical_report.md; \
		printf '\n\n'; \
		cat serving/cost_models/outputs/prediction_notes.md; \
	} > serving/cost_models/outputs/predictions.md
