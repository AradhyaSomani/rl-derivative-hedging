PY ?= python
SCENARIOS := $(wildcard config/scenarios/*.yaml)

.PHONY: install test lint paths baselines train eval report app all

install:
	$(PY) -m pip install -e ".[dev]"
	pre-commit install

test:
	$(PY) -m pytest

lint:
	ruff check . && ruff format --check .

# Train and validation paths only. Test paths are generated once, by `make eval`.
paths:
	@for s in $(SCENARIOS); do $(PY) -m hedge.cli paths --scenario $$s --splits train validation || exit 1; done

# Baselines on validation paths (band width tuned there too).
baselines:
	$(PY) -m hedge.cli baselines --scenarios $(SCENARIOS) --split validation

train:
	@echo "train: not built yet (Phase 3)"; exit 1

eval:
	@echo "eval: not built yet (Phase 4). This is the single test-path run after the agents-frozen tag."; exit 1

report:
	@echo "report: not built yet (Phase 4)"; exit 1

app:
	@echo "app: not built yet (Phase 5)"; exit 1

all: paths baselines train eval report
