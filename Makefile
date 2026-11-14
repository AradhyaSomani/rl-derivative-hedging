PY ?= python
JOBS ?= 4
SCENARIOS := $(wildcard config/scenarios/*.yaml)

.PHONY: install test lint paths baselines train eval stress final-test report db api app all

install:
	$(PY) -m pip install -e ".[dev,rl,serve]"
	pre-commit install

test:
	$(PY) -m pytest

lint:
	ruff check . && ruff format --check .

# Train and validation paths only. Test paths are generated once, by `make final-test`.
paths:
	@for s in $(SCENARIOS); do $(PY) -m hedge.cli paths --scenario $$s --splits train validation || exit 1; done

# Baselines on validation paths (band width tuned there too).
baselines:
	$(PY) -m hedge.cli baselines --scenarios $(SCENARIOS) --split validation

# The frozen setup: PPO and deep hedging on every scenario, SAC on the two 10 bps scenarios,
# 5 training seeds each. Finished runs are skipped, so this can be restarted.
train:
	$(PY) -m hedge.cli experiment --config config/experiments/scenarios_ppo.yaml --jobs $(JOBS)
	$(PY) -m hedge.cli experiment --config config/experiments/scenarios_sac.yaml --jobs $(JOBS)
	$(PY) -m hedge.cli experiment --config config/experiments/scenarios_deep_hedging.yaml --jobs $(JOBS)

# Every strategy on validation paths, with bootstrap confidence intervals. Safe to repeat.
eval:
	$(PY) -m hedge.cli eval --scenarios $(SCENARIOS) --split validation

stress:
	$(PY) -m hedge.cli stress --split validation

# THE SINGLE TEST RUN. Needs the git tag `agents-frozen`. Do this once.
final-test:
	$(PY) -m hedge.cli eval --scenarios $(SCENARIOS) --split test --final-test-run
	$(PY) -m hedge.cli stress --split test --final-test-run

# Figures for docs/ and the small results sample the demo ships with.
report:
	$(PY) -m hedge.cli report
	$(PY) -m hedge.cli export-sample

# Needs DATABASE_URL, e.g. postgresql://localhost:5432/hedging
db:
	$(PY) -m hedge.cli db-load

api:
	uvicorn hedge.api.main:app --reload

app:
	streamlit run app/streamlit_app.py

all: paths baselines train eval stress report
