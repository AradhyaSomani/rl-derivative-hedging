# rl-hedging

Does a reinforcement-learning agent hedge a short option more cheaply, and with less tail risk,
than textbook delta hedging once trading costs exist and the market model is wrong?

**Finding so far (validation paths, provisional):** a deep hedging network cuts tail risk
(CVaR 95%) by 11% to 19% against Black-Scholes delta hedging in all eight scenarios and also
beats a tuned no-trade band. The reinforcement-learning agents (PPO, SAC) make the hedge about
12% cheaper on average at 10 basis points but not safer: they do not beat the band anywhere.

![Hedging cost distribution on Heston paths at 10 basis points](docs/figures/histogram_heston_10bps.png)

CVaR 95% of hedging cost at 10 basis points, 10,000 validation paths, median of 5 training
seeds for the learned strategies (lower is better):

| Strategy | GBM | Heston |
| --- | --- | --- |
| BS delta | 1.368 | 1.557 |
| No-trade band (tuned) | 1.305 | 1.514 |
| PPO | 1.376 | 1.634 |
| SAC | 1.398 | 1.580 |
| **Deep hedging** | **1.153** | **1.384** |

The full table with confidence intervals, the stress tests and everything that did not work
are in [docs/results.md](docs/results.md). The plan written before the experiments is in
[docs/preregistration.md](docs/preregistration.md).

**Status.** These numbers are on validation paths. The test paths (seeds 10,000 to 19,999)
have never been generated. The final run on them is one command, `make final-test`, which
refuses to start until the git tag `agents-frozen` exists.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
make install
make test
```

`make install` runs `pip install -e ".[dev,rl,serve]"`: the core package, the test tools,
stable-baselines3 with PyTorch (large), and FastAPI, Streamlit and the PostgreSQL driver.

## Reproduce everything

Every run is launched from a config file in `config/`, and every run folder in `data/runs/`
records the config hashes, the path-file hash, the seed and the git commit.

| Step | Command | Time on a 4-core CPU |
| --- | --- | --- |
| Paths (train and validation) | `make paths` | about 1 minute |
| Baselines, band tuned on validation | `make baselines` | under 1 minute |
| Train 90 runs (40 PPO, 10 SAC, 40 deep hedging) | `make train` | about 2 hours with `JOBS=4` (8 CPU-hours) |
| Score everything on validation, with bootstrap intervals | `make eval` | under 1 minute |
| Stress tests | `make stress` | about 1 minute |
| Figures and the demo's sample | `make report` | under 1 minute |
| All of the above | `make all` | about 2 hours |
| The single test run, after tagging `agents-frozen` | `make final-test` | about 2 minutes |

No GPU is needed. One PPO run (1,000,000 steps) takes about 450 seconds, one SAC run (100,000
steps) about 670 seconds and one deep hedging run about 100 seconds. The choices that led to
the frozen setup (reward grid, action space, hyperparameter search) can be rerun from the
other files in `config/experiments/` with
`python -m hedge.cli experiment --config <file>` and `python -m hedge.cli tune --config <file>`.

Other commands:

| Command | What it does |
| --- | --- |
| `python -m hedge.cli walkthrough --scenario config/scenarios/gbm_10bps.yaml --seed 0` | Print the step-by-step cash account for one delta-hedged path |
| `python -m hedge.cli sanity --scenarios config/scenarios/*.yaml` | Run the scripted delta agent and a random agent through the environment |

## Database, API and demo

```bash
createdb hedging
export DATABASE_URL=postgresql://localhost:5432/hedging
make db        # creates the schema in src/hedge/db/schema.sql and loads every result
make api       # FastAPI on http://localhost:8000, interactive docs at /docs
make app       # Streamlit demo on http://localhost:8501
```

- **Database.** PostgreSQL holds the scenarios, strategies, training runs and their learning
  curves, every evaluation with its metrics, the per-path results (about 1.1 million rows) and
  step-by-step detail for 25 sample paths per evaluation. `make db` rebuilds it from the files
  in `data/`, so the files stay the source of truth.
- **API.** Read endpoints over the database (`/scenarios`, `/strategies`, `/results`,
  `/frontier`, `/evaluations`, `/evaluations/{id}/distribution`,
  `/evaluations/{id}/paths/{path_id}`, `/runs/{id}/curve`, `/stress`, `/health`), plus two that
  run the frozen agents: `POST /hedge` returns each agent's holding and the BS delta for one
  market state, and `POST /simulate` hedges up to 1,000 fresh paths with every strategy.
- **Demo.** Five pages: a race between two hedgers on one path, cost distributions, the cost
  against tail-risk frontier with the stress tests, a page where you change cost and
  volatility and get fresh paths, and a plain-English summary. The demo does not need the
  database or a training run: it reads `data/results` if present and otherwise the small
  committed copy in `app/sample` (results extract and frozen weights, 7 MB), which is what a
  hosted copy uses. For Streamlit Community Cloud, point it at `app/streamlit_app.py` with
  `app/requirements.txt`.

## Layout

```text
app/               streamlit_app.py, sample/ (results extract and frozen weights for the demo)
config/            scenarios/, agents/, experiments/, eval.yaml (the frozen line-up)
data/              paths/ (Parquet), runs/, results/   (git-ignored)
docs/              theory.md, preregistration.md, results.md
src/hedge/
  seeds.py         seed ranges for train, validation, test; one seed per path
  config.py        scenario configs and their hashes
  pnl.py           the P&L engine every strategy is scored by
  market/          black_scholes.py, gbm.py, heston.py, pathsets.py
  baselines/       no_hedge.py, delta.py, no_trade_band.py, run.py
  eval/            metrics.py, evaluate.py, final.py (the evaluation and the test guard),
                   stress.py, live.py (fresh-path comparisons), plots.py, report.py
  env/             hedging_env.py (gymnasium), rewards.py
  agents/          scripted.py, train_sb3.py (PPO, SAC), deep_hedging.py, experiment.py, tune.py
  results.py       file-based access to finished results
  db/              schema.sql, load.py
  api/             main.py (FastAPI)
tests/
notebooks/         exploration only, never imported
```

## Rules the code enforces

- **One P&L engine.** `hedge.pnl.hedge_pnl` scores every strategy. It is tested against an
  independent cash-account implementation, the exact no-hedge result, and the near-zero
  variance of a frequent, cost-free delta hedge.
- **The environment adds no accounting of its own.** Its step P&Ls sum to the engine's total
  for every reward type, and a scripted agent that returns the Black-Scholes delta reproduces
  the delta baseline path by path.
- **Test paths are sacred.** Seeds 10,000 to 19,999 are the test set. The simulators and the
  path loader refuse them unless called with `allow_test=True`
  (`--final-test-run` on the command line), and the final evaluation also refuses to run
  unless the git tag `agents-frozen` exists.
- **Risk, not mean, and at least five seeds.** Every learned strategy is reported as the
  median of five training seeds with the range, and the comparison with delta is a paired
  bootstrap interval on CVaR 95%.
- **Same paths for every strategy.** One seed makes one path, and scenarios that differ only
  in cost level share the same path file.
