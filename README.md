# rl-hedging

Does a reinforcement-learning agent hedge a short option more cheaply, and with less tail risk,
than textbook delta hedging once trading costs exist and the market model is wrong?

Status: Phase 1 code (simulators, P&L engine, baselines). No agents are trained yet, so there
are no results to report. The experiment plan is in
[docs/preregistration.md](docs/preregistration.md).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
make install        # pip install -e ".[dev]" and pre-commit install
make test
```

## Commands

Every run is launched from a config file in `config/`.

| Command | What it does |
| --- | --- |
| `make paths` | Generate train (100,000) and validation (10,000) paths for every scenario into `data/paths/` |
| `make baselines` | Tune the no-trade band and score the three baselines on validation paths; writes `data/results/` |
| `python -m hedge.cli walkthrough --scenario config/scenarios/gbm_10bps.yaml --seed 0` | Print the step-by-step cash account for one delta-hedged path |
| `make train`, `eval`, `report`, `app` | Not built yet (Phases 3 to 5) |

## Layout

```text
config/            scenarios/*.yaml, agents/*.yaml
data/              paths/ (Parquet), runs/, results/   (git-ignored)
docs/              theory.md, preregistration.md, results.md
src/hedge/
  seeds.py         seed ranges for train, validation, test; one seed per path
  config.py        scenario configs and their hashes
  pnl.py           the P&L engine every strategy is scored by
  market/          black_scholes.py, gbm.py, heston.py, pathsets.py
  baselines/       no_hedge.py, delta.py, no_trade_band.py, run.py
  eval/            metrics.py
  env/ agents/ db/ api/    later phases
tests/
notebooks/         exploration only, never imported
```

## Rules the code enforces

- **One P&L engine.** `hedge.pnl.hedge_pnl` scores every strategy. It is tested against an
  independent cash-account implementation, the exact no-hedge result, and the near-zero
  variance of a frequent, cost-free delta hedge.
- **Test paths are sacred.** Seeds 10,000 to 19,999 are the test set. The simulators and the
  path loader refuse them unless called with `allow_test=True`
  (`--final-test-run` on the command line), which is for the single run after the
  `agents-frozen` tag.
- **Same paths for every strategy.** One seed makes one path, and scenarios that differ only
  in cost level share the same path file.
