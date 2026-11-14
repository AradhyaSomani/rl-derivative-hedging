"""Small Optuna search over PPO/SAC hyperparameters, scored by validation CVaR 95%.

    name: tune_ppo_offset
    agent: config/agents/ppo_offset.yaml
    scenario: config/scenarios/gbm_10bps.yaml
    seed: 0
    n_trials: 30            # the pre-registered cap per algorithm
    reward: {name: quadratic, lam: 1.0}
    search:
      learning_rate: {low: 1.0e-5, high: 1.0e-3, log: true}
      ent_coef: {low: 1.0e-8, high: 1.0e-2, log: true}
      gamma: {low: 0.9, high: 1.0}
      lr_schedule: [constant, linear]

Trials run ``jobs`` at a time. Results go to ``data/results/{name}.csv`` and the best
hyperparameters to ``data/results/{name}_best.json``.
"""

from __future__ import annotations

import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from hedge.agents.train_sb3 import AgentConfig, train
from hedge.config import Scenario
from hedge.market import pathsets

RESULTS_DIR = Path("data/results")


def _suggest(trial, search: dict[str, Any]) -> dict[str, Any]:
    params = {}
    for name, spec in search.items():
        if isinstance(spec, list):
            params[name] = trial.suggest_categorical(name, spec)
        else:
            params[name] = trial.suggest_float(
                name, float(spec["low"]), float(spec["high"]), log=bool(spec.get("log", False))
            )
    return params


def _run(args: tuple[AgentConfig, str, int]) -> dict[str, Any]:
    agent, scenario_file, seed = args
    return train(agent, Scenario.from_yaml(scenario_file), seed)


def tune(config_file: str | Path, jobs: int = 1) -> pd.DataFrame:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    with open(config_file) as f:
        cfg = yaml.safe_load(f)
    base = AgentConfig.from_yaml(cfg["agent"], total_steps=cfg.get("total_steps"))
    if "reward" in cfg:
        base = replace(base, reward=cfg["reward"])
    scenario = Scenario.from_yaml(cfg["scenario"])
    pathsets.build(scenario, "train")
    pathsets.build(scenario, "validation")
    seed = int(cfg.get("seed", 0))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    rows: list[dict[str, Any]] = []
    out = RESULTS_DIR / f"{cfg['name']}.csv"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(jobs, mp_context=mp.get_context("spawn")) as pool:
        while len(rows) < cfg["n_trials"]:
            batch = [study.ask() for _ in range(min(jobs, cfg["n_trials"] - len(rows)))]
            jobs_args = []
            for trial in batch:
                params = _suggest(trial, cfg["search"])
                agent = replace(
                    base,
                    name=f"{base.name}_trial{trial.number:02d}",
                    hyperparams={**base.hyperparams, **params},
                )
                jobs_args.append((agent, cfg["scenario"], seed))
            for trial, metrics in zip(batch, pool.map(_run, jobs_args), strict=True):
                study.tell(trial, metrics["cvar95"])
                rows.append({"trial": trial.number, **trial.params, **metrics})
                print(
                    f"trial {trial.number}: cvar95 {metrics['cvar95']:.4f} "
                    f"mean cost {metrics['mean_cost']:.4f} {trial.params}",
                    flush=True,
                )
            pd.DataFrame(rows).to_csv(out, index=False)
    best = {"trial": study.best_trial.number, "cvar95": study.best_value, **study.best_params}
    (RESULTS_DIR / f"{cfg['name']}_best.json").write_text(json.dumps(best, indent=2) + "\n")
    return pd.DataFrame(rows)
