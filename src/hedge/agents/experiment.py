"""Run a grid of training runs from an experiment YAML, several at a time.

    name: reward_grid
    agent: config/agents/ppo.yaml
    scenarios: [config/scenarios/gbm_10bps.yaml]
    seeds: [0]
    rewards: [quadratic, mean_variance, terminal_cvar]   # optional; default: the agent's own
    lams: [0.1, 0.5, 1, 2]                               # optional
    total_steps: 1000000                                 # optional override

Runs that already have ``eval_validation.json`` are skipped, so an interrupted experiment can
be restarted. The summary goes to ``data/results/{name}_validation.csv`` together with the
baselines on the same validation paths.
"""

from __future__ import annotations

import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from hedge.agents.train_sb3 import RUNS_DIR, AgentConfig, run_dir, train
from hedge.baselines.run import run_baselines, tune_band
from hedge.config import Scenario
from hedge.eval.metrics import summarise
from hedge.market import pathsets

RESULTS_DIR = Path("data/results")


def expand(experiment: dict[str, Any]) -> list[tuple[AgentConfig, str, int]]:
    """All (agent config, scenario file, seed) runs of an experiment."""
    base = AgentConfig.from_yaml(experiment["agent"], total_steps=experiment.get("total_steps"))
    rewards = experiment.get("rewards", [base.reward["name"]])
    lams = experiment.get("lams", [base.reward.get("lam", 1.0)])
    runs = []
    for reward in rewards:
        for lam in lams:
            spec = {**base.reward, "name": reward, "lam": float(lam)}
            if reward == "terminal_cvar":
                spec.setdefault("threshold", "auto")
            agent = replace(base, reward=spec)
            for scenario_file in experiment["scenarios"]:
                for seed in experiment["seeds"]:
                    runs.append((agent, scenario_file, int(seed)))
    return runs


def _one(args: tuple[AgentConfig, str, int]) -> dict[str, Any]:
    agent, scenario_file, seed = args
    scenario = Scenario.from_yaml(scenario_file)
    done = run_dir(agent, scenario, seed) / "eval_validation.json"
    if done.exists():
        return json.loads(done.read_text())
    return train(agent, scenario, seed)


def baseline_rows(scenario_files: list[str]) -> list[dict[str, Any]]:
    with open("config/agents/no_trade_band.yaml") as f:
        band_cfg = yaml.safe_load(f)
    rows = []
    for scenario_file in dict.fromkeys(scenario_files):
        scenario = Scenario.from_yaml(scenario_file)
        validation = pathsets.load_or_build(scenario, "validation")
        band, _ = tune_band(scenario, validation, band_cfg["widths"], band_cfg["rebalance_to"])
        for strategy, result in run_baselines(scenario, validation, band).items():
            rows.append(
                {
                    "strategy": strategy,
                    "scenario": scenario.name,
                    "seed": None,
                    **summarise(result),
                    "split": "validation",
                }
            )
    return rows


def run_experiment(config_file: str | Path, jobs: int = 1) -> pd.DataFrame:
    with open(config_file) as f:
        experiment = yaml.safe_load(f)
    runs = expand(experiment)
    for scenario_file in dict.fromkeys(r[1] for r in runs):  # build paths before forking out
        scenario = Scenario.from_yaml(scenario_file)
        pathsets.build(scenario, "train")
        pathsets.build(scenario, "validation")

    rows = []
    if jobs <= 1:
        for run in runs:
            rows.append(_one(run))
            print(f"done {rows[-1]['strategy']} {rows[-1]['scenario']} seed {rows[-1]['seed']}")
    else:
        with ProcessPoolExecutor(jobs, mp_context=mp.get_context("spawn")) as pool:
            for future in as_completed([pool.submit(_one, run) for run in runs]):
                rows.append(future.result())
                r = rows[-1]
                print(f"done {r['strategy']} {r['scenario']} seed {r['seed']}", flush=True)

    table = pd.DataFrame(rows + baseline_rows([r[1] for r in runs]))
    table = table.sort_values(["scenario", "strategy", "seed"]).reset_index(drop=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULTS_DIR / f"{experiment['name']}_validation.csv", index=False)
    return table


__all__ = ["RUNS_DIR", "baseline_rows", "expand", "run_experiment"]
