"""In-memory computations for the API's POST endpoints and the demo's "Try it" page.

Nothing here reads test paths or writes anything: fresh paths come from the dev seed range.
"""

from __future__ import annotations

import json
import os
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from hedge.agents.train_sb3 import RUNS_DIR, env_kwargs, load_model
from hedge.baselines.no_trade_band import no_trade_band_holdings
from hedge.baselines.run import deltas, score
from hedge.config import Scenario
from hedge.env.hedging_env import HedgingEnv
from hedge.eval.evaluate import evaluate_policy, sb3_predict
from hedge.eval.final import agent_from_run, load_eval_config, tuned_band
from hedge.eval.metrics import summarise
from hedge.market import black_scholes as bs
from hedge.market import pathsets
from hedge.market.pathsets import PathSet
from hedge.seeds import SEED_RANGES

SCENARIO_DIR = Path("config/scenarios")
TRAINED_COSTS = (0, 5, 10, 20)
MAX_PATHS = 1_000


def nearest_scenario(market: str, cost_bps: float) -> Scenario:
    """The trained scenario closest to the requested cost level."""
    if market not in ("gbm", "heston"):
        raise ValueError("market must be 'gbm' or 'heston'")
    nearest = min(TRAINED_COSTS, key=lambda c: abs(c - cost_bps))
    return Scenario.from_yaml(SCENARIO_DIR / f"{market}_{nearest}bps.yaml")


def custom_scenario(market: str, cost_bps: float, volatility: float | None = None) -> Scenario:
    """A scenario with the requested cost and volatility, built on the nearest trained one."""
    base = nearest_scenario(market, cost_bps)
    params = dict(base.market)
    if volatility is not None:
        if market == "gbm":
            params["sigma"] = float(volatility)
        else:
            params["v0"] = params["theta"] = float(volatility) ** 2
    return replace(base, name=f"{market}_custom", market=params, cost_bps=float(cost_bps))


def _runs_dir(runs_dir: Path | None) -> Path:
    """An explicit folder, else HEDGE_RUNS_DIR (set by the demo), else ``data/runs``."""
    return Path(runs_dir or os.environ.get("HEDGE_RUNS_DIR") or RUNS_DIR)


def median_seed(strategy: str, scenario_name: str, runs_dir: Path | None = None) -> int | None:
    """The training seed whose validation CVaR is the median of the available seeds."""
    runs_dir = _runs_dir(runs_dir)
    scores = []
    for file in sorted((runs_dir / strategy / scenario_name).glob("seed*/eval_validation.json")):
        metrics = json.loads(file.read_text())
        scores.append((metrics["cvar95"], metrics["seed"]))
    if not scores:
        return None
    scores.sort()
    return scores[(len(scores) - 1) // 2][1]


@lru_cache(maxsize=64)
def _load(strategy: str, scenario_name: str, seed: int, runs_dir: str):
    directory = Path(runs_dir) / strategy / scenario_name / f"seed{seed}"
    agent = agent_from_run(directory)
    return agent, load_model(agent, directory)


def frozen_agents(scenario_name: str, runs_dir: Path | None = None) -> dict[str, tuple[Any, Any]]:
    """label -> (agent config, model) for every learned strategy trained on this scenario."""
    runs_dir = _runs_dir(runs_dir)
    out = {}
    for entry in load_eval_config()["learned"]:
        seed = median_seed(entry["strategy"], scenario_name, runs_dir)
        if seed is not None:
            out[entry["label"]] = _load(entry["strategy"], scenario_name, seed, str(runs_dir))
    return out


def simulate_comparison(
    market: str = "gbm",
    cost_bps: float = 10.0,
    volatility: float | None = None,
    n_paths: int = 1_000,
    first_seed: int = 0,
    include_costs: bool = True,
) -> dict[str, Any]:
    """Hedge ``n_paths`` fresh paths with every strategy and compare them."""
    n_paths = int(min(max(n_paths, 1), MAX_PATHS))
    dev = SEED_RANGES["dev"]
    first_seed = int(first_seed) % (len(dev) - n_paths + 1)
    scenario = custom_scenario(market, cost_bps, volatility)
    trained = nearest_scenario(market, cost_bps)
    paths = pathsets.simulate(scenario, np.arange(first_seed, first_seed + n_paths))

    target = deltas(scenario, paths)
    band = tuned_band(trained)
    results = {
        "No hedge": score(scenario, paths, np.zeros_like(target)),
        "BS delta": score(scenario, paths, target),
        "No-trade band": score(
            scenario, paths, no_trade_band_holdings(target, band["width"], band["rebalance_to"])
        ),
    }
    for label, (agent, model) in frozen_agents(trained.name).items():
        env = HedgingEnv(scenario, paths, mode="eval", **env_kwargs(agent))
        results[label] = evaluate_policy(env, sb3_predict(model))[0]

    strategies = {}
    for label, result in results.items():
        strategies[label] = summarise(result)
        if include_costs:
            strategies[label]["costs"] = np.round(result.cost, 5).tolist()
    return {
        "market": market,
        "cost_bps": float(cost_bps),
        "volatility": float(volatility) if volatility is not None else None,
        "n_paths": n_paths,
        "first_seed": first_seed,
        "agents_trained_on": trained.name,
        "premium": scenario.premium(),
        "strategies": strategies,
    }


def hedge_decision(
    spot: float,
    strike: float,
    days_left: int,
    volatility: float,
    current_holding: float,
    cost_bps: float,
) -> dict[str, Any]:
    """What the frozen agents and BS delta would hold now, for one market state.

    Uses the agents trained on GBM at the nearest cost level. The option is a call with 30
    trading days of original life, as in training.
    """
    trained = nearest_scenario("gbm", cost_bps)
    n_steps = trained.n_steps
    if not 1 <= days_left <= n_steps:
        raise ValueError(f"days_left must be between 1 and {n_steps}")
    if min(spot, strike, volatility) <= 0:
        raise ValueError("spot, strike and volatility must be positive")
    scenario = replace(
        trained,
        name="gbm_live",
        market={**trained.market, "s0": float(strike), "sigma": float(volatility)},
        cost_bps=float(cost_bps),
    )
    t = n_steps - int(days_left)
    state = PathSet(np.array([0]), np.full((1, n_steps + 1), float(spot)), None)
    held = np.array([float(np.clip(current_holding, 0.0, 1.0))])
    delta = float(bs.delta(spot, strike, days_left * scenario.dt, volatility))
    agents = {}
    for label, (agent, model) in frozen_agents(trained.name).items():
        env = HedgingEnv(scenario, state, mode="eval", **env_kwargs(agent))
        action = sb3_predict(model)(env.batch_observation(t, held))
        agents[label] = float(env.batch_holdings(t, action)[0])
    return {"bs_delta": delta, "agents": agents, "agents_trained_on": trained.name}
