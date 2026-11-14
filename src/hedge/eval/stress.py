"""Stress tests: score frozen strategies on markets they were not trained on.

Each stress keeps the base scenario's premium, cost rate and the hedger's beliefs (the
volatility inside BS delta, the observation scaling) and changes only the price paths:

    vol_up / vol_down   GBM with a different volatility
    model_swap          Heston paths for agents trained on GBM
    jumps               rare jumps added to the base scenario's own paths

The exploratory section (``exploratory_stress`` in config/eval.yaml) repeats the Heston
stresses with the deep hedging paper's parameters; it was added after the test run.

The same seeds as the chosen split are used, so every strategy sees identical stressed paths.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hedge.config import Scenario
from hedge.eval.final import (
    RESULTS_DIR,
    load_eval_config,
    metric_rows,
    require_frozen,
    score_strategies,
    seed_table,
)
from hedge.market import pathsets
from hedge.market.pathsets import PathSet
from hedge.seeds import TestSeedError, split_seeds


def add_jumps(
    prices: np.ndarray, seeds: np.ndarray, dt: float, intensity: float, mean: float, std: float
) -> np.ndarray:
    """Multiply paths by a compound-Poisson jump factor (Merton-style), one stream per seed.

    ``intensity`` is jumps per year; each jump multiplies the price by exp(N(mean, std^2)).
    The jump stream is seeded separately from the path itself, so the base path is unchanged.
    """
    n_steps = prices.shape[1] - 1
    log_jump = np.zeros((prices.shape[0], n_steps))
    for i, seed in enumerate(seeds):
        rng = np.random.Generator(np.random.PCG64([int(seed), 777]))
        counts = rng.poisson(intensity * dt, size=n_steps)
        # The sum of k jumps is normal with mean k * mean and variance k * std^2.
        log_jump[i] = counts * mean + np.sqrt(counts) * std * rng.standard_normal(n_steps)
    factor = np.exp(
        np.concatenate([np.zeros((prices.shape[0], 1)), log_jump.cumsum(axis=1)], axis=1)
    )
    return prices * factor


def stressed_paths(
    name: str, spec: dict[str, Any], split: str, allow_test: bool = False
) -> tuple[Scenario, PathSet]:
    """The base scenario and the stressed path set for one stress test."""
    base = Scenario.from_yaml(spec["base"])
    seeds = split_seeds(split)
    if not any(key in spec for key in ("market", "paths_from", "jumps")):
        raise ValueError(f"stress test {name!r} needs market, paths_from or jumps")
    # The paths come from `paths_from` if given, else from the base scenario; `market` then
    # overrides some of that scenario's market parameters, and `jumps` adds jumps on top.
    source = Scenario.from_yaml(spec["paths_from"]) if "paths_from" in spec else base
    if "market" in spec:
        source = replace(source, market={**source.market, **spec["market"]})
    paths = pathsets.simulate(source, seeds, allow_test=allow_test)
    if "jumps" in spec:
        jumped = add_jumps(paths.prices, seeds, base.dt, **spec["jumps"])
        paths = PathSet(paths.seeds, jumped, paths.variances)
    if base.model == "gbm":
        paths = PathSet(paths.seeds, paths.prices, None)  # a GBM hedger does not see variance
    return base, paths


def run_stress_tests(
    split: str = "validation",
    allow_test: bool = False,
    results_dir: Path = RESULTS_DIR,
    exploratory: bool = False,
) -> pd.DataFrame:
    """Score every strategy under every stress; report how much each CVaR worsens.

    ``exploratory=True`` runs the ``exploratory_stress`` section of ``config/eval.yaml``
    instead of ``stress``: tests added after the test results were known (preregistration,
    section 9). Their results go to separate files, ``stress_exploratory_{split}*.csv``, so
    the preregistered stress results are never overwritten.
    """
    if split == "test":
        if not allow_test:
            raise TestSeedError("the test split needs allow_test=True")
        require_frozen()
    cfg = load_eval_config()
    rows: list[dict[str, Any]] = []
    section = "exploratory_stress" if exploratory else "stress"
    prefix = "stress_exploratory" if exploratory else "stress"
    for name, spec in cfg[section].items():
        base, paths = stressed_paths(name, spec, split, allow_test)
        clean = pathsets.load_or_build(base, split, allow_test=allow_test)
        for label, path_set in (("base", clean), (name, paths)):
            results = score_strategies(base, path_set, cfg)
            for row in metric_rows(base.name, results, cfg):
                rows.append({**row, "stress_test": name, "paths": label, "split": split})
        print(f"stress {name} done", flush=True)

    table = pd.DataFrame(rows)
    summary = seed_table(table, ["stress_test", "paths", "scenario", "strategy"])
    base_cvar = summary[summary["paths"] == "base"].set_index(["stress_test", "strategy"])["cvar95"]
    stressed = summary[summary["paths"] != "base"].copy()
    keys = list(zip(stressed["stress_test"], stressed["strategy"], strict=True))
    stressed["cvar95_base"] = base_cvar.reindex(keys).to_numpy()
    stressed["cvar95_worsening"] = stressed["cvar95"] - stressed["cvar95_base"]
    stressed["cvar95_ratio"] = stressed["cvar95"] / stressed["cvar95_base"]
    results_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(results_dir / f"{prefix}_{split}.csv", index=False)
    stressed.to_csv(results_dir / f"{prefix}_{split}_table.csv", index=False)
    return stressed
