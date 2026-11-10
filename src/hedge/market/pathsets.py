"""Train, validation and test path sets: generate, save as Parquet, load.

Files are named ``{model}_{paths_hash}_{split}.parquet``. The hash covers the market, maturity
and time grid, so scenarios that differ only in cost level share one file, and every strategy
is scored on identical paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from hedge.config import Scenario
from hedge.market.gbm import simulate_gbm
from hedge.market.heston import simulate_heston
from hedge.seeds import SPLITS, guard_test_seeds, split_seeds

DEFAULT_DIR = Path("data/paths")


@dataclass(frozen=True)
class PathSet:
    seeds: np.ndarray  # (n_paths,)
    prices: np.ndarray  # (n_paths, T + 1)
    variances: np.ndarray | None = None  # (n_paths, T + 1), Heston only

    def __len__(self) -> int:
        return self.seeds.size


def simulate(scenario: Scenario, seeds, *, allow_test: bool = False) -> PathSet:
    """Simulate the scenario's market for the given seeds."""
    seeds = np.asarray(seeds, dtype=np.int64)
    m = scenario.market
    if scenario.model == "gbm":
        prices = simulate_gbm(
            seeds, scenario.n_steps, scenario.dt, m["s0"], m["mu"], m["sigma"],
            allow_test=allow_test,
        )  # fmt: skip
        return PathSet(seeds, prices)
    # Sub-steps are per stored step, so the Euler step is the same at any rebalancing frequency.
    substeps = max(1, int(m["substeps_per_day"]) // int(scenario.grid["steps_per_day"]))
    prices, variances = simulate_heston(
        seeds, scenario.n_steps, scenario.dt, m["s0"], m["mu"], **scenario.heston_params,
        substeps=substeps, allow_test=allow_test,
    )  # fmt: skip
    return PathSet(seeds, prices, variances)


def path_file(scenario: Scenario, split: str, directory: Path = DEFAULT_DIR) -> Path:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    return Path(directory) / f"{scenario.model}_{scenario.paths_hash}_{split}.parquet"


def save(path_set: PathSet, file: Path) -> None:
    n_cols = path_set.prices.shape[1]
    data = {"seed": path_set.seeds}
    data.update({f"s_{t}": path_set.prices[:, t] for t in range(n_cols)})
    if path_set.variances is not None:
        data.update({f"v_{t}": path_set.variances[:, t] for t in range(n_cols)})
    file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data).to_parquet(file, index=False)


def load(file: Path, *, allow_test: bool = False) -> PathSet:
    df = pd.read_parquet(file)
    seeds = df["seed"].to_numpy()
    guard_test_seeds(seeds, allow_test)
    prices = df[[c for c in df.columns if c.startswith("s_")]].to_numpy()
    v_cols = [c for c in df.columns if c.startswith("v_")]
    return PathSet(seeds, prices, df[v_cols].to_numpy() if v_cols else None)


def build(
    scenario: Scenario,
    split: str,
    directory: Path = DEFAULT_DIR,
    *,
    n_paths: int | None = None,
    allow_test: bool = False,
    overwrite: bool = False,
) -> Path:
    """Generate one split's paths and save them. Returns the file; skips work if it exists."""
    file = path_file(scenario, split, directory)
    if file.exists() and not overwrite:
        return file
    seeds = split_seeds(split, n_paths)
    save(simulate(scenario, seeds, allow_test=allow_test), file)
    return file


def load_or_build(
    scenario: Scenario, split: str, directory: Path = DEFAULT_DIR, *, allow_test: bool = False
) -> PathSet:
    return load(build(scenario, split, directory, allow_test=allow_test), allow_test=allow_test)
