"""Score every strategy on one split of every scenario, with bootstrap confidence intervals.

On validation paths this can be run as often as needed. On test paths it is the single final
run: it needs ``allow_test=True`` and refuses unless the git tag ``agents-frozen`` exists.

Outputs in ``data/results/``:

    final_{split}.csv          one row per scenario, strategy and training seed
    final_{split}_table.csv    median over seeds, CVaR range, CI of the CVaR gap to BS delta
    costs_{split}.parquet      per-path premium, payoff, hedge gain, cost and P&L
    steps_{split}.parquet      step-by-step detail for the first few paths (the demo's race)
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from hedge.agents.train_sb3 import RUNS_DIR, AgentConfig, _code_commit, env_kwargs, load_model
from hedge.baselines.no_hedge import no_hedge_holdings
from hedge.baselines.no_trade_band import no_trade_band_holdings
from hedge.baselines.run import deltas, score, tune_band
from hedge.config import Scenario
from hedge.env.hedging_env import HedgingEnv
from hedge.env.rewards import option_marks, step_pnls
from hedge.eval.evaluate import evaluate_policy, sb3_predict
from hedge.eval.metrics import cvar, summarise
from hedge.market import pathsets
from hedge.market.pathsets import PathSet
from hedge.pnl import HedgeResult
from hedge.seeds import TestSeedError

RESULTS_DIR = Path("data/results")
EVAL_CONFIG = Path("config/eval.yaml")
BASELINES = ("no_hedge", "bs_delta", "no_trade_band")
N_SAMPLE_PATHS = 25  # paths per evaluation that keep step-by-step detail


class NotFrozenError(RuntimeError):
    """Raised when the test run is asked for before the agents are frozen."""


def load_eval_config(path: Path = EVAL_CONFIG) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def require_frozen() -> None:
    try:
        out = subprocess.run(
            ["git", "tag", "-l", "agents-frozen"], capture_output=True, text=True, check=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as err:
        raise NotFrozenError("cannot check for the agents-frozen tag: git failed") from err
    if not out.stdout.strip():
        raise NotFrozenError(
            "The test paths are opened once, after the setup is frozen. "
            "Commit the chosen configs and run `git tag agents-frozen` first."
        )


def agent_from_run(directory: Path) -> AgentConfig:
    raw = json.loads((directory / "run.json").read_text())["agent"]
    raw["net_arch"] = tuple(raw["net_arch"])
    return AgentConfig(**raw)


_BAND_CACHE: dict[str, dict[str, Any]] = {}


def tuned_band(scenario: Scenario) -> dict[str, Any]:
    """The band setting for a scenario, always tuned on validation paths (cached per run)."""
    if scenario.hash not in _BAND_CACHE:
        with open("config/agents/no_trade_band.yaml") as f:
            band_cfg = yaml.safe_load(f)
        validation = pathsets.load_or_build(scenario, "validation")
        _BAND_CACHE[scenario.hash] = tune_band(
            scenario, validation, band_cfg["widths"], band_cfg["rebalance_to"]
        )[0]
    return _BAND_CACHE[scenario.hash]


def bootstrap_cvar_gap(
    cost: np.ndarray, reference: np.ndarray, n_resamples: int, seed: int, level: float = 0.95
) -> tuple[float, float, float]:
    """CVaR 95% of ``cost`` minus that of ``reference``, with a paired bootstrap interval.

    Both arrays are costs on the same paths, so each resample draws the same paths for both.
    Returns (difference, lower, upper). Negative means lower tail risk than the reference.
    """
    rng = np.random.default_rng(seed)
    n = cost.size
    k = max(1, int(np.ceil(n * 0.05 - 1e-9)))
    gaps = np.empty(n_resamples)
    for start in range(0, n_resamples, 100):
        stop = min(start + 100, n_resamples)
        idx = rng.integers(n, size=(stop - start, n))
        tail = [
            np.partition(x[idx], n - k, axis=1)[:, n - k :].mean(axis=1) for x in (cost, reference)
        ]
        gaps[start:stop] = tail[0] - tail[1]
    lo, hi = np.quantile(gaps, [(1 - level) / 2, 1 - (1 - level) / 2])
    return cvar(cost) - cvar(reference), float(lo), float(hi)


def score_strategies(
    base: Scenario,
    path_set: PathSet,
    cfg: dict[str, Any],
    runs_dir: Path = RUNS_DIR,
    trained_on: str | None = None,
) -> dict[tuple[str, int | None], HedgeResult]:
    """Baselines and every available learned strategy on one path set.

    Each result carries the holdings that produced it in ``result.holdings``.

    ``base`` supplies the premium, strike, cost rate and the volatility inside BS delta.
    ``trained_on`` names the scenario whose trained agents to load (default: ``base``).
    """
    results: dict[tuple[str, int | None], HedgeResult] = {}
    band = tuned_band(base)
    target = deltas(base, path_set)
    baseline_holdings = {
        "no_hedge": no_hedge_holdings(path_set.prices),
        "bs_delta": target,
        "no_trade_band": no_trade_band_holdings(target, band["width"], band["rebalance_to"]),
    }
    for name, holdings in baseline_holdings.items():
        results[(name, None)] = _with_holdings(score(base, path_set, holdings), holdings)
    for entry in cfg["learned"]:
        for seed in cfg["seeds"]:
            directory = runs_dir / entry["strategy"] / (trained_on or base.name) / f"seed{seed}"
            if not (directory / "run.json").exists():
                continue
            agent = agent_from_run(directory)
            env = HedgingEnv(base, path_set, mode="eval", **env_kwargs(agent))
            model = load_model(agent, directory)
            result, holdings = evaluate_policy(env, sb3_predict(model))
            results[(entry["strategy"], seed)] = _with_holdings(result, holdings)
    return results


def _with_holdings(result: HedgeResult, holdings: np.ndarray) -> HedgeResult:
    object.__setattr__(result, "holdings", holdings)  # extra attribute on a frozen dataclass
    return result


def sample_steps(
    base: Scenario, path_set: PathSet, result: HedgeResult, n_paths: int = N_SAMPLE_PATHS
) -> pd.DataFrame:
    """Step-by-step spot, time to expiry, BS delta, holding, cost and running P&L."""
    n = min(n_paths, len(path_set))
    sample = PathSet(
        path_set.seeds[:n],
        path_set.prices[:n],
        None if path_set.variances is None else path_set.variances[:n],
    )
    holdings = result.holdings[:n]
    gross, cost = step_pnls(
        sample.prices, holdings, option_marks(base, sample), base.premium(), base.cost_rate
    )
    target = deltas(base, sample)
    steps = np.arange(base.n_steps)
    frame = pd.DataFrame(
        {
            "path_id": np.repeat(np.arange(n), base.n_steps),
            "step": np.tile(steps, n),
            "spot": sample.prices[:, :-1].ravel(),
            "tau": np.tile(base.dt * (base.n_steps - steps), n),
            "bs_delta": target.ravel(),
            "holding": holdings.ravel(),
            "step_cost": cost.ravel(),
            "cum_pnl": np.cumsum(gross - cost, axis=1).ravel(),
        }
    )
    return frame


def metric_rows(
    scenario_name: str, results: dict[tuple[str, int | None], HedgeResult], cfg: dict[str, Any]
) -> list[dict[str, Any]]:
    boot = cfg["bootstrap"]
    reference = results[("bs_delta", None)].cost
    rows = []
    for (strategy, seed), result in results.items():
        row = {"scenario": scenario_name, "strategy": strategy, "seed": seed, **summarise(result)}
        if strategy != "bs_delta":
            gap, lo, hi = bootstrap_cvar_gap(
                result.cost, reference, boot["n_resamples"], boot["seed"], boot["level"]
            )
            row.update(cvar_vs_delta=gap, cvar_vs_delta_lo=lo, cvar_vs_delta_hi=hi)
        rows.append(row)
    return rows


def seed_table(rows: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Median over training seeds. The CI reported is that of the seed with the median CVaR."""
    out = []
    for key, group in rows.groupby(keys, sort=False, dropna=False):
        group = group.sort_values("cvar95")
        median_row = group.iloc[(len(group) - 1) // 2]
        record = dict(zip(keys, key if isinstance(key, tuple) else (key,), strict=True))
        record["n_seeds"] = int(group["seed"].notna().sum())
        for col in ("mean_cost", "std_cost", "var95", "cvar95", "mean_txn_cost", "mean_turnover"):
            record[col] = float(group[col].median())
        record["cvar95_min"] = float(group["cvar95"].min())
        record["cvar95_max"] = float(group["cvar95"].max())
        for col in ("cvar_vs_delta", "cvar_vs_delta_lo", "cvar_vs_delta_hi"):
            record[col] = median_row.get(col, np.nan)
        out.append(record)
    return pd.DataFrame(out)


def final_evaluation(
    scenario_files: list[str],
    split: str = "validation",
    allow_test: bool = False,
    runs_dir: Path = RUNS_DIR,
    results_dir: Path = RESULTS_DIR,
) -> pd.DataFrame:
    """Score everything on one split. ``split="test"`` is the single final run."""
    if split == "test":
        if not allow_test:
            raise TestSeedError("the test split needs allow_test=True")
        require_frozen()
    cfg = load_eval_config()
    rows: list[dict[str, Any]] = []
    costs: list[pd.DataFrame] = []
    steps: list[pd.DataFrame] = []
    for scenario_file in scenario_files:
        scenario = Scenario.from_yaml(scenario_file)
        path_set = pathsets.load_or_build(scenario, split, allow_test=allow_test)
        results = score_strategies(scenario, path_set, cfg, runs_dir)
        for row in metric_rows(scenario.name, results, cfg):
            rows.append(
                {
                    **row,
                    "split": split,
                    "config_hash": scenario.hash,
                    "paths_hash": scenario.paths_hash,
                    "code_commit": _code_commit(),
                }
            )
        for (strategy, seed), result in results.items():
            frame = result.to_frame()
            frame.insert(0, "path_id", np.arange(len(frame)))
            frame.insert(0, "seed", -1 if seed is None else seed)
            frame.insert(0, "strategy", strategy)
            frame.insert(0, "scenario", scenario.name)
            costs.append(frame)
            detail = sample_steps(scenario, path_set, result)
            detail.insert(0, "seed", -1 if seed is None else seed)
            detail.insert(0, "strategy", strategy)
            detail.insert(0, "scenario", scenario.name)
            steps.append(detail)
        print(f"scored {scenario.name} ({len(results)} strategy runs)", flush=True)

    results_dir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(results_dir / f"final_{split}.csv", index=False)
    summary = seed_table(table, ["scenario", "strategy"])
    summary.to_csv(results_dir / f"final_{split}_table.csv", index=False)
    pd.concat(costs, ignore_index=True).to_parquet(
        results_dir / f"costs_{split}.parquet", index=False
    )
    pd.concat(steps, ignore_index=True).to_parquet(
        results_dir / f"steps_{split}.parquet", index=False
    )
    return summary
