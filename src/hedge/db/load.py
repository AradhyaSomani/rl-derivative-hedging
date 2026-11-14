"""Load the experiment record into PostgreSQL from the config and result files.

    export DATABASE_URL=postgresql://user:password@localhost:5432/hedging
    python -m hedge.cli db-load --split validation

The loader rebuilds the database from scratch each time (it drops and recreates the schema),
so it is safe to re-run after new results. Simulated paths stay in Parquet; only their file
names are stored.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psycopg
from psycopg.types.json import Jsonb

from hedge.agents.train_sb3 import RUNS_DIR
from hedge.config import Scenario
from hedge.env.hedging_env import default_obs_features
from hedge.eval.final import RESULTS_DIR, load_eval_config, tuned_band
from hedge.market import pathsets
from hedge.seeds import SEED_RANGES, SPLITS

SCHEMA = Path(__file__).with_name("schema.sql")
SCENARIO_DIR = Path("config/scenarios")
BASELINE_KINDS = ("no_hedge", "bs_delta", "no_trade_band")


def connect(url: str | None = None) -> psycopg.Connection:
    url = url or os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("set DATABASE_URL, e.g. postgresql://localhost:5432/hedging")
    return psycopg.connect(url)


def reset_schema(conn: psycopg.Connection) -> None:
    conn.execute("DROP SCHEMA public CASCADE; CREATE SCHEMA public;")
    conn.execute(SCHEMA.read_text())


def _none(value: Any) -> Any:
    """NaN and numpy scalars to plain Python values PostgreSQL accepts."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, np.generic):
        value = value.item()
        return None if isinstance(value, float) and np.isnan(value) else value
    return value


def _insert(conn: psycopg.Connection, table: str, key: str, **values: Any) -> int:
    cols = ", ".join(values)
    marks = ", ".join(["%s"] * len(values))
    row = conn.execute(
        f"INSERT INTO {table} ({cols}) VALUES ({marks}) RETURNING {key}",  # noqa: S608
        [_none(v) for v in values.values()],
    ).fetchone()
    return row[0]


def _copy(conn: psycopg.Connection, table: str, frame: pd.DataFrame) -> None:
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False, header=False, na_rep="\\N")
    buffer.seek(0)
    cols = ", ".join(frame.columns)
    with conn.cursor().copy(f"COPY {table} ({cols}) FROM STDIN WITH (FORMAT csv, NULL '\\N')") as c:
        c.write(buffer.read())


def load_setup(conn: psycopg.Connection) -> dict[str, dict[str, int]]:
    """Market models, option specs, scenarios and path sets. Returns the ids by name."""
    ids: dict[str, dict[str, int]] = {"market": {}, "option": {}, "scenario": {}, "path_set": {}}
    for file in sorted(SCENARIO_DIR.glob("*.yaml")):
        scenario = Scenario.from_yaml(file)
        market_name = (
            f"gbm_vol{round(scenario.market['sigma'] * 100)}"
            if scenario.model == "gbm"
            else "heston_base"
        )
        if market_name not in ids["market"]:
            params = {k: v for k, v in scenario.market.items() if k != "model"}
            ids["market"][market_name] = _insert(
                conn, "market_models", "market_model_id",
                name=market_name, model=scenario.model, params=Jsonb(params),
            )  # fmt: skip
        option_key = json.dumps(scenario.option, sort_keys=True)
        if option_key not in ids["option"]:
            ids["option"][option_key] = _insert(
                conn, "option_specs", "option_spec_id",
                kind=scenario.kind, is_short=True,
                moneyness=float(scenario.option["moneyness"]),
                maturity_days=int(scenario.option["maturity_days"]),
            )  # fmt: skip
        ids["scenario"][scenario.name] = _insert(
            conn, "scenarios", "scenario_id",
            name=scenario.name, market_model_id=ids["market"][market_name],
            option_spec_id=ids["option"][option_key], cost_bps=scenario.cost_bps,
            steps_per_day=int(scenario.grid["steps_per_day"]), config_hash=scenario.hash,
        )  # fmt: skip
        for split in SPLITS:
            file_path = pathsets.path_file(scenario, split)
            if not file_path.exists():
                continue
            seeds = SEED_RANGES[split]
            ids["path_set"][f"{scenario.name}/{split}"] = _insert(
                conn, "path_sets", "path_set_id",
                scenario_id=ids["scenario"][scenario.name], split=split, n_paths=len(seeds),
                seed_start=seeds.start, seed_end=seeds.stop - 1, parquet_path=str(file_path),
            )  # fmt: skip
    return ids


def load_strategies(conn: psycopg.Connection, cfg: dict[str, Any]) -> dict[str, int]:
    """Baselines and the learned strategies named in ``config/eval.yaml``."""
    ids: dict[str, int] = {}
    bands = {f.stem: tuned_band(Scenario.from_yaml(f)) for f in sorted(SCENARIO_DIR.glob("*.yaml"))}
    for kind in BASELINE_KINDS:
        hyper = {"band_by_scenario": bands} if kind == "no_trade_band" else None
        ids[kind] = _insert(
            conn, "strategies", "strategy_id",
            name=kind, kind=kind, hyperparams=Jsonb(hyper) if hyper else None,
        )  # fmt: skip
    for entry in cfg["learned"]:
        name = entry["strategy"]
        run_files = sorted((RUNS_DIR / name).glob("*/seed*/run.json"))
        if not run_files:
            continue
        info = json.loads(run_files[0].read_text())
        agent = info["agent"]
        learned_by_reward = agent["algo"] != "deep_hedging"
        scenario = Scenario.from_yaml(SCENARIO_DIR / f"{info['scenario']}.yaml")
        features = agent["env"].get("obs_features") or list(default_obs_features(scenario))
        hyper = {**agent["hyperparams"], "env": agent["env"], "net_arch": agent["net_arch"]}
        ids[name] = _insert(
            conn, "strategies", "strategy_id",
            name=name, kind=agent["algo"],
            reward=agent["reward"]["name"] if learned_by_reward else None,
            risk_lambda=agent["reward"].get("lam") if learned_by_reward else None,
            obs_features=[f for f in features if f != "variance"], hyperparams=Jsonb(hyper),
        )  # fmt: skip
    return ids


def load_runs(
    conn: psycopg.Connection, cfg: dict[str, Any], ids: dict[str, dict[str, int]],
    strategy_ids: dict[str, int],
) -> dict[tuple[str, str, int], int]:  # fmt: skip
    """Training runs and their validation curves."""
    run_ids: dict[tuple[str, str, int], int] = {}
    for entry in cfg["learned"]:
        name = entry["strategy"]
        for run_file in sorted((RUNS_DIR / name).glob("*/seed*/run.json")):
            info = json.loads(run_file.read_text())
            directory = run_file.parent
            checkpoint = next(iter(directory.glob("model.*")), None)
            run_id = _insert(
                conn, "training_runs", "run_id",
                strategy_id=strategy_ids[name], scenario_id=ids["scenario"][info["scenario"]],
                seed=info["seed"], total_steps=info["total_steps"],
                wall_time_s=info["wall_time_s"], status="finished",
                checkpoint=str(checkpoint) if checkpoint else None,
                tb_log_dir=str(directory / "tb") if (directory / "tb").exists() else None,
                code_commit=info["code_commit"],
            )  # fmt: skip
            run_ids[(name, info["scenario"], info["seed"])] = run_id
            curve = pd.read_csv(directory / "curve.csv")
            frame = pd.DataFrame(
                {
                    "run_id": run_id,
                    "step": curve["step"],
                    "mean_reward": curve.get("mean_reward"),
                    "val_mean_cost": curve["val_mean_cost"],
                    "val_cvar95": curve["val_cvar95"],
                }
            )
            _copy(conn, "training_curve", frame)
    return run_ids


def load_evaluations(
    conn: psycopg.Connection, split: str, ids: dict[str, dict[str, int]],
    strategy_ids: dict[str, int], run_ids: dict[tuple[str, str, int], int],
    results_dir: Path = RESULTS_DIR,
) -> int:  # fmt: skip
    """Evaluations with their metrics, per-path results and sample step detail."""
    metrics = pd.read_csv(results_dir / f"final_{split}.csv")
    costs = pd.read_parquet(results_dir / f"costs_{split}.parquet")
    steps = pd.read_parquet(results_dir / f"steps_{split}.parquet")
    costs_by = dict(tuple(costs.groupby(["scenario", "strategy", "seed"])))
    steps_by = dict(tuple(steps.groupby(["scenario", "strategy", "seed"])))
    metric_cols = [
        "n_paths", "mean_cost", "std_cost", "var95", "cvar95", "mean_txn_cost", "mean_turnover",
        "cvar_vs_delta", "cvar_vs_delta_lo", "cvar_vs_delta_hi",
    ]  # fmt: skip
    count = 0
    for _, row in metrics.iterrows():
        seed = None if pd.isna(row["seed"]) else int(row["seed"])
        eval_id = _insert(
            conn, "evaluations", "eval_id",
            strategy_id=strategy_ids[row["strategy"]],
            run_id=run_ids.get((row["strategy"], row["scenario"], seed)),
            path_set_id=ids["path_set"][f"{row['scenario']}/{split}"],
            stress_test=None, code_commit=row["code_commit"],
        )  # fmt: skip
        conn.execute(
            f"INSERT INTO eval_metrics (eval_id, {', '.join(metric_cols)}) "  # noqa: S608
            f"VALUES (%s, {', '.join(['%s'] * len(metric_cols))})",
            [eval_id, *[_none(row.get(c)) for c in metric_cols]],
        )
        key = (row["scenario"], row["strategy"], -1 if seed is None else seed)
        paths = costs_by[key].drop(columns=["scenario", "strategy", "seed"]).copy()
        paths = paths.rename(columns={"txn_cost": "txn_cost"})
        paths.insert(0, "eval_id", eval_id)
        _copy(conn, "eval_paths", paths)
        detail = steps_by[key].drop(columns=["scenario", "strategy", "seed"]).copy()
        detail.insert(0, "eval_id", eval_id)
        _copy(conn, "eval_steps", detail)
        count += 1
    return count


def load_stress(
    conn: psycopg.Connection, split: str, ids: dict[str, dict[str, int]],
    strategy_ids: dict[str, int], run_ids: dict[tuple[str, str, int], int],
    results_dir: Path = RESULTS_DIR,
) -> int:  # fmt: skip
    """Stress-test evaluations (metrics only). They point at the base scenario's path set."""
    file = results_dir / f"stress_{split}.csv"
    if not file.exists():
        return 0
    table = pd.read_csv(file)
    table = table[table["paths"] != "base"]
    count = 0
    for _, row in table.iterrows():
        seed = None if pd.isna(row["seed"]) else int(row["seed"])
        eval_id = _insert(
            conn, "evaluations", "eval_id",
            strategy_id=strategy_ids[row["strategy"]],
            run_id=run_ids.get((row["strategy"], row["scenario"], seed)),
            path_set_id=ids["path_set"][f"{row['scenario']}/{split}"],
            stress_test=row["stress_test"], code_commit="see stress csv",
        )  # fmt: skip
        conn.execute(
            "INSERT INTO eval_metrics (eval_id, n_paths, mean_cost, std_cost, var95, cvar95, "
            "mean_txn_cost, mean_turnover, cvar_vs_delta, cvar_vs_delta_lo, cvar_vs_delta_hi) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            [eval_id]
            + [
                _none(row.get(c))
                for c in (
                    "n_paths", "mean_cost", "std_cost", "var95", "cvar95", "mean_txn_cost",
                    "mean_turnover", "cvar_vs_delta", "cvar_vs_delta_lo", "cvar_vs_delta_hi",
                )
            ],
        )  # fmt: skip
        count += 1
    return count


def load_all(url: str | None = None, splits: tuple[str, ...] = ("validation",)) -> dict[str, int]:
    """Rebuild the whole database. Returns row counts for a quick check."""
    cfg = load_eval_config()
    with connect(url) as conn:
        reset_schema(conn)
        ids = load_setup(conn)
        strategy_ids = load_strategies(conn, cfg)
        run_ids = load_runs(conn, cfg, ids, strategy_ids)
        n_eval = n_stress = 0
        for split in splits:
            if (RESULTS_DIR / f"final_{split}.csv").exists():
                n_eval += load_evaluations(conn, split, ids, strategy_ids, run_ids)
                n_stress += load_stress(conn, split, ids, strategy_ids, run_ids)
        conn.commit()
        counts = {
            table: conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]  # noqa: S608
            for table in (
                "scenarios", "path_sets", "strategies", "training_runs", "training_curve",
                "evaluations", "eval_metrics", "eval_paths", "eval_steps",
            )
        }  # fmt: skip
    return counts
