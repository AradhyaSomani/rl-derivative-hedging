"""Command line entry points: ``python -m hedge.cli <command>``. See the Makefile."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from hedge.baselines.run import deltas, run_baselines, tune_band
from hedge.config import Scenario
from hedge.eval.metrics import summarise
from hedge.market import pathsets
from hedge.pnl import cash_ledger

RESULTS_DIR = Path("data/results")
BAND_CONFIG = Path("config/agents/no_trade_band.yaml")


def _code_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def cmd_paths(args: argparse.Namespace) -> None:
    scenario = Scenario.from_yaml(args.scenario)
    for split in args.splits:
        file = pathsets.build(
            scenario, split, args.dir, allow_test=args.final_test_run, overwrite=args.overwrite
        )
        print(f"{scenario.name} {split}: {file}")


def cmd_baselines(args: argparse.Namespace) -> None:
    """Tune the band on validation paths, then score all baselines on ``--split``."""
    with open(BAND_CONFIG) as f:
        band_cfg = yaml.safe_load(f)
    rows, grids, chosen = [], [], {}
    for scenario_file in args.scenarios:
        scenario = Scenario.from_yaml(scenario_file)
        validation = pathsets.load_or_build(scenario, "validation", args.dir)
        band, grid = tune_band(scenario, validation, band_cfg["widths"], band_cfg["rebalance_to"])
        chosen[scenario.name] = band
        grids += [{"scenario": scenario.name, **row} for row in grid]

        if args.split == "validation":
            path_set = validation
        else:
            path_set = pathsets.load_or_build(
                scenario, args.split, args.dir, allow_test=args.final_test_run
            )
        for strategy, result in run_baselines(scenario, path_set, band).items():
            rows.append(
                {
                    "scenario": scenario.name,
                    "strategy": strategy,
                    **summarise(result),
                    "split": args.split,
                    "config_hash": scenario.hash,
                    "paths_hash": scenario.paths_hash,
                    "code_commit": _code_commit(),
                }
            )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS_DIR / f"baselines_{args.split}.csv", index=False)
    pd.DataFrame(grids).to_csv(RESULTS_DIR / "band_tuning_validation.csv", index=False)
    (RESULTS_DIR / "band_chosen.json").write_text(json.dumps(chosen, indent=2) + "\n")
    cols = [
        "scenario",
        "strategy",
        "mean_cost",
        "std_cost",
        "cvar95",
        "mean_txn_cost",
        "mean_turnover",
    ]
    print(table[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nChosen bands:", json.dumps(chosen))


def cmd_walkthrough(args: argparse.Namespace) -> None:
    """Print the cash account of one delta-hedged path, to check a spreadsheet against."""
    scenario = Scenario.from_yaml(args.scenario)
    path_set = pathsets.simulate(scenario, [args.seed])
    holdings = deltas(scenario, path_set)[0]
    ledger = cash_ledger(
        path_set.prices[0], holdings, scenario.strike, scenario.premium(),
        scenario.cost_rate, scenario.kind,
    )  # fmt: skip
    print(f"{scenario.name}, seed {args.seed}: premium received {scenario.premium():.4f}")
    print(ledger.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nHedging P&L = final cash = {ledger['cash'].iloc[-1]:.4f}")
    if args.csv:
        ledger.to_csv(args.csv, index=False)


def cmd_sanity(args: argparse.Namespace) -> None:
    """Phase 2 gate: scripted delta agent vs the delta baseline, and a random agent, in the env."""
    from hedge.agents.scripted import delta_policy, random_policy, rollout
    from hedge.env.hedging_env import HedgingEnv

    rows = []
    for scenario_file in args.scenarios:
        scenario = Scenario.from_yaml(scenario_file)
        path_set = pathsets.load_or_build(scenario, "validation", args.dir)
        baselines = run_baselines(scenario, path_set, {"width": 0.0, "rebalance_to": "delta"})
        env = HedgingEnv(scenario, path_set, mode="eval")
        scripted = rollout(env, delta_policy, args.n_paths)
        random = rollout(env, random_policy, args.n_paths)
        n = len(scripted.env_pnl)
        gap = np.abs(scripted.result.pnl - baselines["bs_delta"].pnl[:n]).max()
        env_gap = np.abs(scripted.env_pnl - scripted.result.pnl).max()
        print(
            f"{scenario.name}: scripted delta vs baseline, max |P&L difference| over {n} paths "
            f"= {gap:.2e}; env step sum vs engine = {env_gap:.2e}"
        )
        for name, result in [("scripted_delta", scripted.result), ("random", random.result)]:
            rows.append({"scenario": scenario.name, "strategy": name, **summarise(result)})
        rows.append(
            {"scenario": scenario.name, "strategy": "no_hedge", **summarise(baselines["no_hedge"])}
        )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS_DIR / "sanity_validation.csv", index=False)
    cols = ["scenario", "strategy", "mean_cost", "std_cost", "cvar95", "mean_turnover"]
    print(table[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def cmd_train(args: argparse.Namespace) -> None:
    from hedge.agents.train_sb3 import AgentConfig, train

    agent = AgentConfig.from_yaml(args.agent, total_steps=args.total_steps)
    metrics = train(agent, Scenario.from_yaml(args.scenario), args.seed, verbose=args.verbose)
    print(json.dumps(metrics, indent=2))


def cmd_experiment(args: argparse.Namespace) -> None:
    from hedge.agents.experiment import run_experiment

    table = run_experiment(args.config, args.jobs)
    cols = ["scenario", "strategy", "seed", "mean_cost", "std_cost", "cvar95", "mean_turnover"]
    print(table[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def cmd_tune(args: argparse.Namespace) -> None:
    from hedge.agents.tune import tune

    table = tune(args.config, args.jobs)
    print(table.sort_values("cvar95").head(10).to_string(index=False))


def cmd_eval(args: argparse.Namespace) -> None:
    """Score every strategy on one split. With ``--split test`` this is the single final run."""
    from hedge.eval.final import final_evaluation

    summary = final_evaluation(args.scenarios, args.split, allow_test=args.final_test_run)
    cols = ["scenario", "strategy", "n_seeds", "mean_cost", "std_cost", "cvar95"]
    cols += ["cvar_vs_delta", "cvar_vs_delta_lo", "cvar_vs_delta_hi"]
    print(summary[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def cmd_stress(args: argparse.Namespace) -> None:
    from hedge.eval.stress import run_stress_tests

    table = run_stress_tests(args.split, allow_test=args.final_test_run)
    cols = ["stress_test", "strategy", "n_seeds", "cvar95_base", "cvar95", "cvar95_worsening"]
    print(table[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def cmd_db_load(args: argparse.Namespace) -> None:
    from hedge.db.load import load_all

    counts = load_all(args.url, tuple(args.splits))
    print(json.dumps(counts, indent=2))


def cmd_report(args: argparse.Namespace) -> None:
    from hedge.eval.report import report_all

    for file in report_all():
        print(file)


def cmd_export_sample(args: argparse.Namespace) -> None:
    from hedge.results import export_sample

    files = export_sample()
    size = sum(f.stat().st_size for f in files) / 1e6
    print(f"wrote {len(files)} files to app/sample ({size:.1f} MB)")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="hedge")
    sub = parser.add_subparsers(dest="command", required=True)

    final = {
        "action": "store_true",
        "help": "Allow test seeds. Only for the single test run after the agents-frozen tag.",
    }

    p = sub.add_parser("paths", help="generate path sets for one scenario")
    p.add_argument("--scenario", required=True)
    p.add_argument("--splits", nargs="+", default=["train", "validation"])
    p.add_argument("--dir", type=Path, default=pathsets.DEFAULT_DIR)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--final-test-run", **final)
    p.set_defaults(func=cmd_paths)

    p = sub.add_parser("baselines", help="tune the band and score the baselines")
    p.add_argument("--scenarios", nargs="+", required=True)
    p.add_argument("--split", default="validation", choices=["validation", "test"])
    p.add_argument("--dir", type=Path, default=pathsets.DEFAULT_DIR)
    p.add_argument("--final-test-run", **final)
    p.set_defaults(func=cmd_baselines)

    p = sub.add_parser("walkthrough", help="step-by-step cash account for one delta-hedged path")
    p.add_argument("--scenario", required=True)
    p.add_argument("--seed", type=int, default=0, help="a dev seed (0 to 9,999)")
    p.add_argument("--csv", help="also save the ledger to this file")
    p.set_defaults(func=cmd_walkthrough)

    p = sub.add_parser("sanity", help="run the scripted delta and random agents in the env")
    p.add_argument("--scenarios", nargs="+", required=True)
    p.add_argument("--n-paths", type=int, default=None, help="default: all validation paths")
    p.add_argument("--dir", type=Path, default=pathsets.DEFAULT_DIR)
    p.set_defaults(func=cmd_sanity)

    p = sub.add_parser("train", help="train one agent on one scenario with one seed")
    p.add_argument("--agent", required=True, help="a file in config/agents/")
    p.add_argument("--scenario", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--total-steps", type=int, default=None, help="override the agent config")
    p.add_argument("--verbose", type=int, default=0)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("experiment", help="run a grid of training runs")
    p.add_argument("--config", required=True, help="a file in config/experiments/")
    p.add_argument("--jobs", type=int, default=1, help="runs to train in parallel")
    p.set_defaults(func=cmd_experiment)

    p = sub.add_parser("eval", help="score every strategy on one split, with bootstrap CIs")
    p.add_argument("--scenarios", nargs="+", required=True)
    p.add_argument("--split", default="validation", choices=["validation", "test"])
    p.add_argument("--final-test-run", **final)
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("stress", help="stress tests: volatility shock, model swap, jumps")
    p.add_argument("--split", default="validation", choices=["validation", "test"])
    p.add_argument("--final-test-run", **final)
    p.set_defaults(func=cmd_stress)

    p = sub.add_parser("db-load", help="rebuild the PostgreSQL experiment record from files")
    p.add_argument("--url", default=None, help="default: the DATABASE_URL environment variable")
    p.add_argument("--splits", nargs="+", default=["validation", "test"])
    p.set_defaults(func=cmd_db_load)

    p = sub.add_parser("report", help="regenerate the figures in docs/figures")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("export-sample", help="copy what the demo needs into app/sample")
    p.set_defaults(func=cmd_export_sample)

    p = sub.add_parser("tune", help="Optuna hyperparameter search on validation CVaR")
    p.add_argument("--config", required=True, help="a tune_*.yaml file in config/experiments/")
    p.add_argument("--jobs", type=int, default=1, help="trials to train in parallel")
    p.set_defaults(func=cmd_tune)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
