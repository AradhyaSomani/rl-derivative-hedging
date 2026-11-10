"""Command line entry points: ``python -m hedge.cli <command>``. See the Makefile."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

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

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
