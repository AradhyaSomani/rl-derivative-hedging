"""Turn finished training runs into the figures and tables used in ``docs/results.md``."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from hedge.agents.train_sb3 import RUNS_DIR, AgentConfig, env_kwargs, load_model
from hedge.config import Scenario
from hedge.env.hedging_env import HedgingEnv
from hedge.eval import plots
from hedge.eval.evaluate import policy_holdings, sb3_predict
from hedge.market import pathsets

RESULTS_DIR = Path("data/results")


def _agent_from_run(directory: Path) -> AgentConfig:
    raw = json.loads((directory / "run.json").read_text())["agent"]
    raw["net_arch"] = tuple(raw["net_arch"])
    return AgentConfig(**raw)


def first_agent_figures(
    strategy: str, scenario_file: str, seed: int = 0, figures_dir: Path = plots.FIGURES_DIR
) -> list[Path]:
    """Phase 3A: the learning curve and the holding-against-delta plot for one run."""
    scenario = Scenario.from_yaml(scenario_file)
    directory = RUNS_DIR / strategy / scenario.name / f"seed{seed}"
    agent = _agent_from_run(directory)
    baselines = pd.read_csv(RESULTS_DIR / "baselines_validation.csv")
    refs = baselines[baselines["scenario"] == scenario.name].set_index("strategy")["cvar95"]
    curve = pd.read_csv(directory / "curve.csv")
    stem = f"{strategy}_{scenario.name}_seed{seed}"
    files = [
        plots.learning_curve(
            {strategy: curve},
            {"BS delta": refs["bs_delta"], "No-trade band": refs["no_trade_band"]},
            f"{strategy} on {scenario.name}: validation tail risk while training",
            figures_dir / f"curve_{stem}.png",
        )
    ]
    validation = pathsets.load_or_build(scenario, "validation")
    sample = pathsets.PathSet(
        validation.seeds[:5],
        validation.prices[:5],
        None if validation.variances is None else validation.variances[:5],
    )
    env = HedgingEnv(scenario, sample, mode="eval", **env_kwargs(agent))
    holdings = policy_holdings(env, sb3_predict(load_model(agent, directory)))
    files.append(
        plots.holdings_vs_delta(
            holdings,
            env._deltas,
            sample.prices,
            f"{strategy} on {scenario.name}: holding against BS delta, first 5 validation paths",
            figures_dir / f"holdings_{stem}.png",
        )
    )
    return files


def frontier_figure(
    experiment: str, figures_dir: Path = plots.FIGURES_DIR, group: str = "reward"
) -> list[Path]:
    """Phase 3C: mean cost against CVaR for every run of an experiment, one figure per scenario."""
    table = pd.read_csv(RESULTS_DIR / f"{experiment}_validation.csv")
    files = []
    for scenario, rows in table.groupby("scenario"):
        rows = rows[rows["strategy"] != "no_hedge"].copy()
        parts = rows["strategy"].str.extract(
            r"^(?P<agent>.+)_(?P<reward>quadratic|mean_variance|terminal_cvar)_l(?P<lam>[\d.]+)$"
        )
        rows["reward"] = parts["reward"]
        rows["agent"] = parts["agent"]
        rows["label"] = "λ=" + parts["lam"].fillna("")
        # Median over seeds when an experiment has several.
        agents = (
            rows[rows["reward"].notna()]
            .groupby(["strategy", "reward", "agent", "label"], as_index=False)[
                ["mean_cost", "cvar95"]
            ]
            .median()
        )
        baselines = rows[rows["reward"].isna()]
        delta_cvar = float(baselines.loc[baselines["strategy"] == "bs_delta", "cvar95"].iloc[0])
        shown = agents[agents["cvar95"] <= 3 * delta_cvar]
        hidden = len(agents) - len(shown)
        note = f" ({hidden} runs off the scale not shown)" if hidden else ""
        files.append(
            plots.frontier(
                pd.concat([shown, baselines], ignore_index=True),
                f"{scenario}, validation paths: down and left is better{note}",
                figures_dir / f"frontier_{experiment}_{scenario}.png",
                group=group,
            )
        )
    return files


def experiment_table(experiment: str) -> pd.DataFrame:
    """Median and range over seeds for every strategy and scenario of an experiment."""
    table = pd.read_csv(RESULTS_DIR / f"{experiment}_validation.csv")
    grouped = table.groupby(["scenario", "strategy"])
    out = grouped[["mean_cost", "std_cost", "cvar95", "mean_turnover"]].median()
    out["n_seeds"] = grouped["seed"].count().clip(lower=1)
    out["cvar95_min"] = grouped["cvar95"].min()
    out["cvar95_max"] = grouped["cvar95"].max()
    return out.reset_index()


__all__ = ["experiment_table", "first_agent_figures", "frontier_figure"]


STRATEGY_LABELS = {"bs_delta": "BS delta", "no_trade_band": "No-trade band", "no_hedge": "No hedge"}


def seed_summary(experiments: list[str]) -> pd.DataFrame:
    """One row per scenario and strategy: medians over training seeds, with the CVaR range."""
    table = pd.concat(
        [pd.read_csv(RESULTS_DIR / f"{name}_validation.csv") for name in experiments]
    ).drop_duplicates(["scenario", "strategy", "seed"])
    grouped = table.groupby(["scenario", "strategy"], sort=False)
    out = grouped[["mean_cost", "std_cost", "cvar95", "mean_turnover"]].median()
    out["n_seeds"] = grouped["seed"].count()
    out["cvar95_min"] = grouped["cvar95"].min()
    out["cvar95_max"] = grouped["cvar95"].max()
    out = out.reset_index()
    out["market"] = out["scenario"].str.split("_").str[0]
    out["cost_bps"] = out["scenario"].str.extract(r"_(\d+)bps")[0].astype(float)
    return out.sort_values(["market", "cost_bps", "strategy"]).reset_index(drop=True)


def summary_markdown(summary: pd.DataFrame) -> str:
    lines = [
        "| Scenario | Strategy | Seeds | Mean cost | Std | CVaR 95% | CVaR range | Turnover |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, r in summary.iterrows():
        learned = r["n_seeds"] > 0
        spread = f"{r['cvar95_min']:.3f} to {r['cvar95_max']:.3f}" if learned else ""
        lines.append(
            f"| {r['scenario']} | {STRATEGY_LABELS.get(r['strategy'], r['strategy'])} "
            f"| {int(r['n_seeds']) if learned else ''} | {r['mean_cost']:.3f} "
            f"| {r['std_cost']:.3f} | {r['cvar95']:.3f} | {spread} | {r['mean_turnover']:.2f} |"
        )
    return "\n".join(lines)


def cost_frontier_figures(
    summary: pd.DataFrame, agent_strategy: str, agent_label: str,
    figures_dir: Path = plots.FIGURES_DIR,
) -> list[Path]:  # fmt: skip
    """One figure per market: delta, band and one learned agent across the cost levels."""
    labels = {"bs_delta": "BS delta", "no_trade_band": "No-trade band", agent_strategy: agent_label}
    files = []
    for market, rows in summary.groupby("market"):
        rows = rows[rows["strategy"].isin(labels)].copy()
        rows["strategy_label"] = rows["strategy"].map(labels)
        rows["order"] = rows["strategy"].map({k: i for i, k in enumerate(labels)})
        rows.loc[rows["n_seeds"] == 0, ["cvar95_min", "cvar95_max"]] = float("nan")
        files.append(
            plots.cost_frontier(
                rows.sort_values(["order", "cost_bps"]),
                f"{market.upper()}, validation paths: down and left is better. "
                "Bars show the range over 5 training seeds",
                figures_dir / f"cost_frontier_{market}.png",
            )
        )
    return files


def cross_evaluate(
    strategy: str, train_scenario_file: str, eval_scenario_files: list[str], seeds: list[int]
) -> pd.DataFrame:
    """Score agents trained on one scenario on the validation paths of others (generalisation)."""
    from hedge.eval.evaluate import evaluate_policy
    from hedge.eval.metrics import summarise

    trained_on = Scenario.from_yaml(train_scenario_file)
    rows = []
    for scenario_file in eval_scenario_files:
        scenario = Scenario.from_yaml(scenario_file)
        validation = pathsets.load_or_build(scenario, "validation")
        for seed in seeds:
            directory = RUNS_DIR / strategy / trained_on.name / f"seed{seed}"
            agent = _agent_from_run(directory)
            env = HedgingEnv(scenario, validation, mode="eval", **env_kwargs(agent))
            result, _ = evaluate_policy(env, sb3_predict(load_model(agent, directory)))
            rows.append(
                {
                    "strategy": strategy,
                    "trained_on": trained_on.name,
                    "scenario": scenario.name,
                    "seed": seed,
                    **summarise(result),
                }
            )
    return pd.DataFrame(rows)


def median_curve(strategy: str, scenario_name: str, seeds: list[int]) -> pd.DataFrame:
    """Median validation curve over seeds."""
    curves = [
        pd.read_csv(RUNS_DIR / strategy / scenario_name / f"seed{seed}" / "curve.csv")
        for seed in seeds
    ]
    return pd.concat(curves).groupby("step", as_index=False).median(numeric_only=True)


def report_all(figures_dir: Path = plots.FIGURES_DIR) -> list[Path]:
    """Regenerate every figure used in the README and ``docs/results.md`` from saved results."""
    from hedge import results

    files: list[Path] = []
    gbm10 = "config/scenarios/gbm_10bps.yaml"
    if (RUNS_DIR / "ppo_quadratic_l1" / "gbm_10bps" / "seed0" / "run.json").exists():
        files += first_agent_figures("ppo_quadratic_l1", gbm10, figures_dir=figures_dir)
    if (RESULTS_DIR / "reward_grid_validation.csv").exists():
        files += frontier_figure("reward_grid", figures_dir)

    baselines = pd.read_csv(RESULTS_DIR / "baselines_validation.csv")
    seeds = [0, 1, 2, 3, 4]
    for scenario_name in ("gbm_10bps", "heston_10bps"):
        runs = {
            "PPO": "ppo_tuned_quadratic_l1",
            "SAC": "sac_quadratic_l1",
            "Deep hedging": "deep_hedging",
        }
        curves = {
            label: median_curve(strategy, scenario_name, seeds)
            for label, strategy in runs.items()
            if (RUNS_DIR / strategy / scenario_name / "seed4" / "curve.csv").exists()
        }
        if "Deep hedging" in curves:
            # One deep hedging step is a batch of whole episodes: count it in trading days too.
            run = json.loads(
                (RUNS_DIR / "deep_hedging" / scenario_name / "seed0" / "run.json").read_text()
            )
            days = Scenario.from_yaml(f"config/scenarios/{scenario_name}.yaml").n_steps
            per_step = run["hyperparams"]["batch_size"] * days
            curves["Deep hedging"] = curves["Deep hedging"].assign(
                step=lambda c, k=per_step: c["step"] * k
            )
        refs = baselines[baselines["scenario"] == scenario_name].set_index("strategy")["cvar95"]
        if curves:
            files.append(
                plots.learning_curve(
                    curves,
                    {"BS delta": refs["bs_delta"], "No-trade band": refs["no_trade_band"]},
                    f"{scenario_name}: validation CVaR 95% while training, median of 5 seeds",
                    figures_dir / f"curves_{scenario_name}.png",
                    log_x=True,
                    x_label="Simulated trading days used in training (log scale)",
                )
            )

    summary = results.summary_table()
    split = results.available_split()
    shown = ["bs_delta", "no_trade_band", "ppo_tuned_quadratic_l1", "deep_hedging"]
    for market, rows in summary.groupby("market"):
        rows = rows[rows["strategy"].isin(shown)].copy()
        rows["strategy_label"] = rows["label"]
        rows["order"] = rows["strategy"].map({k: i for i, k in enumerate(shown)})
        rows.loc[rows["n_seeds"] == 0, ["cvar95_min", "cvar95_max"]] = float("nan")
        files.append(
            plots.cost_frontier(
                rows.sort_values(["order", "cost_bps"]),
                f"{market.upper()}, {split} paths: down and left is better. "
                "Bars show the range over 5 training seeds",
                figures_dir / f"cost_frontier_{market}.png",
            )
        )
    for scenario_name in ("gbm_10bps", "heston_10bps"):
        costs = results.costs(scenario_name)
        wanted = ["BS delta", "No-trade band", "Deep hedging"]
        data = {
            label: costs.loc[costs["label"] == label, "cost"].to_numpy()
            for label in wanted
            if (costs["label"] == label).any()
        }
        files.append(
            plots.cost_histogram(
                data,
                f"{scenario_name}, {split} paths: hedging cost over 10,000 paths",
                figures_dir / f"histogram_{scenario_name}.png",
            )
        )
    return files
