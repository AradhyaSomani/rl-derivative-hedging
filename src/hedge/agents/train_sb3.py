"""Train a stable-baselines3 agent on the hedging environment, from config files.

One run = one agent config + one scenario + one seed. Outputs go to
``data/runs/{strategy}/{scenario}/seed{n}/``:

    model.zip             trained weights
    curve.csv             step, mean_reward, val_mean_cost, val_std_cost, val_cvar95
    run.json              configs, hashes, code commit, wall time
    eval_validation.json  final metrics on validation paths

Training samples from train paths only. Validation paths are used for the curve and the final
metrics. Test paths are never touched here.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from hedge.baselines.run import run_baselines
from hedge.config import Scenario, config_hash
from hedge.env.hedging_env import HedgingEnv, make_env_fn
from hedge.env.rewards import RewardConfig
from hedge.eval.evaluate import evaluate_policy, sb3_predict
from hedge.eval.metrics import summarise, var
from hedge.market import pathsets

RUNS_DIR = Path("data/runs")
ALGOS = ("ppo", "sac", "td3", "deep_hedging")


@dataclass(frozen=True)
class AgentConfig:
    """An agent YAML from ``config/agents/``."""

    name: str
    algo: str
    total_steps: int
    n_envs: int = 8
    net_arch: tuple[int, ...] = (64, 64)  # the same for every algorithm, for fairness
    # Passed to the SB3 constructor, except ``lr_schedule: linear``, which decays the learning
    # rate to zero over training.
    hyperparams: dict[str, Any] = field(default_factory=dict)
    reward: dict[str, Any] = field(default_factory=lambda: {"name": "quadratic", "lam": 1.0})
    env: dict[str, Any] = field(default_factory=dict)  # action_mode, obs_features, max_offset
    eval_every: int = 50_000
    vec_env: str = "dummy"  # or "subproc"

    @classmethod
    def from_yaml(cls, path: str | Path, **overrides: Any) -> AgentConfig:
        with open(path) as f:
            raw = yaml.safe_load(f)
        raw.update({k: v for k, v in overrides.items() if v is not None})
        raw["net_arch"] = tuple(raw.get("net_arch", (64, 64)))
        cfg = cls(**raw)
        if cfg.algo not in ALGOS:
            raise ValueError(f"algo must be one of {ALGOS}, got {cfg.algo!r}")
        return cfg

    @property
    def strategy(self) -> str:
        """Name used for the run folder and the results table, e.g. ``ppo_quadratic_l0.5``."""
        if self.algo == "deep_hedging":
            return self.name  # trained on a risk measure directly, so no reward in the name
        parts = [self.name, self.reward["name"], f"l{self.reward.get('lam', 1.0):g}"]
        return "_".join(parts)

    @property
    def hash(self) -> str:
        return config_hash(json.loads(json.dumps(asdict(self))))


def _code_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def reward_config(agent: AgentConfig, scenario: Scenario, validation) -> RewardConfig:
    """Build the reward. ``threshold: auto`` uses the delta hedge's validation VaR 95%."""
    raw = dict(agent.reward)
    if raw.get("threshold") == "auto":
        delta = run_baselines(scenario, validation, {"width": 0.0, "rebalance_to": "delta"})
        raw["threshold"] = var(delta["bs_delta"].cost, raw.get("alpha", 0.95))
    return RewardConfig(**raw)


def run_dir(agent: AgentConfig, scenario: Scenario, seed: int, root: Path = RUNS_DIR) -> Path:
    return Path(root) / agent.strategy / scenario.name / f"seed{seed}"


def env_kwargs(agent: AgentConfig) -> dict[str, Any]:
    kwargs = dict(agent.env)
    if kwargs.get("obs_features"):
        kwargs["obs_features"] = tuple(kwargs["obs_features"])
    return kwargs


def load_model(agent: AgentConfig, directory: Path):
    """Load a trained model of either kind; both have ``predict(obs, deterministic=True)``."""
    if agent.algo == "deep_hedging":
        from hedge.agents.deep_hedging import load_deep_hedger
        from hedge.env.hedging_env import OBS_FEATURES

        scenario_name = json.loads((directory / "run.json").read_text())["scenario"]
        features = agent.env.get("obs_features")
        obs_dim = len(features) if features else len(OBS_FEATURES) - ("gbm" in scenario_name)
        return load_deep_hedger(agent, directory, obs_dim)
    import stable_baselines3 as sb3

    return getattr(sb3, agent.algo.upper()).load(directory / "model.zip", device="cpu")


def train(
    agent: AgentConfig,
    scenario: Scenario,
    seed: int,
    paths_dir: Path = pathsets.DEFAULT_DIR,
    runs_dir: Path = RUNS_DIR,
    train_paths=None,
    validation_paths=None,
    verbose: int = 0,
) -> dict[str, Any]:
    """Train one agent and return its final validation metrics."""
    if agent.algo == "deep_hedging":
        from hedge.agents.deep_hedging import train_deep_hedging

        return train_deep_hedging(
            agent, scenario, seed, paths_dir, runs_dir, train_paths, validation_paths, verbose
        )
    import stable_baselines3 as sb3
    import torch
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

    torch.set_num_threads(1)  # small MLPs: more threads only add overhead
    train_set = train_paths or pathsets.load_or_build(scenario, "train", paths_dir)
    validation = validation_paths or pathsets.load_or_build(scenario, "validation", paths_dir)
    reward = reward_config(agent, scenario, validation)
    kwargs = env_kwargs(agent)
    out = run_dir(agent, scenario, seed, runs_dir)
    out.mkdir(parents=True, exist_ok=True)

    vec_cls = SubprocVecEnv if agent.vec_env == "subproc" else DummyVecEnv
    venv = VecMonitor(
        vec_cls(
            [
                make_env_fn(scenario, train_set, rank=seed * 1000 + i, reward=reward, **kwargs)
                for i in range(agent.n_envs)
            ]
        )
    )
    eval_env = HedgingEnv(scenario, validation, reward, mode="eval", **kwargs)

    hyperparams = dict(agent.hyperparams)
    if hyperparams.pop("lr_schedule", "constant") == "linear":
        start_lr = float(hyperparams.get("learning_rate", 3e-4))
        # SB3 passes the share of training still to go, from 1 down to 0.
        hyperparams["learning_rate"] = lambda remaining: remaining * start_lr

    algo_cls = getattr(sb3, agent.algo.upper())
    model = algo_cls(
        "MlpPolicy",
        venv,
        seed=seed,
        device="cpu",
        verbose=verbose,
        policy_kwargs={"net_arch": list(agent.net_arch)},
        tensorboard_log=str(out / "tb"),
        **hyperparams,
    )

    curve: list[dict[str, float]] = []

    def record(step: int) -> None:
        result, _ = evaluate_policy(eval_env, sb3_predict(model))
        metrics = summarise(result)
        rewards = [ep["r"] for ep in model.ep_info_buffer] if model.ep_info_buffer else []
        curve.append(
            {
                "step": step,
                "mean_reward": float(np.mean(rewards)) if rewards else float("nan"),
                "val_mean_cost": metrics["mean_cost"],
                "val_std_cost": metrics["std_cost"],
                "val_cvar95": metrics["cvar95"],
            }
        )
        pd.DataFrame(curve).to_csv(out / "curve.csv", index=False)

    class ValidationCallback(BaseCallback):
        def __init__(self) -> None:
            super().__init__()
            self.next_eval = agent.eval_every

        def _on_step(self) -> bool:
            if self.num_timesteps >= self.next_eval:
                record(self.num_timesteps)
                self.next_eval += agent.eval_every
            return True

    start = time.time()
    record(0)
    model.learn(total_timesteps=agent.total_steps, callback=ValidationCallback())
    wall = time.time() - start
    if curve[-1]["step"] < model.num_timesteps:
        record(model.num_timesteps)
    venv.close()
    model.save(out / "model.zip")

    result, _ = evaluate_policy(eval_env, sb3_predict(model))
    metrics = {
        "strategy": agent.strategy,
        "scenario": scenario.name,
        "seed": seed,
        **summarise(result),
        "split": "validation",
    }
    run_info = {
        "agent": json.loads(json.dumps(asdict(agent))),
        "agent_hash": agent.hash,
        "reward": asdict(reward),
        "scenario": scenario.name,
        "scenario_hash": scenario.hash,
        "paths_hash": scenario.paths_hash,
        "seed": seed,
        "total_steps": int(model.num_timesteps),
        "wall_time_s": round(wall, 1),
        "code_commit": _code_commit(),
    }
    (out / "run.json").write_text(json.dumps(run_info, indent=2) + "\n")
    (out / "eval_validation.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics
