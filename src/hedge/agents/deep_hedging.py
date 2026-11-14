"""Deep hedging (Buehler, Gonon, Teichmann and Wood, 2019) in PyTorch.

One network, shared across time steps, maps the same observation the RL agents see to an
action. The whole episode is run through a torch copy of the P&L engine and the network is
trained on a risk measure of the hedging cost over a batch of training paths: CVaR 95% by
default, or entropic risk. No value function, no per-step reward.

Training uses train paths, early stopping uses validation CVaR, and the final numbers come
from the numpy P&L engine like every other strategy. Outputs have the same layout as the
stable-baselines3 runs (``model.pt`` instead of ``model.zip``).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from hedge.agents.train_sb3 import RUNS_DIR, AgentConfig, _code_commit, env_kwargs, run_dir
from hedge.config import Scenario
from hedge.env.hedging_env import HedgingEnv
from hedge.eval.evaluate import evaluate_policy
from hedge.eval.metrics import summarise
from hedge.market import pathsets

DEFAULTS = {
    "learning_rate": 1e-3,
    "batch_size": 5_000,
    "patience": 6,  # validation checks without improvement before stopping
    "loss": "cvar",  # or "entropic"
    "alpha": 0.95,
    "risk_aversion": 1.0,  # entropic risk only
}


def torch_hedge_pnl(
    paths: torch.Tensor,
    holdings: torch.Tensor,
    strike: float,
    premium: float,
    cost_rate: float,
    kind: str = "call",
) -> torch.Tensor:
    """The P&L engine in torch, differentiable in ``holdings``. Mirrors ``hedge.pnl.hedge_pnl``."""
    s_T = paths[:, -1]
    payoff = torch.relu(s_T - strike) if kind == "call" else torch.relu(strike - s_T)
    hedge_gain = (holdings * (paths[:, 1:] - paths[:, :-1])).sum(dim=1)
    flat = torch.zeros_like(holdings[:, :1])
    trades = torch.diff(torch.cat([flat, holdings, flat], dim=1), dim=1)
    txn_cost = cost_rate * (paths * trades.abs()).sum(dim=1)
    return premium - payoff + hedge_gain - txn_cost


def cvar_loss(cost: torch.Tensor, alpha: float = 0.95) -> torch.Tensor:
    """Mean of the worst ``1 - alpha`` share of costs in the batch."""
    k = max(1, int(np.ceil(cost.numel() * (1.0 - alpha) - 1e-9)))
    return torch.topk(cost, k).values.mean()


def entropic_loss(cost: torch.Tensor, risk_aversion: float = 1.0) -> torch.Tensor:
    """Entropic risk: (1 / a) log E[exp(a * cost)]."""
    n = torch.tensor(float(cost.numel()), dtype=cost.dtype)
    return (torch.logsumexp(risk_aversion * cost, dim=0) - torch.log(n)) / risk_aversion


class DeepHedger(nn.Module):
    """A small MLP from observation to action in [-1, 1], shared across time steps."""

    def __init__(self, obs_dim: int, net_arch: tuple[int, ...] = (64, 64)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = obs_dim
        for hidden in net_arch:
            layers += [nn.Linear(width, hidden), nn.Tanh()]
            width = hidden
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)
        # Start at action 0: the delta hedge in delta_offset mode, half a share in holding mode.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(obs)).squeeze(-1)

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        """Same call as a stable-baselines3 model, so the evaluation code takes either."""
        with torch.no_grad():
            actions = self(torch.as_tensor(np.atleast_2d(obs), dtype=torch.float32))
        return actions.numpy().reshape(-1, 1), None


class EpisodeData:
    """Everything needed to run whole episodes in torch for one path set."""

    def __init__(self, env: HedgingEnv) -> None:
        self.env = env
        zeros = np.zeros(len(env.path_set))
        static = np.stack([env.batch_observation(t, zeros) for t in range(env.n_steps)], axis=1)
        self.features = torch.as_tensor(static)  # (n, T, d), holding column to be filled in
        self.paths = torch.tensor(env.path_set.prices, dtype=torch.float32)
        self.deltas = torch.as_tensor(env._deltas, dtype=torch.float32)
        self.holding_col = (
            env.obs_features.index("holding") if "holding" in env.obs_features else None
        )

    def holdings(self, model: DeepHedger, index: torch.Tensor) -> torch.Tensor:
        """Holdings the network chooses on the selected paths, shape (batch, T), with gradients."""
        env = self.env
        span = env.h_high - env.h_low
        held = torch.zeros(index.numel())
        out = []
        for t in range(env.n_steps):
            obs = self.features[index, t]
            if self.holding_col is not None:
                obs = obs.clone()
                obs[:, self.holding_col] = 2.0 * (held - env.h_low) / span - 1.0
            action = model(obs)
            if env.action_mode == "holding":
                held = env.h_low + 0.5 * (action + 1.0) * span
            else:
                held = torch.clamp(
                    self.deltas[index, t] + action * env.max_offset, env.h_low, env.h_high
                )
            out.append(held)
        return torch.stack(out, dim=1)

    def cost(self, model: DeepHedger, index: torch.Tensor) -> torch.Tensor:
        env = self.env
        pnl = torch_hedge_pnl(
            self.paths[index],
            self.holdings(model, index),
            env.scenario.strike,
            env.premium,
            env.cost_rate,
            env.scenario.kind,
        )
        return -pnl


def load_deep_hedger(agent: AgentConfig, directory: Path, obs_dim: int) -> DeepHedger:
    model = DeepHedger(obs_dim, agent.net_arch)
    model.load_state_dict(torch.load(directory / "model.pt", weights_only=True))
    return model.eval()


def train_deep_hedging(
    agent: AgentConfig,
    scenario: Scenario,
    seed: int,
    paths_dir: Path = pathsets.DEFAULT_DIR,
    runs_dir: Path = RUNS_DIR,
    train_paths=None,
    validation_paths=None,
    verbose: int = 0,
) -> dict[str, Any]:
    """Train one deep-hedging network and return its validation metrics."""
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    hp = {**DEFAULTS, **agent.hyperparams}
    train_set = train_paths or pathsets.load_or_build(scenario, "train", paths_dir)
    validation = validation_paths or pathsets.load_or_build(scenario, "validation", paths_dir)
    kwargs = env_kwargs(agent)
    data = EpisodeData(HedgingEnv(scenario, train_set, mode="eval", **kwargs))
    eval_env = HedgingEnv(scenario, validation, mode="eval", **kwargs)
    out = run_dir(agent, scenario, seed, runs_dir)
    out.mkdir(parents=True, exist_ok=True)

    model = DeepHedger(len(eval_env.obs_features), agent.net_arch)
    optimiser = torch.optim.Adam(model.parameters(), lr=float(hp["learning_rate"]))
    generator = torch.Generator().manual_seed(seed)
    batch = min(int(hp["batch_size"]), len(train_set))

    def loss_fn(cost: torch.Tensor) -> torch.Tensor:
        if hp["loss"] == "entropic":
            return entropic_loss(cost, float(hp["risk_aversion"]))
        return cvar_loss(cost, float(hp["alpha"]))

    curve: list[dict[str, float]] = []
    best = {"cvar": float("inf"), "state": None, "step": 0}
    stale = 0

    def validate(step: int, train_loss: float) -> bool:
        """Record validation metrics; return True when training should stop."""
        nonlocal stale
        metrics = summarise(evaluate_policy(eval_env, lambda obs: model.predict(obs)[0])[0])
        curve.append(
            {
                "step": step,
                "train_loss": train_loss,
                "val_mean_cost": metrics["mean_cost"],
                "val_std_cost": metrics["std_cost"],
                "val_cvar95": metrics["cvar95"],
            }
        )
        if verbose:
            print(f"step {step}: train loss {train_loss:.4f}, val CVaR {metrics['cvar95']:.4f}")
        if metrics["cvar95"] < best["cvar"] - 1e-5:
            best.update(
                cvar=metrics["cvar95"],
                state={k: v.clone() for k, v in model.state_dict().items()},
                step=step,
            )
            stale = 0
        else:
            stale += 1
        return stale >= int(hp["patience"])

    start = time.time()
    validate(0, float("nan"))
    step = 0
    while step < agent.total_steps:
        index = torch.randint(len(train_set), (batch,), generator=generator)
        loss = loss_fn(data.cost(model, index))
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        step += 1
        if step % agent.eval_every == 0 and validate(step, float(loss.detach())):
            break
    wall = time.time() - start

    model.load_state_dict(best["state"])  # early stopping: keep the best validation CVaR
    model.eval()
    torch.save(model.state_dict(), out / "model.pt")
    pd.DataFrame(curve).to_csv(out / "curve.csv", index=False)

    result, _ = evaluate_policy(eval_env, lambda obs: model.predict(obs)[0])
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
        "hyperparams": hp,
        "scenario": scenario.name,
        "scenario_hash": scenario.hash,
        "paths_hash": scenario.paths_hash,
        "seed": seed,
        "total_steps": step,
        "best_step": best["step"],
        "wall_time_s": round(wall, 1),
        "code_commit": _code_commit(),
    }
    (out / "run.json").write_text(json.dumps(run_info, indent=2) + "\n")
    (out / "eval_validation.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics
