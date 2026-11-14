"""Scripted sanity agents and a rollout helper.

Policies here are functions ``policy(env, obs) -> action``. ``rollout`` records the holdings an
agent chooses and scores them with the P&L engine, like every other strategy.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from hedge.baselines.run import score
from hedge.env.hedging_env import HedgingEnv
from hedge.pnl import HedgeResult

Policy = Callable[[HedgingEnv, np.ndarray], np.ndarray]


def delta_policy(env: HedgingEnv, obs: np.ndarray) -> np.ndarray:
    """Always hold the Black-Scholes delta. Must reproduce the Phase 1 delta baseline."""
    delta = env.current_delta
    return env.holding_to_action(delta, delta)


def delta_policy_from_obs(env: HedgingEnv, obs: np.ndarray) -> np.ndarray:
    """Hold BS delta using only the observation, as a trained agent would (float32 precision)."""
    if env.action_mode == "delta_offset":
        return np.zeros(1, dtype=np.float32)
    return obs[[env.obs_features.index("bs_delta")]]


def random_policy(env: HedgingEnv, obs: np.ndarray) -> np.ndarray:
    """Uniform random actions from the env's own seeded generator."""
    return env.np_random.uniform(-1.0, 1.0, size=1)


@dataclass(frozen=True)
class Rollout:
    result: HedgeResult  # from the P&L engine, on the recorded holdings
    holdings: np.ndarray  # (n_episodes, T)
    env_pnl: np.ndarray  # episode P&L as summed by the env, step by step
    reward: np.ndarray  # total reward per episode
    path_index: np.ndarray


def rollout(
    env: HedgingEnv, policy: Policy, n_episodes: int | None = None, seed: int = 0
) -> Rollout:
    """Run ``policy`` on paths 0 .. n_episodes - 1 of the env's path set, in order."""
    n = len(env.path_set) if n_episodes is None else n_episodes
    holdings = np.empty((n, env.n_steps))
    env_pnl = np.empty(n)
    reward = np.zeros(n)
    for i in range(n):
        obs, _ = env.reset(seed=seed if i == 0 else None, options={"path_index": i})
        for t in range(env.n_steps):
            obs, r, terminated, _, info = env.step(policy(env, obs))
            holdings[i, t] = info["holding"]
            reward[i] += r
        assert terminated
        env_pnl[i] = info["episode_pnl"]
    index = np.arange(n)
    subset = type(env.path_set)(
        env.path_set.seeds[index],
        env.path_set.prices[index],
        None if env.path_set.variances is None else env.path_set.variances[index],
    )
    return Rollout(score(env.scenario, subset, holdings), holdings, env_pnl, reward, index)
