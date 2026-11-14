"""Score a learned policy on a whole path set, through the one P&L engine."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from hedge.baselines.run import score
from hedge.env.hedging_env import HedgingEnv
from hedge.pnl import HedgeResult

# Maps a batch of observations (n, d) to a batch of actions (n, 1) or (n,).
PredictFn = Callable[[np.ndarray], np.ndarray]


def policy_holdings(env: HedgingEnv, predict: PredictFn) -> np.ndarray:
    """Holdings the policy chooses on every path of ``env.path_set``, shape (n_paths, T).

    Steps all paths together: T calls to ``predict`` instead of n_paths * T.
    """
    holdings = np.empty((len(env.path_set), env.n_steps))
    held = np.zeros(len(env.path_set))
    for t in range(env.n_steps):
        held = env.batch_holdings(t, predict(env.batch_observation(t, held)))
        holdings[:, t] = held
    return holdings


def evaluate_policy(env: HedgingEnv, predict: PredictFn) -> tuple[HedgeResult, np.ndarray]:
    """Run the policy on every path and score the holdings with the P&L engine."""
    holdings = policy_holdings(env, predict)
    return score(env.scenario, env.path_set, holdings), holdings


def sb3_predict(model) -> PredictFn:
    """Deterministic actions from a stable-baselines3 model."""
    return lambda obs: model.predict(obs, deterministic=True)[0]
