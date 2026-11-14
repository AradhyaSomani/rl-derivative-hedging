"""Gymnasium environment: one episode is one short option, hedged from sale to expiry.

The environment owns no finance of its own. Paths come from ``hedge.market.pathsets``, deltas
from the Phase 1 baseline code, and step P&L from ``hedge.env.rewards``, whose episode total is
tested against the P&L engine.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from hedge.baselines.run import deltas
from hedge.config import Scenario
from hedge.env.rewards import RewardConfig, compute_reward, option_marks, step_components
from hedge.market.pathsets import PathSet

OBS_FEATURES = ("tau", "log_moneyness", "holding", "bs_delta", "variance")
ACTION_MODES = ("holding", "delta_offset")
OBS_CLIP = 10.0  # bound on the two features with no natural limit; never reached in practice


def default_obs_features(scenario: Scenario) -> tuple[str, ...]:
    """Everything a trader would know: the variance only exists on Heston."""
    return OBS_FEATURES if scenario.model == "heston" else OBS_FEATURES[:-1]


class HedgingEnv(gym.Env):
    """Hedge a short European option with daily trades in the underlying.

    Observation (each scaled to roughly [-1, 1]), in the order of ``obs_features``:
        tau            time to expiry: 1 at the sale, -1 at expiry
        log_moneyness  log(S / K) divided by three standard deviations of the log return to expiry
        holding        shares currently held, before this step's trade
        bs_delta       the Black-Scholes delta, the textbook target
        variance       current volatility relative to its long-run level, minus 1 (Heston only)

    Action: one number in [-1, 1].
        action_mode="holding"       mapped linearly onto the allowed holding, [0, 1] shares for a
                                    call and [-1, 0] for a put. -1 is the least hedged.
        action_mode="delta_offset"  the holding is BS delta + action * max_offset, clipped to
                                    the allowed range, so action 0 is the delta hedge.

    The task list specifies ``Box(0, 1)``; the symmetric range is what stable-baselines3 asks
    for, and the mapping makes the two equivalent.

    mode="train" draws a random path each episode; mode="eval" walks through the path set in
    order, wrapping around at the end. ``reset(options={"path_index": i})`` picks a path.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        scenario: Scenario,
        path_set: PathSet,
        reward: RewardConfig | None = None,
        mode: str = "train",
        obs_features: tuple[str, ...] | None = None,
        action_mode: str = "holding",
        max_offset: float = 0.25,
    ) -> None:
        super().__init__()
        if mode not in ("train", "eval"):
            raise ValueError(f"mode must be 'train' or 'eval', got {mode!r}")
        if action_mode not in ACTION_MODES:
            raise ValueError(f"action_mode must be one of {ACTION_MODES}, got {action_mode!r}")
        features = tuple(obs_features or default_obs_features(scenario))
        unknown = set(features) - set(OBS_FEATURES)
        if unknown:
            raise ValueError(f"unknown observation features: {sorted(unknown)}")
        if "variance" in features and path_set.variances is None:
            raise ValueError("the 'variance' feature needs a Heston path set")

        self.scenario = scenario
        self.path_set = path_set
        self.reward_cfg = reward or RewardConfig()
        self.mode = mode
        self.obs_features = features
        self.action_mode = action_mode
        self.max_offset = float(max_offset)

        self.n_steps = scenario.n_steps
        if path_set.prices.shape[1] != self.n_steps + 1:
            raise ValueError("path set does not match the scenario's number of steps")
        self.h_low, self.h_high = (0.0, 1.0) if scenario.kind == "call" else (-1.0, 0.0)
        self.premium = scenario.premium()
        self.cost_rate = scenario.cost_rate

        # Per-path quantities, computed once with the Phase 1 code.
        self._deltas = deltas(scenario, path_set)  # (n_paths, T)
        self._marks = option_marks(scenario, path_set)  # (n_paths, T + 1)
        ref_vol = (
            scenario.market["sigma"] if scenario.model == "gbm" else scenario.market["theta"] ** 0.5
        )
        self._ref_vol = float(ref_vol)
        self._moneyness_scale = 3.0 * self._ref_vol * np.sqrt(scenario.maturity)

        bounded = {"tau", "holding", "bs_delta"}
        high = np.array([1.0 if f in bounded else OBS_CLIP for f in features], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

        self._next_index = 0
        self._path = -1
        self._t = 0
        self._holding = 0.0
        self._episode_pnl = 0.0

    # Mapping between actions and holdings
    def _scale(self, holding):
        """Holding in [h_low, h_high] -> [-1, 1]."""
        return 2.0 * (holding - self.h_low) / (self.h_high - self.h_low) - 1.0

    def action_to_holding(self, action, bs_delta: float) -> float:
        a = float(np.clip(np.asarray(action, dtype=float).reshape(-1)[0], -1.0, 1.0))
        if self.action_mode == "holding":
            return self.h_low + 0.5 * (a + 1.0) * (self.h_high - self.h_low)
        return float(np.clip(bs_delta + a * self.max_offset, self.h_low, self.h_high))

    def holding_to_action(self, holding: float, bs_delta: float) -> np.ndarray:
        """The action that produces ``holding`` (float64, so scripted agents are exact)."""
        if self.action_mode == "holding":
            return np.array([self._scale(holding)])
        return np.array([(holding - bs_delta) / self.max_offset])

    @property
    def current_delta(self) -> float:
        """BS delta at the current step: what the scripted delta agent holds."""
        return float(self._deltas[self._path, self._t])

    # Batched versions, used to evaluate a policy on a whole path set at once.
    # tests/test_train.py checks they agree with stepping the env path by path.
    def batch_observation(self, t: int, holdings: np.ndarray) -> np.ndarray:
        """Observations of every path at step ``t`` given current holdings, shape (n_paths, d)."""
        prices = self.path_set.prices
        columns = []
        for feature in self.obs_features:
            if feature == "tau":
                columns.append(np.full(len(self.path_set), 1.0 - 2.0 * t / self.n_steps))
            elif feature == "log_moneyness":
                columns.append(np.log(prices[:, t] / self.scenario.strike) / self._moneyness_scale)
            elif feature == "holding":
                columns.append(self._scale(holdings))
            elif feature == "bs_delta":
                columns.append(self._scale(self._deltas[:, min(t, self.n_steps - 1)]))
            else:
                columns.append(np.sqrt(self.path_set.variances[:, t]) / self._ref_vol - 1.0)
        obs = np.stack(columns, axis=1).astype(np.float32)
        return np.clip(obs, -OBS_CLIP, OBS_CLIP)

    def batch_holdings(self, t: int, actions: np.ndarray) -> np.ndarray:
        """Holdings of every path at step ``t`` for a batch of actions, shape (n_paths,)."""
        a = np.clip(np.asarray(actions, dtype=float).reshape(len(self.path_set)), -1.0, 1.0)
        if self.action_mode == "holding":
            return self.h_low + 0.5 * (a + 1.0) * (self.h_high - self.h_low)
        return np.clip(self._deltas[:, t] + a * self.max_offset, self.h_low, self.h_high)

    # Gymnasium API
    def _observation(self) -> np.ndarray:
        p, t = self._path, min(self._t, self.n_steps)
        values = []
        for feature in self.obs_features:
            if feature == "tau":
                values.append(1.0 - 2.0 * t / self.n_steps)
            elif feature == "log_moneyness":
                spot = self.path_set.prices[p, t]
                values.append(np.log(spot / self.scenario.strike) / self._moneyness_scale)
            elif feature == "holding":
                values.append(self._scale(self._holding))
            elif feature == "bs_delta":
                # At expiry there is no decision left; report the last delta.
                values.append(self._scale(self._deltas[p, min(t, self.n_steps - 1)]))
            else:
                values.append(np.sqrt(self.path_set.variances[p, t]) / self._ref_vol - 1.0)
        return np.clip(np.asarray(values, dtype=np.float32), -OBS_CLIP, OBS_CLIP)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        n_paths = len(self.path_set)
        if options and "path_index" in options:
            self._path = int(options["path_index"]) % n_paths
        elif self.mode == "train":
            self._path = int(self.np_random.integers(n_paths))
        else:
            self._path = self._next_index % n_paths
            self._next_index += 1
        self._t = 0
        self._holding = 0.0
        self._episode_pnl = 0.0
        info = {"path_index": self._path, "seed": int(self.path_set.seeds[self._path])}
        return self._observation(), info

    def step(self, action):
        if self._path < 0 or self._t >= self.n_steps:
            raise RuntimeError("call reset() before step(), and after an episode ends")
        p, t = self._path, self._t
        prices, marks = self.path_set.prices, self._marks
        holding = self.action_to_holding(action, self._deltas[p, t])
        done = t == self.n_steps - 1
        gross, cost = step_components(
            prices[p, t],
            prices[p, t + 1],
            self._holding,
            holding,
            marks[p, t],
            marks[p, t + 1],
            self.cost_rate,
            self.premium - marks[p, 0] if t == 0 else 0.0,
            closing=done,
        )
        gross, cost = float(gross), float(cost)
        self._episode_pnl += gross - cost
        self._holding = holding
        self._t += 1
        reward = compute_reward(self.reward_cfg, gross, cost, done, self._episode_pnl)
        info = {"holding": holding, "step_pnl": gross - cost, "step_cost": cost}
        if done:
            info["episode_pnl"] = self._episode_pnl
            info["path_index"] = p
        return self._observation(), float(reward), done, False, info


def make_env_fn(scenario: Scenario, path_set: PathSet, rank: int = 0, **kwargs):
    """A picklable factory for ``DummyVecEnv`` / ``SubprocVecEnv``.

    Each copy gets its own seed (``rank``), so parallel training envs draw different paths.
    """

    def _init() -> HedgingEnv:
        env = HedgingEnv(scenario, path_set, **kwargs)
        env.reset(seed=rank)
        return env

    return _init
