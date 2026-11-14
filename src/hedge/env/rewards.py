"""Step P&L and the three reward types.

Step P&L uses a mark-to-market view of the short option. With ``M_t`` the option's
Black-Scholes mark at time t (true sigma on GBM, sqrt of the current variance on Heston):

    gross_t = h_t (S_{t+1} - S_t) - (M_{t+1} - M_t)        [+ premium - M_0 at t = 0]
    cost_t  = rate * S_t |h_t - h_{t-1}|                   [+ rate * S_T |h_{T-1}| at the last step]
    pnl_t   = gross_t - cost_t

``M_T`` is the payoff, so the marks telescope and the step P&Ls of an episode sum to exactly
the P&L engine's total, whatever the marks are in between. The marks only decide how the total
is spread over the steps. ``tests/test_env.py`` checks the sum for every reward type.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hedge.config import Scenario
from hedge.market import black_scholes as bs
from hedge.market.pathsets import PathSet

REWARDS = ("quadratic", "mean_variance", "terminal_cvar")


@dataclass(frozen=True)
class RewardConfig:
    """Which reward, and its parameters.

    quadratic:      -gross_t^2 - lam * cost_t
                    Squared hedging error before costs, plus a linear charge for costs.
    mean_variance:  pnl_t - (lam / 2) * pnl_t^2
                    Net step P&L with a risk-aversion penalty.
    terminal_cvar:  0 until expiry, then
                    -L - lam * (threshold + max(L - threshold, 0) / (1 - alpha))
                    where L is the episode's hedging cost. Its expectation is minus (mean cost
                    + lam * an upper bound on CVaR_alpha), tight when ``threshold`` is the VaR
                    (Rockafellar and Uryasev).
    """

    name: str = "quadratic"
    lam: float = 1.0
    alpha: float = 0.95
    threshold: float = 0.0

    def __post_init__(self) -> None:
        if self.name not in REWARDS:
            raise ValueError(f"reward must be one of {REWARDS}, got {self.name!r}")
        if self.lam < 0:
            raise ValueError("lam must be non-negative")
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be in (0, 1)")


def option_marks(scenario: Scenario, path_set: PathSet) -> np.ndarray:
    """Black-Scholes mark of the option, shape (n_paths, T + 1). Column T is the payoff."""
    n_steps = path_set.prices.shape[1] - 1
    tau = scenario.dt * (n_steps - np.arange(n_steps + 1))
    sigma = scenario.market["sigma"] if scenario.model == "gbm" else np.sqrt(path_set.variances)
    return bs.price(path_set.prices, scenario.strike, tau, sigma, 0.0, scenario.kind)


def step_components(
    s_t, s_next, h_prev, h_t, mark_t, mark_next, cost_rate: float, premium_adj, closing: bool
):
    """Gross P&L and cost of one step. Works on numbers or arrays (one entry per path).

    ``premium_adj`` is ``premium - M_0`` at the first step and 0 afterwards. ``closing`` is True
    at the last step, which also pays for unwinding the position at expiry.
    """
    gross = h_t * (s_next - s_t) - (mark_next - mark_t) + premium_adj
    cost = cost_rate * s_t * np.abs(h_t - h_prev)
    if closing:
        cost = cost + cost_rate * s_next * np.abs(h_t)
    return gross, cost


def step_pnls(
    paths: np.ndarray, holdings: np.ndarray, marks: np.ndarray, premium: float, cost_rate: float
) -> tuple[np.ndarray, np.ndarray]:
    """Gross P&L and cost for every step of every path, each of shape (n_paths, T)."""
    n_steps = holdings.shape[1]
    gross = np.empty_like(holdings, dtype=float)
    cost = np.empty_like(holdings, dtype=float)
    h_prev = np.zeros(holdings.shape[0])
    for t in range(n_steps):
        gross[:, t], cost[:, t] = step_components(
            paths[:, t],
            paths[:, t + 1],
            h_prev,
            holdings[:, t],
            marks[:, t],
            marks[:, t + 1],
            cost_rate,
            premium - marks[:, 0] if t == 0 else 0.0,
            closing=t == n_steps - 1,
        )
        h_prev = holdings[:, t]
    return gross, cost


def compute_reward(
    cfg: RewardConfig, gross: float, cost: float, done: bool, episode_pnl: float
) -> float:
    """Reward for one step. ``episode_pnl`` is the running total including this step."""
    if cfg.name == "quadratic":
        return -(gross**2) - cfg.lam * cost
    if cfg.name == "mean_variance":
        pnl = gross - cost
        return pnl - 0.5 * cfg.lam * pnl**2
    if not done:
        return 0.0
    loss = -episode_pnl
    shortfall = cfg.threshold + max(loss - cfg.threshold, 0.0) / (1.0 - cfg.alpha)
    return -loss - cfg.lam * shortfall
