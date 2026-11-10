"""The hedging P&L engine. Every strategy, baseline or agent, is scored by ``hedge_pnl``.

Setting: you sold one European option at time 0 for ``premium`` and trade the underlying at
times 0 .. T-1. The interest rate is zero, so cash earns nothing.

    P&L = premium - payoff + sum_{t=0}^{T-1} h_t (S_{t+1} - S_t)
                           - sum_{t=0}^{T} kappa S_t |h_t - h_{t-1}|

with ``h_{-1} = 0`` (you start flat) and ``h_T = 0`` (the position is closed at expiry), so
the cost sum includes the initial trade and the closing trade.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HedgeResult:
    """Per-path results, each of shape (n_paths,).

    pnl = premium - payoff + hedge_gain - txn_cost
    """

    premium: np.ndarray
    payoff: np.ndarray
    hedge_gain: np.ndarray
    txn_cost: np.ndarray
    pnl: np.ndarray
    turnover: np.ndarray  # shares traded, including the opening and closing trades

    @property
    def cost(self) -> np.ndarray:
        """Hedging cost, the quantity all risk metrics are computed on: minus the P&L."""
        return -self.pnl

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "premium": self.premium,
                "payoff": self.payoff,
                "hedge_gain": self.hedge_gain,
                "txn_cost": self.txn_cost,
                "pnl": self.pnl,
                "turnover": self.turnover,
            }
        )


def option_payoff(s_T, strike: float, kind: str = "call") -> np.ndarray:
    if kind == "call":
        return np.maximum(np.asarray(s_T) - strike, 0.0)
    if kind == "put":
        return np.maximum(strike - np.asarray(s_T), 0.0)
    raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")


def _trades(holdings: np.ndarray) -> np.ndarray:
    """Trades at t = 0 .. T, shape (n_paths, T + 1): open from flat, rebalance, close at expiry."""
    flat = np.zeros((holdings.shape[0], 1))
    return np.diff(holdings, axis=1, prepend=flat, append=flat)


def hedge_pnl(
    paths: np.ndarray,
    holdings: np.ndarray,
    strike: float,
    premium: float,
    cost_rate: float = 0.0,
    kind: str = "call",
) -> HedgeResult:
    """Score a hedging strategy on a set of paths.

    paths:     prices, shape (n_paths, T + 1), column t is S_t.
    holdings:  shares held over each step, shape (n_paths, T); column t is the position chosen
               at time t (knowing S_t) and held until t + 1.
    cost_rate: proportional cost as a fraction of traded value (10 bps = 0.001).
    """
    paths = np.atleast_2d(np.asarray(paths, dtype=float))
    holdings = np.atleast_2d(np.asarray(holdings, dtype=float))
    if holdings.shape != (paths.shape[0], paths.shape[1] - 1):
        raise ValueError(
            f"holdings must have shape (n_paths, T) = {(paths.shape[0], paths.shape[1] - 1)}, "
            f"got {holdings.shape}"
        )
    if cost_rate < 0:
        raise ValueError("cost_rate must be non-negative")

    abs_trades = np.abs(_trades(holdings))
    payoff = option_payoff(paths[:, -1], strike, kind)
    hedge_gain = np.sum(holdings * np.diff(paths, axis=1), axis=1)
    txn_cost = cost_rate * np.sum(paths * abs_trades, axis=1)
    premium_arr = np.full(paths.shape[0], float(premium))
    return HedgeResult(
        premium=premium_arr,
        payoff=payoff,
        hedge_gain=hedge_gain,
        txn_cost=txn_cost,
        pnl=premium_arr - payoff + hedge_gain - txn_cost,
        turnover=abs_trades.sum(axis=1),
    )


def cash_ledger(
    path,
    holdings,
    strike: float,
    premium: float,
    cost_rate: float = 0.0,
    kind: str = "call",
) -> pd.DataFrame:
    """Step-by-step cash account for one path, the way you would do it in a spreadsheet.

    This is a deliberately separate, slow implementation: it tracks cash and stock instead of
    using the closed-form sum. ``tests/test_pnl.py`` checks that its final cash equals
    ``hedge_pnl`` on the same path. One row per time t = 0 .. T; the last row closes the stock
    position and pays the option payoff.
    """
    path = np.asarray(path, dtype=float)
    holdings = np.asarray(holdings, dtype=float)
    n_steps = path.size - 1
    if holdings.shape != (n_steps,):
        raise ValueError(f"holdings must have shape ({n_steps},), got {holdings.shape}")

    rows = []
    cash = float(premium)  # premium received at t = 0
    held = 0.0
    for t in range(n_steps + 1):
        target = holdings[t] if t < n_steps else 0.0
        trade = target - held
        stock_cash_flow = -trade * path[t]  # buying shares costs cash
        cost = cost_rate * path[t] * abs(trade)
        payoff = float(option_payoff(path[t], strike, kind)) if t == n_steps else 0.0
        cash += stock_cash_flow - cost - payoff
        held = target
        rows.append(
            {
                "t": t,
                "spot": path[t],
                "holding": held,
                "trade": trade,
                "stock_cash_flow": stock_cash_flow,
                "txn_cost": cost,
                "payoff_paid": payoff,
                "cash": cash,
                "cash_plus_stock": cash + held * path[t],
            }
        )
    return pd.DataFrame(rows)
