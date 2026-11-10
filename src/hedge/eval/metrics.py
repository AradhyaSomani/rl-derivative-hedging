"""Risk metrics on hedging cost (minus the hedging P&L). Higher cost is worse."""

from __future__ import annotations

import numpy as np

from hedge.pnl import HedgeResult


def var(costs, level: float = 0.95) -> float:
    """Value at risk: the ``level`` quantile of cost."""
    return float(np.quantile(np.asarray(costs), level))


def cvar(costs, level: float = 0.95) -> float:
    """Expected shortfall: the mean of the worst ``1 - level`` share of costs.

    With n samples it averages the ``ceil(n * (1 - level))`` largest, so CVaR 95% on 10,000
    paths is the mean of the worst 500.
    """
    costs = np.sort(np.asarray(costs, dtype=float))
    n_tail = max(1, int(np.ceil(costs.size * (1.0 - level) - 1e-9)))
    return float(costs[-n_tail:].mean())


def summarise(result: HedgeResult) -> dict[str, float]:
    """The metrics of ``eval_metrics`` in the database schema, for one strategy on one path set."""
    cost = result.cost
    return {
        "n_paths": int(cost.size),
        "mean_cost": float(cost.mean()),
        "std_cost": float(cost.std(ddof=1)),
        "var95": var(cost, 0.95),
        "cvar95": cvar(cost, 0.95),
        "mean_txn_cost": float(result.txn_cost.mean()),
        "mean_turnover": float(result.turnover.mean()),
    }
