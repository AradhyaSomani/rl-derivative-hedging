"""Run the three baselines on a scenario's paths, through the one P&L engine."""

from __future__ import annotations

from typing import Any

import numpy as np

from hedge.baselines.delta import bs_delta_holdings
from hedge.baselines.no_hedge import no_hedge_holdings
from hedge.baselines.no_trade_band import no_trade_band_holdings
from hedge.config import Scenario
from hedge.eval.metrics import summarise
from hedge.market.pathsets import PathSet
from hedge.pnl import HedgeResult, hedge_pnl


def baseline_sigma(scenario: Scenario, path_set: PathSet):
    """Volatility the delta hedger feeds Black-Scholes: true sigma or sqrt(current variance)."""
    if scenario.model == "gbm":
        return float(scenario.market["sigma"])
    return np.sqrt(path_set.variances)


def deltas(scenario: Scenario, path_set: PathSet) -> np.ndarray:
    return bs_delta_holdings(
        path_set.prices,
        scenario.strike,
        scenario.dt,
        baseline_sigma(scenario, path_set),
        kind=scenario.kind,
    )


def score(scenario: Scenario, path_set: PathSet, holdings: np.ndarray) -> HedgeResult:
    return hedge_pnl(
        path_set.prices,
        holdings,
        scenario.strike,
        scenario.premium(),
        scenario.cost_rate,
        scenario.kind,
    )


def tune_band(
    scenario: Scenario, path_set: PathSet, widths, modes=("delta", "edge")
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Pick the band (width, rebalance_to) with the lowest CVaR 95%. Use validation paths only.

    Returns the best setting and the whole grid, so the grid can be reported.
    """
    target = deltas(scenario, path_set)
    grid = []
    for mode in modes:
        for width in widths:
            result = score(scenario, path_set, no_trade_band_holdings(target, width, mode))
            grid.append({"width": float(width), "rebalance_to": mode, **summarise(result)})
    best = min(grid, key=lambda row: row["cvar95"])
    return {"width": best["width"], "rebalance_to": best["rebalance_to"]}, grid


def run_baselines(
    scenario: Scenario, path_set: PathSet, band: dict[str, Any]
) -> dict[str, HedgeResult]:
    """Score no hedge, BS delta and the no-trade band on the same paths."""
    target = deltas(scenario, path_set)
    band_holdings = no_trade_band_holdings(target, band["width"], band["rebalance_to"])
    return {
        "no_hedge": score(scenario, path_set, no_hedge_holdings(path_set.prices)),
        "bs_delta": score(scenario, path_set, target),
        "no_trade_band": score(scenario, path_set, band_holdings),
    }
