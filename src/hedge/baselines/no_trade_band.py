"""Delta hedge with a no-trade band: only trade when the holding has drifted too far from delta."""

from __future__ import annotations

import numpy as np

REBALANCE_TO = ("delta", "edge")


def no_trade_band_holdings(
    deltas: np.ndarray, width: float, rebalance_to: str = "delta"
) -> np.ndarray:
    """Apply a no-trade band of half-width ``width`` (in shares) around the target deltas.

    At each step, if the current holding is within ``width`` of delta, do nothing. Otherwise
    trade either all the way to delta (``rebalance_to="delta"``) or just to the nearest edge of
    the band (``"edge"``, the form the transaction-cost literature finds optimal). The position
    starts flat, so the first step trades unless delta is already within the band of zero.
    ``width = 0`` reproduces the plain delta hedge.
    """
    if width < 0:
        raise ValueError("width must be non-negative")
    if rebalance_to not in REBALANCE_TO:
        raise ValueError(f"rebalance_to must be one of {REBALANCE_TO}, got {rebalance_to!r}")
    deltas = np.atleast_2d(np.asarray(deltas, dtype=float))
    holdings = np.empty_like(deltas)
    held = np.zeros(deltas.shape[0])
    for t in range(deltas.shape[1]):
        target = deltas[:, t]
        gap = held - target
        outside = np.abs(gap) > width
        new = target if rebalance_to == "delta" else target + np.sign(gap) * width
        held = np.where(outside, new, held)
        holdings[:, t] = held
    return holdings
