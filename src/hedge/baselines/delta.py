"""Black-Scholes delta hedge, rebalanced at every step."""

from __future__ import annotations

import numpy as np

from hedge.market import black_scholes as bs


def bs_delta_holdings(
    paths: np.ndarray,
    strike: float,
    dt: float,
    sigma,
    r: float = 0.0,
    kind: str = "call",
) -> np.ndarray:
    """Holdings that offset a short option: the option's Black-Scholes delta at each step.

    ``sigma`` is a number (GBM: the true volatility) or an array shaped like ``paths``
    (Heston: the square root of the current variance, which makes this the misspecified hedge).
    Returns shape (n_paths, T).
    """
    paths = np.atleast_2d(paths)
    n_steps = paths.shape[1] - 1
    tau = dt * (n_steps - np.arange(n_steps))  # time to expiry at t = 0 .. T-1
    sigma = np.asarray(sigma, dtype=float)
    if sigma.ndim == 2:
        sigma = sigma[:, :-1]
    return bs.delta(paths[:, :-1], strike, tau, sigma, r, kind)
