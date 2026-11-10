"""Geometric Brownian motion paths (the Black-Scholes world)."""

from __future__ import annotations

import numpy as np

from hedge.seeds import guard_test_seeds, standard_normals


def simulate_gbm(
    seeds,
    n_steps: int,
    dt: float,
    s0: float = 100.0,
    mu: float = 0.0,
    sigma: float = 0.2,
    *,
    allow_test: bool = False,
) -> np.ndarray:
    """Simulate one price path per seed.

    Returns an array of shape (len(seeds), n_steps + 1); column 0 is ``s0``. The scheme is
    exact: log returns are normal with mean ``(mu - sigma^2 / 2) dt`` and variance
    ``sigma^2 dt``, so the step size adds no bias.
    """
    guard_test_seeds(seeds, allow_test)
    z = standard_normals(seeds, n_steps)
    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    log_paths = np.concatenate([np.zeros((z.shape[0], 1)), np.cumsum(log_returns, axis=1)], axis=1)
    return s0 * np.exp(log_paths)
