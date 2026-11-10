"""Heston stochastic-volatility model: simulator and semi-analytic European price.

    dS = mu S dt + sqrt(v) S dW1
    dv = kappa (theta - v) dt + xi sqrt(v) dW2,    corr(dW1, dW2) = rho

Parameters: ``v0`` starting variance, ``kappa`` speed of mean reversion, ``theta`` long-run
variance, ``xi`` volatility of variance, ``rho`` correlation between price and variance shocks.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad

from hedge.seeds import guard_test_seeds, standard_normals


def simulate_heston(
    seeds,
    n_steps: int,
    dt: float,
    s0: float = 100.0,
    mu: float = 0.0,
    v0: float = 0.04,
    kappa: float = 2.0,
    theta: float = 0.04,
    xi: float = 0.3,
    rho: float = -0.7,
    substeps: int = 8,
    *,
    allow_test: bool = False,
    chunk: int = 20_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate one (price, variance) path per seed with Euler and full truncation.

    Each of the ``n_steps`` steps of length ``dt`` is split into ``substeps`` Euler steps; only
    the values at the ends of the full steps are kept. Full truncation: the variance state may
    go negative, but ``max(v, 0)`` is what enters the drift and both diffusion terms.

    Returns ``(prices, variances)``, each of shape (len(seeds), n_steps + 1). The stored
    variance is the truncated one, so it is never negative.
    """
    guard_test_seeds(seeds, allow_test)
    seeds = np.asarray(seeds, dtype=np.int64)
    n = seeds.size
    prices = np.empty((n, n_steps + 1))
    variances = np.empty((n, n_steps + 1))
    h = dt / substeps
    sqrt_h = np.sqrt(h)
    rho_perp = np.sqrt(1.0 - rho**2)
    total = n_steps * substeps

    for lo in range(0, n, chunk):
        sl = slice(lo, min(lo + chunk, n))
        # Per seed: the first `total` draws drive the price, the next `total` the variance.
        z = standard_normals(seeds[sl], 2 * total)
        z1 = z[:, :total]
        z2 = rho * z1 + rho_perp * z[:, total:]
        log_s = np.full(z.shape[0], np.log(s0))
        v = np.full(z.shape[0], float(v0))
        prices[sl, 0] = s0
        variances[sl, 0] = max(v0, 0.0)
        for j in range(total):
            v_pos = np.maximum(v, 0.0)
            root = np.sqrt(v_pos) * sqrt_h
            log_s += (mu - 0.5 * v_pos) * h + root * z1[:, j]
            v += kappa * (theta - v_pos) * h + xi * root * z2[:, j]
            if (j + 1) % substeps == 0:
                k = (j + 1) // substeps
                prices[sl, k] = np.exp(log_s)
                variances[sl, k] = np.maximum(v, 0.0)
    return prices, variances


def _char_fn(u, tau, s0, r, v0, kappa, theta, xi, rho):
    """Characteristic function of log S_T under the pricing measure.

    Uses the formulation of Albrecher et al. (2007), "The little Heston trap", which stays on
    the right branch of the complex logarithm for all maturities.
    """
    iu = 1j * u
    b = kappa - rho * xi * iu
    d = np.sqrt(b**2 + xi**2 * (iu + u**2))
    g = (b - d) / (b + d)
    e = np.exp(-d * tau)
    c = r * iu * tau + kappa * theta / xi**2 * ((b - d) * tau - 2.0 * np.log((1 - g * e) / (1 - g)))
    big_d = (b - d) / xi**2 * (1 - e) / (1 - g * e)
    return np.exp(c + big_d * v0 + iu * np.log(s0))


def heston_price(
    s0: float,
    k: float,
    tau: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    r: float = 0.0,
    kind: str = "call",
) -> float:
    """Semi-analytic Heston price of a European option (Heston 1993, two-probability form)."""
    if kind not in ("call", "put"):
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")
    args = (tau, s0, r, v0, kappa, theta, xi, rho)
    log_k = np.log(k)
    forward = s0 * np.exp(r * tau)

    def integrand_p2(u):
        return (np.exp(-1j * u * log_k) * _char_fn(u, *args) / (1j * u)).real

    def integrand_p1(u):
        return (np.exp(-1j * u * log_k) * _char_fn(u - 1j, *args) / (1j * u * forward)).real

    p1 = 0.5 + quad(integrand_p1, 1e-10, 500.0, limit=2000)[0] / np.pi
    p2 = 0.5 + quad(integrand_p2, 1e-10, 500.0, limit=2000)[0] / np.pi
    call = s0 * p1 - k * np.exp(-r * tau) * p2
    if kind == "call":
        return float(call)
    return float(call - s0 + k * np.exp(-r * tau))  # put-call parity
