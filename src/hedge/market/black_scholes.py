"""Black-Scholes prices and Greeks for European calls and puts.

Everything is vectorised: any argument may be a numpy array and the usual broadcasting
rules apply. ``tau`` is time to expiry in years. At ``tau == 0`` the price is the payoff,
delta is the exercise indicator, and gamma and vega are zero.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr

_KINDS = ("call", "put")


def _check_kind(kind: str) -> None:
    if kind not in _KINDS:
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")


def _d1_d2(s, k, tau, sigma, r):
    """Return d1, d2 and a mask of the points where they are defined (tau > 0, sigma > 0)."""
    s, k, tau, sigma, r = np.broadcast_arrays(
        *(np.asarray(x, dtype=float) for x in (s, k, tau, sigma, r))
    )
    live = (tau > 0) & (sigma > 0)
    safe_tau = np.where(live, tau, 1.0)
    safe_sigma = np.where(live, sigma, 1.0)
    vol_sqrt_t = safe_sigma * np.sqrt(safe_tau)
    d1 = (np.log(s / k) + (r + 0.5 * safe_sigma**2) * safe_tau) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    return s, k, tau, r, d1, d2, live


def price(s, k, tau, sigma, r=0.0, kind: str = "call"):
    """Option price."""
    _check_kind(kind)
    s, k, tau, r, d1, d2, live = _d1_d2(s, k, tau, sigma, r)
    disc_k = k * np.exp(-r * tau)
    if kind == "call":
        value = s * ndtr(d1) - disc_k * ndtr(d2)
        dead = np.maximum(s - disc_k, 0.0)
    else:
        value = disc_k * ndtr(-d2) - s * ndtr(-d1)
        dead = np.maximum(disc_k - s, 0.0)
    return np.where(live, value, dead)


def delta(s, k, tau, sigma, r=0.0, kind: str = "call"):
    """Sensitivity of the price to the underlying. In [0, 1] for a call, [-1, 0] for a put."""
    _check_kind(kind)
    s, k, tau, r, d1, _, live = _d1_d2(s, k, tau, sigma, r)
    disc_k = k * np.exp(-r * tau)
    if kind == "call":
        return np.where(live, ndtr(d1), (s > disc_k).astype(float))
    return np.where(live, ndtr(d1) - 1.0, -(s < disc_k).astype(float))


def gamma(s, k, tau, sigma, r=0.0):
    """Second derivative of the price in the underlying (same for calls and puts)."""
    s, k, tau, r, d1, _, live = _d1_d2(s, k, tau, sigma, r)
    sigma = np.broadcast_to(np.asarray(sigma, dtype=float), s.shape)
    denom = np.where(live, s * sigma * np.sqrt(np.where(live, tau, 1.0)), 1.0)
    return np.where(live, _pdf(d1) / denom, 0.0)


def vega(s, k, tau, sigma, r=0.0):
    """Derivative of the price in volatility, per unit of sigma (same for calls and puts)."""
    s, k, tau, r, d1, _, live = _d1_d2(s, k, tau, sigma, r)
    return np.where(live, s * _pdf(d1) * np.sqrt(np.where(live, tau, 0.0)), 0.0)


def _pdf(x):
    return np.exp(-0.5 * x**2) / np.sqrt(2.0 * np.pi)
