import numpy as np
import pytest

from hedge.market import black_scholes as bs
from hedge.market.heston import heston_price, simulate_heston

DT = 1 / 252
BASE = {"v0": 0.04, "kappa": 2.0, "theta": 0.04, "xi": 0.3, "rho": -0.7}


def test_price_matches_published_reference():
    # Test case from Broadie and Kaya (2006), exact simulation of the Heston model: 6.8061.
    price = heston_price(
        100, 100, 1.0, v0=0.010201, kappa=6.21, theta=0.019, xi=0.61, rho=-0.70, r=0.0319
    )
    assert price == pytest.approx(6.8061, abs=2e-4)


@pytest.mark.parametrize("strike", [90.0, 100.0, 110.0])
def test_price_tends_to_black_scholes_without_vol_of_vol(strike):
    tau = 30 * DT
    heston = heston_price(100, strike, tau, v0=0.04, kappa=2.0, theta=0.04, xi=1e-4, rho=-0.7)
    assert heston == pytest.approx(float(bs.price(100, strike, tau, 0.2)), abs=1e-4)


def test_put_call_parity():
    call = heston_price(100, 105, 0.5, **BASE, r=0.03, kind="call")
    put = heston_price(100, 105, 0.5, **BASE, r=0.03, kind="put")
    assert call - put == pytest.approx(100 - 105 * np.exp(-0.03 * 0.5), abs=1e-10)


def test_shapes_variance_and_reproducibility():
    seeds = np.arange(200)
    s, v = simulate_heston(seeds, 30, DT, **BASE)
    assert s.shape == v.shape == (200, 31)
    assert (s[:, 0] == 100.0).all() and (v[:, 0] == 0.04).all()
    assert (v >= 0).all() and (s > 0).all()
    s2, v2 = simulate_heston(seeds[50:60], 30, DT, **BASE, chunk=3)
    np.testing.assert_array_equal(s[50:60], s2)
    np.testing.assert_array_equal(v[50:60], v2)


def test_full_truncation_survives_a_violated_feller_condition():
    # 2 kappa theta < xi^2: the variance hits zero often. It must stay non-negative and finite.
    s, v = simulate_heston(np.arange(500), 30, DT, v0=0.04, kappa=1.0, theta=0.04, xi=2.0, rho=-0.7)
    assert (v >= 0).all() and np.isfinite(s).all()
    assert (v == 0).any()


def test_reduces_to_gbm_without_vol_of_vol():
    s, v = simulate_heston(
        np.arange(2_000), 30, DT, v0=0.04, kappa=2.0, theta=0.04, xi=0.0, rho=0.0
    )
    np.testing.assert_allclose(v, 0.04)
    log_ret = np.log(s[:, -1] / 100.0)
    tau = 30 * DT
    assert abs(log_ret.mean() + 0.5 * 0.04 * tau) < 4 * np.sqrt(0.04 * tau / 2_000)
    assert log_ret.var(ddof=1) == pytest.approx(0.04 * tau, rel=0.15)


@pytest.mark.slow
def test_monte_carlo_prices_match_semi_analytic_within_one_percent():
    n_paths, n_steps = 60_000, 30
    s, _ = simulate_heston(np.arange(100_000, 100_000 + n_paths), n_steps, DT, **BASE, substeps=8)
    s_T = s[:, -1]
    for strike in (95.0, 100.0, 105.0):
        payoff = np.maximum(s_T - strike, 0.0)
        # Control variate: E[S_T] = S0 exactly when the drift is zero.
        beta = np.cov(payoff, s_T)[0, 1] / s_T.var(ddof=1)
        mc = (payoff - beta * (s_T - 100.0)).mean()
        exact = heston_price(100.0, strike, n_steps * DT, **BASE)
        assert mc == pytest.approx(exact, rel=0.01), f"strike {strike}: MC {mc:.4f} vs {exact:.4f}"
