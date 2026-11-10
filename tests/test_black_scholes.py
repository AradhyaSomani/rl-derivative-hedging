import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hedge.market import black_scholes as bs

spots = st.floats(50.0, 200.0)
strikes = st.floats(50.0, 200.0)
taus = st.floats(0.01, 2.0)
sigmas = st.floats(0.05, 0.8)
rates = st.floats(0.0, 0.1)


def test_reference_values():
    # Standard textbook case: S = K = 100, one year, 20% volatility, 5% rate.
    assert bs.price(100, 100, 1.0, 0.2, 0.05, "call") == pytest.approx(10.450583572, abs=1e-6)
    assert bs.price(100, 100, 1.0, 0.2, 0.05, "put") == pytest.approx(5.573526022, abs=1e-6)
    assert bs.delta(100, 100, 1.0, 0.2, 0.05, "call") == pytest.approx(0.636830651, abs=1e-6)


@given(spots, strikes, taus, sigmas, rates)
def test_put_call_parity(s, k, tau, sigma, r):
    call = bs.price(s, k, tau, sigma, r, "call")
    put = bs.price(s, k, tau, sigma, r, "put")
    assert call - put == pytest.approx(s - k * np.exp(-r * tau), abs=1e-9)
    assert bs.delta(s, k, tau, sigma, r, "call") - bs.delta(s, k, tau, sigma, r, "put") == (
        pytest.approx(1.0, abs=1e-12)
    )


@given(spots, strikes, taus, sigmas, rates, st.sampled_from(["call", "put"]))
@settings(max_examples=200)
def test_greeks_match_finite_differences(s, k, tau, sigma, r, kind):
    h = 1e-3
    up, down = bs.price(s + h, k, tau, sigma, r, kind), bs.price(s - h, k, tau, sigma, r, kind)
    mid = bs.price(s, k, tau, sigma, r, kind)
    assert bs.delta(s, k, tau, sigma, r, kind) == pytest.approx((up - down) / (2 * h), abs=1e-5)
    assert bs.gamma(s, k, tau, sigma, r) == pytest.approx((up - 2 * mid + down) / h**2, abs=1e-4)
    e = 1e-5
    vega_fd = bs.price(s, k, tau, sigma + e, r, kind) - bs.price(s, k, tau, sigma - e, r, kind)
    assert bs.vega(s, k, tau, sigma, r) == pytest.approx(vega_fd / (2 * e), abs=1e-4)


@given(spots, strikes, taus, sigmas, rates)
def test_bounds(s, k, tau, sigma, r):
    call = bs.price(s, k, tau, sigma, r, "call")
    assert max(s - k * np.exp(-r * tau), 0.0) - 1e-9 <= call <= s
    assert 0.0 <= bs.delta(s, k, tau, sigma, r, "call") <= 1.0
    assert -1.0 <= bs.delta(s, k, tau, sigma, r, "put") <= 0.0
    assert bs.gamma(s, k, tau, sigma, r) >= 0.0


def test_expiry_is_the_payoff():
    s = np.array([90.0, 100.0, 110.0])
    np.testing.assert_allclose(bs.price(s, 100.0, 0.0, 0.2, kind="call"), [0.0, 0.0, 10.0])
    np.testing.assert_allclose(bs.price(s, 100.0, 0.0, 0.2, kind="put"), [10.0, 0.0, 0.0])
    np.testing.assert_allclose(bs.delta(s, 100.0, 0.0, 0.2, kind="call"), [0.0, 0.0, 1.0])
    np.testing.assert_allclose(bs.delta(s, 100.0, 0.0, 0.2, kind="put"), [-1.0, 0.0, 0.0])
    assert not np.isnan(bs.gamma(s, 100.0, 0.0, 0.2)).any()
    assert not np.isnan(bs.vega(s, 100.0, 0.0, 0.2)).any()


def test_broadcasting():
    s = np.linspace(80, 120, 5)[:, None]
    tau = np.linspace(0.0, 1.0, 4)[None, :]
    assert bs.price(s, 100.0, tau, 0.2).shape == (5, 4)
    assert bs.delta(s, 100.0, tau, 0.2, kind="put").shape == (5, 4)


def test_unknown_kind():
    with pytest.raises(ValueError):
        bs.price(100, 100, 1.0, 0.2, kind="straddle")
