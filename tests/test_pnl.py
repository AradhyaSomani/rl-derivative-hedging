import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from hedge.baselines.delta import bs_delta_holdings
from hedge.baselines.no_hedge import no_hedge_holdings
from hedge.market import black_scholes as bs
from hedge.market.gbm import simulate_gbm
from hedge.pnl import cash_ledger, hedge_pnl

T_YEARS = 30 / 252


def test_no_hedge_is_premium_minus_payoff_exactly():
    paths = simulate_gbm(np.arange(1_000), 30, 1 / 252)
    for kind in ("call", "put"):
        result = hedge_pnl(paths, no_hedge_holdings(paths), 100.0, 2.75, cost_rate=0.001, kind=kind)
        intrinsic = paths[:, -1] - 100.0 if kind == "call" else 100.0 - paths[:, -1]
        np.testing.assert_array_equal(result.pnl, 2.75 - np.maximum(intrinsic, 0.0))
        assert (result.txn_cost == 0).all() and (result.turnover == 0).all()
        assert (result.hedge_gain == 0).all()


def test_hand_computed_path():
    # Two steps. Buy 0.5 at 100, go to 0.8 at 110, close at 105. Costs at 1%.
    result = hedge_pnl(
        [[100.0, 110.0, 105.0]], [[0.5, 0.8]], strike=100.0, premium=4.0, cost_rate=0.01
    )
    assert result.payoff[0] == pytest.approx(5.0)
    assert result.hedge_gain[0] == pytest.approx(0.5 * 10 + 0.8 * -5)  # 1.0
    assert result.txn_cost[0] == pytest.approx(0.01 * (100 * 0.5 + 110 * 0.3 + 105 * 0.8))  # 1.67
    assert result.turnover[0] == pytest.approx(0.5 + 0.3 + 0.8)
    assert result.pnl[0] == pytest.approx(4.0 - 5.0 + 1.0 - 1.67)
    assert result.cost[0] == pytest.approx(1.67)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_frequent_costless_delta_hedge_has_near_zero_pnl(kind):
    """Design rule 1: with no costs and 1,000 rebalances, delta hedging replicates the option."""
    sigma, n_steps = 0.2, 1_000
    dt = T_YEARS / n_steps
    # A non-zero drift on purpose: replication must not depend on where the price is heading.
    paths = simulate_gbm(np.arange(2_000), n_steps, dt, mu=0.1, sigma=sigma)
    premium = float(bs.price(100.0, 100.0, T_YEARS, sigma, kind=kind))
    holdings = bs_delta_holdings(paths, 100.0, dt, sigma, kind=kind)
    result = hedge_pnl(paths, holdings, 100.0, premium, cost_rate=0.0, kind=kind)

    unhedged = hedge_pnl(paths, no_hedge_holdings(paths), 100.0, premium, kind=kind)
    assert result.pnl.std() < 0.03 * premium
    assert result.pnl.std() < 0.03 * unhedged.pnl.std()
    assert abs(result.pnl.mean()) < 4 * result.pnl.std() / np.sqrt(len(paths))


def test_hedging_error_shrinks_like_one_over_root_n():
    sigma, stds = 0.2, []
    for n_steps in (25, 100, 400):
        dt = T_YEARS / n_steps
        paths = simulate_gbm(np.arange(4_000), n_steps, dt, sigma=sigma)
        holdings = bs_delta_holdings(paths, 100.0, dt, sigma)
        premium = float(bs.price(100.0, 100.0, T_YEARS, sigma))
        stds.append(hedge_pnl(paths, holdings, 100.0, premium).pnl.std())
    # Four times the steps should about halve the error.
    assert stds[0] / stds[1] == pytest.approx(2.0, rel=0.2)
    assert stds[1] / stds[2] == pytest.approx(2.0, rel=0.2)


def test_costs_only_ever_reduce_pnl():
    paths = simulate_gbm(np.arange(500), 30, 1 / 252)
    holdings = bs_delta_holdings(paths, 100.0, 1 / 252, 0.2)
    free = hedge_pnl(paths, holdings, 100.0, 2.75, cost_rate=0.0)
    costly = hedge_pnl(paths, holdings, 100.0, 2.75, cost_rate=0.001)
    assert (costly.pnl < free.pnl).all()
    np.testing.assert_allclose(free.pnl - costly.pnl, costly.txn_cost)
    # Cost is linear in the rate.
    double = hedge_pnl(paths, holdings, 100.0, 2.75, cost_rate=0.002)
    np.testing.assert_allclose(double.txn_cost, 2 * costly.txn_cost)


path_strategy = st.integers(2, 40).flatmap(
    lambda n: st.tuples(
        arrays(float, n + 1, elements=st.floats(1.0, 500.0)),
        arrays(float, n, elements=st.floats(-2.0, 2.0)),
    )
)


@given(
    path_strategy,
    st.floats(1.0, 500.0),
    st.floats(0.0, 50.0),
    st.floats(0.0, 0.05),
    st.sampled_from(["call", "put"]),
)
@settings(max_examples=200, deadline=None)
def test_engine_matches_cash_account(path_and_holdings, strike, premium, cost_rate, kind):
    """The closed-form engine equals a step-by-step cash and stock ledger on any path."""
    path, holdings = path_and_holdings
    result = hedge_pnl(path[None, :], holdings[None, :], strike, premium, cost_rate, kind)
    ledger = cash_ledger(path, holdings, strike, premium, cost_rate, kind)
    assert ledger["holding"].iloc[-1] == 0.0
    assert result.pnl[0] == pytest.approx(ledger["cash"].iloc[-1], abs=1e-7)
    assert result.txn_cost[0] == pytest.approx(ledger["txn_cost"].sum(), abs=1e-9)
    assert result.turnover[0] == pytest.approx(ledger["trade"].abs().sum(), abs=1e-9)
    # The identity stored in the database: pnl = premium - payoff + hedge_gain - txn_cost.
    assert result.pnl[0] == pytest.approx(
        result.premium[0] - result.payoff[0] + result.hedge_gain[0] - result.txn_cost[0]
    )


def test_shape_checks():
    with pytest.raises(ValueError):
        hedge_pnl(np.ones((3, 31)), np.ones((3, 31)), 100.0, 1.0)
    with pytest.raises(ValueError):
        hedge_pnl(np.ones((3, 31)), np.ones((3, 30)), 100.0, 1.0, cost_rate=-0.1)
