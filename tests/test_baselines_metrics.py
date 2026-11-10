import numpy as np
import pytest

from hedge.baselines.no_trade_band import no_trade_band_holdings
from hedge.baselines.run import deltas, run_baselines, tune_band
from hedge.eval.metrics import cvar, summarise, var
from hedge.market import black_scholes as bs
from hedge.market import pathsets


def test_cvar_by_hand():
    costs = np.arange(1.0, 101.0)  # 1 .. 100
    assert cvar(costs, 0.95) == pytest.approx(np.mean([96, 97, 98, 99, 100]))
    assert cvar(costs, 0.0) == pytest.approx(costs.mean())
    assert cvar(costs, 0.95) >= var(costs, 0.95) >= np.median(costs)
    assert cvar(np.arange(10_000.0), 0.95) == pytest.approx(np.arange(9_500, 10_000).mean())


def test_band_of_zero_is_delta_and_wide_band_never_trades():
    rng = np.random.default_rng(0)
    target = rng.uniform(0.2, 0.8, size=(20, 30))
    np.testing.assert_array_equal(no_trade_band_holdings(target, 0.0), target)
    np.testing.assert_array_equal(no_trade_band_holdings(target, 0.0, "edge"), target)
    np.testing.assert_array_equal(no_trade_band_holdings(target, 1.0), np.zeros_like(target))


@pytest.mark.parametrize("mode", ["delta", "edge"])
def test_band_stays_within_width_and_trades_less(mode):
    rng = np.random.default_rng(1)
    target = np.clip(0.5 + np.cumsum(rng.normal(0, 0.03, size=(50, 30)), axis=1), 0, 1)
    held = no_trade_band_holdings(target, 0.05, mode)
    assert (np.abs(held - target) <= 0.05 + 1e-12).all()
    turnover = lambda h: np.abs(np.diff(h, axis=1, prepend=0.0)).sum()  # noqa: E731
    assert turnover(held) < turnover(target)


def test_band_by_hand():
    target = np.array([[0.50, 0.53, 0.58, 0.55, 0.40]])
    np.testing.assert_allclose(
        no_trade_band_holdings(target, 0.05, "delta"), [[0.50, 0.50, 0.58, 0.58, 0.40]]
    )
    np.testing.assert_allclose(
        no_trade_band_holdings(target, 0.05, "edge"), [[0.45, 0.48, 0.53, 0.53, 0.45]]
    )


def test_delta_holdings_use_time_to_expiry(gbm_scenario):
    ps = pathsets.simulate(gbm_scenario, np.arange(5))
    d = deltas(gbm_scenario, ps)
    assert d.shape == (5, 30)
    assert d[0, 0] == pytest.approx(float(bs.delta(100.0, 100.0, 30 / 252, 0.2)))
    assert d[0, -1] == pytest.approx(float(bs.delta(ps.prices[0, 29], 100.0, 1 / 252, 0.2)))


def test_heston_delta_uses_current_variance(heston_scenario):
    ps = pathsets.simulate(heston_scenario, np.arange(5))
    d = deltas(heston_scenario, ps)
    expected = bs.delta(ps.prices[2, 7], 100.0, 23 / 252, np.sqrt(ps.variances[2, 7]))
    assert d[2, 7] == pytest.approx(float(expected))


@pytest.mark.parametrize("fixture", ["gbm_scenario", "heston_scenario"])
def test_baselines_end_to_end(fixture, request):
    scenario = request.getfixturevalue(fixture)
    ps = pathsets.simulate(scenario, np.arange(3_000))
    band, grid = tune_band(scenario, ps, [0.0, 0.02, 0.05], ["delta", "edge"])
    assert len(grid) == 6
    results = {k: summarise(v) for k, v in run_baselines(scenario, ps, band).items()}
    # Hedging cuts risk a lot; the tuned band is at least as good as delta (width 0 is in the grid).
    assert results["bs_delta"]["std_cost"] < 0.5 * results["no_hedge"]["std_cost"]
    assert results["bs_delta"]["cvar95"] < results["no_hedge"]["cvar95"]
    assert results["no_trade_band"]["cvar95"] <= results["bs_delta"]["cvar95"] + 1e-12
    assert results["no_hedge"]["mean_turnover"] == 0.0
    # With zero drift the premium is fair, so the unhedged mean cost is about zero.
    assert abs(results["no_hedge"]["mean_cost"]) < 0.25
