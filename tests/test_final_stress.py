import numpy as np
import pytest

from hedge.eval.final import NotFrozenError, bootstrap_cvar_gap, final_evaluation, seed_table
from hedge.eval.metrics import cvar
from hedge.eval.stress import add_jumps, stressed_paths
from hedge.market import pathsets
from hedge.seeds import TestSeedError


def test_bootstrap_gap():
    rng = np.random.default_rng(0)
    reference = rng.normal(0.0, 1.0, size=5_000)
    worse = reference * 1.5  # a fatter tail on the same paths
    gap, lo, hi = bootstrap_cvar_gap(worse, reference, 300, seed=1)
    assert gap == pytest.approx(cvar(worse) - cvar(reference))
    assert 0 < lo < gap < hi
    # A strategy identical to the reference has a gap of exactly zero in every resample.
    assert bootstrap_cvar_gap(reference, reference, 50, seed=1) == (0.0, 0.0, 0.0)
    # Unrelated noise of the same size: the interval should cover zero.
    other = reference + rng.normal(0.0, 0.01, size=5_000)
    _, lo, hi = bootstrap_cvar_gap(other, reference, 300, seed=1)
    assert lo < 0 < hi


def test_seed_table_takes_medians():
    import pandas as pd

    rows = pd.DataFrame(
        {
            "scenario": ["a"] * 3,
            "strategy": ["ppo"] * 3,
            "seed": [0, 1, 2],
            "mean_cost": [0.1, 0.2, 0.3],
            "std_cost": [1.0, 1.0, 1.0],
            "var95": [1.0, 1.0, 1.0],
            "cvar95": [1.5, 1.2, 1.9],
            "mean_txn_cost": [0.1, 0.1, 0.1],
            "mean_turnover": [2.0, 2.0, 2.0],
            "cvar_vs_delta": [0.5, 0.2, 0.9],
            "cvar_vs_delta_lo": [0.4, 0.1, 0.8],
            "cvar_vs_delta_hi": [0.6, 0.3, 1.0],
        }
    )
    out = seed_table(rows, ["scenario", "strategy"]).iloc[0]
    assert out["n_seeds"] == 3 and out["cvar95"] == 1.5
    assert (out["cvar95_min"], out["cvar95_max"]) == (1.2, 1.9)
    assert out["cvar_vs_delta"] == 0.5  # the CI of the median-CVaR seed


def test_final_test_run_is_guarded(tmp_path, monkeypatch):
    files = ["config/scenarios/gbm_10bps.yaml"]
    with pytest.raises(TestSeedError):
        final_evaluation(files, split="test", results_dir=tmp_path)
    # Even with permission, it refuses until the agents-frozen tag exists.
    monkeypatch.chdir(tmp_path)  # not a git repository, so no tag
    with pytest.raises(NotFrozenError):
        final_evaluation(files, split="test", allow_test=True, results_dir=tmp_path)


def test_jumps(gbm_scenario):
    ps = pathsets.simulate(gbm_scenario, np.arange(2_000))
    jumped = add_jumps(ps.prices, ps.seeds, gbm_scenario.dt, intensity=2.0, mean=-0.05, std=0.08)
    assert jumped.shape == ps.prices.shape
    np.testing.assert_array_equal(jumped[:, 0], ps.prices[:, 0])
    changed = (np.abs(jumped[:, -1] / ps.prices[:, -1] - 1) > 1e-12).mean()
    expected = 1 - np.exp(-2.0 * 30 / 252)  # chance of at least one jump in 30 days
    assert changed == pytest.approx(expected, abs=0.03)
    again = add_jumps(ps.prices, ps.seeds, gbm_scenario.dt, intensity=2.0, mean=-0.05, std=0.08)
    np.testing.assert_array_equal(jumped, again)
    none = add_jumps(ps.prices, ps.seeds, gbm_scenario.dt, intensity=0.0, mean=-0.05, std=0.08)
    np.testing.assert_array_equal(none, ps.prices)


def test_stressed_paths():
    base, up = stressed_paths(
        "vol_up",
        {"base": "config/scenarios/gbm_10bps.yaml", "market": {"sigma": 0.30}},
        "validation",
    )
    clean = pathsets.simulate(base, up.seeds[:500])
    assert base.market["sigma"] == 0.2  # the hedger's beliefs are unchanged
    ratio = np.log(up.prices[:500, -1] / 100).std() / np.log(clean.prices[:, -1] / 100).std()
    assert ratio == pytest.approx(1.5, rel=0.02)
    np.testing.assert_array_equal(up.seeds, np.arange(20_000, 30_000))

    base, swapped = stressed_paths(
        "model_swap",
        {
            "base": "config/scenarios/gbm_10bps.yaml",
            "paths_from": "config/scenarios/heston_10bps.yaml",
        },
        "validation",
    )
    assert base.model == "gbm" and swapped.variances is None

    with pytest.raises(TestSeedError):
        stressed_paths("x", {"base": "config/scenarios/gbm_10bps.yaml", "jumps": {}}, "test")
