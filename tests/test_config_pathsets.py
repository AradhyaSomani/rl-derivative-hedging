import numpy as np
import pytest

from hedge.config import Scenario, config_hash
from hedge.market import black_scholes as bs
from hedge.market import pathsets
from hedge.seeds import TestSeedError

from .conftest import SCENARIO_DIR


def test_all_scenarios_load():
    files = sorted(SCENARIO_DIR.glob("*.yaml"))
    assert len(files) == 8
    for f in files:
        scenario = Scenario.from_yaml(f)
        assert scenario.name == f.stem
        assert scenario.n_steps == 30
        assert scenario.premium() > 0


def test_derived_quantities(gbm_scenario):
    assert gbm_scenario.strike == 100.0
    assert gbm_scenario.dt == pytest.approx(1 / 252)
    assert gbm_scenario.cost_rate == pytest.approx(0.001)
    assert gbm_scenario.premium() == pytest.approx(float(bs.price(100, 100, 30 / 252, 0.2)))


def test_hashes():
    a = Scenario.from_yaml(SCENARIO_DIR / "gbm_5bps.yaml")
    b = Scenario.from_yaml(SCENARIO_DIR / "gbm_20bps.yaml")
    h = Scenario.from_yaml(SCENARIO_DIR / "heston_5bps.yaml")
    assert a.hash != b.hash  # different scenarios
    assert a.paths_hash == b.paths_hash  # same market, so the same paths
    assert a.paths_hash != h.paths_hash
    assert config_hash({"x": 1, "y": 2}) == config_hash({"y": 2, "x": 1})


def test_nonzero_rate_is_rejected(gbm_scenario):
    raw = {
        "name": "x",
        "market": {**gbm_scenario.market, "r": 0.05},
        "option": gbm_scenario.option,
        "grid": gbm_scenario.grid,
        "cost_bps": 10,
    }
    with pytest.raises(ValueError):
        Scenario.from_dict(raw)


@pytest.mark.parametrize("fixture", ["gbm_scenario", "heston_scenario"])
def test_parquet_round_trip(fixture, request, tmp_path):
    scenario = request.getfixturevalue(fixture)
    file = pathsets.build(scenario, "validation", tmp_path, n_paths=64)
    assert scenario.paths_hash in file.name and file.name.endswith("_validation.parquet")
    loaded = pathsets.load(file)
    fresh = pathsets.simulate(scenario, np.arange(20_000, 20_064))
    np.testing.assert_array_equal(loaded.seeds, fresh.seeds)
    np.testing.assert_array_equal(loaded.prices, fresh.prices)
    if scenario.model == "heston":
        np.testing.assert_array_equal(loaded.variances, fresh.variances)
    else:
        assert loaded.variances is None


def test_test_split_needs_explicit_permission(gbm_scenario, tmp_path):
    with pytest.raises(TestSeedError):
        pathsets.build(gbm_scenario, "test", tmp_path, n_paths=8)
    file = pathsets.build(gbm_scenario, "test", tmp_path, n_paths=8, allow_test=True)
    with pytest.raises(TestSeedError):
        pathsets.load(file)
    assert len(pathsets.load(file, allow_test=True)) == 8
