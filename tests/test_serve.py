"""The live computations, the file-based results access and the API's model endpoints.

These use the committed sample in ``app/sample`` (frozen weights and a results extract), so
they run without a database or a training run. They are skipped if the sample is absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

SAMPLE = Path("app/sample")
pytestmark = pytest.mark.skipif(
    not (SAMPLE / "runs").exists(), reason="app/sample not exported (make sample)"
)

pytest.importorskip("torch")
pytest.importorskip("fastapi")

from hedge import results  # noqa: E402
from hedge.eval import live  # noqa: E402
from hedge.market import black_scholes as bs  # noqa: E402
from hedge.seeds import SEED_RANGES  # noqa: E402


@pytest.fixture(autouse=True)
def sample_runs(monkeypatch):
    monkeypatch.setenv("HEDGE_RUNS_DIR", str(SAMPLE / "runs"))


def test_nearest_scenario_picks_closest_cost():
    assert live.nearest_scenario("gbm", 7).name == "gbm_5bps"
    assert live.nearest_scenario("heston", 30).name == "heston_20bps"
    with pytest.raises(ValueError):
        live.nearest_scenario("sabr", 10)


def test_custom_scenario_changes_only_what_was_asked():
    scenario = live.custom_scenario("gbm", 12.0, volatility=0.3)
    assert scenario.cost_bps == 12.0
    assert scenario.market["sigma"] == 0.3
    heston = live.custom_scenario("heston", 10.0, volatility=0.3)
    assert heston.market["v0"] == pytest.approx(0.09)


def test_hedge_decision_matches_bs_delta_and_stays_in_range():
    out = live.hedge_decision(
        spot=102.0, strike=100.0, days_left=12, volatility=0.2, current_holding=0.5, cost_bps=10
    )
    expected = bs.delta(102.0, 100.0, 12 / 252, 0.2, 0.0, "call")
    assert out["bs_delta"] == pytest.approx(float(expected), abs=1e-9)
    assert out["agents"], "no frozen agents were loaded"
    for holding in out["agents"].values():
        assert 0.0 <= holding <= 1.0
        assert abs(holding - out["bs_delta"]) <= 0.25 + 1e-6  # the delta-offset action cap


def test_simulate_uses_dev_seeds_only_and_is_reproducible():
    test_range = SEED_RANGES["test"]
    first = live.simulate_comparison(market="gbm", cost_bps=10, n_paths=200, first_seed=0)
    again = live.simulate_comparison(market="gbm", cost_bps=10, n_paths=200, first_seed=0)
    assert first["strategies"] == again["strategies"]
    # A first seed inside the test range is folded back into the dev range, never used as is.
    folded = live.simulate_comparison(
        market="gbm", cost_bps=10, n_paths=200, first_seed=test_range.start + 5
    )
    assert folded["first_seed"] < SEED_RANGES["dev"].stop
    assert {"No hedge", "BS delta", "No-trade band"} <= set(first["strategies"])
    hedged = first["strategies"]["BS delta"]
    assert hedged["cvar95"] < first["strategies"]["No hedge"]["cvar95"]


def test_results_sample_tables_are_consistent(monkeypatch, tmp_path):
    monkeypatch.setattr(results, "FULL_DIR", tmp_path / "absent")  # force the sample
    summary = results.summary_table()
    assert {"scenario", "strategy", "label", "cvar95", "mean_cost"} <= set(summary.columns)
    delta = summary[summary["strategy"] == "bs_delta"]
    assert len(delta) == summary["scenario"].nunique()
    costs = results.costs("gbm_10bps")
    # Baselines have no training seed, so their per-path costs must reproduce the table.
    for label in ("BS delta", "No-trade band"):
        rows = costs[costs["label"] == label]
        row = summary[(summary["scenario"] == "gbm_10bps") & (summary["label"] == label)]
        assert np.isclose(rows["cost"].mean(), row["mean_cost"].iloc[0], atol=1e-3), label


def test_api_model_endpoints():
    from fastapi.testclient import TestClient

    from hedge.api.main import app

    with TestClient(app) as client:
        health = client.get("/health").json()
        assert health["models"]["ok"], health
        ok = client.post(
            "/hedge",
            json={"spot": 100, "strike": 100, "days_left": 30, "volatility": 0.2,
                  "current_holding": 0.0, "cost_bps": 10},
        )  # fmt: skip
        assert ok.status_code == 200 and 0.4 < ok.json()["bs_delta"] < 0.6
        assert client.post("/hedge", json={"spot": -1}).status_code == 422
        sim = client.post("/simulate", json={"market": "heston", "n_paths": 100})
        assert sim.status_code == 200 and sim.json()["n_paths"] == 100
        assert client.post("/simulate", json={"n_paths": 10**6}).status_code == 422
