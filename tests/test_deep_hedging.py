import numpy as np
import pytest

from hedge.baselines.run import run_baselines
from hedge.env.hedging_env import HedgingEnv
from hedge.eval.evaluate import evaluate_policy, sb3_predict
from hedge.eval.metrics import cvar
from hedge.market import pathsets
from hedge.pnl import hedge_pnl

torch = pytest.importorskip("torch")

from hedge.agents.deep_hedging import (  # noqa: E402
    DeepHedger,
    EpisodeData,
    cvar_loss,
    entropic_loss,
    torch_hedge_pnl,
)
from hedge.agents.train_sb3 import AgentConfig, load_model, run_dir, train  # noqa: E402

BAND = {"width": 0.0, "rebalance_to": "delta"}


@pytest.mark.parametrize("kind", ["call", "put"])
def test_torch_engine_matches_numpy_engine(kind):
    rng = np.random.default_rng(0)
    paths = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, size=(50, 31)), axis=1))
    holdings = rng.uniform(-1, 1, size=(50, 30))
    expected = hedge_pnl(paths, holdings, 100.0, 2.5, 0.001, kind).pnl
    got = torch_hedge_pnl(
        torch.tensor(paths), torch.tensor(holdings), 100.0, 2.5, 0.001, kind
    ).numpy()
    np.testing.assert_allclose(got, expected, atol=1e-10)


def test_losses():
    cost = torch.tensor(np.random.default_rng(1).normal(size=10_000))
    assert float(cvar_loss(cost, 0.95)) == pytest.approx(cvar(cost.numpy(), 0.95), abs=1e-12)
    # Entropic risk of a constant is the constant; it is at least the mean otherwise.
    assert float(entropic_loss(torch.full((100,), 2.0, dtype=torch.float64))) == pytest.approx(2.0)
    assert float(entropic_loss(cost, 1.0)) >= float(cost.mean())


@pytest.mark.parametrize("fixture", ["gbm_scenario", "heston_scenario"])
def test_untrained_network_is_the_delta_hedge(fixture, request):
    """The network starts at action 0, which in delta_offset mode is exactly BS delta."""
    scenario = request.getfixturevalue(fixture)
    ps = pathsets.simulate(scenario, np.arange(300))
    env = HedgingEnv(scenario, ps, mode="eval", action_mode="delta_offset")
    model = DeepHedger(len(env.obs_features))
    baseline = run_baselines(scenario, ps, BAND)["bs_delta"]

    # Through the numpy evaluation path.
    result, _ = evaluate_policy(env, sb3_predict(model))
    np.testing.assert_allclose(result.pnl, baseline.pnl, atol=1e-10)
    # Through the torch training path (float32).
    data = EpisodeData(env)
    cost = data.cost(model, torch.arange(300)).detach().numpy()
    np.testing.assert_allclose(cost, baseline.cost, atol=2e-3)


@pytest.mark.parametrize("action_mode", ["holding", "delta_offset"])
def test_torch_rollout_matches_numpy_evaluation(gbm_scenario, action_mode):
    ps = pathsets.simulate(gbm_scenario, np.arange(200))
    env = HedgingEnv(gbm_scenario, ps, mode="eval", action_mode=action_mode)
    torch.manual_seed(0)
    model = DeepHedger(len(env.obs_features))
    torch.nn.init.normal_(model.net[-1].weight, std=0.5)  # a non-trivial policy
    data = EpisodeData(env)
    torch_holdings = data.holdings(model, torch.arange(200)).detach().numpy()
    _, numpy_holdings = evaluate_policy(env, sb3_predict(model))
    np.testing.assert_allclose(torch_holdings, numpy_holdings, atol=1e-5)


@pytest.mark.slow
def test_training_lowers_validation_cvar(gbm_scenario, tmp_path):
    agent = AgentConfig(
        name="deep_hedging",
        algo="deep_hedging",
        total_steps=60,
        hyperparams={"batch_size": 2000, "patience": 3},
        env={"action_mode": "delta_offset", "max_offset": 0.25},
        eval_every=20,
    )
    assert agent.strategy == "deep_hedging"
    train_set = pathsets.simulate(gbm_scenario, np.arange(100_000, 104_000))
    validation = pathsets.simulate(gbm_scenario, np.arange(20_000, 22_000))
    metrics = train(
        agent, gbm_scenario, 0, runs_dir=tmp_path,
        train_paths=train_set, validation_paths=validation,
    )  # fmt: skip
    delta = run_baselines(gbm_scenario, validation, BAND)["bs_delta"]
    assert metrics["cvar95"] < cvar(delta.cost)  # starts at delta and keeps the best checkpoint

    out = run_dir(agent, gbm_scenario, 0, tmp_path)
    env = HedgingEnv(gbm_scenario, validation, mode="eval", action_mode="delta_offset")
    result, holdings = evaluate_policy(env, sb3_predict(load_model(agent, out)))
    assert cvar(result.cost) == pytest.approx(metrics["cvar95"], abs=1e-9)
    assert (holdings >= 0).all() and (holdings <= 1).all()
