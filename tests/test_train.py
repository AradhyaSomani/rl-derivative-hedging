import json

import numpy as np
import pandas as pd
import pytest

from hedge.agents.scripted import delta_policy_from_obs, rollout
from hedge.baselines.run import run_baselines
from hedge.env.hedging_env import HedgingEnv
from hedge.eval.evaluate import evaluate_policy
from hedge.market import pathsets

BAND = {"width": 0.0, "rebalance_to": "delta"}


@pytest.mark.parametrize("fixture", ["gbm_scenario", "heston_scenario"])
@pytest.mark.parametrize("action_mode", ["holding", "delta_offset"])
def test_batched_evaluation_matches_stepping_the_env(fixture, action_mode, request):
    scenario = request.getfixturevalue(fixture)
    ps = pathsets.simulate(scenario, np.arange(200))
    env = HedgingEnv(scenario, ps, mode="eval", action_mode=action_mode)

    def policy(obs):  # any deterministic function of the observation, in batch
        obs = np.atleast_2d(obs)
        return np.tanh(obs.sum(axis=1, keepdims=True))

    batched, holdings = evaluate_policy(env, policy)
    stepped = rollout(env, lambda e, obs: policy(obs)[0])
    np.testing.assert_allclose(holdings, stepped.holdings, atol=1e-12)
    np.testing.assert_allclose(batched.pnl, stepped.result.pnl, atol=1e-10)


def test_batched_delta_policy_matches_baseline(gbm_scenario):
    ps = pathsets.simulate(gbm_scenario, np.arange(200))
    env = HedgingEnv(gbm_scenario, ps, mode="eval")
    idx = env.obs_features.index("bs_delta")
    result, _ = evaluate_policy(env, lambda obs: obs[:, [idx]])
    baseline = run_baselines(gbm_scenario, ps, BAND)["bs_delta"]
    assert np.abs(result.pnl - baseline.pnl).max() < 1e-3  # float32 observation
    stepped = rollout(env, delta_policy_from_obs)
    np.testing.assert_allclose(result.pnl, stepped.result.pnl, atol=1e-10)


@pytest.mark.slow
@pytest.mark.parametrize("algo", ["ppo", "sac"])
def test_training_run_end_to_end(algo, gbm_scenario, tmp_path):
    pytest.importorskip("stable_baselines3")
    from hedge.agents.train_sb3 import AgentConfig, load_model, run_dir, train
    from hedge.eval.evaluate import sb3_predict

    steps = 2048 if algo == "ppo" else 400
    agent = AgentConfig(
        name=algo,
        algo=algo,
        total_steps=steps,
        n_envs=2 if algo == "ppo" else 1,
        hyperparams={"n_steps": 256} if algo == "ppo" else {"learning_starts": 100},
        reward={"name": "terminal_cvar", "lam": 0.5, "threshold": "auto"},
        eval_every=1024 if algo == "ppo" else 200,
    )
    assert agent.strategy == f"{algo}_terminal_cvar_l0.5"
    train_set = pathsets.simulate(gbm_scenario, np.arange(100_000, 100_500))
    validation = pathsets.simulate(gbm_scenario, np.arange(20_000, 20_300))
    metrics = train(
        agent, gbm_scenario, 3, runs_dir=tmp_path,
        train_paths=train_set, validation_paths=validation,
    )  # fmt: skip

    out = run_dir(agent, gbm_scenario, 3, tmp_path)
    curve = pd.read_csv(out / "curve.csv")
    assert curve["step"].iloc[0] == 0 and curve["step"].iloc[-1] >= steps
    assert curve["val_cvar95"].notna().all()
    info = json.loads((out / "run.json").read_text())
    assert info["seed"] == 3 and info["scenario_hash"] == gbm_scenario.hash
    assert info["reward"]["threshold"] > 0  # "auto" resolved to the delta hedge's VaR
    assert metrics["n_paths"] == 300 and metrics["split"] == "validation"

    # The saved model reproduces the recorded metrics.
    env = HedgingEnv(gbm_scenario, validation, mode="eval")
    result, holdings = evaluate_policy(env, sb3_predict(load_model(agent, out)))
    assert float(result.cost.mean()) == pytest.approx(metrics["mean_cost"], abs=1e-9)
    assert (holdings >= 0).all() and (holdings <= 1).all()


def test_experiment_expansion(tmp_path):
    pytest.importorskip("stable_baselines3")
    from hedge.agents.experiment import expand

    runs = expand(
        {
            "agent": "config/agents/ppo.yaml",
            "scenarios": ["config/scenarios/gbm_10bps.yaml", "config/scenarios/gbm_5bps.yaml"],
            "seeds": [0, 1],
            "rewards": ["quadratic", "terminal_cvar"],
            "lams": [0.1, 2],
            "total_steps": 5000,
        }
    )
    assert len(runs) == 16
    names = {agent.strategy for agent, _, _ in runs}
    assert names == {
        "ppo_quadratic_l0.1", "ppo_quadratic_l2", "ppo_terminal_cvar_l0.1", "ppo_terminal_cvar_l2",
    }  # fmt: skip
    assert all(agent.total_steps == 5000 for agent, _, _ in runs)
    cvar = next(a for a, _, _ in runs if a.reward["name"] == "terminal_cvar")
    assert cvar.reward["threshold"] == "auto"
