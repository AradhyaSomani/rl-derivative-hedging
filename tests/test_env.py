import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from hedge.agents.scripted import (
    delta_policy,
    delta_policy_from_obs,
    random_policy,
    rollout,
)
from hedge.baselines.run import run_baselines
from hedge.config import Scenario
from hedge.env.hedging_env import HedgingEnv, make_env_fn
from hedge.env.rewards import REWARDS, RewardConfig, option_marks, step_pnls
from hedge.market import pathsets
from hedge.pnl import hedge_pnl

N = 300
BAND = {"width": 0.0, "rebalance_to": "delta"}


@pytest.fixture(scope="module", params=["gbm_scenario", "heston_scenario"])
def setup(request):
    scenario = request.getfixturevalue(request.param)
    return scenario, pathsets.simulate(scenario, np.arange(N))


@pytest.fixture(scope="module")
def put_setup(gbm_scenario):
    raw = {
        "name": "gbm_put",
        "market": gbm_scenario.market,
        "option": {**gbm_scenario.option, "kind": "put"},
        "grid": gbm_scenario.grid,
        "cost_bps": 10,
    }
    scenario = Scenario.from_dict(raw)
    return scenario, pathsets.simulate(scenario, np.arange(N))


# Phase 2 gate: both env checkers pass.
@pytest.mark.parametrize("reward", REWARDS)
@pytest.mark.parametrize("action_mode", ["holding", "delta_offset"])
def test_gymnasium_check_env(setup, reward, action_mode):
    scenario, ps = setup
    env = HedgingEnv(scenario, ps, RewardConfig(reward), action_mode=action_mode)
    check_env(env, skip_render_check=True)


@pytest.mark.parametrize("mode", ["train", "eval"])
def test_stable_baselines3_check_env(setup, mode, recwarn):
    sb3_checker = pytest.importorskip("stable_baselines3.common.env_checker")
    scenario, ps = setup
    sb3_checker.check_env(HedgingEnv(scenario, ps, mode=mode), warn=True)
    assert not [w for w in recwarn if "stable_baselines3" in str(w.filename)]


def test_vectorised_envs(gbm_scenario):
    vec = pytest.importorskip("stable_baselines3.common.vec_env")
    ps = pathsets.simulate(gbm_scenario, np.arange(50))
    for cls in (vec.DummyVecEnv, vec.SubprocVecEnv):
        envs = cls([make_env_fn(gbm_scenario, ps, rank=i) for i in range(2)])
        obs = envs.reset()
        assert obs.shape == (2, 4)
        obs, rewards, dones, infos = envs.step(np.zeros((2, 1), dtype=np.float32))
        assert rewards.shape == (2,) and not dones.any()
        envs.close()


# Phase 2 gate: the reward sum test, for all three reward types.
@pytest.mark.parametrize("reward", REWARDS)
@pytest.mark.parametrize("policy", [delta_policy, random_policy])
def test_step_pnls_sum_to_the_engine_total(setup, reward, policy):
    scenario, ps = setup
    env = HedgingEnv(scenario, ps, RewardConfig(reward, lam=0.5), mode="eval")
    run = rollout(env, policy, seed=1)
    np.testing.assert_allclose(run.env_pnl, run.result.pnl, rtol=0, atol=1e-10)


def test_vectorised_step_pnls_match_engine(setup):
    scenario, ps = setup
    rng = np.random.default_rng(0)
    holdings = rng.uniform(0, 1, size=(N, scenario.n_steps))
    gross, cost = step_pnls(
        ps.prices, holdings, option_marks(scenario, ps), scenario.premium(), scenario.cost_rate
    )
    engine = hedge_pnl(
        ps.prices, holdings, scenario.strike, scenario.premium(), scenario.cost_rate, scenario.kind
    )
    np.testing.assert_allclose((gross - cost).sum(axis=1), engine.pnl, rtol=0, atol=1e-10)
    np.testing.assert_allclose(cost.sum(axis=1), engine.txn_cost, rtol=0, atol=1e-12)


def test_reward_definitions(setup):
    scenario, ps = setup
    # mean_variance with lam = 0 is the plain P&L, so rewards sum to the engine total.
    env = HedgingEnv(scenario, ps, RewardConfig("mean_variance", lam=0.0), mode="eval")
    run = rollout(env, random_policy, 50)
    np.testing.assert_allclose(run.reward, run.result.pnl, atol=1e-10)
    # mean_variance never pays more than the P&L; quadratic is never positive.
    env = HedgingEnv(scenario, ps, RewardConfig("mean_variance", lam=2.0), mode="eval")
    run = rollout(env, random_policy, 50)
    assert (run.reward <= run.result.pnl + 1e-12).all()
    env = HedgingEnv(scenario, ps, RewardConfig("quadratic", lam=1.0), mode="eval")
    assert (rollout(env, random_policy, 50).reward <= 0).all()
    # terminal_cvar: one reward at expiry, a function of the episode cost alone.
    cfg = RewardConfig("terminal_cvar", lam=0.5, alpha=0.95, threshold=1.0)
    env = HedgingEnv(scenario, ps, cfg, mode="eval")
    run = rollout(env, random_policy, 50)
    loss = -run.result.pnl
    expected = -loss - 0.5 * (1.0 + np.maximum(loss - 1.0, 0.0) / 0.05)
    np.testing.assert_allclose(run.reward, expected, atol=1e-9)
    env.reset(options={"path_index": 0})
    assert env.step(np.zeros(1))[1] == 0.0


# Phase 2 gate: the scripted delta agent reproduces the Phase 1 baseline.
@pytest.mark.parametrize("action_mode", ["holding", "delta_offset"])
def test_scripted_delta_agent_matches_baseline(setup, action_mode):
    scenario, ps = setup
    baseline = run_baselines(scenario, ps, BAND)["bs_delta"]
    env = HedgingEnv(scenario, ps, mode="eval", action_mode=action_mode)
    run = rollout(env, delta_policy)
    np.testing.assert_allclose(run.result.pnl, baseline.pnl, rtol=0, atol=1e-10)
    np.testing.assert_allclose(run.result.turnover, baseline.turnover, rtol=0, atol=1e-10)
    # Reading delta off the float32 observation still matches to well under a cent.
    run32 = rollout(env, delta_policy_from_obs)
    assert np.abs(run32.result.pnl - baseline.pnl).max() < 1e-3


def test_put_is_hedged_with_a_short_position(put_setup):
    scenario, ps = put_setup
    env = HedgingEnv(scenario, ps, mode="eval")
    run = rollout(env, delta_policy)
    assert (run.holdings <= 0).all() and (run.holdings >= -1).all()
    baseline = run_baselines(scenario, ps, BAND)["bs_delta"]
    np.testing.assert_allclose(run.result.pnl, baseline.pnl, rtol=0, atol=1e-10)
    np.testing.assert_allclose(run.env_pnl, baseline.pnl, rtol=0, atol=1e-10)


def test_random_agent_is_far_worse_than_delta(setup):
    scenario, ps = setup
    env = HedgingEnv(scenario, ps, mode="eval")
    random_run, delta_run = rollout(env, random_policy), rollout(env, delta_policy)
    assert random_run.result.cost.std() > 2 * delta_run.result.cost.std()
    assert random_run.result.txn_cost.mean() > 2 * delta_run.result.txn_cost.mean()


def test_path_selection(gbm_scenario):
    ps = pathsets.simulate(gbm_scenario, np.arange(20))
    env = HedgingEnv(gbm_scenario, ps, mode="eval")
    assert [env.reset()[1]["path_index"] for _ in range(22)] == [*range(20), 0, 1]
    assert env.reset(options={"path_index": 7})[1] == {"path_index": 7, "seed": 7}

    train = HedgingEnv(gbm_scenario, ps, mode="train")
    first = [train.reset(seed=3)[1]["path_index"]] + [
        train.reset()[1]["path_index"] for _ in range(30)
    ]
    again = [train.reset(seed=3)[1]["path_index"]] + [
        train.reset()[1]["path_index"] for _ in range(30)
    ]
    assert first == again and len(set(first)) > 5


def test_observation_and_action_mapping(setup):
    scenario, ps = setup
    env = HedgingEnv(scenario, ps, mode="eval")
    obs, _ = env.reset(options={"path_index": 3})
    names = env.obs_features
    assert obs.dtype == np.float32 and obs.shape == (len(names),)
    assert obs[names.index("tau")] == 1.0
    assert obs[names.index("log_moneyness")] == 0.0  # at the money at the sale
    assert obs[names.index("holding")] == -1.0  # flat
    assert obs[names.index("bs_delta")] == pytest.approx(2 * env.current_delta - 1, abs=1e-6)
    if "variance" in names:
        assert obs[names.index("variance")] == pytest.approx(0.0, abs=1e-6)  # v0 = theta
    obs, _, _, _, info = env.step(np.array([0.5], dtype=np.float32))
    assert info["holding"] == pytest.approx(0.75)
    assert obs[names.index("holding")] == pytest.approx(0.5)
    assert env.step(np.array([7.0]))[4]["holding"] == 1.0  # clipped

    # Observations stay in a sensible range over whole episodes.
    seen = []
    for i in range(50):
        obs, _ = env.reset(options={"path_index": i})
        for _ in range(env.n_steps):
            obs, _, done, _, _ = env.step(env.np_random.uniform(-1, 1, size=1))
            seen.append(obs)
    assert done and obs[names.index("tau")] == -1.0
    assert np.abs(np.array(seen)).max() < 3.0


def test_ablation_without_bs_delta(gbm_scenario):
    ps = pathsets.simulate(gbm_scenario, np.arange(10))
    env = HedgingEnv(gbm_scenario, ps, obs_features=("tau", "log_moneyness", "holding"))
    assert env.observation_space.shape == (3,)
    check_env(env, skip_render_check=True)


def test_bad_arguments(gbm_scenario):
    ps = pathsets.simulate(gbm_scenario, np.arange(5))
    with pytest.raises(ValueError):
        HedgingEnv(gbm_scenario, ps, obs_features=("tau", "variance"))
    with pytest.raises(ValueError):
        HedgingEnv(gbm_scenario, ps, mode="test")
    with pytest.raises(ValueError):
        RewardConfig("sharpe")
    env = HedgingEnv(gbm_scenario, ps)
    with pytest.raises(RuntimeError):
        env.step(np.zeros(1))
