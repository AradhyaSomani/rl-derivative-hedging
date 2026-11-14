"""Warm-up exercise: train PPO (and SAC for comparison) on Gymnasium's Pendulum-v1.

Exploration only (lives in notebooks/, never imported by src/ or tests/). It exists to learn
the stable-baselines3 training loop and how to read TensorBoard curves before touching the
hedging environment.

    python notebooks/pendulum_warmup.py --algo ppo --timesteps 200000 --seed 0
    python notebooks/pendulum_warmup.py --algo ppo_default --timesteps 200000 --seed 0
    python notebooks/pendulum_warmup.py --algo sac --timesteps 30000 --seed 0 --eval-every 2000
    python notebooks/pendulum_warmup.py --plot          # learning-curve figure from the runs
    tensorboard --logdir data/pendulum/tb

Pendulum-v1: swing a pendulum up and hold it upright. Each episode is 200 steps; the reward
per step is -(angle^2 + 0.1 * angular_velocity^2 + 0.001 * torque^2), so a return of 0 is
perfect, a random policy scores about -1,200, and about -200 or better means it has learned
to swing up and balance.

PPO hyperparameters are rl-baselines3-zoo's tuned set for Pendulum-v1; with SB3's defaults
PPO barely learns this task, which is itself a lesson (see the reading notes).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

OUT = Path("data/pendulum")  # git-ignored, like every other run output


def make_model(algo: str, env, seed: int, tb: Path):
    from stable_baselines3 import PPO, SAC

    if algo == "ppo_default":
        return PPO("MlpPolicy", env, seed=seed, tensorboard_log=str(tb), verbose=0)
    if algo == "ppo":
        return PPO(
            "MlpPolicy",
            env,
            seed=seed,
            tensorboard_log=str(tb),
            verbose=0,
            n_steps=1024,
            gae_lambda=0.95,
            gamma=0.9,
            n_epochs=10,
            ent_coef=0.0,
            learning_rate=1e-3,
            clip_range=0.2,
            use_sde=True,
            sde_sample_freq=4,
        )
    if algo == "sac":
        return SAC(
            "MlpPolicy", env, seed=seed, tensorboard_log=str(tb), verbose=0, learning_rate=1e-3
        )
    raise ValueError(algo)


def train(algo: str, timesteps: int, seed: int, eval_every: int) -> dict:
    from stable_baselines3.common.callbacks import EvalCallback
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.evaluation import evaluate_policy

    run = OUT / f"{algo}_seed{seed}"
    run.mkdir(parents=True, exist_ok=True)
    # PPO is on-policy: 4 parallel envs give it more fresh samples per update.
    # SAC is off-policy: one env, and it reuses old samples from its replay buffer.
    n_envs = 1 if algo == "sac" else 4
    env = make_vec_env("Pendulum-v1", n_envs=n_envs, seed=seed)
    eval_env = make_vec_env("Pendulum-v1", n_envs=1, seed=seed + 1000)  # separate seeds
    cb = EvalCallback(
        eval_env,
        n_eval_episodes=10,
        eval_freq=max(eval_every // n_envs, 1),
        log_path=str(run),
        deterministic=True,
        verbose=0,
    )
    model = make_model(algo, env, seed, OUT / "tb")
    model.learn(total_timesteps=timesteps, callback=cb, tb_log_name=f"{algo}_seed{seed}")
    model.save(run / "model")
    mean, std = evaluate_policy(model, eval_env, n_eval_episodes=20, deterministic=True)
    summary = {
        "algo": algo,
        "seed": seed,
        "timesteps": timesteps,
        "final_mean_return": float(mean),
        "final_std_return": float(std),
    }
    (run / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def random_baseline(seed: int, episodes: int = 20) -> float:
    import gymnasium as gym

    env = gym.make("Pendulum-v1")
    env.action_space.seed(seed)
    totals = []
    for ep in range(episodes):
        env.reset(seed=seed + ep)
        done, total = False, 0.0
        while not done:
            _, r, term, trunc, _ = env.step(env.action_space.sample())
            total += r
            done = term or trunc
        totals.append(total)
    return float(np.mean(totals))


def plot() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for path in sorted(OUT.glob("*_seed*/evaluations.npz")):
        d = np.load(path)
        ax.plot(d["timesteps"], d["results"].mean(axis=1), marker="o", ms=3, label=path.parent.name)
    rb = random_baseline(0)
    ax.axhline(rb, color="grey", ls="--", lw=1, label=f"random policy ({rb:.0f})")
    ax.axhline(-200, color="grey", ls=":", lw=1, label="about solved (-200)")
    ax.set_xlabel("environment steps")
    ax.set_ylabel("mean evaluation return (10 episodes)")
    ax.set_title("Pendulum-v1: PPO (on-policy, tuned and default) vs SAC (off-policy)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "pendulum_curves.png", dpi=130)
    print(f"wrote {OUT / 'pendulum_curves.png'}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=["ppo", "ppo_default", "sac"], default="ppo")
    p.add_argument("--timesteps", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=10_000)
    p.add_argument("--plot", action="store_true")
    a = p.parse_args()
    if a.plot:
        plot()
        return
    print(json.dumps(train(a.algo, a.timesteps, a.seed, a.eval_every), indent=2))


if __name__ == "__main__":
    main()
