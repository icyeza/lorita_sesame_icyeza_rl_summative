"""Policy-gradient training on `UltrasoundProbeEnv`: REINFORCE (custom), A2C, PPO.

All three share the DQN harness's env-construction convention so
`training/sweep.py` and `evaluation/evaluate.py` can treat all four
algorithms uniformly.
"""
from __future__ import annotations

import os

from stable_baselines3 import A2C, PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

from environment.custom_env import UltrasoundProbeEnv
from training.dqn_training import make_env as _make_single_env, make_vec_env
from training.callbacks import WallClockLimitCallback
from training.reinforce import REINFORCE


def train_reinforce(config: dict, log_dir: str, model_dir: str, seed: int = 0,
                     total_timesteps: int | None = None, env_kwargs: dict | None = None,
                     n_envs: int = 1, max_wall_clock_seconds: float | None = None,
                     info_keywords: tuple = ()):
    """REINFORCE is single-env only: the from-scratch implementation collects
    one full episode at a time on the Python side (no VecEnv batching), so
    `n_envs` is accepted for interface parity with the other three trainers
    but ignored here (documented, not a silent bug) -- see status.md."""
    env_kwargs = env_kwargs or {}
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    env = _make_single_env(env_kwargs, log_dir, seed, info_keywords=info_keywords)()

    model = REINFORCE(
        env,
        lr=config.get("learning_rate", 3e-4),
        gamma=config.get("gamma", 0.99),
        hidden=tuple(config.get("net_arch", [64, 64])),
        use_baseline=config.get("use_baseline", False),
        entropy_coef=config.get("entropy_coef", 0.0),
        seed=seed,
        tensorboard_log=log_dir,
    )
    steps = total_timesteps or config.get("total_timesteps", 20_000)
    model.learn(total_timesteps=steps, log_path=os.path.join(log_dir, "progress.csv"),
                max_wall_clock_seconds=max_wall_clock_seconds)

    save_path = os.path.join(model_dir, "model")
    model.save(save_path)
    return model, save_path + ".pt"


def train_a2c(config: dict, log_dir: str, model_dir: str, seed: int = 0,
              total_timesteps: int | None = None, env_kwargs: dict | None = None,
              n_envs: int = 1, max_wall_clock_seconds: float | None = None,
              info_keywords: tuple = ()):
    env_kwargs = env_kwargs or {}
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    env = make_vec_env(env_kwargs, log_dir, seed, n_envs, info_keywords)

    model = A2C(
        "MlpPolicy", env,
        learning_rate=config.get("learning_rate", 7e-4),
        gamma=config.get("gamma", 0.99),
        n_steps=config.get("n_steps", 8),
        ent_coef=config.get("entropy_coef", 0.01),
        gae_lambda=config.get("gae_lambda", 1.0),
        policy_kwargs=dict(net_arch=config.get("net_arch", [64, 64])),
        seed=seed,
        verbose=0,
    )
    model.set_logger(configure(log_dir, ["csv", "tensorboard"]))
    steps = total_timesteps or config.get("total_timesteps", 20_000)
    callback = WallClockLimitCallback(max_wall_clock_seconds) if max_wall_clock_seconds else None
    model.learn(total_timesteps=steps, progress_bar=False, callback=callback)

    save_path = os.path.join(model_dir, "model.zip")
    model.save(save_path)
    # RESOURCE LEAK FIX (status.md "corrected grid launch" pass): see
    # training/dqn_training.py::train_dqn's identical fix comment --
    # SubprocVecEnv workers must be closed deterministically here, not left
    # to garbage collection, or a multi-combo sweep accumulates orphaned
    # worker processes until the machine runs out of RAM.
    env.close()
    return model, save_path


def train_ppo(config: dict, log_dir: str, model_dir: str, seed: int = 0,
              total_timesteps: int | None = None, env_kwargs: dict | None = None,
              n_envs: int = 1, max_wall_clock_seconds: float | None = None,
              info_keywords: tuple = ()):
    env_kwargs = env_kwargs or {}
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    env = make_vec_env(env_kwargs, log_dir, seed, n_envs, info_keywords)

    model = PPO(
        "MlpPolicy", env,
        learning_rate=config.get("learning_rate", 3e-4),
        gamma=config.get("gamma", 0.99),
        n_steps=config.get("n_steps", 128),
        batch_size=config.get("batch_size", 64),
        n_epochs=config.get("n_epochs", 10),
        ent_coef=config.get("entropy_coef", 0.0),
        gae_lambda=config.get("gae_lambda", 0.95),
        clip_range=config.get("clip_range", 0.2),
        policy_kwargs=dict(net_arch=config.get("net_arch", [64, 64])),
        seed=seed,
        verbose=0,
    )
    model.set_logger(configure(log_dir, ["csv", "tensorboard"]))
    steps = total_timesteps or config.get("total_timesteps", 20_000)
    callback = WallClockLimitCallback(max_wall_clock_seconds) if max_wall_clock_seconds else None
    model.learn(total_timesteps=steps, progress_bar=False, callback=callback)

    save_path = os.path.join(model_dir, "model.zip")
    model.save(save_path)
    # RESOURCE LEAK FIX (status.md "corrected grid launch" pass): see
    # training/dqn_training.py::train_dqn's identical fix comment --
    # SubprocVecEnv workers must be closed deterministically here, not left
    # to garbage collection, or a multi-combo sweep accumulates orphaned
    # worker processes until the machine runs out of RAM.
    env.close()
    return model, save_path


TRAINERS = {
    "dqn": None,  # filled in by training.dqn_training to avoid circular import
    "reinforce": train_reinforce,
    "a2c": train_a2c,
    "ppo": train_ppo,
}
