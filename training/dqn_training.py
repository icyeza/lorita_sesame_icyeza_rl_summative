"""DQN training on `UltrasoundProbeEnv` via Stable-Baselines3."""
from __future__ import annotations

import os

from stable_baselines3 import DQN
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from environment.custom_env import UltrasoundProbeEnv
from training.callbacks import WallClockLimitCallback

# NOTE on DQN + n_envs>1: SB3's DQN replay buffer supports vectorized
# collection (VecEnv) fine for experience collection, but DQN itself does
# not require -- and gets no gradient-step speedup from -- multiple envs
# the way on-policy algorithms (A2C/PPO) do. We still allow it for
# throughput (more env-steps/sec = more replay-buffer fill rate), see
# Phase-3 notes in training/sweep.py.


def make_env(env_kwargs: dict, log_dir: str, seed: int, rank: int = 0, info_keywords: tuple = ()):
    def _init():
        env = UltrasoundProbeEnv(seed=seed + rank, **env_kwargs)
        # SB3's Monitor appends ".monitor.csv" to whatever filename you pass
        # UNLESS it already ends with exactly "monitor.csv". "monitor.csv"
        # itself is therefore already the "no suffix appended" case (used
        # for rank 0, and hardcoded elsewhere e.g. evaluation/plots.py --
        # do not change it). "monitor_1.csv" does NOT end with "monitor.csv"
        # so it got double-suffixed into "monitor_1.csv.monitor.csv"; using
        # "monitor_1" (no extension) instead produces the clean
        # "monitor_1.monitor.csv".
        base = "monitor.csv" if rank == 0 else f"monitor_{rank}"
        # info_keywords: () by default (unchanged monitor.csv columns for
        # every existing training path). Pass e.g.
        # ("success", "freeze_attempted", "d_m", "alpha_deg") to have
        # Monitor log those UltrasoundProbeEnv.step() info-dict fields
        # (diagnostic-only, no reward/geometry effect) directly per episode
        # -- see scripts/run_freeze_placement_confirmation.py.
        return Monitor(env, filename=os.path.join(log_dir, base), info_keywords=info_keywords)
    return _init


def make_vec_env(env_kwargs: dict, log_dir: str, seed: int, n_envs: int, info_keywords: tuple = ()):
    os.makedirs(log_dir, exist_ok=True)
    if n_envs <= 1:
        return DummyVecEnv([make_env(env_kwargs, log_dir, seed, 0, info_keywords)])
    return SubprocVecEnv([make_env(env_kwargs, log_dir, seed, i, info_keywords) for i in range(n_envs)])


def train_dqn(config: dict, log_dir: str, model_dir: str, seed: int = 0,
              total_timesteps: int | None = None, env_kwargs: dict | None = None,
              n_envs: int = 1, max_wall_clock_seconds: float | None = None,
              info_keywords: tuple = ()):
    """config keys: learning_rate, gamma, buffer_size, batch_size,
    target_update_interval, exploration_fraction, net_arch (list[int]),
    total_timesteps (used if `total_timesteps` arg not overridden)."""
    env_kwargs = env_kwargs or {}
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    env = make_vec_env(env_kwargs, log_dir, seed, n_envs, info_keywords)

    model = DQN(
        "MlpPolicy", env,
        learning_rate=config.get("learning_rate", 1e-3),
        gamma=config.get("gamma", 0.99),
        buffer_size=config.get("buffer_size", 50_000),
        batch_size=config.get("batch_size", 64),
        target_update_interval=config.get("target_update_interval", 1000),
        exploration_fraction=config.get("exploration_fraction", 0.2),
        policy_kwargs=dict(net_arch=config.get("net_arch", [64, 64])),
        learning_starts=config.get("learning_starts", 200),
        seed=seed,
        verbose=0,
    )
    model.set_logger(configure(log_dir, ["csv", "tensorboard"]))

    steps = total_timesteps or config.get("total_timesteps", 20_000)
    callback = WallClockLimitCallback(max_wall_clock_seconds) if max_wall_clock_seconds else None
    model.learn(total_timesteps=steps, progress_bar=False, callback=callback)

    save_path = os.path.join(model_dir, "model.zip")
    model.save(save_path)
    # RESOURCE LEAK FIX (status.md "corrected grid launch" pass): SubprocVecEnv
    # worker processes are NOT torn down until this VecEnv is garbage
    # collected, which is NOT guaranteed to happen promptly between grid
    # combos -- a real multi-hour sweep (many combos back-to-back) was
    # observed to accumulate orphaned worker processes (12 alive from just
    # 3 DQN combos) until the machine ran out of RAM. env.close() tears
    # down the workers deterministically, right here, every time.
    env.close()
    return model, save_path
