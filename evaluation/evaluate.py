"""Load a trained model, run N evaluation episodes, report real metrics.

Works uniformly across all four algorithms because DQN/A2C/PPO (Stable-
Baselines3) and REINFORCE (custom, `training/reinforce.py`) all expose the
same `.predict(obs, deterministic=True) -> (action, state)` interface.

Usage:
    uv run python -m evaluation.evaluate --algo ppo --model-path models/ppo/best/model.zip --n-episodes 50
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from environment.custom_env import UltrasoundProbeEnv
from training.reinforce import REINFORCE

ALGO_CLASSES = {}


def _sb3_loader(name):
    def _load(path, env):
        from stable_baselines3 import DQN, A2C, PPO
        cls = {"dqn": DQN, "a2c": A2C, "ppo": PPO}[name]
        return cls.load(path, env=env)
    return _load


LOADERS = {
    "dqn": _sb3_loader("dqn"),
    "a2c": _sb3_loader("a2c"),
    "ppo": _sb3_loader("ppo"),
    "reinforce": lambda path, env: REINFORCE.load(path, env),
}


def load_model(algo: str, model_path: str, env):
    if algo not in LOADERS:
        raise ValueError(f"unknown algo {algo}")
    return LOADERS[algo](model_path, env)


def run_episode(model, env):
    obs, _ = env.reset()
    done = False
    ep_reward = 0.0
    ep_len = 0
    info = {}
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        ep_reward += reward
        ep_len += 1
        done = terminated or truncated
    return ep_reward, ep_len, info


def evaluate(algo: str, model_path: str, n_episodes: int = 50, env_kwargs: dict | None = None,
             seed: int = 1000):
    env_kwargs = env_kwargs or {}
    env = UltrasoundProbeEnv(seed=seed, **env_kwargs)
    model = load_model(algo, model_path, env)

    rewards, lengths, successes, flags, n_acquired = [], [], [], [], []
    for i in range(n_episodes):
        env.reset(seed=seed + i)
        r, l, info = run_episode(model, env)
        rewards.append(r)
        lengths.append(l)
        acquired = info.get("acquired", [])
        n_acquired.append(len(acquired))
        target_count = len(env.targets)
        successes.append(len(acquired) == target_count)
        flags.append(info.get("flag"))

    results = dict(
        algo=algo, n_episodes=n_episodes,
        mean_reward=float(np.mean(rewards)), std_reward=float(np.std(rewards)),
        mean_length=float(np.mean(lengths)),
        success_rate=float(np.mean(successes)),
        mean_targets_acquired=float(np.mean(n_acquired)),
        classification_rate=float(np.mean([f is not None for f in flags])),
    )
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", required=True, choices=list(LOADERS.keys()))
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--n-episodes", type=int, default=50)
    parser.add_argument("--single-target", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    env_kwargs = dict(single_target=args.single_target)
    results = evaluate(args.algo, args.model_path, args.n_episodes, env_kwargs)
    print(json.dumps(results, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
