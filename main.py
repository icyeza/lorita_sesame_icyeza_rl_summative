"""Entrypoint: run a demo episode with the best available saved model.

Looks under `models/<algo>/best/` for each of the four algorithms (in the
order dqn, ppo, a2c, reinforce) and loads the first one found. If none
exist yet (a clean clone before any training has run), falls back to a
randomly-initialized PPO policy and says so explicitly -- it never silently
pretends to be a trained agent.

Runs entirely headless (matplotlib "Agg" backend): renders each step to an
RGB frame and saves the episode as an animated GIF under
`logs/demo/demo_episode.gif`, plus prints a step-by-step summary.

Usage: uv run main.py
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from environment.custom_env import UltrasoundProbeEnv
from evaluation.evaluate import load_model

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(REPO_ROOT, "models")
DEMO_OUT_DIR = os.path.join(REPO_ROOT, "logs", "demo")

ALGO_PRIORITY = ["ppo", "a2c", "dqn", "reinforce"]
MODEL_EXT = {"ppo": ".zip", "a2c": ".zip", "dqn": ".zip", "reinforce": ".pt"}


def find_best_model():
    for algo in ALGO_PRIORITY:
        path = os.path.join(MODELS_DIR, algo, "best", "model" + MODEL_EXT[algo])
        if os.path.exists(path):
            return algo, path
    return None, None


def main():
    env = UltrasoundProbeEnv(seed=42, render_mode="rgb_array")
    algo, path = find_best_model()

    if algo is None:
        print("No trained model found under models/<algo>/best/. "
              "Falling back to a RANDOMLY-INITIALIZED PPO policy -- "
              "this demo is NOT showing a trained agent.")
        from stable_baselines3 import PPO
        model = PPO("MlpPolicy", env, seed=42, verbose=0)
    else:
        print(f"Loading best saved model: algo={algo}, path={path}")
        model = load_model(algo, path, env)

    obs, info = env.reset(seed=42)
    frames = [env.render()]
    total_reward = 0.0
    for step in range(180):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        total_reward += reward
        frames.append(env.render())
        print(f"step={step:3d} target={info.get('target')} "
              f"alpha={info.get('alpha_deg', 0):6.1f}deg d={info.get('d_m', 0) * 1000:6.1f}mm "
              f"reward={reward:+.2f} acquired={info.get('acquired')}")
        if terminated or truncated:
            print(f"Episode finished. flag={info.get('flag')} total_reward={total_reward:.2f}")
            break

    os.makedirs(DEMO_OUT_DIR, exist_ok=True)
    out_path = os.path.join(DEMO_OUT_DIR, "demo_episode.gif")
    _save_gif(frames, out_path)
    print(f"Saved demo animation to {out_path}")


def _save_gif(frames: list[np.ndarray], out_path: str, fps: int = 5):
    fig, ax = plt.subplots(figsize=(frames[0].shape[1] / 100, frames[0].shape[0] / 100))
    ax.axis("off")
    im = ax.imshow(frames[0])

    def update(i):
        im.set_data(frames[i])
        return [im]

    anim = animation.FuncAnimation(fig, update, frames=len(frames), interval=1000 / fps)
    anim.save(out_path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)


if __name__ == "__main__":
    main()
