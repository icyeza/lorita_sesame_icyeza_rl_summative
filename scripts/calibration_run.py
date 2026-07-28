"""Calibration run: one PPO run on the LOCKED environment (defaults now
alpha_tol_deg=18, tilt_step_deg=3.0, shaping_mode="multiplicative",
start_curriculum=True/radius=8, d_tol_m=0.012 -- see status.md "lock the
environment" pass) to read off where success saturates, so the grid
launch's `grid_timesteps` is set from real data rather than a guess.

single_target=True (the curriculum STAGE the grid tables use -- samples
head/abdomen/femur uniformly at random per episode, NOT femur-only; this
is deliberately the harder, mixed-target distribution the grids will
actually run on, not the isolated femur-only config the lock's 92% number
was validated under).

PPO, SubprocVecEnv n_envs=4, uncapped, seed=0. Emits a success-rate +
reward vs timesteps plot to logs/calibration/.

Usage: uv run python scripts/calibration_run.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from training.pg_training import train_ppo

REPO_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = REPO_ROOT / "logs" / "calibration"
MODEL_DIR = REPO_ROOT / "models" / "calibration"

TOTAL_TIMESTEPS = 100_000
N_ENVS = 4
SEED = 0
ROLLING_WINDOW = 50
# saturation = rolling success rate stays within this fraction of its own
# eventual (last-10%-of-run) plateau value for the rest of the run
SATURATION_TOLERANCE = 0.05


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    env_kwargs = dict(single_target=True)  # LOCKED defaults apply otherwise
    config = dict(entropy_coef=0.05)
    info_keywords = ("success", "freeze_attempted", "d_m", "alpha_deg")

    print(f"Calibration run: PPO, single_target (mixed head/abdomen/femur), LOCKED env defaults "
          f"(alpha_tol=18deg, tilt_step=3deg, shaping=multiplicative, start_curriculum=True/radius=8, "
          f"d_tol=12mm), n_envs={N_ENVS}, total_timesteps={TOTAL_TIMESTEPS}, seed={SEED}, uncapped")

    model, save_path = train_ppo(
        config, str(LOG_DIR), str(MODEL_DIR), seed=SEED,
        total_timesteps=TOTAL_TIMESTEPS, env_kwargs=env_kwargs,
        n_envs=N_ENVS, max_wall_clock_seconds=None, info_keywords=info_keywords,
    )
    print(f"Calibration run complete. Model saved to {save_path}")

    monitor_paths = sorted(LOG_DIR.glob("monitor*.csv"))
    frames = []
    for p in monitor_paths:
        df = pd.read_csv(p, skiprows=1)
        if "r" in df.columns and len(df) > 0:
            frames.append(df)
    combined = pd.concat(frames, ignore_index=True).sort_values("t").reset_index(drop=True)
    combined["cum_timesteps"] = combined["l"].cumsum()
    combined.to_csv(LOG_DIR / "calibration_episodes.csv", index=False)

    rolling_success = combined["success"].astype(float).rolling(ROLLING_WINDOW, min_periods=1).mean()
    rolling_reward = combined["r"].rolling(ROLLING_WINDOW, min_periods=1).mean()
    rolling_freeze = combined["freeze_attempted"].astype(float).rolling(ROLLING_WINDOW, min_periods=1).mean()

    # Read saturation timestep off the REAL curve: the plateau value is the
    # mean rolling success over the last 10% of episodes; saturation
    # timestep = first cum_timesteps where rolling success gets within
    # SATURATION_TOLERANCE of that plateau AND stays there for the rest of
    # the run (checked by requiring the min of the remaining tail to also
    # clear plateau - tolerance).
    n = len(combined)
    plateau = float(rolling_success.iloc[int(n * 0.9):].mean())
    threshold = plateau - SATURATION_TOLERANCE
    saturation_idx = None
    for i in range(n):
        if rolling_success.iloc[i:].min() >= threshold:
            saturation_idx = i
            break
    saturation_timestep = int(combined["cum_timesteps"].iloc[saturation_idx]) if saturation_idx is not None else None

    print(f"\nPlateau success rate (last 10% of episodes): {plateau:.4f}")
    if saturation_timestep is not None:
        print(f"Saturation timestep (first point success stays within "
              f"{SATURATION_TOLERANCE} of plateau for the rest of the run): {saturation_timestep}")
    else:
        print("Success rate did NOT clearly saturate within this run's budget -- "
              "consider a longer calibration run before setting grid_timesteps.")

    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    axes[0].plot(combined["cum_timesteps"], rolling_reward, color="steelblue", linewidth=2)
    axes[0].set_ylabel(f"rolling mean episode reward, window={ROLLING_WINDOW}")
    axes[0].set_title("Reward vs timesteps")
    axes[1].plot(combined["cum_timesteps"], rolling_success, color="green", linewidth=2)
    axes[1].axhline(plateau, color="red", linestyle="--", label=f"plateau={plateau:.2f}")
    if saturation_timestep is not None:
        axes[1].axvline(saturation_timestep, color="gray", linestyle=":",
                         label=f"saturation={saturation_timestep}")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].set_ylabel(f"rolling success rate, window={ROLLING_WINDOW}")
    axes[1].set_title("Success rate vs timesteps (single_target, mixed targets)")
    axes[1].legend()
    axes[2].plot(combined["cum_timesteps"], rolling_freeze, color="orange", linewidth=2)
    axes[2].set_ylim(-0.02, 1.02)
    axes[2].set_xlabel("timesteps")
    axes[2].set_ylabel(f"rolling freeze-attempt fraction, window={ROLLING_WINDOW}")
    axes[2].set_title("Freeze-attempt fraction vs timesteps")
    fig.suptitle(f"Calibration run -- PPO, single_target, locked env, {TOTAL_TIMESTEPS} steps",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(LOG_DIR / "calibration_curve.png", dpi=130)
    plt.close(fig)

    summary = dict(
        total_timesteps=TOTAL_TIMESTEPS, n_envs=N_ENVS, seed=SEED, n_episodes=n,
        plateau_success_rate=plateau, saturation_timestep=saturation_timestep,
    )
    with open(LOG_DIR / "calibration_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {LOG_DIR / 'calibration_summary.json'}")


if __name__ == "__main__":
    main()
