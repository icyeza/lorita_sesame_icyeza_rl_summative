"""Single calibration training run (NOT a sweep): produces one real
learning curve so the human can read the grid-timestep budget off where it
leaves the noise floor. One algorithm (PPO), one config, one run.

Configuration (see module-level constants below): single_target curriculum,
n_envs=4 (SubprocVecEnv), uncapped (no wall-clock limit -- the documented
SubprocVecEnv + wall-clock-cap hang is specifically the capped+vectorized
combination; uncapped vectorized is safe), deterministic seed.

Usage: uv run python scripts/run_calibration.py
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

ALGO = "ppo"
CURRICULUM = "single_target"
N_ENVS = 4
TOTAL_TIMESTEPS = 200_000
SEED = 0
ROLLING_WINDOW = 20
NOISE_MARGIN = 1.0  # reward units above the initial noise band's mean+std to call "learning starts"


def find_learning_start(df: pd.DataFrame, window: int = ROLLING_WINDOW, margin: float = NOISE_MARGIN):
    """Rough automatic estimate: the first timestep where the rolling-mean
    reward exceeds the initial noise band's (mean + margin), sustained for
    at least `window` episodes. This is presented as an ESTIMATE for the
    human to confirm visually from the plot, not a decision."""
    if len(df) < window * 3:
        return None, None
    initial_band = df["r"].iloc[:window]
    noise_mean, noise_std = initial_band.mean(), initial_band.std()
    threshold = noise_mean + margin
    rolling = df["r"].rolling(window, min_periods=window).mean()
    above = rolling > threshold
    # require `window` consecutive episodes above threshold to avoid a single spike
    sustained = above.rolling(window).sum() >= window
    idx = sustained[sustained].index
    if len(idx) == 0:
        return None, threshold
    first_idx = idx[0]
    return int(df["cum_timesteps"].iloc[first_idx]), threshold


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Calibration run: algo={ALGO} curriculum={CURRICULUM} n_envs={N_ENVS} "
          f"total_timesteps={TOTAL_TIMESTEPS} seed={SEED}")
    print("Uncapped (no --max-wall-clock-seconds) -- safe since n_envs>1 is only "
          "unsafe when COMBINED with a wall-clock cap.")

    env_kwargs = dict(single_target=(CURRICULUM == "single_target"))
    config = {}  # default PPO hyperparameters (see training/pg_training.py::train_ppo)

    model, save_path = train_ppo(
        config, str(LOG_DIR), str(MODEL_DIR), seed=SEED,
        total_timesteps=TOTAL_TIMESTEPS, env_kwargs=env_kwargs,
        n_envs=N_ENVS, max_wall_clock_seconds=None,
    )
    print(f"Training complete. Model saved to {save_path}")

    # Aggregate monitor_*.csv (one per vectorized worker)
    monitor_paths = sorted(LOG_DIR.glob("monitor*.csv"))
    frames = []
    for p in monitor_paths:
        df = pd.read_csv(p, skiprows=1)
        if "r" in df.columns and len(df) > 0:
            df["worker"] = p.name
            frames.append(df)
    if not frames:
        print("No monitor data found -- cannot plot.")
        return

    # GLOBAL cumulative timesteps: each worker's own `l.cumsum()` only
    # reaches total_timesteps/n_envs (e.g. 50k of a 200k run for n_envs=4),
    # since it's local to that worker's episodes -- concatenating those
    # local values directly would mislabel the x-axis. Instead, sort ALL
    # workers' episodes by wall-clock completion time `t` (comparable
    # across workers since they start simultaneously) and take a cumulative
    # sum of `l` over that combined chronological order, which approximates
    # true global env-steps-so-far across all n_envs workers.
    combined = pd.concat(frames, ignore_index=True).sort_values("t").reset_index(drop=True)
    combined["cum_timesteps"] = combined["l"].cumsum()
    combined.to_csv(LOG_DIR / "calibration_episodes.csv", index=False)

    learning_start_step, threshold = find_learning_start(combined)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(combined["cum_timesteps"], combined["r"], s=6, alpha=0.25, color="steelblue",
               label="episode reward (raw)")
    rolling = combined["r"].rolling(ROLLING_WINDOW, min_periods=1).mean()
    ax.plot(combined["cum_timesteps"], rolling, color="darkorange", linewidth=2,
            label=f"rolling mean (window={ROLLING_WINDOW})")
    if learning_start_step is not None:
        ax.axvline(learning_start_step, color="green", linestyle="--", alpha=0.7,
                    label=f"auto-estimated learning start (~{learning_start_step} steps)")
    ax.set_xlabel("timesteps")
    ax.set_ylabel("episode reward")
    ax.set_title(
        f"SINGLE-RUN CALIBRATION CURVE -- PPO, single_target curriculum\n"
        f"(n_envs={N_ENVS}, total_timesteps={TOTAL_TIMESTEPS}, seed={SEED}) -- "
        f"NOT a benchmark, one real run only"
    )
    ax.legend(fontsize=9)
    fig.tight_layout()
    out_path = LOG_DIR / "calibration_curve.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)

    print(f"\nSaved learning curve to {out_path}")
    print(f"Total episodes logged: {len(combined)}")
    if learning_start_step is not None:
        print(f"AUTO-ESTIMATE (for human confirmation from the plot, not a decision): "
              f"rolling-mean reward first sustainably exceeds the initial noise band "
              f"(threshold={threshold:.2f}) around timestep {learning_start_step}.")
    else:
        print("AUTO-ESTIMATE: no sustained rise above the initial noise band was detected "
              "in this run -- inspect the plot directly.")

    summary = dict(
        algo=ALGO, curriculum=CURRICULUM, n_envs=N_ENVS, total_timesteps=TOTAL_TIMESTEPS,
        seed=SEED, n_episodes=len(combined),
        learning_start_estimate_timesteps=learning_start_step,
        noise_band_threshold=float(threshold) if threshold is not None else None,
    )
    with open(LOG_DIR / "calibration_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
