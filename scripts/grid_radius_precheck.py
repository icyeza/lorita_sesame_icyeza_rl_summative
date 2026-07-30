"""Differentiation pre-check for the grid launch (status.md "corrected grid
launch -- differentiating start radius" pass).

The locked-default start radius (start_curriculum_max_random_steps=8) was
found to saturate success to ~95-100% within ~1000-3000 steps (see the
stopped calibration run) -- meaning every grid hyperparameter combo would
converge to the same ceiling and produce 10 IDENTICAL table rows, exactly
what the assignment's differentiation requirement rules out.

Scope note: `alpha_tol_deg`, `tilt_step_deg`, `d_tol_m`, `shaping_mode`
(the locked PHYSICS) are untouched here. Only the grid SWEEP's start
distribution is overridden via explicit env_kwargs
(`start_curriculum_max_random_steps=MID_RADIUS`) -- the environment's own
class defaults (used by everything else, e.g. main.py, the deployed
policy) are not touched.

This script runs ONE default-hyperparameter PPO config at the candidate
mid radius for a short budget and reports the resulting success-rate
curve, so the radius can be confirmed non-saturated (and non-zero) BEFORE
committing the full 4-algorithm x 10-run batch to it.

Usage: uv run python scripts/grid_radius_precheck.py --radius 40 --timesteps 5000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from training.pg_training import train_ppo

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "logs" / "grid_radius_precheck"
MODEL_DIR = REPO_ROOT / "models" / "grid_radius_precheck"

N_ENVS = 4
SEED = 0
ROLLING_WINDOW = 20
SATURATED_FLOOR = 0.90   # if success stays >= this the whole run, radius is too easy
DEAD_CEILING = 0.05      # if success stays <= this the whole run, radius is too hard


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--radius", type=int, required=True)
    parser.add_argument("--timesteps", type=int, default=5000)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = OUT_DIR / f"radius_{args.radius}"
    model_dir = MODEL_DIR / f"radius_{args.radius}"
    run_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    env_kwargs = dict(single_target=True, start_curriculum=True,
                       start_curriculum_max_random_steps=args.radius)
    config = dict()  # PPO defaults
    info_keywords = ("success", "freeze_attempted", "d_m", "alpha_deg")

    print(f"Pre-check: PPO default config, single_target (mixed targets), "
          f"start_curriculum_max_random_steps={args.radius} (locked physics unchanged: "
          f"alpha_tol=18deg, tilt_step=3deg, shaping=multiplicative, d_tol=12mm), "
          f"n_envs={N_ENVS}, total_timesteps={args.timesteps}, seed={SEED}, uncapped")

    t0 = time.monotonic()
    model, save_path = train_ppo(
        config, str(run_dir), str(model_dir), seed=SEED,
        total_timesteps=args.timesteps, env_kwargs=env_kwargs,
        n_envs=N_ENVS, max_wall_clock_seconds=None, info_keywords=info_keywords,
    )
    elapsed = time.monotonic() - t0
    print(f"Pre-check run complete in {elapsed:.1f}s. Model saved to {save_path}")

    monitor_paths = sorted(run_dir.glob("monitor*.csv"))
    frames = []
    for p in monitor_paths:
        df = pd.read_csv(p, skiprows=1)
        if "r" in df.columns and len(df) > 0:
            frames.append(df)
    combined = pd.concat(frames, ignore_index=True).sort_values("t").reset_index(drop=True)
    combined["cum_timesteps"] = combined["l"].cumsum()
    combined.to_csv(run_dir / "precheck_episodes.csv", index=False)

    n = len(combined)
    throughput = combined["cum_timesteps"].max() / elapsed
    mean_episode_len = combined["l"].mean()
    rolling_success = combined["success"].astype(float).rolling(ROLLING_WINDOW, min_periods=1).mean()
    overall_success = float(combined["success"].mean())
    first_half_success = float(combined["success"].iloc[:n // 2].mean())
    second_half_success = float(combined["success"].iloc[n // 2:].mean())

    print(f"\nN episodes: {n}, mean episode length: {mean_episode_len:.2f} steps")
    print(f"Throughput: {throughput:.2f} steps/sec (wall-clock {elapsed:.1f}s for "
          f"{combined['cum_timesteps'].max()} steps)")
    print(f"Overall success rate: {overall_success:.4f}")
    print(f"First half success: {first_half_success:.4f}, second half: {second_half_success:.4f}")

    if second_half_success >= SATURATED_FLOOR:
        verdict = f"SATURATED (second-half success {second_half_success:.2f} >= {SATURATED_FLOOR}) -- radius too easy, widen it"
    elif second_half_success <= DEAD_CEILING:
        verdict = f"DEAD (second-half success {second_half_success:.2f} <= {DEAD_CEILING}) -- radius too hard, narrow it"
    else:
        verdict = f"GOOD -- non-saturated, non-zero learning curve at this radius"
    print(f"\nVERDICT: {verdict}")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(combined["cum_timesteps"], rolling_success, color="green", linewidth=2)
    ax.axhline(SATURATED_FLOOR, color="red", linestyle="--", label=f"saturated floor={SATURATED_FLOOR}")
    ax.axhline(DEAD_CEILING, color="gray", linestyle="--", label=f"dead ceiling={DEAD_CEILING}")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("timesteps")
    ax.set_ylabel(f"rolling success rate, window={ROLLING_WINDOW}")
    ax.set_title(f"Differentiation pre-check -- PPO default, radius={args.radius}\n{verdict}", fontsize=10)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "precheck_curve.png", dpi=130)
    plt.close(fig)

    summary = dict(
        radius=args.radius, timesteps=args.timesteps, n_envs=N_ENVS, seed=SEED,
        n_episodes=n, mean_episode_length=mean_episode_len, throughput_steps_per_sec=throughput,
        wall_clock_s=elapsed, overall_success=overall_success,
        first_half_success=first_half_success, second_half_success=second_half_success,
        verdict=verdict,
    )
    with open(run_dir / "precheck_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {run_dir / 'precheck_summary.json'}")


if __name__ == "__main__":
    main()
