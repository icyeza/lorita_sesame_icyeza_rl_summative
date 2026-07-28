"""Short CONFIRMATION calibration run (NOT a sweep, NOT a budget decision):
after raising POTENTIAL_D_SCALE (0.015 -> 0.05, see environment/custom_env.py),
checks whether the training wall documented in status.md ("single_target PPO
calibration wall") breaks. Same methodology as scripts/run_calibration.py +
scripts/diagnose_calibration_plateau.py, run fresh at a modest budget (tens
of thousands of steps -- enough to see the wall break or not, not a
convergence run) so this run's own data can show success/freeze/terminal-d
evolving, unlike the walled run where every checkpoint would have looked
the same anyway.

Logs to logs/calibration_confirm/ (separate from logs/calibration/, which
documents the PRE-fix walled run and must not be overwritten).

Usage: uv run python scripts/run_calibration_confirm.py
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

from environment.custom_env import UltrasoundProbeEnv, ACTIONS
from evaluation.evaluate import load_model
from training.pg_training import train_ppo

REPO_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = REPO_ROOT / "logs" / "calibration_confirm"
MODEL_DIR = REPO_ROOT / "models" / "calibration_confirm"

ALGO = "ppo"
CURRICULUM = "single_target"
N_ENVS = 4
TOTAL_TIMESTEPS = 60_000
SEED = 0
ROLLING_WINDOW_REWARD = 20
ROLLING_WINDOW_SUCCESS = 30
N_EVAL_EPISODES = 100
FREEZE_ACTION = ACTIONS.index("freeze_and_measure")
ALPHA_TOL_DEG = 15.0
D_TOL_MM = 12.0


def train_and_collect():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"CONFIRMATION calibration run (post distance-scale fix): algo={ALGO} "
          f"curriculum={CURRICULUM} n_envs={N_ENVS} total_timesteps={TOTAL_TIMESTEPS} seed={SEED}")
    env_kwargs = dict(single_target=True)
    model, save_path = train_ppo(
        {}, str(LOG_DIR), str(MODEL_DIR), seed=SEED,
        total_timesteps=TOTAL_TIMESTEPS, env_kwargs=env_kwargs,
        n_envs=N_ENVS, max_wall_clock_seconds=None,
    )
    print(f"Training complete. Model saved to {save_path}")

    monitor_paths = sorted(LOG_DIR.glob("monitor*.csv"))
    frames = []
    for p in monitor_paths:
        df = pd.read_csv(p, skiprows=1)
        if "r" in df.columns and len(df) > 0:
            df["worker"] = p.name
            frames.append(df)
    combined = pd.concat(frames, ignore_index=True).sort_values("t").reset_index(drop=True)
    combined["cum_timesteps"] = combined["l"].cumsum()
    combined["success_proxy"] = combined["l"] < 60
    combined.to_csv(LOG_DIR / "confirm_episodes.csv", index=False)
    return combined


def plot_reward_and_success(combined: pd.DataFrame):
    fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True)

    axes[0].scatter(combined["cum_timesteps"], combined["r"], s=6, alpha=0.25, color="steelblue")
    rolling_r = combined["r"].rolling(ROLLING_WINDOW_REWARD, min_periods=1).mean()
    axes[0].plot(combined["cum_timesteps"], rolling_r, color="darkorange", linewidth=2,
                 label=f"rolling mean (window={ROLLING_WINDOW_REWARD})")
    axes[0].set_ylabel("episode reward")
    axes[0].set_title(f"CONFIRMATION run post distance-scale fix -- PPO, single_target\n"
                       f"(n_envs={N_ENVS}, total_timesteps={TOTAL_TIMESTEPS}, seed={SEED})")
    axes[0].legend(fontsize=9)

    rolling_success = combined["success_proxy"].astype(float).rolling(ROLLING_WINDOW_SUCCESS, min_periods=1).mean()
    axes[1].plot(combined["cum_timesteps"], rolling_success, color="crimson", linewidth=2)
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].set_xlabel("timesteps")
    axes[1].set_ylabel(f"rolling success rate (window={ROLLING_WINDOW_SUCCESS})")
    axes[1].set_title("success rate vs timesteps (exact proxy: episode length < 60)")

    fig.tight_layout()
    out_path = LOG_DIR / "confirm_reward_and_success.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Saved {out_path}")
    return out_path


def checkpoint_snapshot(seed: int = 54321):
    print("\n=== End-of-training checkpoint snapshot (final model) ===")
    env = UltrasoundProbeEnv(seed=seed, single_target=True)
    model = load_model("ppo", str(MODEL_DIR / "model.zip"), env)

    rows = []
    for ep in range(N_EVAL_EPISODES):
        obs, info = env.reset(seed=seed + ep)
        freeze_attempted = False
        step_rewards = []
        done = False
        final_info = info
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
            if action == FREEZE_ACTION:
                freeze_attempted = True
            obs, reward, terminated, truncated, info = env.step(action)
            step_rewards.append(reward)
            final_info = info
            done = terminated or truncated
        success = len(step_rewards) < 60
        rows.append(dict(
            episode=ep, length=len(step_rewards), success=success,
            freeze_attempted=freeze_attempted,
            terminal_alpha_deg=final_info.get("alpha_deg"),
            terminal_d_mm=(final_info.get("d_m") or 0) * 1000,
            total_reward=float(sum(step_rewards)),
        ))
    df = pd.DataFrame(rows)
    df.to_csv(LOG_DIR / "confirm_checkpoint_snapshot.csv", index=False)

    stats = dict(
        success_rate=float(df["success"].mean()),
        freeze_attempted_rate=float(df["freeze_attempted"].mean()),
        median_terminal_alpha_deg=float(df["terminal_alpha_deg"].median()),
        median_terminal_d_mm=float(df["terminal_d_mm"].median()),
    )
    print(f"  success_rate={stats['success_rate']:.3f}, "
          f"freeze_attempted_rate={stats['freeze_attempted_rate']:.3f}")
    print(f"  median terminal alpha={stats['median_terminal_alpha_deg']:.2f}deg (tol={ALPHA_TOL_DEG})")
    print(f"  median terminal d={stats['median_terminal_d_mm']:.2f}mm (tol={D_TOL_MM})")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].hist(df["terminal_alpha_deg"], bins=20, color="steelblue", alpha=0.8)
    axes[0].axvline(ALPHA_TOL_DEG, color="red", linestyle="--", label=f"alpha_tol={ALPHA_TOL_DEG}deg")
    axes[0].set_xlabel("terminal alpha (deg)")
    axes[0].set_title(f"Terminal alpha (n={N_EVAL_EPISODES}, end-of-run snapshot)")
    axes[0].legend()
    axes[1].hist(df["terminal_d_mm"], bins=20, color="seagreen", alpha=0.8)
    axes[1].axvline(D_TOL_MM, color="red", linestyle="--", label=f"d_tol={D_TOL_MM}mm")
    axes[1].set_xlabel("terminal d (mm)")
    axes[1].set_title(f"Terminal d (n={N_EVAL_EPISODES}, end-of-run snapshot)")
    axes[1].legend()
    fig.tight_layout()
    out_path = LOG_DIR / "confirm_terminal_alpha_d.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  saved {out_path}")

    return stats


def reward_breakdown(seed: int = 54321):
    print("\n=== Reward-component breakdown (typical episode, final checkpoint) ===")
    env = UltrasoundProbeEnv(seed=seed, single_target=True)
    model = load_model("ppo", str(MODEL_DIR / "model.zip"), env)
    obs, info = env.reset(seed=seed)
    shaping_total, event_total, step_cost_total = 0.0, 0.0, 0.0
    n_steps = 0
    done = False
    froze = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)
        obs, reward, terminated, truncated, info = env.step(action)
        n_steps += 1
        step_cost_total += -0.05
        remainder = reward - (-0.05)
        if action == FREEZE_ACTION:
            froze = True
            event_total += remainder
        elif n_steps == 60:
            event_total += remainder
        else:
            shaping_total += remainder
        done = terminated or truncated
    total = shaping_total + event_total + step_cost_total
    print(f"  episode length: {n_steps} steps, froze={froze}, terminated_early={n_steps < 60}")
    print(f"  accumulated shaping-delta reward: {shaping_total:+.3f}")
    print(f"  accumulated event reward: {event_total:+.3f}")
    print(f"  accumulated step-cost: {step_cost_total:+.3f}")
    print(f"  total: {total:+.3f}")
    return dict(shaping_total=shaping_total, event_total=event_total,
                step_cost_total=step_cost_total, n_steps=n_steps, froze=froze)


def main():
    combined = train_and_collect()
    plot_reward_and_success(combined)
    snapshot_stats = checkpoint_snapshot()
    breakdown = reward_breakdown()

    summary = dict(
        algo=ALGO, curriculum=CURRICULUM, n_envs=N_ENVS, total_timesteps=TOTAL_TIMESTEPS, seed=SEED,
        n_episodes=len(combined),
        whole_run_success_rate=float(combined["success_proxy"].mean()),
        checkpoint_snapshot=snapshot_stats,
        reward_breakdown=breakdown,
    )
    with open(LOG_DIR / "confirm_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nWhole-run success rate (length<60 proxy, N={len(combined)}): "
          f"{summary['whole_run_success_rate']:.4f}")
    print(f"Saved full confirmation summary to {LOG_DIR / 'confirm_summary.json'}")


if __name__ == "__main__":
    main()
