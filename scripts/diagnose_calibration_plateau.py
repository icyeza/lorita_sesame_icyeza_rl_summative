"""Disambiguate the calibration run's reward plateau: solved-early (agent
acquires the target almost instantly, -6 is the ceiling) vs walled (a
hover-for-shaping local optimum where the agent never commits to `freeze`).

NO RETRAINING. Two data sources, both from the existing calibration run:

(A) Episode length, already logged in logs/calibration/*.csv (SB3 Monitor's
    default r/l/t columns -- no custom info_keywords were configured, so
    per-episode success/freeze/alpha/d were NOT logged live). In
    single_target mode, SUBTASK_MAX_STEPS=60 and EPISODE_MAX_STEPS=180 --
    since the single sub-task's 60-step timeout always fires before the
    180-step episode cap could ever matter, an episode can ONLY end two
    ways: a successful freeze (terminates early, length < 60) or a
    sub-task timeout (length == 60 exactly). This makes "length < 60" an
    exact (not approximate) proxy for "successfully froze within
    tolerance," recoverable from data already logged during training --
    genuinely time-resolved across the whole run, no re-evaluation needed.

(B) Only ONE PPO checkpoint exists for this run (models/calibration/model.zip,
    saved once at the end of training -- no CheckpointCallback was
    configured, so no intermediate checkpoints exist and none can be
    created without retraining, which is out of scope here). Freeze-ATTEMPT
    fraction and terminal alpha/d are not recoverable from the existing
    r/l/t logs at all (a failed freeze doesn't change episode length or
    terminate the episode). These are measured via ONE deterministic
    evaluation of that single final checkpoint -- an end-of-training
    snapshot, NOT a time series -- clearly labeled as such.

Usage: uv run python scripts/diagnose_calibration_plateau.py
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

REPO_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = REPO_ROOT / "logs" / "calibration"
OUT_DIR = REPO_ROOT / "logs" / "calibration_plateau_diagnosis"
MODEL_PATH = REPO_ROOT / "models" / "calibration" / "model.zip"

ROLLING_WINDOW = 50  # episodes -- wider than the calibration plot's 20 since success is a 0/1 signal
N_EVAL_EPISODES = 100
FREEZE_ACTION = ACTIONS.index("freeze_and_measure")
ALPHA_TOL_DEG = 15.0
D_TOL_MM = 12.0


def task_a_success_proxy_from_existing_logs():
    """Derive the success-rate-vs-timesteps curve entirely from data
    already logged during the calibration run -- no re-evaluation."""
    combined = pd.read_csv(LOG_DIR / "calibration_episodes.csv").sort_values("cum_timesteps").reset_index(drop=True)
    combined["success_proxy"] = combined["l"] < 60  # exact, see module docstring
    n_success = int(combined["success_proxy"].sum())
    n_total = len(combined)
    print(f"=== TASK A: success proxy from existing logs (length < 60) ===")
    print(f"  {n_success}/{n_total} episodes ({n_success / n_total:.4f}) show length < 60 "
          f"(i.e. a successful freeze occurred before the 60-step sub-task timeout)")
    if n_success > 0:
        success_rows = combined[combined["success_proxy"]]
        print(f"  successful episodes occurred at cum_timesteps: {success_rows['cum_timesteps'].tolist()}")

    rolling_success = combined["success_proxy"].astype(float).rolling(ROLLING_WINDOW, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(combined["cum_timesteps"], rolling_success, color="crimson", linewidth=2)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("timesteps")
    ax.set_ylabel(f"rolling success rate (window={ROLLING_WINDOW} episodes)")
    ax.set_title("Task A: single-target SUCCESS RATE vs timesteps\n"
                  "(exact proxy: episode length < 60 = successful freeze before sub-task timeout)\n"
                  "derived from existing calibration logs, no re-evaluation")
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "A_success_rate.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  saved {out_path}")
    return combined, n_success, n_total


def task_bc_checkpoint_snapshot(n_episodes: int = N_EVAL_EPISODES, seed: int = 12345):
    """ONE deterministic evaluation of the ONLY available checkpoint
    (end-of-training, 200k steps). Not a time series -- an end-of-training
    snapshot, clearly labeled. Collects: success, freeze-attempted (any
    freeze_and_measure action, regardless of outcome), terminal alpha/d,
    and a full reward-component breakdown for a "walled" diagnosis if
    warranted."""
    print(f"\n=== TASKS B/C: end-of-training checkpoint snapshot "
          f"(ONLY {1} checkpoint available: {MODEL_PATH.name}, saved at 200k steps) ===")
    env = UltrasoundProbeEnv(seed=seed, single_target=True)
    model = load_model("ppo", str(MODEL_PATH), env)

    rows = []
    for ep in range(n_episodes):
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

        success = len(step_rewards) < 60  # same exact proxy, cross-check
        rows.append(dict(
            episode=ep, length=len(step_rewards), success=success,
            freeze_attempted=freeze_attempted,
            terminal_alpha_deg=final_info.get("alpha_deg"),
            terminal_d_mm=(final_info.get("d_m") or 0) * 1000,
            total_reward=float(sum(step_rewards)),
        ))

    df = pd.DataFrame(rows)
    success_rate = df["success"].mean()
    freeze_rate = df["freeze_attempted"].mean()
    median_alpha = df["terminal_alpha_deg"].median()
    median_d = df["terminal_d_mm"].median()

    print(f"  n_episodes={n_episodes}, success_rate={success_rate:.3f}, "
          f"freeze_attempted_rate={freeze_rate:.3f}")
    print(f"  terminal alpha (deg): median={median_alpha:.2f}, "
          f"min={df['terminal_alpha_deg'].min():.2f}, max={df['terminal_alpha_deg'].max():.2f} "
          f"(alpha_tol={ALPHA_TOL_DEG}deg)")
    print(f"  terminal d (mm): median={median_d:.2f}, "
          f"min={df['terminal_d_mm'].min():.2f}, max={df['terminal_d_mm'].max():.2f} "
          f"(d_tol={D_TOL_MM}mm)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "checkpoint_snapshot_episodes.csv", index=False)

    # Plot B: terminal alpha/d as a single labeled snapshot (NOT a trend --
    # only one checkpoint exists) with tolerance lines for reference.
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].hist(df["terminal_alpha_deg"], bins=20, color="steelblue", alpha=0.8)
    axes[0].axvline(ALPHA_TOL_DEG, color="red", linestyle="--", label=f"alpha_tol={ALPHA_TOL_DEG}deg")
    axes[0].set_xlabel("terminal alpha (deg)")
    axes[0].set_title(f"Terminal alpha at episode end\n(END-OF-TRAINING SNAPSHOT, n={n_episodes}, NOT a trend)")
    axes[0].legend()

    axes[1].hist(df["terminal_d_mm"], bins=20, color="seagreen", alpha=0.8)
    axes[1].axvline(D_TOL_MM, color="red", linestyle="--", label=f"d_tol={D_TOL_MM}mm")
    axes[1].set_xlabel("terminal d (mm)")
    axes[1].set_title(f"Terminal d at episode end\n(END-OF-TRAINING SNAPSHOT, n={n_episodes}, NOT a trend)")
    axes[1].legend()
    fig.tight_layout()
    out_path_b = OUT_DIR / "B_terminal_alpha_d_snapshot.png"
    fig.savefig(out_path_b, dpi=130)
    plt.close(fig)
    print(f"  saved {out_path_b}")

    # Plot C: freeze-attempt vs success, single bar snapshot (again, ONE
    # checkpoint -- not a time series).
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(["freeze attempted", "succeeded"], [freeze_rate, success_rate], color=["orange", "green"])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("fraction of episodes")
    ax.set_title(f"Task C: freeze-attempt vs success\n(END-OF-TRAINING SNAPSHOT, n={n_episodes}, NOT a trend)")
    fig.tight_layout()
    out_path_c = OUT_DIR / "C_freeze_fraction_snapshot.png"
    fig.savefig(out_path_c, dpi=130)
    plt.close(fig)
    print(f"  saved {out_path_c}")

    return df, dict(success_rate=success_rate, freeze_rate=freeze_rate,
                     median_alpha=median_alpha, median_d=median_d)


def reward_component_breakdown(seed: int = 12345):
    """For a typical episode from the final checkpoint, break down
    accumulated shaping reward vs event rewards vs step-cost, to inform
    (not decide) a freeze-reward/step-cost rebalance if the regime is
    walled."""
    print("\n=== Reward-component breakdown (typical episode, final checkpoint, deterministic) ===")
    env = UltrasoundProbeEnv(seed=seed, single_target=True)
    model = load_model("ppo", str(MODEL_PATH), env)
    obs, info = env.reset(seed=seed)
    shaping_total, event_total, step_cost_total = 0.0, 0.0, 0.0
    n_steps = 0
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)
        prev_potential = env._prev_potential
        obs, reward, terminated, truncated, info = env.step(action)
        n_steps += 1
        # step-cost is always exactly -0.05/step (custom_env.py); everything
        # else in `reward` is shaping delta + any event bonus/penalty this step
        step_cost_total += -0.05
        remainder = reward - (-0.05)
        # event rewards only occur on freeze or timeout steps (large in
        # magnitude); shaping deltas are the potential-based term otherwise.
        if action == FREEZE_ACTION or n_steps == 60:
            event_total += remainder
        else:
            shaping_total += remainder
        done = terminated or truncated

    total = shaping_total + event_total + step_cost_total
    print(f"  episode length: {n_steps} steps")
    print(f"  accumulated shaping-delta reward: {shaping_total:+.3f}")
    print(f"  accumulated event reward (freeze/timeout steps): {event_total:+.3f}")
    print(f"  accumulated step-cost: {step_cost_total:+.3f}")
    print(f"  total (should match episode reward): {total:+.3f}")
    return dict(shaping_total=shaping_total, event_total=event_total,
                step_cost_total=step_cost_total, n_steps=n_steps)


def main():
    combined, n_success, n_total = task_a_success_proxy_from_existing_logs()
    snapshot_df, snapshot_stats = task_bc_checkpoint_snapshot()

    success_rate_overall = n_success / n_total
    freeze_rate = snapshot_stats["freeze_rate"]
    median_alpha = snapshot_stats["median_alpha"]
    median_d = snapshot_stats["median_d"]

    if success_rate_overall < 0.05 and freeze_rate < 0.2 and median_alpha > ALPHA_TOL_DEG:
        regime = "WALLED"
    elif success_rate_overall > 0.8 and freeze_rate > 0.8 and median_alpha <= ALPHA_TOL_DEG and median_d <= D_TOL_MM:
        regime = "SOLVED-EARLY"
    else:
        regime = "MIXED/PARTIAL"

    print(f"\n=== REGIME: {regime} ===")
    print(f"  whole-run success rate (length<60 proxy, N={n_total}): {success_rate_overall:.4f}")
    print(f"  end-of-training snapshot: freeze_attempted_rate={freeze_rate:.3f}, "
          f"success_rate={snapshot_stats['success_rate']:.3f}, "
          f"median terminal alpha={median_alpha:.2f}deg (tol={ALPHA_TOL_DEG}), "
          f"median terminal d={median_d:.2f}mm (tol={D_TOL_MM})")

    breakdown = None
    if regime == "WALLED":
        breakdown = reward_component_breakdown()

    summary = dict(
        regime=regime,
        whole_run_success_rate=success_rate_overall,
        n_episodes_total=n_total,
        n_episodes_success=n_success,
        checkpoint_snapshot=snapshot_stats,
        reward_breakdown=breakdown,
    )
    with open(OUT_DIR / "diagnosis_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved full diagnosis to {OUT_DIR}")


if __name__ == "__main__":
    main()
