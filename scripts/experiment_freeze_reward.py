"""EXPERIMENT (not a commit): find the minimal freeze-reward intervention
that produces real PLACEMENT learning (terminal d marching toward the
12mm tolerance, success lifting off zero) without freeze-attempt runaway.

Prior results this experiment builds on (see status.md Addenda 6-7):
  - freeze_miss_penalty=-2.0 (original): freeze never explored (0.000 attempt rate)
  - freeze_miss_penalty=0.0 + ent_coef=0.05, 200k steps: RUNAWAY
    (freeze-attempt rate ~0.987, success 0.000, terminal d never improves)

Hypothesis this experiment tests: the "cliff" freeze reward (flat penalty
for ANY miss, regardless of how close) gives the agent no gradient toward
better placement -- a freeze at 13mm and a freeze at 90mm are rewarded
identically outside tolerance. Arms A/B test whether the miss-PENALTY dial
alone (still a cliff) has an interior value that works; Arm C removes the
cliff itself via a placement-graded reward
(`10*exp(-d/freeze_grade_sigma_m) - freeze_attempt_cost`).

The INSIDE-TOLERANCE success definition (alpha<=alpha_tol, d<=d_tol -> +10
scaled by alpha, acquisition, termination) is IDENTICAL across every arm --
only what happens on a MISS changes. Shaping potential, distance scale,
alpha term, tolerances, phantom geometry, features.py, actuator clamp,
classifier, and head are untouched by this whole experiment.

All arms: PPO, femur-only, SubprocVecEnv n_envs=4, uncapped,
total_timesteps=200_000, seed=0. Live per-episode logging via
Monitor(info_keywords=(...)).

Usage: uv run python scripts/experiment_freeze_reward.py [--arm A|B|C|all]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from training.pg_training import train_ppo

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = REPO_ROOT / "logs" / "freeze_reward_experiment"
MODEL_ROOT = REPO_ROOT / "models" / "freeze_reward_experiment"

N_ENVS = 4
TOTAL_TIMESTEPS = 200_000
SEED = 0
D_TOL_MM = 12.0
ROLLING_WINDOW = 50
INFO_KEYWORDS = ("success", "freeze_attempted", "d_m", "alpha_deg")

# WIN thresholds (behavioral, not reward-based)
WIN_D_THRESHOLD_MM = D_TOL_MM * 2.0   # "near tolerance": within 2x d_tol
WIN_SUCCESS_FLOOR = 0.1
RUNAWAY_FREEZE_THRESHOLD = 0.9
RUNAWAY_SUCCESS_CEILING = 0.1
STALL_FREEZE_FLOOR = 0.1  # "attempts healthy" needs at least this


ARMS = {
    "armA_ent_original_penalty": dict(
        env_kwargs=dict(freeze_miss_penalty=-2.0),
        ent_coef=0.05,
    ),
    "armB_penalty_neg0.5": dict(
        env_kwargs=dict(freeze_miss_penalty=-0.5),
        ent_coef=0.05,
    ),
    "armB2_penalty_neg0.3": dict(
        env_kwargs=dict(freeze_miss_penalty=-0.3),
        ent_coef=0.05,
    ),
    "armC_graded_sigma0.03_cost0.3": dict(
        env_kwargs=dict(freeze_reward_mode="graded", freeze_grade_sigma_m=0.03, freeze_attempt_cost=0.3),
        ent_coef=0.05,
    ),
    "armC2_graded_sigma0.03_cost1.0": dict(
        env_kwargs=dict(freeze_reward_mode="graded", freeze_grade_sigma_m=0.03, freeze_attempt_cost=1.0),
        ent_coef=0.05,
    ),
}


def classify(late_freeze, late_success, late_d):
    if late_freeze >= RUNAWAY_FREEZE_THRESHOLD and late_success <= RUNAWAY_SUCCESS_CEILING:
        return "RUNAWAY"
    if late_d <= WIN_D_THRESHOLD_MM and late_success > WIN_SUCCESS_FLOOR:
        return "WIN"
    if late_freeze < STALL_FREEZE_FLOOR:
        return "EXTINGUISHED"
    if late_d > WIN_D_THRESHOLD_MM and late_success <= WIN_SUCCESS_FLOOR:
        return "STALLED"
    return "AMBIGUOUS"


def run_arm(name: str, env_kwargs: dict, ent_coef: float):
    log_dir = OUT_ROOT / name
    model_dir = MODEL_ROOT / name
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    full_env_kwargs = dict(single_target=True, single_target_which="femur", **env_kwargs)
    print(f"\n=== ARM: {name} ===")
    print(f"  env_kwargs={env_kwargs}, ent_coef={ent_coef}")

    config = dict(entropy_coef=ent_coef)
    model, save_path = train_ppo(
        config, str(log_dir), str(model_dir), seed=SEED,
        total_timesteps=TOTAL_TIMESTEPS, env_kwargs=full_env_kwargs,
        n_envs=N_ENVS, max_wall_clock_seconds=None,
        info_keywords=INFO_KEYWORDS,
    )

    monitor_paths = sorted(log_dir.glob("monitor*.csv"))
    frames = []
    for p in monitor_paths:
        df = pd.read_csv(p, skiprows=1)
        if "r" in df.columns and len(df) > 0:
            frames.append(df)
    combined = pd.concat(frames, ignore_index=True).sort_values("t").reset_index(drop=True)
    combined["cum_timesteps"] = combined["l"].cumsum()
    combined["d_mm"] = combined["d_m"] * 1000.0
    combined.to_csv(log_dir / "episodes.csv", index=False)

    n = len(combined)
    last_quarter = combined.iloc[3 * n // 4:]
    overall_success = float(combined["success"].mean())
    overall_freeze = float(combined["freeze_attempted"].mean())
    late_success = float(last_quarter["success"].mean())
    late_freeze = float(last_quarter["freeze_attempted"].mean())
    late_d = float(last_quarter["d_mm"].median())

    verdict = classify(late_freeze, late_success, late_d)

    print(f"  N={n} episodes. Overall: success={overall_success:.4f}, freeze={overall_freeze:.4f}")
    print(f"  Last quarter: success={late_success:.4f}, freeze={late_freeze:.4f}, "
          f"median terminal d={late_d:.2f}mm")
    print(f"  ARM VERDICT: {verdict}")

    # trajectory plot
    rolling_d = combined["d_mm"].rolling(ROLLING_WINDOW, min_periods=1).median()
    rolling_success = combined["success"].astype(float).rolling(ROLLING_WINDOW, min_periods=1).mean()
    rolling_freeze = combined["freeze_attempted"].astype(float).rolling(ROLLING_WINDOW, min_periods=1).mean()

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    axes[0].plot(combined["cum_timesteps"], rolling_d, color="steelblue", linewidth=2)
    axes[0].axhline(D_TOL_MM, color="red", linestyle="--", label=f"d_tol={D_TOL_MM}mm")
    axes[0].set_ylabel(f"rolling median terminal d (mm), window={ROLLING_WINDOW}")
    axes[0].set_title("Terminal d vs timesteps (leading indicator)")
    axes[0].legend()

    axes[1].plot(combined["cum_timesteps"], rolling_success, color="green", linewidth=2)
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].set_ylabel(f"rolling success rate, window={ROLLING_WINDOW}")
    axes[1].set_title("Success rate vs timesteps")

    axes[2].plot(combined["cum_timesteps"], rolling_freeze, color="orange", linewidth=2)
    axes[2].set_ylim(-0.02, 1.02)
    axes[2].set_xlabel("timesteps")
    axes[2].set_ylabel(f"rolling freeze-attempt fraction, window={ROLLING_WINDOW}")
    axes[2].set_title("Freeze-attempt fraction vs timesteps (runaway guard)")

    fig.suptitle(f"Arm: {name} -- {env_kwargs}, ent_coef={ent_coef}\nVERDICT: {verdict}", fontsize=11)
    fig.tight_layout()
    fig.savefig(log_dir / "trajectories.png", dpi=130)
    plt.close(fig)

    stats = dict(
        arm=name, env_kwargs=env_kwargs, ent_coef=ent_coef, n_episodes=n,
        overall_success_rate=overall_success, overall_freeze_attempted_rate=overall_freeze,
        late_success_rate=late_success, late_freeze_attempted_rate=late_freeze,
        late_median_terminal_d_mm=late_d, verdict=verdict,
    )
    with open(log_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", default="A", choices=["A", "B", "C", "all"])
    args = parser.parse_args()

    all_stats = []
    if args.arm in ("A", "all"):
        s = run_arm("armA_ent_original_penalty", **ARMS["armA_ent_original_penalty"])
        all_stats.append(s)
        if s["verdict"] == "WIN":
            print("\nArm A WINS -- stopping early per 'stop early if a cheaper arm clearly succeeds'.")
            _save_summary(all_stats)
            return

    if args.arm in ("B", "all"):
        s = run_arm("armB_penalty_neg0.5", **ARMS["armB_penalty_neg0.5"])
        all_stats.append(s)
        if s["verdict"] == "WIN":
            print("\nArm B (-0.5) WINS -- stopping early.")
            _save_summary(all_stats)
            return
        s2 = run_arm("armB2_penalty_neg0.3", **ARMS["armB2_penalty_neg0.3"])
        all_stats.append(s2)
        if s2["verdict"] == "WIN":
            print("\nArm B2 (-0.3) WINS -- stopping early.")
            _save_summary(all_stats)
            return

    if args.arm in ("C", "all"):
        s = run_arm("armC_graded_sigma0.03_cost0.3", **ARMS["armC_graded_sigma0.03_cost0.3"])
        all_stats.append(s)
        if s["verdict"] == "WIN":
            print("\nArm C (cost=0.3) WINS.")
            _save_summary(all_stats)
            return
        print(f"\nArm C (cost=0.3) verdict={s['verdict']} -- trying a stronger anti-spam cost...")
        s2 = run_arm("armC2_graded_sigma0.03_cost1.0", **ARMS["armC2_graded_sigma0.03_cost1.0"])
        all_stats.append(s2)

    _save_summary(all_stats)


def _save_summary(all_stats):
    print("\n\n=== SUMMARY (all arms run) ===")
    header = f"{'arm':<32} {'success':>9} {'freeze':>9} {'terminal_d_mm':>14} {'verdict':>12}"
    print(header)
    for s in all_stats:
        print(f"{s['arm']:<32} {s['late_success_rate']:>9.4f} {s['late_freeze_attempted_rate']:>9.4f} "
              f"{s['late_median_terminal_d_mm']:>14.2f} {s['verdict']:>12}")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    with open(OUT_ROOT / "experiment_summary.json", "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"\nSaved to {OUT_ROOT / 'experiment_summary.json'}")


if __name__ == "__main__":
    main()
