"""EXPERIMENT (not a commit): find the minimal lever that makes the agent
ever attempt `freeze_and_measure` on a fully-recovered target (femur,
improvement ratio 0.766 post distance-shaping-fix). Freeze-attempt fraction
was measured at exactly 0.000 in the post-distance-fix confirmation run --
the hypothesis is exploration collapse on that one action (early training,
every freeze is a miss -> -2 penalty -> policy drives its probability to
~0 and never resamples it once poses improve).

Arms (all femur-only, PPO, SubprocVecEnv n_envs=4, uncapped, same seed):
  baseline : current config (distance fix only, default ent_coef=0.0,
             default freeze_miss_penalty=-2.0) -- confirms the wall
             reproduces on a clean, confound-free target.
  arm1a    : ent_coef=0.01 (pure hyperparameter, NO env change)
  arm1b    : ent_coef=0.05 (pure hyperparameter, NO env change)
  arm2     : freeze_miss_penalty=0.0 (ENV reward change -- tested only,
             NOT committed as the new default) -- only run if arm1 is
             insufficient.

Nothing here changes the default environment. `single_target_which="femur"`
and `freeze_miss_penalty` are experimental constructor arguments (see
environment/custom_env.py) that default to the original behavior; this
script is the only caller that sets them away from default.

Usage: uv run python scripts/experiment_freeze_wall.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from environment.custom_env import UltrasoundProbeEnv, ACTIONS
from evaluation.evaluate import load_model
from training.pg_training import train_ppo

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = REPO_ROOT / "logs" / "freeze_wall_experiment"
MODEL_ROOT = REPO_ROOT / "models" / "freeze_wall_experiment"

N_ENVS = 4
TOTAL_TIMESTEPS = 40_000
SEED = 0
N_EVAL_EPISODES = 100
FREEZE_ACTION = ACTIONS.index("freeze_and_measure")
ROLLING_WINDOW = 30

ARMS = {
    "baseline": dict(ent_coef=0.0, freeze_miss_penalty=-2.0),
    "arm1a_ent0.01": dict(ent_coef=0.01, freeze_miss_penalty=-2.0),
    "arm1b_ent0.05": dict(ent_coef=0.05, freeze_miss_penalty=-2.0),
}
# arm2 added conditionally at runtime only if arm1 is insufficient -- see main()
ARM2 = {"arm2_miss0": dict(ent_coef=0.0, freeze_miss_penalty=0.0)}


def run_arm(name: str, ent_coef: float, freeze_miss_penalty: float):
    log_dir = OUT_ROOT / name
    model_dir = MODEL_ROOT / name
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== ARM: {name} (ent_coef={ent_coef}, freeze_miss_penalty={freeze_miss_penalty}) ===")
    env_kwargs = dict(single_target=True, single_target_which="femur",
                       freeze_miss_penalty=freeze_miss_penalty)
    config = dict(entropy_coef=ent_coef)
    model, save_path = train_ppo(
        config, str(log_dir), str(model_dir), seed=SEED,
        total_timesteps=TOTAL_TIMESTEPS, env_kwargs=env_kwargs,
        n_envs=N_ENVS, max_wall_clock_seconds=None,
    )

    # success-rate-vs-timesteps from existing logs (exact proxy: length<60)
    monitor_paths = sorted(log_dir.glob("monitor*.csv"))
    frames = []
    for p in monitor_paths:
        df = pd.read_csv(p, skiprows=1)
        if "r" in df.columns and len(df) > 0:
            frames.append(df)
    combined = pd.concat(frames, ignore_index=True).sort_values("t").reset_index(drop=True)
    combined["cum_timesteps"] = combined["l"].cumsum()
    combined["success_proxy"] = combined["l"] < 60
    combined.to_csv(log_dir / "episodes.csv", index=False)
    whole_run_success_rate = float(combined["success_proxy"].mean())

    # end-of-run checkpoint snapshot: freeze-attempt fraction + terminal d
    eval_env = UltrasoundProbeEnv(seed=SEED + 999, single_target=True, single_target_which="femur",
                                   freeze_miss_penalty=freeze_miss_penalty)
    model_eval = load_model("ppo", str(Path(save_path)), eval_env)
    rows = []
    for ep in range(N_EVAL_EPISODES):
        obs, info = eval_env.reset(seed=SEED + 999 + ep)
        freeze_attempted = False
        step_rewards = []
        done = False
        final_info = info
        while not done:
            action, _ = model_eval.predict(obs, deterministic=True)
            action = int(action)
            if action == FREEZE_ACTION:
                freeze_attempted = True
            obs, reward, terminated, truncated, info = eval_env.step(action)
            step_rewards.append(reward)
            final_info = info
            done = terminated or truncated
        rows.append(dict(
            length=len(step_rewards), success=len(step_rewards) < 60,
            freeze_attempted=freeze_attempted,
            terminal_d_mm=(final_info.get("d_m") or 0) * 1000,
        ))
    snap = pd.DataFrame(rows)
    snap.to_csv(log_dir / "snapshot.csv", index=False)

    stats = dict(
        arm=name, ent_coef=ent_coef, freeze_miss_penalty=freeze_miss_penalty,
        whole_run_success_rate=whole_run_success_rate,
        n_episodes=len(combined),
        snapshot_success_rate=float(snap["success"].mean()),
        snapshot_freeze_attempted_rate=float(snap["freeze_attempted"].mean()),
        snapshot_median_terminal_d_mm=float(snap["terminal_d_mm"].median()),
    )
    print(f"  whole-run success rate: {stats['whole_run_success_rate']:.4f} (N={stats['n_episodes']})")
    print(f"  snapshot: freeze_attempted_rate={stats['snapshot_freeze_attempted_rate']:.3f}, "
          f"success_rate={stats['snapshot_success_rate']:.3f}, "
          f"median terminal d={stats['snapshot_median_terminal_d_mm']:.2f}mm")

    # plot: rolling success rate vs timesteps for this arm
    fig, ax = plt.subplots(figsize=(9, 4.5))
    rolling = combined["success_proxy"].astype(float).rolling(ROLLING_WINDOW, min_periods=1).mean()
    ax.plot(combined["cum_timesteps"], rolling, color="crimson", linewidth=2)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("timesteps")
    ax.set_ylabel(f"rolling success rate (window={ROLLING_WINDOW})")
    ax.set_title(f"Arm: {name} -- femur-only, ent_coef={ent_coef}, "
                 f"freeze_miss_penalty={freeze_miss_penalty}")
    fig.tight_layout()
    fig.savefig(log_dir / "success_rate.png", dpi=130)
    plt.close(fig)

    with open(log_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    return stats


def main():
    all_stats = []
    for name, cfg in ARMS.items():
        all_stats.append(run_arm(name, **cfg))

    arm1_broke_wall = any(
        s["snapshot_freeze_attempted_rate"] > 0.1 for s in all_stats if s["arm"].startswith("arm1")
    )
    if not arm1_broke_wall:
        print("\nArm 1 (entropy alone) did not clearly break the wall -- running Arm 2 "
              "(freeze_miss_penalty=0.0, an ENV reward change, TEST ONLY)...")
        for name, cfg in ARM2.items():
            all_stats.append(run_arm(name, **cfg))
    else:
        print("\nArm 1 (entropy alone) broke the wall -- skipping Arm 2 (no env change needed).")

    print("\n\n=== SUMMARY (all arms) ===")
    header = f"{'arm':<16} {'ent_coef':>9} {'miss_pen':>9} {'freeze_rate':>12} {'success_rate':>13} {'median_d_mm':>12}"
    print(header)
    for s in all_stats:
        print(f"{s['arm']:<16} {s['ent_coef']:>9} {s['freeze_miss_penalty']:>9} "
              f"{s['snapshot_freeze_attempted_rate']:>12.3f} {s['snapshot_success_rate']:>13.3f} "
              f"{s['snapshot_median_terminal_d_mm']:>12.2f}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    with open(OUT_ROOT / "experiment_summary.json", "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"\nSaved summary to {OUT_ROOT / 'experiment_summary.json'}")


if __name__ == "__main__":
    main()
