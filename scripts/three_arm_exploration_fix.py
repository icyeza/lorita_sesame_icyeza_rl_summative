"""Three-arm exploration fix: break the far-field wall.

Addendum 12 fixed the oracle-level geometry (action-granularity/tolerance
co-calibration, 92% oracle success at alpha_tol=18deg) but a 40k-step PPO
smoke train under that config showed a pure exploration failure: success
stuck at exactly 0%, and critically terminal alpha DRIFTED THE WRONG WAY
(67.6deg -> 74.6deg) while terminal d barely moved. Likely mechanism: the
multiplicative reward f(alpha)*g(d) (which correctly fixed the near-goal
bank-one-axis problem) goes nearly flat in the far field -- when both
factors are small, their product is smaller still, so a fresh policy
starting far from the goal has almost no usable gradient and wanders.

Three independent levers attack this, each changing exactly ONE thing,
each a short 40k-step smoke train (confirm learning, not convergence):
  Arm 1 (hybrid reward):    shaping_mode="hybrid" -- keeps the
                             multiplicative term but adds a weak additive
                             breadcrumb so the far field is not flat.
  Arm 2 (start curriculum): start_curriculum=True -- initializes the probe
                             near a solvable pose (small random real-action
                             offsets from the fine-search optimizer's
                             pose) so training begins where the gradient
                             is live.
  Arm 3 (entropy bump):     ent_coef 0.05 -> 0.1, everything else
                             (multiplicative shaping, normal starts)
                             unchanged -- tests whether wider sampling
                             alone finds the ridge.

HARD RULE: judge BEHAVIORALLY. A prior pass's automated one-line heuristic
("LEARNING SIGNAL PRESENT") falsely fired on a mere terminal-d drop while
terminal alpha was getting WORSE -- this script prints an alpha-direction
check explicitly and never reports a single boolean verdict without it.

Common config: PPO, femur-only, alpha_tol_deg=18 (deg), tilt_step_deg=3.0
(the co-calibrated, gate-passing config from Addendum 12), d_tol/geometry/
features/actuator/classifier untouched, SubprocVecEnv n_envs=4, uncapped,
total_timesteps=40_000, seed=0. Nothing committed as default.

Usage: uv run python scripts/three_arm_exploration_fix.py
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
OUT_ROOT = REPO_ROOT / "logs" / "three_arm_exploration_fix"
MODEL_ROOT = REPO_ROOT / "models" / "three_arm_exploration_fix"

TOTAL_TIMESTEPS = 40_000
N_ENVS = 4
SEED = 0
ROLLING_WINDOW = 50
INFO_KEYWORDS = ("success", "freeze_attempted", "d_m", "alpha_deg")

# Common, co-calibrated config (Addendum 12), held fixed across all arms.
COMMON_ENV_KWARGS = dict(single_target=True, single_target_which="femur",
                          alpha_tol_deg=18.0, tilt_step_deg=3.0)

ARMS = {
    "arm1_hybrid_reward": dict(
        env_kwargs=dict(shaping_mode="hybrid", hybrid_weight=0.2),
        ent_coef=0.05,
    ),
    "arm2_start_curriculum": dict(
        env_kwargs=dict(shaping_mode="multiplicative", start_curriculum=True,
                         start_curriculum_max_random_steps=8),
        ent_coef=0.05,
    ),
    "arm3_entropy_bump": dict(
        env_kwargs=dict(shaping_mode="multiplicative"),
        ent_coef=0.1,
    ),
}


def run_arm(name: str, env_kwargs: dict, ent_coef: float):
    log_dir = OUT_ROOT / name
    model_dir = MODEL_ROOT / name
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    full_env_kwargs = dict(**COMMON_ENV_KWARGS, **env_kwargs)
    print(f"\n=== ARM: {name} ===")
    print(f"  env_kwargs={full_env_kwargs}, ent_coef={ent_coef}")

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
    first_quarter = combined.iloc[:n // 4]
    last_quarter = combined.iloc[3 * n // 4:]
    overall_success = float(combined["success"].mean())
    overall_freeze = float(combined["freeze_attempted"].mean())
    late_success = float(last_quarter["success"].mean())
    late_freeze = float(last_quarter["freeze_attempted"].mean())
    first_freeze = float(first_quarter["freeze_attempted"].mean())
    late_d = float(last_quarter["d_mm"].median())
    first_d = float(first_quarter["d_mm"].median())
    late_alpha = float(last_quarter["alpha_deg"].median())
    first_alpha = float(first_quarter["alpha_deg"].median())

    # BEHAVIORAL verdict -- explicit, no single-metric heuristic. Genuine
    # learning requires success visibly off 0 AND alpha trending TOWARD
    # tolerance (not just d moving, which a prior pass showed can happen
    # while alpha gets worse -- that is NOT learning).
    success_off_zero = late_success > 0.0
    alpha_toward_tolerance = late_alpha < first_alpha - 1.0  # >=1deg real improvement, not noise
    alpha_away_from_tolerance = late_alpha > first_alpha + 1.0
    d_toward_tolerance = late_d < first_d * 0.9
    freeze_declining = late_freeze < first_freeze * 0.7
    freeze_runaway = late_freeze > 0.9

    if success_off_zero and alpha_toward_tolerance:
        verdict = "GENUINE LEARNING (success off 0 AND alpha trending toward tolerance)"
    elif success_off_zero:
        verdict = "PARTIAL (success off 0 but alpha not clearly trending toward tolerance)"
    elif alpha_toward_tolerance and d_toward_tolerance:
        verdict = "EARLY PROGRESS, NO SUCCESS YET (both axes trending right way, 0 success at 40k)"
    elif alpha_away_from_tolerance:
        verdict = "NOT LEARNING (alpha drifting AWAY from tolerance, same failure as before)"
    else:
        verdict = "FLAT (no clear directional signal at 40k)"

    print(f"  N={n} episodes.")
    print(f"  Overall: success={overall_success:.4f}, freeze={overall_freeze:.4f}")
    print(f"  First quarter: freeze={first_freeze:.4f}, median terminal d={first_d:.2f}mm, "
          f"median terminal alpha={first_alpha:.2f}deg")
    print(f"  Last quarter:  freeze={late_freeze:.4f}, median terminal d={late_d:.2f}mm, "
          f"median terminal alpha={late_alpha:.2f}deg, success={late_success:.4f}")
    print(f"  alpha direction: {'TOWARD tolerance' if alpha_toward_tolerance else ('AWAY from tolerance' if alpha_away_from_tolerance else 'no clear trend')} "
          f"({first_alpha:.2f}deg -> {late_alpha:.2f}deg)")
    print(f"  freeze trend: {'declining' if freeze_declining else ('runaway' if freeze_runaway else 'stable')} "
          f"({first_freeze:.4f} -> {late_freeze:.4f})")
    print(f"  BEHAVIORAL VERDICT: {verdict}")

    # trajectory plot
    rolling_d = combined["d_mm"].rolling(ROLLING_WINDOW, min_periods=1).median()
    rolling_alpha = combined["alpha_deg"].rolling(ROLLING_WINDOW, min_periods=1).median()
    rolling_success = combined["success"].astype(float).rolling(ROLLING_WINDOW, min_periods=1).mean()
    rolling_freeze = combined["freeze_attempted"].astype(float).rolling(ROLLING_WINDOW, min_periods=1).mean()

    fig, axes = plt.subplots(4, 1, figsize=(10, 15), sharex=True)
    axes[0].plot(combined["cum_timesteps"], rolling_d, color="steelblue", linewidth=2)
    axes[0].axhline(12.0, color="red", linestyle="--", label="d_tol=12mm")
    axes[0].set_ylabel(f"rolling median terminal d (mm), window={ROLLING_WINDOW}")
    axes[0].set_title("Terminal d vs timesteps")
    axes[0].legend()
    axes[1].plot(combined["cum_timesteps"], rolling_alpha, color="purple", linewidth=2)
    axes[1].axhline(18.0, color="red", linestyle="--", label="alpha_tol=18deg")
    axes[1].set_ylabel(f"rolling median terminal alpha (deg), window={ROLLING_WINDOW}")
    axes[1].set_title("Terminal alpha vs timesteps (the direction check)")
    axes[1].legend()
    axes[2].plot(combined["cum_timesteps"], rolling_success, color="green", linewidth=2)
    axes[2].set_ylim(-0.02, 1.02)
    axes[2].set_ylabel(f"rolling success rate, window={ROLLING_WINDOW}")
    axes[2].set_title("Success rate vs timesteps")
    axes[3].plot(combined["cum_timesteps"], rolling_freeze, color="orange", linewidth=2)
    axes[3].set_ylim(-0.02, 1.02)
    axes[3].set_xlabel("timesteps")
    axes[3].set_ylabel(f"rolling freeze-attempt fraction, window={ROLLING_WINDOW}")
    axes[3].set_title("Freeze-attempt fraction vs timesteps")
    fig.suptitle(f"Arm: {name} -- {env_kwargs}, ent_coef={ent_coef}\n{verdict}", fontsize=10)
    fig.tight_layout()
    fig.savefig(log_dir / "trajectories.png", dpi=130)
    plt.close(fig)

    stats = dict(
        arm=name, env_kwargs=env_kwargs, ent_coef=ent_coef, n_episodes=n,
        overall_success_rate=overall_success, overall_freeze_attempted_rate=overall_freeze,
        first_quarter_freeze_rate=first_freeze, late_freeze_rate=late_freeze,
        first_quarter_median_terminal_d_mm=first_d, late_median_terminal_d_mm=late_d,
        first_quarter_median_terminal_alpha_deg=first_alpha, late_median_terminal_alpha_deg=late_alpha,
        late_success_rate=late_success,
        alpha_toward_tolerance=alpha_toward_tolerance, alpha_away_from_tolerance=alpha_away_from_tolerance,
        verdict=verdict,
    )
    with open(log_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    return stats


def main():
    all_stats = []
    for name, params in ARMS.items():
        s = run_arm(name, **params)
        all_stats.append(s)

    print("\n\n=== SUMMARY (all three arms) ===")
    header = f"{'arm':<24} {'success':>9} {'alpha_first->last':>20} {'d_first->last':>18} {'verdict':>55}"
    print(header)
    for s in all_stats:
        alpha_str = f"{s['first_quarter_median_terminal_alpha_deg']:.1f}->{s['late_median_terminal_alpha_deg']:.1f}"
        d_str = f"{s['first_quarter_median_terminal_d_mm']:.1f}->{s['late_median_terminal_d_mm']:.1f}"
        print(f"{s['arm']:<24} {s['late_success_rate']:>9.4f} {alpha_str:>20} {d_str:>18} {s['verdict']:>55}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    with open(OUT_ROOT / "experiment_summary.json", "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"\nSaved to {OUT_ROOT / 'experiment_summary.json'}")


if __name__ == "__main__":
    main()
