"""Go/no-go confirmation run for the arm3 freeze-exploration fix
(ent_coef=0.05, freeze_miss_penalty=0.0) found in
scripts/experiment_freeze_wall.py. That experiment broke freeze-attempts
off 0.000 (-> 0.460) but only ran 40k steps -- freeze probability was near
0 for most of that budget, so there were almost no steps during which
PLACEMENT (not just attempting freeze) could be shaped. Success staying at
0.000 there meant "hasn't started learning placement," not "can't."

This run holds arm3's config EXACTLY fixed and gives it 200k steps to show
whether placement is learnable. Per-episode success/freeze-attempted/
terminal-d/alpha are now logged LIVE via Monitor(info_keywords=...) --
see environment/custom_env.py's info dict and training/dqn_training.py's
make_env -- so this is a genuine trajectory, not a single end-of-training
checkpoint snapshot like the previous pass needed.

NO commit to the default environment happens here: `freeze_miss_penalty`
and `single_target_which` remain experimental constructor arguments with
unchanged defaults; only this script's env_kwargs sets them away from
default.

Usage: uv run python scripts/run_freeze_placement_confirmation.py
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

from training.pg_training import train_ppo

REPO_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = REPO_ROOT / "logs" / "freeze_placement_confirmation"
MODEL_DIR = REPO_ROOT / "models" / "freeze_placement_confirmation"

# Arm3 config, held EXACTLY fixed.
ENT_COEF = 0.05
FREEZE_MISS_PENALTY = 0.0
N_ENVS = 4
TOTAL_TIMESTEPS = 200_000
SEED = 0
D_TOL_MM = 12.0
ROLLING_WINDOW = 50
RUNAWAY_FREEZE_THRESHOLD = 0.9
RUNAWAY_SUCCESS_CEILING = 0.1


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Confirmation run: PPO, femur-only, ent_coef={ENT_COEF}, "
          f"freeze_miss_penalty={FREEZE_MISS_PENALTY}, n_envs={N_ENVS}, "
          f"total_timesteps={TOTAL_TIMESTEPS}, seed={SEED}, uncapped")

    env_kwargs = dict(single_target=True, single_target_which="femur",
                       freeze_miss_penalty=FREEZE_MISS_PENALTY)
    config = dict(entropy_coef=ENT_COEF)
    info_keywords = ("success", "freeze_attempted", "d_m", "alpha_deg")

    model, save_path = train_ppo(
        config, str(LOG_DIR), str(MODEL_DIR), seed=SEED,
        total_timesteps=TOTAL_TIMESTEPS, env_kwargs=env_kwargs,
        n_envs=N_ENVS, max_wall_clock_seconds=None,
        info_keywords=info_keywords,
    )
    print(f"Training complete. Model saved to {save_path}")

    monitor_paths = sorted(LOG_DIR.glob("monitor*.csv"))
    frames = []
    for p in monitor_paths:
        df = pd.read_csv(p, skiprows=1)
        if "r" in df.columns and len(df) > 0:
            frames.append(df)
    combined = pd.concat(frames, ignore_index=True).sort_values("t").reset_index(drop=True)
    combined["cum_timesteps"] = combined["l"].cumsum()
    combined["d_mm"] = combined["d_m"] * 1000.0
    combined.to_csv(LOG_DIR / "confirmation_episodes.csv", index=False)

    n = len(combined)
    overall_success_rate = float(combined["success"].mean())
    overall_freeze_rate = float(combined["freeze_attempted"].mean())
    last_quarter = combined.iloc[3 * n // 4:]
    late_success_rate = float(last_quarter["success"].mean())
    late_freeze_rate = float(last_quarter["freeze_attempted"].mean())
    late_median_d = float(last_quarter["d_mm"].median())

    print(f"\nOverall (N={n}): success_rate={overall_success_rate:.4f}, "
          f"freeze_attempted_rate={overall_freeze_rate:.4f}")
    print(f"Last quarter (N={len(last_quarter)}): success_rate={late_success_rate:.4f}, "
          f"freeze_attempted_rate={late_freeze_rate:.4f}, median terminal d={late_median_d:.2f}mm")

    # Verdict
    if late_freeze_rate >= RUNAWAY_FREEZE_THRESHOLD and late_success_rate <= RUNAWAY_SUCCESS_CEILING:
        verdict = "RUNAWAY"
    elif late_median_d <= D_TOL_MM * 2.0 and late_success_rate > 0.1:
        verdict = "GO"
    elif late_freeze_rate > 0.1 and late_median_d > D_TOL_MM * 2.0 and late_success_rate <= 0.1:
        verdict = "NO-GO"
    else:
        verdict = "AMBIGUOUS"

    print(f"\n=== VERDICT: {verdict} ===")

    # --- Plots: three trajectories vs timesteps ---
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
    axes[1].set_title("Success rate vs timesteps (the greenlight metric)")

    axes[2].plot(combined["cum_timesteps"], rolling_freeze, color="orange", linewidth=2)
    axes[2].set_ylim(-0.02, 1.02)
    axes[2].set_xlabel("timesteps")
    axes[2].set_ylabel(f"rolling freeze-attempt fraction, window={ROLLING_WINDOW}")
    axes[2].set_title("Freeze-attempt fraction vs timesteps (exploration + runaway guard)")

    fig.suptitle(f"Freeze-placement confirmation run -- PPO, femur-only, "
                 f"ent_coef={ENT_COEF}, freeze_miss_penalty={FREEZE_MISS_PENALTY}\n"
                 f"n_envs={N_ENVS}, total_timesteps={TOTAL_TIMESTEPS}, seed={SEED} -- "
                 f"VERDICT: {verdict}", fontsize=11)
    fig.tight_layout()
    out_path = LOG_DIR / "confirmation_trajectories.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"\nSaved trajectories to {out_path}")

    summary = dict(
        ent_coef=ENT_COEF, freeze_miss_penalty=FREEZE_MISS_PENALTY, n_envs=N_ENVS,
        total_timesteps=TOTAL_TIMESTEPS, seed=SEED, n_episodes=n,
        overall_success_rate=overall_success_rate, overall_freeze_attempted_rate=overall_freeze_rate,
        late_quarter_success_rate=late_success_rate, late_quarter_freeze_attempted_rate=late_freeze_rate,
        late_quarter_median_terminal_d_mm=late_median_d,
        verdict=verdict,
    )
    with open(LOG_DIR / "confirmation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {LOG_DIR / 'confirmation_summary.json'}")


if __name__ == "__main__":
    main()
