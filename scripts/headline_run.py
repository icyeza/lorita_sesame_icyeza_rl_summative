"""Headline run: THE agent filmed for the demo video (status.md "headline
run" pass). One clean PPO training run, then save the model -- NOT a
sweep, no tuning loop.

Config rationale (see status.md for the full writeup):
  - PPO: A2C ties it on grid success but is less stable (bigger reward_std
    swings); PPO's clipped objective is the more filmable, less erratic
    choice.
  - Hyperparameters, stability-weighted from the grid table
    (logs/tables/ppo_hyperparameter_table.csv): lr=0.0003 (clearly beat
    0.0001), n_steps=256 (beat 128 on BOTH final_mean_reward AND
    reward_std), clip_range=0.1 (no clear grid trend -> take the more
    conservative/tighter clip for a showcase run), entropy_coef=0.01
    (mildly helped PPO's reward in the grid). This matches grid combo6,
    which had the LOWEST reward_std in the whole PPO table -- deliberately
    chosen over the marginally-higher-reward combo4 because this run
    optimizes for stability/reliability at convergence, not peak reward.
  - Environment: the LOCKED DEPLOYMENT config (alpha_tol=18deg,
    tilt_step=3deg, shaping=multiplicative, d_tol=12mm, start_curriculum
    radius=8 -- all class defaults, not overridden), with
    `single_target=False` (the FULL 3-target task + AGA/SGA
    classification) -- NOT the radius=40 single-target config the grid
    used to differentiate hyperparameters. The grid's job was comparison;
    this run's job is the deployment/demo setting.
  - Budget: 40,000 steps -- single-target curriculum success saturates by
    ~t=1200-3000 (see the stopped calibration run); the full 3-target task
    is harder (three separate acquisitions + classification), so a longer
    but still modest budget is used, not 100k.

Logs success/freeze_attempted/full_task_success/d_m/alpha_deg/true_is_iugr
via Monitor(info_keywords=...) so this run's metrics are REAL LOGGED DATA,
not a reconstructed proxy (the sweep's info_keywords bug is fixed and
confirmed active here).

FALLBACK (protects the video): if the full 3-target task doesn't reach a
filmable converged success rate (<~60%), this script automatically falls
back to single_target=True (radius=8, the proven ~90% setting) for ONE
additional run and reports which was used, rather than tuning the 3-target
task further.

Usage: uv run python scripts/headline_run.py
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
LOG_DIR = REPO_ROOT / "logs" / "headline"
MODEL_DIR = REPO_ROOT / "models" / "headline"

TOTAL_TIMESTEPS = 40_000
N_ENVS = 4
SEED = 0
ROLLING_WINDOW = 20
TAIL_FRACTION = 0.2
FILMABLE_SUCCESS_FLOOR = 0.60

# Stability-weighted config from the grid table (== grid combo6).
PPO_CONFIG = dict(
    learning_rate=0.0003, gamma=0.99, n_steps=256, gae_lambda=0.95,
    entropy_coef=0.01, clip_range=0.1, net_arch=[64, 64],
)
INFO_KEYWORDS = ("success", "full_task_success", "freeze_attempted", "d_m", "alpha_deg", "flag", "true_is_iugr")


def run(single_target: bool, tag: str):
    run_log_dir = LOG_DIR / tag
    run_model_dir = MODEL_DIR / tag
    run_log_dir.mkdir(parents=True, exist_ok=True)
    run_model_dir.mkdir(parents=True, exist_ok=True)

    env_kwargs = dict(single_target=single_target)
    print(f"\n=== Headline run [{tag}]: PPO, single_target={single_target}, "
          f"{PPO_CONFIG}, n_envs={N_ENVS}, total_timesteps={TOTAL_TIMESTEPS}, seed={SEED}, uncapped ===")

    model, save_path = train_ppo(
        PPO_CONFIG, str(run_log_dir), str(run_model_dir), seed=SEED,
        total_timesteps=TOTAL_TIMESTEPS, env_kwargs=env_kwargs,
        n_envs=N_ENVS, max_wall_clock_seconds=None, info_keywords=INFO_KEYWORDS,
    )
    print(f"Training complete. Model saved to {save_path}")

    monitor_paths = sorted(run_log_dir.glob("monitor*.csv"))
    frames = []
    for p in monitor_paths:
        df = pd.read_csv(p, skiprows=1)
        if "r" in df.columns and len(df) > 0:
            frames.append(df)
    combined = pd.concat(frames, ignore_index=True).sort_values("t").reset_index(drop=True)
    combined["cum_timesteps"] = combined["l"].cumsum()
    combined["d_mm"] = combined["d_m"] * 1000.0
    combined.to_csv(run_log_dir / "headline_episodes.csv", index=False)

    n = len(combined)
    tail = combined.iloc[int(n * (1 - TAIL_FRACTION)):]
    success_col = "full_task_success" if not single_target else "success"
    converged_success_rate = float(tail[success_col].astype(float).mean())
    median_terminal_alpha = float(tail["alpha_deg"].median())
    median_terminal_d_mm = float(tail["d_mm"].median())
    converged_freeze_rate = float(tail["freeze_attempted"].astype(float).mean())

    # Classification accuracy: only defined for episodes with a non-null
    # flag (i.e. all 3 acquired + classification actually ran) --
    # comparing the predicted AGA/SGA flag against the phantom's REAL
    # ground-truth IUGR label (true_is_iugr), both real logged fields.
    classified = tail.dropna(subset=["flag"]) if "flag" in tail.columns else tail.iloc[0:0]
    if len(classified) > 0 and "true_is_iugr" in classified.columns:
        predicted_sga = classified["flag"].astype(str).str.contains("SGA")
        true_iugr = classified["true_is_iugr"].astype(bool)
        classification_accuracy = float((predicted_sga == true_iugr).mean())
        n_classified = len(classified)
    else:
        classification_accuracy = None
        n_classified = 0

    print(f"\nHeadline run [{tag}] results (N={n} episodes, last {int(TAIL_FRACTION*100)}% tail):")
    print(f"  converged {'full-task ' if not single_target else ''}success rate: {converged_success_rate:.4f}")
    print(f"  median terminal alpha: {median_terminal_alpha:.2f}deg, median terminal d: {median_terminal_d_mm:.2f}mm")
    print(f"  freeze-attempted rate: {converged_freeze_rate:.4f}")
    if classification_accuracy is not None:
        print(f"  classification accuracy (AGA/SGA vs true_is_iugr, N={n_classified}): {classification_accuracy:.4f}")
    else:
        print(f"  classification accuracy: N/A (0 fully-classified episodes in tail)")

    # Plot: success rate + reward vs timesteps (the headline figure)
    rolling_success = combined[success_col].astype(float).rolling(ROLLING_WINDOW, min_periods=1).mean()
    rolling_reward = combined["r"].rolling(ROLLING_WINDOW, min_periods=1).mean()
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(combined["cum_timesteps"], rolling_reward, color="steelblue", linewidth=2)
    axes[0].set_ylabel(f"rolling mean reward, window={ROLLING_WINDOW}")
    axes[0].set_title(f"Headline run [{tag}] -- reward vs timesteps")
    axes[1].plot(combined["cum_timesteps"], rolling_success, color="green", linewidth=2)
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].set_xlabel("timesteps")
    axes[1].set_ylabel(f"rolling {'full-task ' if not single_target else ''}success rate, window={ROLLING_WINDOW}")
    axes[1].set_title("Success rate vs timesteps")
    fig.suptitle(f"PPO headline run -- single_target={single_target}, {PPO_CONFIG}\n"
                 f"converged success={converged_success_rate:.3f}", fontsize=10)
    fig.tight_layout()
    fig.savefig(run_log_dir / "headline_curve.png", dpi=130)
    plt.close(fig)

    summary = dict(
        tag=tag, single_target=single_target, config=PPO_CONFIG,
        total_timesteps=TOTAL_TIMESTEPS, n_envs=N_ENVS, seed=SEED, n_episodes=n,
        converged_success_rate=converged_success_rate,
        median_terminal_alpha_deg=median_terminal_alpha,
        median_terminal_d_mm=median_terminal_d_mm,
        converged_freeze_attempted_rate=converged_freeze_rate,
        classification_accuracy=classification_accuracy,
        n_classified_episodes=n_classified,
        model_path=save_path,
    )
    with open(run_log_dir / "headline_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    return summary


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    summary = run(single_target=False, tag="full_task")

    if summary["converged_success_rate"] < FILMABLE_SUCCESS_FLOOR:
        print(f"\n[headline_run] Full-task converged success rate "
              f"{summary['converged_success_rate']:.4f} < {FILMABLE_SUCCESS_FLOOR} floor -- "
              f"NOT filmable. Falling back to single_target curriculum (radius=8, proven ~90% setting).")
        summary = run(single_target=True, tag="single_target_fallback")
        chosen_tag = summary["tag"]
    else:
        print(f"\n[headline_run] Full-task converged success rate "
              f"{summary['converged_success_rate']:.4f} >= {FILMABLE_SUCCESS_FLOOR} floor -- filmable, keeping it.")
        chosen_tag = summary["tag"]

    # Copy the chosen model to a stable, unambiguous path for traceability,
    # AND overwrite models/ppo/best/model.zip -- the exact path
    # `main.py::find_best_model()` checks first (ALGO_PRIORITY = ["ppo", ...])
    # -- so `uv run main.py` picks up THIS headline model for filming, not
    # the grid sweep's best-by-reward PPO checkpoint that was there before.
    import shutil
    chosen_model_src = Path(summary["model_path"])
    final_model_path = MODEL_DIR / ("model" + chosen_model_src.suffix)
    shutil.copy(chosen_model_src, final_model_path)

    ppo_best_dir = REPO_ROOT / "models" / "ppo" / "best"
    ppo_best_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(chosen_model_src, ppo_best_dir / "model.zip")

    with open(LOG_DIR / "headline_final_choice.json", "w") as f:
        json.dump(dict(chosen_tag=chosen_tag, summary=summary, final_model_path=str(final_model_path),
                        deployed_to=str(ppo_best_dir / "model.zip")),
                   f, indent=2, default=str)

    print(f"\n=== FINAL CHOICE: {chosen_tag} ===")
    print(f"Model copied to {final_model_path}")
    print(f"Deployed to {ppo_best_dir / 'model.zip'} (main.py's lookup path)")
    print(f"Learning curve: {LOG_DIR / chosen_tag / 'headline_curve.png'}")


if __name__ == "__main__":
    main()
