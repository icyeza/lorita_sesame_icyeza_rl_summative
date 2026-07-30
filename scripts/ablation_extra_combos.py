"""Three single-knob ablation runs appended to the existing grids
(status.md "ablations, protocol fix and report figures" addendum).

Adds runs into the SAME run-id namespaces the canonical batches use, so
`scripts/reconstruct_grid_tables.py` picks them up as extra rows without
disturbing combos 0-9:

  DQN combo10 -- combo3 with buffer_size 50000 -> 10000
  DQN combo11 -- combo3 with exploration_fraction 0.2 -> 0.5
  REINFORCE combo10 -- lr=1e-4, use_baseline=False, entropy_coef=0.0
                       (the natural index-10 cell of reinforce.yaml's grid,
                       completing the baseline-on/off pair at lr=1e-4)

WHY combo3 IS THE DQN BASE: highest final_mean_reward (-2.9235) of all ten
rows in logs/tables/dqn_hyperparameter_table.csv, and also best on
success_proxy_auc (0.5248) and reward_auc (-2.809). Confirmed before this
script was written, not assumed.

MATCHED SETTINGS -- PROVENANCE CAVEAT: combo3's saved run_config.json
records combo/seed/total_timesteps/curriculum/budget/n_envs only. It does
NOT record env kwargs. `single_target=True`, `start_curriculum=True` and
`start_curriculum_max_random_steps=40` below are matched against
`scripts/launch_grids.py`'s source constants (EXTRA_ENV_KWARGS,
curriculum="single_target"), NOT against any saved artifact of the
original run. `training/sweep.py` now writes env_kwargs into
run_config.json so future runs don't have this gap; these three runs
write it explicitly below.

Resilient: one run failing does not abort the others.

Usage: uv run python scripts/ablation_extra_combos.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.dqn_training import train_dqn
from training.pg_training import train_reinforce

REPO_ROOT = Path(__file__).resolve().parent.parent

# Env kwargs matched to launch_grids.py (see PROVENANCE CAVEAT above).
GRID_ENV_KWARGS = dict(single_target=True, start_curriculum=True,
                       start_curriculum_max_random_steps=40)

# --- DQN: combo3's exact config, from its saved run_config.json ---
DQN_COMBO3 = dict(learning_rate=0.001, gamma=0.99, buffer_size=50000, batch_size=64,
                  target_update_interval=2000, exploration_fraction=0.2,
                  net_arch=[128, 128])
DQN_PREFIX = "20260729_231349_single_target_grid"  # the canonical (bumped, 15000-step) DQN batch
DQN_TOTAL_TIMESTEPS = 15000  # combo3's saved total_timesteps
DQN_N_ENVS = 4               # combo3's saved n_envs

# --- REINFORCE: matches combos 0-9 exactly ---
REINFORCE_COMBO10 = dict(learning_rate=0.0001, gamma=0.99, use_baseline=False,
                         entropy_coef=0.0, net_arch=[64, 64])
REINFORCE_PREFIX = "20260730_083939_single_target_grid"
REINFORCE_TOTAL_TIMESTEPS = 7500
REINFORCE_N_ENVS = 1

SEED = 0

JOBS = [
    dict(algo="dqn", idx=10, trainer=train_dqn, prefix=DQN_PREFIX, n_envs=DQN_N_ENVS,
         total_timesteps=DQN_TOTAL_TIMESTEPS,
         combo=dict(DQN_COMBO3, buffer_size=10000),
         note="combo3 ablation: buffer_size 50000 -> 10000"),
    dict(algo="dqn", idx=11, trainer=train_dqn, prefix=DQN_PREFIX, n_envs=DQN_N_ENVS,
         total_timesteps=DQN_TOTAL_TIMESTEPS,
         combo=dict(DQN_COMBO3, exploration_fraction=0.5),
         note="combo3 ablation: exploration_fraction 0.2 -> 0.5"),
    dict(algo="reinforce", idx=10, trainer=train_reinforce, prefix=REINFORCE_PREFIX,
         n_envs=REINFORCE_N_ENVS, total_timesteps=REINFORCE_TOTAL_TIMESTEPS,
         combo=REINFORCE_COMBO10,
         note="completes the baseline-on/off matched pair at lr=1e-4"),
]

INFO_KEYWORDS = ("success", "freeze_attempted", "d_m", "alpha_deg")


def main():
    results, failures = [], []
    overall_t0 = time.monotonic()
    for job in JOBS:
        run_id = f"{job['prefix']}_combo{job['idx']}_seed{SEED}"
        log_dir = REPO_ROOT / "logs" / job["algo"] / run_id
        model_dir = REPO_ROOT / "models" / job["algo"] / run_id
        log_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n--- {job['algo']} combo{job['idx']} (start {time.strftime('%H:%M:%S')}) ---")
        print(f"    {job['note']}")
        print(f"    combo={job['combo']}")
        print(f"    total_timesteps={job['total_timesteps']} n_envs={job['n_envs']} "
              f"seed={SEED} env_kwargs={GRID_ENV_KWARGS}")
        t0 = time.monotonic()
        try:
            model, save_path = job["trainer"](
                job["combo"], str(log_dir), str(model_dir), seed=SEED,
                total_timesteps=job["total_timesteps"], env_kwargs=dict(GRID_ENV_KWARGS),
                n_envs=job["n_envs"], max_wall_clock_seconds=None,
                info_keywords=INFO_KEYWORDS,
            )
            elapsed = time.monotonic() - t0
            with open(log_dir / "run_config.json", "w") as f:
                json.dump(dict(
                    combo=job["combo"], seed=SEED, total_timesteps=job["total_timesteps"],
                    curriculum="single_target", budget="grid", n_envs=job["n_envs"],
                    max_wall_clock_seconds=None, env_kwargs=GRID_ENV_KWARGS,
                    ablation_note=job["note"], wall_clock_s=elapsed,
                ), f, indent=2)
            print(f"  done in {elapsed/60:.1f} min -> {save_path}")
            results.append(dict(run_id=run_id, wall_clock_s=elapsed))
        except Exception as e:
            elapsed = time.monotonic() - t0
            print(f"  FAILED after {elapsed/60:.1f} min: {e!r}")
            failures.append(dict(run_id=run_id, combo=job["combo"], error=repr(e)))
            with open(log_dir / "run_error.json", "w") as f:
                json.dump(dict(combo=job["combo"], error=repr(e)), f, indent=2, default=str)

    total = time.monotonic() - overall_t0
    print(f"\n=== Done: {len(results)} completed, {len(failures)} failed, {total/60:.1f} min total ===")
    with open(REPO_ROOT / "logs" / "ablation_launch_summary.json", "w") as f:
        json.dump(dict(results=results, failures=failures, total_wall_clock_s=total),
                  f, indent=2, default=str)
    if failures:
        print(f"Failures: {failures}")


if __name__ == "__main__":
    main()
