"""Fill out REINFORCE to 10 combos, matching the other three algorithms.

REINFORCE was deliberately run at only 5 of its 12 grid combos in the
original "grid launch with differentiation guardrails" pass (status.md),
since it's single-env and the slowest of the four per-timestep -- a
legitimate time-budget trim under the overnight deadline, not a bug. This
script runs combos 5-9 (0-indexed, in the same itertools.product order
`training.sweep._grid_combinations` uses) at the EXACT SAME settings as
the existing 5 (mid start radius=40, total_timesteps=7500, seed=0), and
writes them into the SAME run-id namespace
(`20260730_083939_single_target_grid_combo{5..9}_seed0`) so
`scripts/reconstruct_grid_tables.py` picks up all 10 as one batch.

Resilient: one combo failing does not abort the rest.

Usage: uv run python scripts/reinforce_extra_combos.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.sweep import _grid_combinations, DEFAULT_CONFIG_DIR
from training.pg_training import train_reinforce

REPO_ROOT = Path(__file__).resolve().parent.parent
PREFIX = "20260730_083939_single_target_grid"  # matches the existing 5 REINFORCE runs
EXTRA_COMBO_INDICES = [5, 6, 7, 8, 9]
TOTAL_TIMESTEPS = 7500  # matches the existing 5 runs' bumped budget
SEED = 0
EXTRA_ENV_KWARGS = dict(single_target=True, start_curriculum=True, start_curriculum_max_random_steps=40)


def main():
    with open(Path(DEFAULT_CONFIG_DIR) / "reinforce.yaml") as f:
        cfg = yaml.safe_load(f)
    combos = _grid_combinations(cfg.get("grid", {}))
    print(f"reinforce.yaml has {len(combos)} total combos; running indices {EXTRA_COMBO_INDICES}")

    results, failures = [], []
    overall_t0 = time.monotonic()
    for idx in EXTRA_COMBO_INDICES:
        combo = combos[idx]
        run_id = f"{PREFIX}_combo{idx}_seed0"
        log_dir = REPO_ROOT / "logs" / "reinforce" / run_id
        model_dir = REPO_ROOT / "models" / "reinforce" / run_id
        log_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n--- combo{idx} (start {time.strftime('%H:%M:%S')}): {combo} ---")
        t0 = time.monotonic()
        try:
            model, save_path = train_reinforce(
                combo, str(log_dir), str(model_dir), seed=SEED,
                total_timesteps=TOTAL_TIMESTEPS, env_kwargs=EXTRA_ENV_KWARGS,
                n_envs=1, max_wall_clock_seconds=None,
            )
            with open(log_dir / "run_config.json", "w") as f:
                json.dump(dict(
                    combo=combo, seed=SEED, total_timesteps=TOTAL_TIMESTEPS,
                    curriculum="single_target", budget="grid", n_envs=1,
                    max_wall_clock_seconds=None,
                ), f, indent=2)
            elapsed = time.monotonic() - t0
            print(f"  combo{idx} done in {elapsed/60:.1f} min -> {save_path}")
            results.append(run_id)
        except Exception as e:
            elapsed = time.monotonic() - t0
            print(f"  combo{idx} FAILED after {elapsed/60:.1f} min: {e!r}")
            failures.append(dict(run_id=run_id, combo=combo, error=repr(e)))
            with open(log_dir / "run_error.json", "w") as f:
                json.dump(dict(combo=combo, error=repr(e)), f, indent=2, default=str)

    total_elapsed = time.monotonic() - overall_t0
    print(f"\n=== Done: {len(results)} completed, {len(failures)} failed, "
          f"{total_elapsed/60:.1f} min total ===")
    if failures:
        print(f"Failures: {failures}")


if __name__ == "__main__":
    main()
