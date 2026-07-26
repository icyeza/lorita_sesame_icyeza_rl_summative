"""Config-driven multi-run sweep runner.

Reads a hyperparameter grid from `training/configs/<algo>.yaml`, runs each
combination as a separate seeded training run, logs to
`logs/<algo>/<run_id>/` (TensorBoard + CSV), and copies the best model (by
final mean episode reward from its monitor.csv) to `models/<algo>/best`.

CURRICULUM (Phase 2): `single_target` and the full 3-target sequential task
are two curriculum STAGES of the SAME environment/env class/action space --
not two environments. `--curriculum single_target` (the default for grid
runs) samples one random target per episode and is what the 10-run
hyperparameter tables are meant to use, since it's far more tractable to
get comparable cross-algorithm learning signal on it. `--curriculum
full_task` is the actual graded task (head->abdomen->femur plus AGA/SGA
classification) and is reserved for a small number of *headline* runs using
each algorithm's best grid config, plus the `main.py` demo.

BUDGET TIERS (Phase 3): `--budget grid` uses each config's (short)
`grid_timesteps`, meant to produce differentiated behavior across
hyperparameters without costing a full training run per cell. `--budget
headline` uses the (longer) `headline_timesteps`, meant for a handful of
best-config runs that produce clean report-quality learning curves.

IMPORTANT: real sweeps (the full grids in `training/configs/`) are meant to
be run by the project owner over multiple days -- do not run a full sweep
here. Use `--smoke` for a tiny-budget pipeline validation run (a handful of
timesteps, one config combination) to prove the path works end to end. The
one bounded exception is a single `--budget headline --curriculum
full_task` run capped with `--max-wall-clock-seconds`, used once in Phase 2
to check the full task produces a learning signal at all -- never expand
that into a grid.

Usage:
    uv run python -m training.sweep --algo dqn --smoke
    uv run python -m training.sweep --algo ppo --curriculum single_target --budget grid
    uv run python -m training.sweep --algo ppo --curriculum full_task --budget headline \\
        --max-wall-clock-seconds 1800
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import shutil
from datetime import datetime

import numpy as np
import pandas as pd
import yaml

from training.dqn_training import train_dqn
from training.pg_training import train_reinforce, train_a2c, train_ppo

TRAINERS = {
    "dqn": train_dqn,
    "reinforce": train_reinforce,
    "a2c": train_a2c,
    "ppo": train_ppo,
}

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG_DIR = os.path.join(REPO_ROOT, "training", "configs")
LOGS_DIR = os.path.join(REPO_ROOT, "logs")
MODELS_DIR = os.path.join(REPO_ROOT, "models")

SMOKE_TIMESTEPS = 2_000


def _grid_combinations(grid: dict) -> list[dict]:
    keys = list(grid.keys())
    values = [grid[k] if isinstance(grid[k], list) else [grid[k]] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def _final_mean_reward(log_dir: str, window: int = 10) -> float:
    """Mean of the last `window` episode rewards across all monitor*.csv
    files in log_dir (plural to support vectorized envs, see
    training/dqn_training.py::make_vec_env, where rank>0 workers write
    monitor_<rank>.csv instead of monitor.csv)."""
    import glob
    paths = sorted(glob.glob(os.path.join(log_dir, "monitor*.csv")))
    if not paths:
        return float("-inf")
    all_r = []
    for path in paths:
        try:
            df = pd.read_csv(path, skiprows=1)
        except Exception:
            continue
        if "r" in df.columns and len(df) > 0:
            all_r.extend(df["r"].tail(window).tolist())
    return float(np.mean(all_r)) if all_r else float("-inf")


class UnsafeWallClockVecEnvCombo(RuntimeError):
    """Raised when a run combines a wall-clock cap with n_envs>1 -- see the
    guard note above `_check_wall_clock_vecenv_combo`."""


def _check_wall_clock_vecenv_combo(n_envs: int, max_wall_clock_seconds: float | None):
    """KNOWN ISSUE, not fixed (multiprocessing deadlocks are open-ended to
    root-cause and this was already investigated once -- see status.md
    Phase 2 "Bug #3"): a `SubprocVecEnv` (n_envs>1) run capped with
    `--max-wall-clock-seconds` was observed to hang indefinitely on Windows
    after hitting the cap -- logging stopped but the process kept consuming
    CPU for 28+ minutes with no clean exit, requiring a manual kill. A
    follow-up isolation test confirmed `n_envs=1` (DummyVecEnv) terminates
    cleanly within seconds of the cap every time, across several repeated
    short-capped runs. Refusing this combination outright (rather than just
    warning) so the hang can't be silently rediscovered hours into an
    unattended real sweep -- capped runs should use `--n-envs 1`; vectorized
    runs should be uncapped."""
    if n_envs > 1 and max_wall_clock_seconds is not None:
        raise UnsafeWallClockVecEnvCombo(
            f"Refusing to run with n_envs={n_envs} AND max_wall_clock_seconds="
            f"{max_wall_clock_seconds}: this combination is known to hang "
            f"indefinitely (SubprocVecEnv shutdown deadlock after the cap "
            f"fires -- see status.md Phase 2 'Bug #3'). Use --n-envs 1 for "
            f"any wall-clock-capped run, or drop --max-wall-clock-seconds "
            f"for a vectorized (n_envs>1) run."
        )


def run_sweep(algo: str, config_path: str | None, smoke: bool,
              curriculum: str = "single_target", budget: str = "grid",
              n_envs: int = 1, max_wall_clock_seconds: float | None = None,
              algo_subdir: str = None, max_combos: int | None = None,
              seeds_override: list[int] | None = None,
              total_timesteps_override: int | None = None,
              extra_env_kwargs: dict | None = None):
    """max_combos: if set, use only the first N grid combinations (in
    `_grid_combinations`'s deterministic itertools.product order) instead
    of the full grid -- used by the locked-environment grid launch (status.md
    "lock the environment" pass) to run exactly 10 differentiated
    combinations per algorithm rather than the full grid x seeds (meant for
    a multi-day run by the project owner, per this module's own docstring).
    seeds_override / total_timesteps_override: same idea, for seeds and the
    timestep budget -- None (default) preserves the original config-driven
    behavior for all other callers.
    extra_env_kwargs: merged into the per-run env_kwargs (which otherwise
    is just `dict(single_target=...)`) -- used by the "corrected grid
    launch" pass to override the grid SWEEP's start distribution
    (`start_curriculum_max_random_steps`) to a harder-than-default radius
    so hyperparameter combos actually differentiate, WITHOUT touching the
    locked environment's own class defaults (which everything else, e.g.
    main.py, still uses unmodified). None (default) changes nothing for
    other callers."""
    _check_wall_clock_vecenv_combo(n_envs, max_wall_clock_seconds)
    algo_subdir = algo_subdir or algo
    if smoke:
        combos = [{}]
        seeds = [0]
        total_timesteps = SMOKE_TIMESTEPS
        curriculum = "single_target"  # smoke always uses the easier stage
    else:
        config_path = config_path or os.path.join(DEFAULT_CONFIG_DIR, f"{algo}.yaml")
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        combos = _grid_combinations(cfg.get("grid", {}))
        if max_combos is not None:
            combos = combos[:max_combos]
        seeds = seeds_override if seeds_override is not None else cfg.get("seeds", [0])
        budget_key = "headline_timesteps" if budget == "headline" else "grid_timesteps"
        total_timesteps = total_timesteps_override or cfg.get(budget_key, cfg.get("total_timesteps", 20_000))

    trainer = TRAINERS[algo]
    results = []
    failures = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    single_target = curriculum == "single_target"

    for combo_idx, combo in enumerate(combos):
        for seed in seeds:
            run_id = (f"smoke_{timestamp}" if smoke else
                      f"{timestamp}_{curriculum}_{budget}_combo{combo_idx}_seed{seed}")
            log_dir = os.path.join(LOGS_DIR, algo_subdir, run_id)
            model_dir = os.path.join(MODELS_DIR, algo_subdir, run_id)
            os.makedirs(log_dir, exist_ok=True)
            os.makedirs(model_dir, exist_ok=True)

            env_kwargs = dict(single_target=single_target, **(extra_env_kwargs or {}))
            print(f"[sweep] {algo} run_id={run_id} combo={combo} seed={seed} "
                  f"curriculum={curriculum} total_timesteps={total_timesteps} "
                  f"n_envs={n_envs}"
                  + (f" max_wall_clock_seconds={max_wall_clock_seconds}" if max_wall_clock_seconds else ""))
            # RESILIENCE: one run's exception must not abort the rest of the
            # batch (an unattended multi-hour grid launch can't afford a
            # single bad combo taking down everything after it) -- log the
            # failure and continue to the next combo/seed.
            try:
                model, save_path = trainer(combo, log_dir, model_dir, seed=seed,
                                            total_timesteps=total_timesteps, env_kwargs=env_kwargs,
                                            n_envs=n_envs, max_wall_clock_seconds=max_wall_clock_seconds,
                                            info_keywords=("success", "freeze_attempted", "d_m", "alpha_deg"))
            except Exception as e:
                print(f"[sweep] FAILED {algo} run_id={run_id} combo={combo} seed={seed}: {e!r}")
                failures.append(dict(run_id=run_id, combo=combo, seed=seed, error=repr(e)))
                with open(os.path.join(log_dir, "run_error.json"), "w") as f:
                    json.dump(dict(combo=combo, seed=seed, error=repr(e)), f, indent=2, default=str)
                continue

            score = _final_mean_reward(log_dir)
            results.append(dict(run_id=run_id, combo=combo, seed=seed, score=score,
                                 log_dir=log_dir, model_path=save_path))

            with open(os.path.join(log_dir, "run_config.json"), "w") as f:
                # env_kwargs recorded from this pass onward (status.md
                # "ablations, protocol fix and report figures" addendum):
                # previously run_config.json captured only the algorithm
                # combo, so a run's start distribution (single_target,
                # start_curriculum, start_curriculum_max_random_steps) was
                # recoverable ONLY from the launching script's source, not
                # from any saved artifact. Not retroactive -- the existing
                # 40 grid runs still lack it.
                json.dump(dict(combo=combo, seed=seed, total_timesteps=total_timesteps,
                                curriculum=curriculum, budget=budget if not smoke else "smoke",
                                n_envs=n_envs, max_wall_clock_seconds=max_wall_clock_seconds,
                                env_kwargs=env_kwargs), f, indent=2)

    if failures:
        print(f"[sweep] {algo}: {len(failures)}/{len(combos) * len(seeds)} runs FAILED: {failures}")

    results_sorted = sorted(results, key=lambda r: r["score"], reverse=True)
    if not results_sorted:
        print(f"[sweep] {algo}: ALL runs failed -- no best model to save.")
        return results_sorted, failures
    best = results_sorted[0]
    best_dir = os.path.join(MODELS_DIR, algo_subdir, "best")
    os.makedirs(best_dir, exist_ok=True)
    best_model_src = best["model_path"]
    ext = os.path.splitext(best_model_src)[1]
    shutil.copy(best_model_src, os.path.join(best_dir, "model" + ext))
    with open(os.path.join(best_dir, "run_info.json"), "w") as f:
        json.dump(best, f, indent=2, default=str)

    print(f"[sweep] best run: {best['run_id']} score={best['score']:.3f} -> {best_dir}")
    return results_sorted, failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", required=True, choices=list(TRAINERS.keys()))
    parser.add_argument("--config", default=None)
    parser.add_argument("--smoke", action="store_true",
                         help="Run a tiny-budget single-combo pipeline validation only.")
    parser.add_argument("--curriculum", choices=["single_target", "full_task"], default="single_target",
                         help="Curriculum stage: single_target (grid/table runs) or full_task "
                              "(the graded task; headline runs + main.py demo only).")
    parser.add_argument("--budget", choices=["grid", "headline"], default="grid",
                         help="grid = short budget for differentiated hyperparameter behavior; "
                              "headline = longer budget for report-quality learning curves.")
    parser.add_argument("--n-envs", type=int, default=1,
                         help="Vectorized env count (SubprocVecEnv if >1). Ignored by REINFORCE.")
    parser.add_argument("--max-wall-clock-seconds", type=float, default=None,
                         help="Hard wall-clock cap on this run's .learn() call, regardless of "
                              "total_timesteps. Use this for the single bounded full_task probe run.")
    args = parser.parse_args()
    run_sweep(args.algo, args.config, args.smoke, curriculum=args.curriculum, budget=args.budget,
              n_envs=args.n_envs, max_wall_clock_seconds=args.max_wall_clock_seconds)


if __name__ == "__main__":
    main()
