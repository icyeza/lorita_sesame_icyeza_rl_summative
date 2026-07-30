"""Launch the four-algorithm hyperparameter grids on the LOCKED environment
(status.md "grid launch with differentiation guardrails" pass) -- the
graded deliverable.

Builds on two prior corrections in this same overall pass (see status.md):
  - Mid start radius (40) for the grid SWEEP only (env_kwargs override via
    `training.sweep.run_sweep`'s `extra_env_kwargs`, NOT the locked class
    defaults) -- confirmed non-saturated by scripts/grid_radius_precheck.py.
  - SubprocVecEnv resource leak fixed (env.close() added to the SB3
    trainers) -- a multi-combo sweep no longer accumulates orphaned worker
    processes.

THIS pass adds a mandatory DQN-first differentiation gate: since success
converges near ~85% for most reasonable configs at radius=40, "are the 10
rows actually different" must be judged on sample-efficiency/stability
metrics (see scripts/grid_report.py's HEADLINE_METRIC_ORDER), not final
success. DQN's grid runs FIRST; its resulting table is checked with
scripts/differentiation_check.py BEFORE A2C/PPO/REINFORCE are allowed to
run. If DQN's rows are near-clones on every metric, the WHOLE plan
(DQN included) is bumped to a longer per-run budget and restarted, rather
than burning hours producing four undifferentiated tables.

Usage: uv run python scripts/launch_grids.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.sweep import run_sweep
from scripts.grid_report import generate_all, build_hyperparam_table, render_table_png
from scripts.differentiation_check import check_differentiation, print_report

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "logs"

SEED = 0
MID_RADIUS = 40  # confirmed non-saturated by scripts/grid_radius_precheck.py
EXTRA_ENV_KWARGS = dict(start_curriculum=True, start_curriculum_max_random_steps=MID_RADIUS)

# Initial per-algo plan. May be bumped (see BUMPED_TIMESTEPS below) if
# DQN's differentiation check fails.
BASE_PLAN = {
    "dqn":       dict(n_envs=4, max_combos=10, total_timesteps=6000),
    "a2c":       dict(n_envs=4, max_combos=10, total_timesteps=6000),
    "ppo":       dict(n_envs=4, max_combos=10, total_timesteps=6000),
    "reinforce": dict(n_envs=1, max_combos=5,  total_timesteps=3000),
}
BUMPED_TIMESTEPS = dict(dqn=15000, a2c=15000, ppo=15000, reinforce=7500)


def run_one_algo(algo: str, p: dict, all_results: dict, all_failures: dict, wall_clock: dict, overall_t0: float):
    print(f"\n--- {algo.upper()} (start {time.strftime('%H:%M:%S')}, "
          f"{p['max_combos']} combos x total_timesteps={p['total_timesteps']} x n_envs={p['n_envs']}) ---")
    t0 = time.monotonic()
    try:
        results, failures = run_sweep(
            algo, None, smoke=False, curriculum="single_target", budget="grid",
            n_envs=p["n_envs"], max_wall_clock_seconds=None,
            max_combos=p["max_combos"], seeds_override=[SEED],
            total_timesteps_override=p["total_timesteps"],
            extra_env_kwargs=EXTRA_ENV_KWARGS,
        )
    except Exception as e:
        print(f"[launch_grids] {algo} grid launch FAILED ENTIRELY: {e!r}")
        results, failures = [], [dict(error=repr(e), note="entire algo grid failed to launch")]
    elapsed = time.monotonic() - t0
    all_results[algo] = results
    all_failures[algo] = failures
    wall_clock[algo] = elapsed
    print(f"[launch_grids] {algo}: {len(results)} completed, {len(failures)} failed, "
          f"{elapsed/60:.1f} min (end {time.strftime('%H:%M:%S')})")

    with open(OUT_DIR / "grid_launch_summary.json", "w") as f:
        json.dump(dict(
            plan={a: p for a, p in [(algo, p)]}, mid_radius=MID_RADIUS, seed=SEED,
            total_wall_clock_s_so_far=time.monotonic() - overall_t0,
            per_algo_wall_clock_s=wall_clock,
            n_completed={a: len(r) for a, r in all_results.items()},
            n_failed={a: len(f_) for a, f_ in all_failures.items()},
            failures=all_failures,
            algos_done=list(all_results.keys()),
        ), f, indent=2, default=str)
    return results


def main():
    print("=== Launching hyperparameter grids on the LOCKED environment "
          f"(mid start radius={MID_RADIUS}, DQN-first differentiation gate) ===")

    all_results, all_failures, wall_clock = {}, {}, {}
    overall_t0 = time.monotonic()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plan = dict(BASE_PLAN)

    # --- DQN first, alone, as the differentiation canary ---
    dqn_results = run_one_algo("dqn", plan["dqn"], all_results, all_failures, wall_clock, overall_t0)
    dqn_table = build_hyperparam_table("dqn", dqn_results)
    report = check_differentiation(dqn_table, plan["dqn"]["total_timesteps"])
    print_report("dqn", report)

    if not report["differentiated"]:
        print(f"\n[launch_grids] DQN grid NOT differentiated at total_timesteps="
              f"{plan['dqn']['total_timesteps']} -- BUMPING all algos to a longer budget and "
              f"RESTARTING THE WHOLE PLAN (DQN included).")
        plan = {a: dict(p, total_timesteps=BUMPED_TIMESTEPS[a]) for a, p in BASE_PLAN.items()}
        all_results, all_failures, wall_clock = {}, {}, {}
        dqn_results = run_one_algo("dqn", plan["dqn"], all_results, all_failures, wall_clock, overall_t0)
        dqn_table = build_hyperparam_table("dqn", dqn_results)
        report = check_differentiation(dqn_table, plan["dqn"]["total_timesteps"])
        print_report("dqn (bumped)", report)
        print(f"[launch_grids] Post-bump DQN differentiated={report['differentiated']} -- "
              f"proceeding regardless (this is the second attempt; report the outcome either way).")
    else:
        print(f"\n[launch_grids] DQN grid IS differentiated at total_timesteps="
              f"{plan['dqn']['total_timesteps']} -- continuing to A2C/PPO/REINFORCE unchanged.")

    # --- Remaining algorithms at the (possibly bumped) plan ---
    for algo in ["a2c", "ppo", "reinforce"]:
        run_one_algo(algo, plan[algo], all_results, all_failures, wall_clock, overall_t0)

    total_wall_clock = time.monotonic() - overall_t0
    print(f"\n=== All grids done. Total wall-clock: {total_wall_clock/60:.1f} min "
          f"({total_wall_clock/3600:.2f} hr) ===")

    with open(OUT_DIR / "grid_launch_summary.json", "w") as f:
        json.dump(dict(
            plan=plan, mid_radius=MID_RADIUS, seed=SEED,
            differentiation_report=report,
            total_wall_clock_s=total_wall_clock,
            per_algo_wall_clock_s=wall_clock,
            n_completed={a: len(r) for a, r in all_results.items()},
            n_failed={a: len(f_) for a, f_ in all_failures.items()},
            failures=all_failures,
        ), f, indent=2, default=str)

    print("\n=== Generating report-ready tables and plots ===")
    summaries = generate_all(all_results, all_failures)
    for algo, s in summaries.items():
        best = s["best"]
        print(f"\n[{algo.upper()}] completed={s['n_completed']} failed={s['n_failed']}")
        if best:
            print(f"  best: run_id={best['run_id']} combo={best['combo']} "
                  f"final_mean_reward={best['final_mean_reward']:.3f} "
                  f"final_success_rate={best.get('final_success_rate', 'n/a')}")
        else:
            print("  NO successful runs to report.")

    print(f"\nTables: {OUT_DIR / 'tables'}")
    print(f"Plots: {OUT_DIR / 'plots'}")


if __name__ == "__main__":
    main()
