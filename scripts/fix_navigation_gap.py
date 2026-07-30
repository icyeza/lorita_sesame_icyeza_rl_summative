"""Fix the "navigation-skill gap" found in the live-viz investigation
(status.md "navigation-skill gap" pass): the deployed headline PPO model
(models/ppo/best/model.zip) scores ~100% success when an episode starts
already inside alpha_tol/d_tol (the curriculum's default, ~95% of
episodes), but 0/9 (0%, N=200 check) when it doesn't -- every one of
those runs to the full 60-step subtask timeout with no partial progress.
Root cause: the locked curriculum's random-walk start (0-8 random real
actions, symmetric +/-) mostly cancels itself out and lands back inside
the generous 18deg/12mm tolerance regardless of radius (measured: even
radius=25 leaves ~90% still inside tolerance) -- so the model almost never
saw a genuine "close a real gap" episode during training.

Fix: retrain from the SAME real headline config (PPO, single_target=True,
radius=8 curriculum, alpha_tol=18deg, d_tol=12mm, shaping=multiplicative --
all class defaults, matching scripts/headline_run.py's
"single_target_fallback" run), but with `start_curriculum_push_prob`
(UltrasoundProbeEnv, see its docstring) ramped up on a STAGED,
SUCCESS-GATED schedule (training.callbacks.StartCurriculumPushScheduleCallback)
that guarantees genuinely non-trivial starts for an increasing fraction of
episodes -- WITHOUT ever widening `start_curriculum_max_random_steps`
itself or approaching true uniform-random starts (the earlier
widening-schedule attempt in scripts/generalization_check.py jumped
straight to that regime and caused catastrophic forgetting; this fix
deliberately stays inside the already-proven-safe radius=8 neighborhood
the whole time).

Does NOT touch the deployed model (models/ppo/best/model.zip) or the
locked environment defaults (start_curriculum_push_prob defaults to 0.0,
unchanged behavior unless explicitly set here). Saves to
models/nav_fix/final/model.zip plus a per-stage checkpoint under
models/nav_fix/stages/ -- promote to the deployed path manually after
reviewing the bucketed eval results this script prints.

Usage:
    uv run python scripts/fix_navigation_gap.py --smoke   # quick wiring + throughput check
    uv run python scripts/fix_navigation_gap.py           # full run
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure

from environment.custom_env import UltrasoundProbeEnv
from training.callbacks import StartCurriculumPushScheduleCallback
from training.dqn_training import make_vec_env

REPO_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = REPO_ROOT / "logs" / "nav_fix"
MODEL_DIR = REPO_ROOT / "models" / "nav_fix"
STAGE_DIR = MODEL_DIR / "stages"
BASELINE_MODEL_PATH = REPO_ROOT / "models" / "ppo" / "best" / "model.zip"

# Same real deployed config as scripts/headline_run.py's
# "single_target_fallback" run (grid combo6, stability-weighted).
PPO_CONFIG = dict(
    learning_rate=0.0003, gamma=0.99, n_steps=256, gae_lambda=0.95,
    entropy_coef=0.01, clip_range=0.1, net_arch=[64, 64],
)
ENV_KWARGS = dict(single_target=True)  # radius=8 curriculum, all other class defaults
N_ENVS = 4
SEED = 0
INFO_KEYWORDS = ("success", "freeze_attempted", "d_m", "alpha_deg")

# Staged, success-gated push_prob schedule -- see
# StartCurriculumPushScheduleCallback's docstring for the gating logic.
# Stage 0 (push_prob=0.0) replicates the exact original curriculum first,
# so the model re-establishes the already-known trivial-start behavior
# before any harder episodes are mixed in.
FULL_SCHEDULE = [
    dict(push_prob=0.0, min_timesteps=5_000, max_timesteps=10_000, success_floor=0.85),
    dict(push_prob=0.2, min_timesteps=15_000, max_timesteps=30_000, success_floor=0.55),
    dict(push_prob=0.4, min_timesteps=15_000, max_timesteps=30_000, success_floor=0.40),
    dict(push_prob=0.55, min_timesteps=15_000, max_timesteps=30_000, success_floor=0.30),
]
FULL_TOTAL_TIMESTEPS = sum(s["max_timesteps"] for s in FULL_SCHEDULE)  # worst case if every stage maxes out

# --smoke: same shape, ~20x smaller -- just to verify wiring (no crashes,
# stage transitions actually fire) and measure real steps/sec on this
# machine before committing to the full run's wall-clock cost.
SMOKE_SCHEDULE = [
    dict(push_prob=0.0, min_timesteps=250, max_timesteps=500, success_floor=0.85),
    dict(push_prob=0.2, min_timesteps=750, max_timesteps=1_500, success_floor=0.55),
    dict(push_prob=0.4, min_timesteps=750, max_timesteps=1_500, success_floor=0.40),
    dict(push_prob=0.55, min_timesteps=750, max_timesteps=1_500, success_floor=0.30),
]
SMOKE_TOTAL_TIMESTEPS = sum(s["max_timesteps"] for s in SMOKE_SCHEDULE)


def bucketed_eval(model, n: int = 200, seed_start: int = 30_000) -> dict:
    """The SAME diagnostic used to find this bug in the first place (this
    session's live-viz investigation): bucket episodes by whether they
    start already inside alpha_tol/d_tol, and report success separately
    per bucket. A single blended success number is exactly what hid this
    problem behind the original 98% headline figure -- this is the
    correct way to report it going forward."""
    env = UltrasoundProbeEnv(seed=1, **ENV_KWARGS)
    already_ok = dict(total=0, success=0)
    not_ok = dict(total=0, success=0, steps=[])
    for i in range(n):
        obs, info = env.reset(seed=seed_start + i)
        target = env.targets[env.target_idx]
        alpha0, d0 = env._pose_error(target)
        started_ok = alpha0 <= env.alpha_tol and d0 <= env.d_tol
        done = False
        steps = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(int(action))
            steps += 1
            done = term or trunc
        succ = bool(info.get("success"))
        bucket = already_ok if started_ok else not_ok
        bucket["total"] += 1
        bucket["success"] += int(succ)
        if not started_ok:
            not_ok["steps"].append(steps)

    result = dict(
        already_ok_total=already_ok["total"], already_ok_success=already_ok["success"],
        already_ok_rate=already_ok["success"] / already_ok["total"] if already_ok["total"] else None,
        not_ok_total=not_ok["total"], not_ok_success=not_ok["success"],
        not_ok_rate=not_ok["success"] / not_ok["total"] if not_ok["total"] else None,
        not_ok_mean_steps=float(np.mean(not_ok["steps"])) if not_ok["steps"] else None,
    )
    return result


def print_bucketed(label: str, r: dict):
    print(f"\n=== bucketed eval: {label} ===")
    print(f"  already-in-tolerance: {r['already_ok_success']}/{r['already_ok_total']}"
          f" ({r['already_ok_rate']:.3f})" if r["already_ok_rate"] is not None else "  already-in-tolerance: n/a")
    print(f"  NOT in tolerance:     {r['not_ok_success']}/{r['not_ok_total']}"
          f" ({r['not_ok_rate']:.3f})" if r["not_ok_rate"] is not None else "  NOT in tolerance: n/a")
    if r["not_ok_mean_steps"] is not None:
        print(f"  NOT-in-tolerance mean steps: {r['not_ok_mean_steps']:.1f}")


def main(smoke: bool):
    schedule = SMOKE_SCHEDULE if smoke else FULL_SCHEDULE
    total_timesteps = SMOKE_TOTAL_TIMESTEPS if smoke else FULL_TOTAL_TIMESTEPS
    tag = "smoke" if smoke else "full"

    log_dir = LOG_DIR / tag
    stage_dir = STAGE_DIR / tag
    log_dir.mkdir(parents=True, exist_ok=True)
    stage_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== fix_navigation_gap [{tag}] ===")
    print(f"config: {PPO_CONFIG}")
    print(f"env_kwargs: {ENV_KWARGS}, n_envs={N_ENVS}, seed={SEED}")
    print(f"schedule: {schedule}")
    print(f"total_timesteps budget (worst case): {total_timesteps}")

    print("\n--- baseline (current deployed model) bucketed eval, for comparison ---")
    if BASELINE_MODEL_PATH.exists():
        baseline_model = PPO.load(str(BASELINE_MODEL_PATH))
        baseline_n = 30 if smoke else 200
        baseline_result = bucketed_eval(baseline_model, n=baseline_n)
        print_bucketed(f"BASELINE (deployed, N={baseline_n})", baseline_result)
    else:
        print("  no deployed model found at", BASELINE_MODEL_PATH, "-- skipping baseline comparison")
        baseline_result = None

    env = make_vec_env(ENV_KWARGS, str(log_dir), SEED, N_ENVS, INFO_KEYWORDS)
    model = PPO(
        "MlpPolicy", env,
        learning_rate=PPO_CONFIG["learning_rate"], gamma=PPO_CONFIG["gamma"],
        n_steps=PPO_CONFIG["n_steps"], gae_lambda=PPO_CONFIG["gae_lambda"],
        ent_coef=PPO_CONFIG["entropy_coef"], clip_range=PPO_CONFIG["clip_range"],
        policy_kwargs=dict(net_arch=PPO_CONFIG["net_arch"]),
        seed=SEED, verbose=0,
    )
    model.set_logger(configure(str(log_dir), ["csv", "tensorboard"]))

    push_cb = StartCurriculumPushScheduleCallback(schedule, str(stage_dir), verbose=1)

    t0 = time.monotonic()
    model.learn(total_timesteps=total_timesteps, progress_bar=False, callback=push_cb)
    elapsed = time.monotonic() - t0
    steps_per_sec = total_timesteps / elapsed if elapsed > 0 else float("nan")
    print(f"\n[fix_navigation_gap] training loop finished: {elapsed:.1f}s wall clock, "
          f"{total_timesteps} timesteps requested, {steps_per_sec:.1f} steps/sec")

    final_path = MODEL_DIR / tag / "model.zip"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(final_path))
    env.close()
    print(f"Final model saved to {final_path}")

    print(f"\nStage log: {push_cb.stage_log}")
    print(f"Final stage reached: {push_cb._stage_idx} "
          f"(push_prob={schedule[push_cb._stage_idx]['push_prob']}), "
          f"advancing_stopped={push_cb._advancing_stopped}")

    print("\n--- new model bucketed eval ---")
    eval_n = 30 if smoke else 200
    new_result = bucketed_eval(model, n=eval_n)
    print_bucketed(f"NEW (N={eval_n})", new_result)

    summary = dict(
        tag=tag, config=PPO_CONFIG, env_kwargs=ENV_KWARGS, schedule=schedule,
        total_timesteps_requested=total_timesteps, elapsed_sec=elapsed, steps_per_sec=steps_per_sec,
        stage_log=push_cb.stage_log, final_stage_idx=push_cb._stage_idx,
        advancing_stopped=push_cb._advancing_stopped,
        baseline_result=baseline_result, new_result=new_result,
        final_model_path=str(final_path),
    )
    with open(log_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary saved to {log_dir / 'summary.json'}")
    print("\nNOTE: the deployed model at models/ppo/best/model.zip was NOT overwritten. "
          "Review the bucketed results above before promoting this model.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                         help="tiny version of the schedule/eval, just to verify wiring and measure real throughput")
    args = parser.parse_args()
    main(smoke=args.smoke)
