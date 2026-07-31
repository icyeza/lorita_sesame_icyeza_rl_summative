"""Held-out evaluation of the DEPLOYED PPO agent + generalization sweep
across start conditions (status.md "ablations, protocol fix and report
figures" addendum).

WHY THIS EXISTS: the previously-reported deployed-agent numbers (98.0%
success, 3.49deg, 1.47mm) were NOT an evaluation. They are the last-20%
TAIL OF THE HEADLINE RUN'S OWN TRAINING EPISODES -- stochastic
action sampling, 1737 training episodes, computed inside
scripts/headline_run.py (see logs/headline/single_target_fallback/
headline_summary.json). This script produces the missing thing: a
deterministic, held-out evaluation of the same weights, on seeds disjoint
from training AND from the 9000-9099 range scripts/generalization_check.py
used.

PART A -- deployed held-out eval:
  model      models/ppo/best/model.zip (the headline weights)
  env        UltrasoundProbeEnv() at CLASS DEFAULTS: single_target=True,
             start_curriculum=True, start_curriculum_max_random_steps=8,
             alpha_tol_deg=18, d_tol_m=0.012, tilt_step_deg=3.0,
             shaping_mode="multiplicative", all 3 targets sampled
  policy     deterministic=True
  N          300 episodes, eval seeds 20000-20299
  budget     subtask_max_steps=60 (env default) -- in single_target mode
             the episode ends the moment that subtask ends, so 60 env
             steps is the whole per-episode budget
  Stratified by target (head / abdomen / femur) as well as pooled.

PART B -- generalization sweep, same model/policy/protocol, varying ONLY
  the start distribution, N=100 per condition, eval seeds 30000+ (1000
  apart per condition, so no condition shares a seed with another or with
  Part A):
    curriculum (radius=8, the training distribution), small (20),
    medium (40), large (80), uniform-random (start_curriculum=False)

PART C -- if uniform-random collapses, quantify WHY arithmetically:
  distance from the start pose to the reachability-search optimum, and
  the minimum number of actions needed to close it given the env's own
  action granularity (COARSE_ARC_DEG=2.0deg per theta/phi action,
  tilt_step_deg=3.0deg per roll/pitch/yaw action), versus the 60-step
  budget. See `analyze_traversal` for the upper-bound caveat.

FAILURE BREAKDOWN -- DEFINITION CORRECTED AFTER SEEING THE DATA. In
single_target mode a failed episode can only end one way: the subtask
timeout at 60 steps (a freeze OUTSIDE tolerance does not terminate; it is
penalized and the episode continues). The first version of this script
split failures on "took >=1 freeze action" vs "took zero" -- which turned
out to be degenerate: EVERY failure in EVERY condition ran the full 60
steps AND took many freeze actions (median 20-38 per failed episode; the
agent spams freeze). So "off-target freeze" and "timeout" are not
alternatives here -- both are true of essentially every failure, and that
binary carries no information.

The informative split, used below, is on TERMINAL GEOMETRY -- did the
agent get the probe into the right neighbourhood and merely fail to land
a freeze inside tolerance, or did it never arrive?
  near_miss_freeze -- failed, terminal pose within 2x tolerance on BOTH
                      axes (alpha <= 36deg AND d <= 24mm): the agent was
                      in the neighbourhood and its freezes kept missing
  navigation_failure -- failed and outside that: never arrived
The raw ingredients (all failures exhaust the budget; freeze-action
counts; fraction inside each single tolerance) are reported alongside, so
the degenerate split is still recoverable and nothing is hidden.

Usage: uv run python scripts/deployed_eval.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from environment.custom_env import ACTIONS, COARSE_ARC_DEG, UltrasoundProbeEnv
from evaluation.evaluate import load_model

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = REPO_ROOT / "models" / "ppo" / "best" / "model.zip"
TABLES_DIR = REPO_ROOT / "logs" / "tables"
OUT_DIR = REPO_ROOT / "logs" / "deployed_eval"

FREEZE_ACTION = ACTIONS.index("freeze_and_measure")

# Env acceptance tolerances (UltrasoundProbeEnv class defaults), used for
# the failure breakdown's "within 2x tolerance" neighbourhood test.
ALPHA_TOL_DEG = 18.0
D_TOL_MM = 12.0

DEPLOYED_N = 300
DEPLOYED_SEED_START = 20_000

GEN_N = 100
GEN_SEED_BASE = 30_000
CONDITIONS = [
    ("curriculum (radius=8)", dict(start_curriculum=True, start_curriculum_max_random_steps=8)),
    ("small (radius=20)", dict(start_curriculum=True, start_curriculum_max_random_steps=20)),
    ("medium (radius=40)", dict(start_curriculum=True, start_curriculum_max_random_steps=40)),
    ("large (radius=80)", dict(start_curriculum=True, start_curriculum_max_random_steps=80)),
    ("uniform-random", dict(start_curriculum=False)),
]


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion. Used instead of the
    normal approximation because several conditions here are expected near
    0% or 100%, where the normal interval leaves [0,1]."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (float(max(0.0, centre - half)), float(min(1.0, centre + half)))


def run_episodes(model, env_kwargs: dict, n: int, seed_start: int,
                 capture_traversal: bool = False) -> pd.DataFrame:
    """One row per episode. `capture_traversal` additionally records the
    start pose and the reachability-search optimum for Part C."""
    env = UltrasoundProbeEnv(seed=1, **env_kwargs)
    rows = []
    for i in range(n):
        obs, _ = env.reset(seed=seed_start + i)
        target = env.targets[0]
        row = dict(episode=i, seed=seed_start + i, target=target)

        if capture_traversal:
            start_pose = (env.probe.theta, env.probe.phi, env.probe.roll,
                          env.probe.pitch, env.probe.yaw)
            env._search_min_pose_error(target)  # deterministic; populates _last_search_pose
            opt_pose = env._last_search_pose
            row.update(_traversal_metrics(env, start_pose, opt_pose))

        done = False
        steps = 0
        n_freeze = 0
        total_reward = 0.0
        info = {}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
            if action == FREEZE_ACTION:
                n_freeze += 1
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            steps += 1

        row.update(
            success=bool(info["success"]),
            steps=steps,
            n_freeze_actions=n_freeze,
            freeze_attempted=bool(info["freeze_attempted"]),
            terminal_alpha_deg=float(info["alpha_deg"]),
            terminal_d_mm=float(info["d_m"]) * 1000.0,
            episode_reward=total_reward,
        )
        rows.append(row)
    env.close()
    return pd.DataFrame(rows)


def _traversal_metrics(env, start_pose, opt_pose) -> dict:
    """Part C arithmetic. See `analyze_traversal` for what this does and
    does NOT establish."""
    th0, ph0, r0, p0, y0 = start_pose
    th1, ph1, r1, p1, y1 = opt_pose

    dth = abs(th1 - th0)
    dph = abs((ph1 - ph0 + np.pi) % (2 * np.pi) - np.pi)  # wrapped shortest way round

    # Angular separation of the two surface directions on the (theta, phi)
    # parameter sphere -- the "geodesic" angle in parameter space.
    cos_sep = (np.cos(th0) * np.cos(th1)
               + np.sin(th0) * np.sin(th1) * np.cos(ph1 - ph0))
    sep = float(np.arccos(np.clip(cos_sep, -1.0, 1.0)))

    # Straight-line distance between the two actual 3-D probe positions on
    # the abdomen surface (mm) -- a physical, not parametric, distance.
    saved = (env.probe.theta, env.probe.phi, env.probe.roll, env.probe.pitch, env.probe.yaw)
    env.probe.theta, env.probe.phi = th0, ph0
    pos0 = env._probe_frame()[0]
    env.probe.theta, env.probe.phi = th1, ph1
    pos1 = env._probe_frame()[0]
    env.probe.theta, env.probe.phi, env.probe.roll, env.probe.pitch, env.probe.yaw = saved
    chord_mm = float(np.linalg.norm(pos1 - pos0)) * 1000.0

    # Minimum action count. theta/phi each move by exactly COARSE_ARC_DEG
    # per action; roll/pitch/yaw by tilt_step_deg. The action set is
    # axis-aligned, so the reachable minimum is the SUM over axes
    # (Manhattan in parameter space), not the geodesic.
    arc = np.radians(COARSE_ARC_DEG)
    tilt = env.tilt_step_deg
    steps_pos = int(np.ceil(dth / arc)) + int(np.ceil(dph / arc))
    steps_tilt = (int(np.ceil(abs(r1 - r0) / tilt))
                  + int(np.ceil(abs(p1 - p0) / tilt))
                  + int(np.ceil(abs(y1 - y0) / tilt)))
    return dict(
        parameter_sphere_separation_deg=float(np.degrees(sep)),
        surface_chord_mm=chord_mm,
        min_positional_actions=steps_pos,
        min_tilt_actions=steps_tilt,
        min_total_actions=steps_pos + steps_tilt + 1,  # +1 for the freeze
    )


def summarize(df: pd.DataFrame, label: str) -> dict:
    n = len(df)
    k = int(df["success"].sum())
    lo, hi = wilson_ci(k, n)
    succ = df[df["success"]]
    fail = df[~df["success"]]

    def q(series):
        if len(series) == 0:
            return dict(median=None, q1=None, q3=None)
        return dict(median=float(series.median()),
                    q1=float(series.quantile(0.25)),
                    q3=float(series.quantile(0.75)))

    return dict(
        label=label, n=n, n_success=k,
        success_rate=k / n if n else float("nan"),
        wilson_lo=lo, wilson_hi=hi,
        alpha_deg_success=q(succ["terminal_alpha_deg"]),
        d_mm_success=q(succ["terminal_d_mm"]),
        alpha_deg_all=q(df["terminal_alpha_deg"]),
        d_mm_all=q(df["terminal_d_mm"]),
        n_fail=len(fail),
        # Informative split (see module docstring): terminal geometry.
        fail_near_miss_freeze=int(((fail["terminal_alpha_deg"] <= 2 * ALPHA_TOL_DEG)
                                   & (fail["terminal_d_mm"] <= 2 * D_TOL_MM)).sum()),
        fail_navigation=int((~((fail["terminal_alpha_deg"] <= 2 * ALPHA_TOL_DEG)
                               & (fail["terminal_d_mm"] <= 2 * D_TOL_MM))).sum()),
        # Degenerate-but-requested split, kept so it is on the record.
        fail_took_a_freeze_action=int((fail["n_freeze_actions"] > 0).sum()),
        fail_never_froze=int((fail["n_freeze_actions"] == 0).sum()),
        fail_exhausted_budget=int((fail["steps"] >= 60).sum()),
        fail_median_freeze_actions=(float(fail["n_freeze_actions"].median()) if len(fail) else None),
        fail_frac_within_alpha_tol=(float((fail["terminal_alpha_deg"] <= ALPHA_TOL_DEG).mean())
                                    if len(fail) else None),
        fail_frac_within_d_tol=(float((fail["terminal_d_mm"] <= D_TOL_MM).mean())
                                if len(fail) else None),
        median_steps=float(df["steps"].median()),
        mean_episode_reward=float(df["episode_reward"].mean()),
    )


def analyze_traversal(df: pd.DataFrame) -> dict:
    """Part C. CAVEAT, stated so the report can state it: the optimum used
    here is the single pose `_search_min_pose_error` converges to. The
    acceptance region (alpha<=18deg AND d<=12mm) is a MANIFOLD, not a
    point, so the true minimum action count to *some* acceptable pose is
    <= the number computed here. These figures are therefore an UPPER
    BOUND on the minimum, and the honest reading is: "the distance to a
    known-good pose is X steps", not "the task provably needs X steps"."""
    return dict(
        n=len(df),
        mean_parameter_sphere_separation_deg=float(df["parameter_sphere_separation_deg"].mean()),
        median_parameter_sphere_separation_deg=float(df["parameter_sphere_separation_deg"].median()),
        mean_surface_chord_mm=float(df["surface_chord_mm"].mean()),
        median_surface_chord_mm=float(df["surface_chord_mm"].median()),
        mean_min_positional_actions=float(df["min_positional_actions"].mean()),
        median_min_positional_actions=float(df["min_positional_actions"].median()),
        mean_min_total_actions=float(df["min_total_actions"].mean()),
        median_min_total_actions=float(df["min_total_actions"].median()),
        step_budget=60,
        frac_episodes_over_budget=float((df["min_total_actions"] > 60).mean()),
    )


def main(from_cache: bool = False):
    """from_cache: recompute every summary/table from the per-episode CSVs
    already in logs/deployed_eval/ instead of re-running the episodes.
    Used when only the SUMMARY definitions change (as they did for the
    failure breakdown) -- the episode-level data is unchanged and
    re-rolling 800 deterministic episodes would produce identical rows at
    ~30 min cost."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()

    model = None
    if not from_cache:
        env_defaults = UltrasoundProbeEnv(seed=0)
        print(f"Loading DEPLOYED model: {MODEL_PATH}")
        model = load_model("ppo", str(MODEL_PATH), env_defaults)

    # ---------------- PART A ----------------
    print(f"\n=== PART A: deployed held-out eval (N={DEPLOYED_N}, seeds "
          f"{DEPLOYED_SEED_START}-{DEPLOYED_SEED_START + DEPLOYED_N - 1}, deterministic, class defaults) ===")
    if from_cache:
        df_a = pd.read_csv(OUT_DIR / "deployed_episodes.csv")
    else:
        df_a = run_episodes(model, {}, DEPLOYED_N, DEPLOYED_SEED_START)
        df_a.to_csv(OUT_DIR / "deployed_episodes.csv", index=False)

    pooled = summarize(df_a, "pooled (all 3 targets)")
    per_target = [summarize(df_a[df_a["target"] == t], t)
                  for t in ["head", "abdomen", "femur"]]
    for s in [pooled] + per_target:
        print(f"  {s['label']:<24} n={s['n']:>3} success={s['success_rate']:.4f} "
              f"[{s['wilson_lo']:.4f}, {s['wilson_hi']:.4f}]  "
              f"fail: near_miss={s['fail_near_miss_freeze']} navigation={s['fail_navigation']}")

    # ---------------- PART B ----------------
    print(f"\n=== PART B: generalization sweep (N={GEN_N}/condition, deterministic) ===")
    gen_rows, gen_frames = [], {}
    for idx, (label, kwargs) in enumerate(CONDITIONS):
        seed_start = GEN_SEED_BASE + 1000 * idx
        capture = kwargs.get("start_curriculum", True) is False
        if from_cache:
            df = pd.read_csv(OUT_DIR / f"gen_episodes_{idx}.csv")
        else:
            df = run_episodes(model, kwargs, GEN_N, seed_start, capture_traversal=capture)
        gen_frames[label] = df
        s = summarize(df, label)
        s["seed_start"] = seed_start
        s["seed_end"] = seed_start + GEN_N - 1
        gen_rows.append(s)
        print(f"  {label:<24} success={s['success_rate']:.4f} "
              f"[{s['wilson_lo']:.4f}, {s['wilson_hi']:.4f}]  "
              f"med_alpha={s['alpha_deg_all']['median']:.2f}deg "
              f"med_d={s['d_mm_all']['median']:.2f}mm  "
              f"fail: near_miss={s['fail_near_miss_freeze']} navigation={s['fail_navigation']}")
        if not from_cache:
            df.to_csv(OUT_DIR / f"gen_episodes_{idx}.csv", index=False)

    # ---------------- PART C ----------------
    traversal = None
    uniform = next(r for r in gen_rows if r["label"] == "uniform-random")
    if uniform["success_rate"] < 0.5:
        print(f"\n=== PART C: uniform-random success={uniform['success_rate']:.4f} -- "
              f"quantifying traversal requirement ===")
        traversal = analyze_traversal(gen_frames["uniform-random"])
        for k, v in traversal.items():
            print(f"  {k} = {v}")
    else:
        print(f"\n=== PART C skipped: uniform-random did not collapse "
              f"(success={uniform['success_rate']:.4f}) ===")

    # ---------------- write outputs ----------------
    gen_table = pd.DataFrame([dict(
        condition=s["label"], n=s["n"], seed_start=s["seed_start"], seed_end=s["seed_end"],
        success_rate=s["success_rate"], wilson_lo=s["wilson_lo"], wilson_hi=s["wilson_hi"],
        median_terminal_alpha_deg=s["alpha_deg_all"]["median"],
        median_terminal_d_mm=s["d_mm_all"]["median"],
        median_terminal_alpha_deg_success=s["alpha_deg_success"]["median"],
        median_terminal_d_mm_success=s["d_mm_success"]["median"],
        n_fail=s["n_fail"],
        fail_near_miss_freeze=s["fail_near_miss_freeze"],
        fail_navigation=s["fail_navigation"],
        fail_exhausted_budget=s["fail_exhausted_budget"],
        fail_took_a_freeze_action=s["fail_took_a_freeze_action"],
        fail_never_froze=s["fail_never_froze"],
        fail_median_freeze_actions=s["fail_median_freeze_actions"],
        fail_frac_within_alpha_tol=s["fail_frac_within_alpha_tol"],
        fail_frac_within_d_tol=s["fail_frac_within_d_tol"],
        median_steps=s["median_steps"],
    ) for s in gen_rows])
    gen_table.to_csv(TABLES_DIR / "generalization_eval.csv", index=False)

    payload = dict(
        model_path=str(MODEL_PATH),
        protocol=dict(
            policy="deterministic=True",
            env="UltrasoundProbeEnv class defaults (single_target=True, alpha_tol_deg=18, "
                "d_tol_m=0.012, tilt_step_deg=3.0, shaping_mode=multiplicative), all 3 targets sampled",
            step_budget_per_episode=60,
            deployed_n=DEPLOYED_N,
            deployed_seeds=[DEPLOYED_SEED_START, DEPLOYED_SEED_START + DEPLOYED_N - 1],
            generalization_n=GEN_N,
            generalization_seed_base=GEN_SEED_BASE,
            failure_breakdown_definition=(
                "near_miss_freeze = failed episode whose terminal pose was within 2x tolerance "
                "on BOTH axes (alpha <= 36deg AND d <= 24mm); navigation_failure = failed "
                "episode outside that. The originally-requested 'off-target freeze vs timeout' "
                "split is degenerate on this data and is reported alongside as "
                "fail_took_a_freeze_action / fail_never_froze / fail_exhausted_budget: every "
                "failure in every condition ran the full 60 steps AND took many freeze actions "
                "(median 20-38), so those two labels are not alternatives."),
        ),
        deployed_pooled=pooled,
        deployed_per_target=per_target,
        generalization=gen_rows,
        traversal_analysis=traversal,
        training_tail_reference=dict(
            source="logs/headline/single_target_fallback/headline_summary.json",
            note="NOT an evaluation -- last-20% tail of the headline run's own training episodes, "
                 "stochastic sampling, N=1737 of 8685",
            converged_success_rate=0.9804260218767991,
            median_terminal_alpha_deg=3.4948708756785103,
            median_terminal_d_mm=1.4729258427482999,
        ),
        wall_clock_s=time.monotonic() - t0,
    )
    with open(OUT_DIR / "deployed_eval.json", "w") as f:
        json.dump(payload, f, indent=2, default=str)

    print(f"\nWrote {TABLES_DIR / 'generalization_eval.csv'}")
    print(f"Wrote {OUT_DIR / 'deployed_eval.json'}")
    print(f"Total wall-clock: {(time.monotonic() - t0)/60:.1f} min")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-cache", action="store_true",
                        help="Recompute summaries/tables from the saved per-episode CSVs "
                             "instead of re-running the episodes.")
    args = parser.parse_args()
    main(from_cache=args.from_cache)
