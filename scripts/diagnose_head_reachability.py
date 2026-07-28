"""DIAGNOSTIC ONLY -- decide whether head-target reachability failures are
alpha-bound (tilt/actuator-cone limited) or d-bound (position/distance
limited), before anyone touches the +-60deg cone or the tolerances.

This script changes NOTHING about env behavior: no reward math, no
geometry, no tolerances, no actuator limits, no reset() sampling. It only
runs additional read-only searches (independently minimizing alpha alone
and d alone, in addition to calling the env's own unmodified
`_is_target_reachable` / `_search_min_pose_error`) and reports numbers.

Reuses `UltrasoundProbeEnv._pose_error` (the exact reward criterion) and
`ACTUATOR_POSE_BOUNDS` (the exact actuator limits) directly from
`environment/custom_env.py` -- the geometry itself is never reimplemented
here, only additional optimizer objectives are added on top of it.

Usage: uv run python scripts/diagnose_head_reachability.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.optimize import minimize

from environment.custom_env import UltrasoundProbeEnv, TARGET_SEQUENCE

OUT_DIR = Path(__file__).resolve().parent.parent / "logs" / "head_reachability_diagnosis"
N_HEAD_SAMPLES = 500
N_CONTEXT_SAMPLES = 100  # abdomen/femur, informational only
N_RESTARTS = 5
SEARCH_OPTIONS = dict(xatol=1e-3, fatol=1e-4, maxiter=300, maxfev=300)


def _set_pose(env: UltrasoundProbeEnv, params: np.ndarray):
    env.probe.theta, env.probe.phi, env.probe.roll, env.probe.pitch, env.probe.yaw = params


def _minimize_scalar(env: UltrasoundProbeEnv, target: str, scalar_fn, rng: np.random.Generator,
                      n_restarts: int = N_RESTARTS):
    """Bounded Nelder-Mead search (env._actuator_pose_bounds(), the same
    real actuator limits the env's own reachability predicate uses --
    tracks whatever actuator_limit_deg THIS env instance was constructed
    with, not a fixed module-level default) minimizing `scalar_fn(alpha,
    d)`. Returns (best_alpha, best_d, best_pose)."""
    best_x, best_val = None, np.inf
    bounds = env._actuator_pose_bounds()

    def objective(params):
        _set_pose(env, params)
        alpha, d = env._pose_error(target)
        return scalar_fn(alpha, d)

    for _ in range(n_restarts):
        x0 = np.array([rng.uniform(lo, hi) for lo, hi in bounds])
        res = minimize(objective, x0, method="Nelder-Mead", bounds=bounds,
                        options=SEARCH_OPTIONS)
        if res.fun < best_val:
            best_val = res.fun
            best_x = res.x

    _set_pose(env, best_x)
    alpha, d = env._pose_error(target)
    return alpha, d, best_x


def analyze_phantom(env: UltrasoundProbeEnv, target: str, rng: np.random.Generator) -> dict:
    """For the CURRENT env.phantom, find:
      - best achievable alpha (minimize alpha alone) and the d at that pose
      - best achievable d (minimize d alone) and the alpha at that pose
      - the combined-criterion pose (env's own unmodified
        `_search_min_pose_error`, exactly what `_is_target_reachable` uses)
      - the env's own reachability verdict (`_is_target_reachable`,
        unmodified)
    """
    # NOTE: _minimize_scalar always returns (alpha, d, best_x) regardless of
    # which scalar was minimized -- the unpacking order below must stay
    # (alpha, d, _) for BOTH calls (an earlier version of this script
    # swapped the second call's unpacking to (d_min, alpha_at_d_min, _),
    # which silently stored alpha-in-radians into "d_min" and multiplied it
    # by 1000 as if converting meters to mm, producing impossible
    # 800-1500mm "distances" -- caught by those values being geometrically
    # impossible given the phantom's ~0.1-0.2m scale, not by any assertion).
    alpha_min, d_at_alpha_min, _ = _minimize_scalar(env, target, lambda a, d: a, rng)
    alpha_at_d_min, d_min, _ = _minimize_scalar(env, target, lambda a, d: d, rng)
    alpha_combined, d_combined = env._search_min_pose_error(target)  # env's own unmodified method
    reachable = env._is_target_reachable(target)  # env's own unmodified predicate

    return dict(
        alpha_min_deg=float(np.degrees(alpha_min)), d_at_alpha_min_mm=float(d_at_alpha_min * 1000),
        d_min_mm=float(d_min * 1000), alpha_at_d_min_deg=float(np.degrees(alpha_at_d_min)),
        alpha_combined_deg=float(np.degrees(alpha_combined)), d_combined_mm=float(d_combined * 1000),
        reachable=bool(reachable),
    )


def classify(record: dict, alpha_tol_deg: float, d_tol_mm: float) -> str:
    """alpha-bound: best achievable d is within tolerance but best
    achievable alpha never gets within tolerance. d-bound: the reverse.
    both-bound: neither individually satisfiable."""
    alpha_ok = record["alpha_min_deg"] <= alpha_tol_deg
    d_ok = record["d_min_mm"] <= d_tol_mm
    if alpha_ok and not d_ok:
        return "d-bound"
    if d_ok and not alpha_ok:
        return "alpha-bound"
    if not alpha_ok and not d_ok:
        return "both-bound"
    return "both-satisfiable-individually"  # only fails jointly at the combined pose, if at all


def run(seed: int = 0):
    env = UltrasoundProbeEnv(seed=seed, guarantee_reachable=False)
    alpha_tol_deg = float(np.degrees(env.alpha_tol))
    d_tol_mm = env.d_tol * 1000.0
    print(f"Active tolerances: alpha_tol={alpha_tol_deg:.1f}deg, d_tol={d_tol_mm:.1f}mm\n")

    rng = np.random.default_rng(seed + 1)  # search restarts: separate from phantom-sampling rng

    print(f"Sampling {N_HEAD_SAMPLES} head-target phantoms (reachability filtering OFF)...")
    head_records = []
    for i in range(N_HEAD_SAMPLES):
        env.reset(seed=seed + i)  # guarantee_reachable=False: raw, unfiltered sampling
        rec = analyze_phantom(env, "head", rng)
        head_records.append(rec)
        if (i + 1) % 50 == 0:
            print(f"  ...{i + 1}/{N_HEAD_SAMPLES}")

    failing = [r for r in head_records if not r["reachable"]]
    reachable = [r for r in head_records if r["reachable"]]

    classes = [classify(r, alpha_tol_deg, d_tol_mm) for r in failing]
    n_alpha_bound = classes.count("alpha-bound")
    n_d_bound = classes.count("d-bound")
    n_both_bound = classes.count("both-bound")
    n_other = len(classes) - n_alpha_bound - n_d_bound - n_both_bound
    n_fail = len(failing)

    def dist(values):
        arr = np.array(values)
        return dict(min=float(arr.min()), median=float(np.median(arr)),
                    max=float(arr.max()), mean=float(arr.mean()))

    failing_alpha_min_dist = dist([r["alpha_min_deg"] for r in failing]) if failing else None
    failing_d_min_dist = dist([r["d_min_mm"] for r in failing]) if failing else None
    reachable_d_combined_dist = dist([r["d_combined_mm"] for r in reachable]) if reachable else None
    reachable_alpha_combined_dist = dist([r["alpha_combined_deg"] for r in reachable]) if reachable else None

    print(f"\n=== HEAD: {n_fail}/{N_HEAD_SAMPLES} unreachable "
          f"({n_fail / N_HEAD_SAMPLES:.3f}), {len(reachable)}/{N_HEAD_SAMPLES} reachable ===")
    if n_fail:
        print(f"Of the {n_fail} failing (unreachable) head phantoms:")
        print(f"  alpha-bound   : {n_alpha_bound} ({n_alpha_bound / n_fail:.3f})")
        print(f"  d-bound       : {n_d_bound} ({n_d_bound / n_fail:.3f})")
        print(f"  both-bound    : {n_both_bound} ({n_both_bound / n_fail:.3f})")
        if n_other:
            print(f"  other (fails only jointly, both individually satisfiable): {n_other} ({n_other / n_fail:.3f})")

        print(f"\nFailing-case best-achievable ALPHA distribution (deg): {failing_alpha_min_dist}")
        print(f"Failing-case best-achievable D distribution (mm):     {failing_d_min_dist}")

    if reachable:
        print(f"\nReachable head cases: best-achievable D at the combined-criterion pose (mm): "
              f"{reachable_d_combined_dist}")
        print(f"Reachable head cases: best-achievable ALPHA at the combined-criterion pose (deg): "
              f"{reachable_alpha_combined_dist}")

    # Optional context: abdomen/femur d_min distributions, informational only
    context = {}
    for target in ["abdomen", "femur"]:
        print(f"\nSampling {N_CONTEXT_SAMPLES} {target}-target phantoms for context (d_min only)...")
        d_mins = []
        for i in range(N_CONTEXT_SAMPLES):
            env.reset(seed=seed + 10_000 + i)
            _, d_min, _ = _minimize_scalar(env, target, lambda a, d: d, rng)  # returns (alpha, d, best_x)
            d_mins.append(d_min * 1000)
        context[target] = dist(d_mins)
        print(f"  {target} best-achievable D distribution (mm): {context[target]}")

    report = dict(
        alpha_tol_deg=alpha_tol_deg, d_tol_mm=d_tol_mm,
        n_head_samples=N_HEAD_SAMPLES,
        n_unreachable=n_fail, n_reachable=len(reachable),
        unreachable_fraction=n_fail / N_HEAD_SAMPLES,
        split=dict(alpha_bound=n_alpha_bound, d_bound=n_d_bound, both_bound=n_both_bound, other=n_other),
        split_fraction=dict(
            alpha_bound=n_alpha_bound / n_fail if n_fail else None,
            d_bound=n_d_bound / n_fail if n_fail else None,
            both_bound=n_both_bound / n_fail if n_fail else None,
        ),
        failing_alpha_min_deg_distribution=failing_alpha_min_dist,
        failing_d_min_mm_distribution=failing_d_min_dist,
        reachable_d_combined_mm_distribution=reachable_d_combined_dist,
        reachable_alpha_combined_deg_distribution=reachable_alpha_combined_dist,
        context_d_min_mm_distribution=context,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "report.json", "w") as f:
        json.dump(report, f, indent=2)
    with open(OUT_DIR / "head_records_raw.json", "w") as f:
        json.dump(head_records, f, indent=2)

    verdict = "alpha-bound" if n_alpha_bound > n_d_bound and n_alpha_bound > n_both_bound else (
        "d-bound" if n_d_bound > n_alpha_bound and n_d_bound > n_both_bound else "both/mixed"
    )
    print(f"\n=== VERDICT: head reachability failures are predominantly {verdict} "
          f"(alpha-bound={n_alpha_bound}, d-bound={n_d_bound}, both-bound={n_both_bound} "
          f"out of {n_fail} failing cases) ===")

    print(f"\nSaved full report to {OUT_DIR}")
    return report


if __name__ == "__main__":
    run()
