"""AUDIT: verify the reachability search itself before trusting the
"head is only ~55% reachable" result. Does NOT touch reward math, phantom
geometry, tolerances, or the actuator clamp -- only inspects/tests/fixes
the SEARCH used to measure reachability.

Task A: confirm `_actuator_pose_bounds()` actually responds to
`actuator_limit_deg` (empirically, on a known cone-adjacent phantom).

Task B: for a fixed d~0 position (found via the existing marginal d-search),
do an exhaustive FINE GRID sweep over roll/pitch/yaw ONLY (position held
fixed) to find the TRUE best-achievable alpha at that position -- bypassing
Nelder-Mead's local-optimum fragility entirely. Compare to what
`_search_min_pose_error` (3 Nelder-Mead restarts) reports.

Task C: for 3 of those cases, also report the analytic geometric angle
between the unrotated local elevational axis and the target normal, as an
independent sanity anchor.

Usage: uv run python scripts/audit_reachability_search.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from environment.custom_env import UltrasoundProbeEnv, TARGET_SEQUENCE
from scripts.diagnose_head_reachability import _minimize_scalar, _set_pose

OUT_DIR = Path(__file__).resolve().parent.parent / "logs" / "reachability_audit"
N_AUDIT_CASES = 20
GRID_RES = 31  # per-axis resolution for the roll/pitch/yaw fine grid


def task_a_cone_wiring_check(seed_list=(6, 7, 41, 43, 44)):
    """Empirically confirm actuator_limit_deg reaches the search: for
    several known cone-adjacent phantoms, report the joint search's result
    across cone widths and flag any non-monotonicity (which would itself
    indicate optimizer unreliability, not a wiring bug -- see docstring)."""
    print("=== TASK A: does the search respond to actuator_limit_deg? ===")
    results = []
    for seed in seed_list:
        row = {"seed": seed}
        for cone in [60.0, 70.0, 80.0]:
            env = UltrasoundProbeEnv(seed=0, guarantee_reachable=False, actuator_limit_deg=cone)
            env.reset(seed=seed)
            alpha, d = env._search_min_pose_error("head")
            row[f"alpha_deg_cone{int(cone)}"] = float(np.degrees(alpha))
            row[f"d_mm_cone{int(cone)}"] = float(d * 1000)
        results.append(row)
        print(f"  seed={seed}: " + ", ".join(
            f"cone{c}=alpha{row[f'alpha_deg_cone{c}']:.1f}deg" for c in [60, 70, 80]
        ))
    responds = any(
        abs(r["alpha_deg_cone60"] - r["alpha_deg_cone80"]) > 1.0 for r in results
    )
    print(f"  Verdict: search DOES change with cone width for at least one case "
          f"({'confirms wiring is live' if responds else 'SUSPICIOUS -- may be hardcoded'})")
    non_monotonic = [r for r in results
                     if not (r["alpha_deg_cone60"] >= r["alpha_deg_cone70"] - 0.5 >= r["alpha_deg_cone80"] - 1.0)
                     and not (r["alpha_deg_cone60"] >= r["alpha_deg_cone70"] and r["alpha_deg_cone70"] >= r["alpha_deg_cone80"])]
    print(f"  Non-monotonic cases (widening the cone should never make the TRUE optimum "
          f"worse -- non-monotonicity here indicates optimizer unreliability, not "
          f"reduced reachability): {len(non_monotonic)}/{len(results)}")
    return results


def fine_grid_tilt_sweep(env: UltrasoundProbeEnv, target: str, theta: float, phi: float,
                          grid_res: int = GRID_RES):
    """Exhaustive grid over (roll, pitch, yaw) with (theta, phi) FIXED --
    bypasses Nelder-Mead entirely. Returns (best_alpha, d_at_best, roll, pitch, yaw)."""
    limit = env.actuator_limit_deg
    axis_vals = np.linspace(-limit, limit, grid_res)
    best_alpha, best_d, best_rpy = np.inf, None, None
    env.probe.theta, env.probe.phi = theta, phi
    for roll in axis_vals:
        env.probe.roll = roll
        for pitch in axis_vals:
            env.probe.pitch = pitch
            for yaw in axis_vals:
                env.probe.yaw = yaw
                alpha, d = env._pose_error(target)
                if alpha < best_alpha:
                    best_alpha, best_d, best_rpy = alpha, d, (roll, pitch, yaw)
    return best_alpha, best_d, best_rpy


def analytic_min_tilt_angle(env: UltrasoundProbeEnv, target: str, theta: float, phi: float) -> float:
    """The GEOMETRIC lower bound: the angle between the UNROTATED local
    elevational axis (roll=pitch=yaw=0) and the target plane's normal.
    This is the minimum possible alpha achievable by ANY single-axis
    rotation from this position -- a true rotation (not constrained to the
    roll/pitch/yaw decomposition) could always achieve exactly this angle.
    If the grid-swept best_alpha is close to this, the actuator's
    decomposition is not the bottleneck; if best_alpha is much larger, the
    roll/pitch/yaw decomposition itself is inefficient at this position."""
    saved = (env.probe.theta, env.probe.phi, env.probe.roll, env.probe.pitch, env.probe.yaw)
    env.probe.theta, env.probe.phi = theta, phi
    env.probe.roll = env.probe.pitch = env.probe.yaw = 0.0
    alpha0, _ = env._pose_error(target)
    env.probe.theta, env.probe.phi, env.probe.roll, env.probe.pitch, env.probe.yaw = saved
    return alpha0


def task_b_and_c_joint_verification(n_cases: int = N_AUDIT_CASES, seed: int = 0):
    print(f"\n=== TASK B: fine-grid joint verification for {n_cases} failing head cases ===")
    env = UltrasoundProbeEnv(seed=seed, guarantee_reachable=False)
    alpha_tol_deg = float(np.degrees(env.alpha_tol))
    d_tol_mm = env.d_tol * 1000.0

    with open(Path(__file__).resolve().parent.parent / "logs" / "head_reachability_diagnosis" / "head_records_raw.json") as f:
        prior_records = json.load(f)

    failing_seeds = [i for i, r in enumerate(prior_records) if not r["reachable"]][:n_cases]

    rows = []
    for case_seed in failing_seeds:
        env.reset(seed=case_seed)
        rng = np.random.default_rng(case_seed + 1)

        # find a d~0 position via the existing (fixed) marginal d-search
        alpha_at_d_min, d_min, best_x = _minimize_scalar(env, "head", lambda a, d: d, rng)
        theta_at_d_min, phi_at_d_min = best_x[0], best_x[1]

        # ground truth: original (weak) joint search's verdict
        alpha_orig, d_orig = env._search_min_pose_error("head")
        reachable_orig = env._is_target_reachable("head")

        # fine-grid tilt sweep AT THE FIXED d~0 POSITION
        best_alpha_grid, d_at_best_grid, best_rpy = fine_grid_tilt_sweep(
            env, "head", theta_at_d_min, phi_at_d_min
        )
        analytic_angle = analytic_min_tilt_angle(env, "head", theta_at_d_min, phi_at_d_min)

        jointly_reachable_grid = (np.degrees(best_alpha_grid) <= alpha_tol_deg) and (d_at_best_grid * 1000 <= d_tol_mm)

        rows.append(dict(
            seed=case_seed,
            d_at_dmin_pose_mm=float(d_min * 1000),
            orig_search_alpha_deg=float(np.degrees(alpha_orig)), orig_search_d_mm=float(d_orig * 1000),
            orig_reachable=bool(reachable_orig),
            grid_best_alpha_deg=float(np.degrees(best_alpha_grid)), grid_d_mm=float(d_at_best_grid * 1000),
            analytic_min_angle_deg=float(np.degrees(analytic_angle)),
            grid_jointly_reachable=bool(jointly_reachable_grid),
            best_roll_pitch_yaw=best_rpy,
        ))
        print(f"  seed={case_seed}: orig_search(alpha={np.degrees(alpha_orig):.1f}deg, reachable={reachable_orig}) "
              f"vs grid-at-fixed-pos(alpha={np.degrees(best_alpha_grid):.1f}deg, d={d_at_best_grid*1000:.2f}mm, "
              f"jointly_reachable={jointly_reachable_grid}) | analytic_min_angle={np.degrees(analytic_angle):.1f}deg")

    n_orig_reachable = sum(r["orig_reachable"] for r in rows)
    n_grid_reachable = sum(r["grid_jointly_reachable"] for r in rows)
    print(f"\n  Of {len(rows)} cases the ORIGINAL search called unreachable:")
    print(f"    grid-sweep confirms {n_grid_reachable}/{len(rows)} are ACTUALLY jointly reachable "
          f"(search under-reported reachability if this is > 0)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "joint_verification.json", "w") as f:
        json.dump(rows, f, indent=2, default=str)

    return rows


if __name__ == "__main__":
    task_a_cone_wiring_check()
    task_b_and_c_joint_verification()
