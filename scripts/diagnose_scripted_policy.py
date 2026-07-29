"""NON-TRAINING diagnostic: is the never-succeeds-in-training signature
(alpha -> ~0.3deg perfect, d parked at ~30-35mm against a 12mm tolerance)
a LEARNING/REWARD problem (env is solvable, PPO's reward lets it stop
short) or a SETUP problem (action granularity / alpha-d coupling makes
12mm unreachable by discrete control)?

No RL training happens anywhere in this file. Only scripted/analytic
policies and the existing reachability fine-search optimizer
(`UltrasoundProbeEnv._search_min_pose_error` / `._actuator_pose_bounds` /
`._pose_error`) are used. Nothing in environment/, reward, tolerances, or
actions is modified -- this file only measures.

Usage: uv run python scripts/diagnose_scripted_policy.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.optimize import minimize

from environment.custom_env import (
    UltrasoundProbeEnv, ACTIONS, ProbeState, compute_potential,
    POTENTIAL_ALPHA_WEIGHT, POTENTIAL_ALPHA_SCALE,
    POTENTIAL_D_WEIGHT, POTENTIAL_D_SCALE,
    COARSE_ARC_DEG, COARSE_ANGLE_DEG,
)

FREEZE_IDX = ACTIONS.index("freeze_and_measure")
MOVE_ACTIONS = [a for a in ACTIONS if a != "freeze_and_measure"]
TARGET = "femur"
N_PHANTOMS_CHECK1 = 20
N_EPISODES_CHECK2 = 100
GREEDY_MAX_STEPS = 200


# ---------------------------------------------------------------------
# Reuses the exact search from UltrasoundProbeEnv._search_min_pose_error
# (position-first + tilt-polish, plus blind joint restarts), but does NOT
# restore env.probe afterward -- so the caller can teleport to (and use)
# the found pose, which the original method (by design, for reset()'s
# read-only reachability CHECK) intentionally does not expose.
# ---------------------------------------------------------------------
def find_best_pose(env: UltrasoundProbeEnv, target: str, restarts: int = 3, seed: int = 20260726):
    bounds = env._actuator_pose_bounds()
    pos_bounds, tilt_bounds = bounds[:2], bounds[2:]
    rng = np.random.default_rng(seed)

    def objective(params):
        env.probe.theta, env.probe.phi, env.probe.roll, env.probe.pitch, env.probe.yaw = params
        alpha, d = env._pose_error(target)
        return alpha / env.alpha_tol + d / max(env.d_tol, 1e-9)

    def position_only_objective(pos_params):
        env.probe.theta, env.probe.phi = pos_params
        env.probe.roll = env.probe.pitch = env.probe.yaw = 0.0
        _, d = env._pose_error(target)
        return d

    def tilt_only_objective(rpy_params, theta_fixed, phi_fixed):
        env.probe.theta, env.probe.phi = theta_fixed, phi_fixed
        env.probe.roll, env.probe.pitch, env.probe.yaw = rpy_params
        alpha, d = env._pose_error(target)
        return alpha / env.alpha_tol + d / max(env.d_tol, 1e-9)

    best_x, best_val = None, np.inf
    for _ in range(restarts):
        x0 = np.array([rng.uniform(lo, hi) for lo, hi in bounds])
        res = minimize(objective, x0, method="Nelder-Mead", bounds=bounds,
                        options=dict(xatol=1e-3, fatol=1e-3, maxiter=200, maxfev=200))
        if res.fun < best_val:
            best_val, best_x = res.fun, res.x
    for _ in range(restarts):
        pos_x0 = np.array([rng.uniform(lo, hi) for lo, hi in pos_bounds])
        pos_res = minimize(position_only_objective, pos_x0, method="Nelder-Mead", bounds=pos_bounds,
                            options=dict(xatol=1e-4, fatol=1e-6, maxiter=200, maxfev=200))
        theta_c, phi_c = pos_res.x
        tilt_x0 = np.array([rng.uniform(lo, hi) for lo, hi in tilt_bounds])
        tilt_res = minimize(tilt_only_objective, tilt_x0, method="Nelder-Mead", bounds=tilt_bounds,
                             args=(theta_c, phi_c),
                             options=dict(xatol=1e-3, fatol=1e-3, maxiter=200, maxfev=200))
        candidate_x = np.array([theta_c, phi_c, *tilt_res.x])
        val = objective(candidate_x)
        if val < best_val:
            best_val, best_x = val, candidate_x

    env.probe.theta, env.probe.phi, env.probe.roll, env.probe.pitch, env.probe.yaw = best_x
    alpha, d = env._pose_error(target)
    return alpha, d


def find_min_d_position(env: UltrasoundProbeEnv, target: str, restarts: int = 3, seed: int = 20260726):
    """Position-only optimum (roll=pitch=yaw=0): the (theta,phi) that
    minimizes d alone, ignoring alpha entirely. Used to quantify the
    alpha-d coupling in Check 3: what alpha results at the position that
    minimizes d, with no tilt correction applied?"""
    bounds = env._actuator_pose_bounds()
    pos_bounds = bounds[:2]
    rng = np.random.default_rng(seed)

    def position_only_objective(pos_params):
        env.probe.theta, env.probe.phi = pos_params
        env.probe.roll = env.probe.pitch = env.probe.yaw = 0.0
        _, d = env._pose_error(target)
        return d

    best_x, best_val = None, np.inf
    for _ in range(restarts):
        x0 = np.array([rng.uniform(lo, hi) for lo, hi in pos_bounds])
        res = minimize(position_only_objective, x0, method="Nelder-Mead", bounds=pos_bounds,
                        options=dict(xatol=1e-4, fatol=1e-6, maxiter=200, maxfev=200))
        if res.fun < best_val:
            best_val, best_x = res.fun, res.x

    env.probe.theta, env.probe.phi = best_x
    env.probe.roll = env.probe.pitch = env.probe.yaw = 0.0
    alpha_raw, d_min = env._pose_error(target)
    return best_x[0], best_x[1], alpha_raw, d_min


def simulate_action(probe: ProbeState, action_name: str, actuator_limit_deg: float,
                     tilt_step_deg: float = COARSE_ANGLE_DEG) -> ProbeState:
    """Exact duplicate of step()'s movement math (clip logic included),
    applied to a COPY of probe state -- used by the greedy controller to
    evaluate candidate actions without mutating the live env.
    tilt_step_deg: mirrors env.tilt_step_deg (defaults to the original
    COARSE_ANGLE_DEG=3deg for backward-compatible standalone use) -- pass
    the LIVE env's own tilt_step_deg so a finer-step experiment is actually
    exercised by this simulation, not silently ignored."""
    p = ProbeState(probe.theta, probe.phi, probe.roll, probe.pitch, probe.yaw, probe.fine_mode)
    arc_deg = COARSE_ARC_DEG / 2.0 if p.fine_mode else COARSE_ARC_DEG
    ang_deg = tilt_step_deg / 2.0 if p.fine_mode else tilt_step_deg
    arc = np.radians(arc_deg)
    if action_name == "theta_plus":
        p.theta = float(np.clip(p.theta + arc, 0.01, np.pi / 2 - 0.01))
    elif action_name == "theta_minus":
        p.theta = float(np.clip(p.theta - arc, 0.01, np.pi / 2 - 0.01))
    elif action_name == "phi_plus":
        p.phi = float((p.phi + arc) % (2 * np.pi))
    elif action_name == "phi_minus":
        p.phi = float((p.phi - arc) % (2 * np.pi))
    elif action_name == "roll_plus":
        p.roll = float(np.clip(p.roll + ang_deg, -actuator_limit_deg, actuator_limit_deg))
    elif action_name == "roll_minus":
        p.roll = float(np.clip(p.roll - ang_deg, -actuator_limit_deg, actuator_limit_deg))
    elif action_name == "pitch_plus":
        p.pitch = float(np.clip(p.pitch + ang_deg, -actuator_limit_deg, actuator_limit_deg))
    elif action_name == "pitch_minus":
        p.pitch = float(np.clip(p.pitch - ang_deg, -actuator_limit_deg, actuator_limit_deg))
    elif action_name == "yaw_plus":
        p.yaw = float(np.clip(p.yaw + ang_deg, -actuator_limit_deg, actuator_limit_deg))
    elif action_name == "yaw_minus":
        p.yaw = float(np.clip(p.yaw - ang_deg, -actuator_limit_deg, actuator_limit_deg))
    elif action_name == "toggle_fine":
        p.fine_mode = not p.fine_mode
    return p


def greedy_episode(env: UltrasoundProbeEnv, fine_near_target: bool = False, fine_trigger_mult: float = 3.0,
                    max_steps: int | None = None):
    """Greedy one-step-lookahead controller. At each step, evaluates all 11
    non-freeze actions by SIMULATING them (via simulate_action + the env's
    real _pose_error), picks whichever most reduces the normalized combined
    error alpha/alpha_tol + d/d_tol, and issues freeze once within
    tolerance. If fine_near_target, also greedily considers toggling fine
    mode once combined error drops under fine_trigger_mult (Check 4).
    max_steps: outer step cap for THIS controller's own loop (independent
    of env.subtask_max_steps/EPISODE_MAX_STEPS, which the env enforces on
    its own and will terminate/truncate the episode before this fires in
    the normal case). Defaults to GREEDY_MAX_STEPS (200) for the original
    Check 2/4 callers; scripts probing LARGER env budgets (e.g.
    scripts/budget_headroom_gate.py testing subtask_max_steps=250/400)
    must pass a max_steps comfortably above the budget under test, or this
    default 200 would silently truncate the walk before the env's own,
    larger budget ever gets a chance to fire."""
    if max_steps is None:
        max_steps = GREEDY_MAX_STEPS
    obs, info = env.reset()
    target = env.targets[env.target_idx]
    steps = 0
    engaged_fine = False

    while True:
        alpha, d = env._pose_error(target)
        combined = alpha / env.alpha_tol + d / env.d_tol
        if alpha <= env.alpha_tol and d <= env.d_tol:
            obs, reward, terminated, truncated, info = env.step(FREEZE_IDX)
            return dict(success=info["success"], steps=steps, terminal_d_mm=d * 1000,
                        terminal_alpha_deg=np.degrees(alpha), reward=reward, outcome="froze_in_tol")

        candidates = list(MOVE_ACTIONS)
        if fine_near_target and not env.probe.fine_mode and combined <= fine_trigger_mult:
            engaged_fine = True

        saved = (env.probe.theta, env.probe.phi, env.probe.roll, env.probe.pitch,
                 env.probe.yaw, env.probe.fine_mode)
        best_action, best_val = None, np.inf
        for name in candidates:
            if name == "toggle_fine" and not (fine_near_target and engaged_fine and not env.probe.fine_mode):
                continue  # only ever consider toggle_fine when Check 4 explicitly engages it
            p = simulate_action(env.probe, name, env.actuator_limit_deg, env.tilt_step_deg)
            env.probe.theta, env.probe.phi = p.theta, p.phi
            env.probe.roll, env.probe.pitch, env.probe.yaw = p.roll, p.pitch, p.yaw
            env.probe.fine_mode = p.fine_mode
            a2, d2 = env._pose_error(target)
            val = a2 / env.alpha_tol + d2 / env.d_tol
            env.probe.theta, env.probe.phi, env.probe.roll, env.probe.pitch, env.probe.yaw, env.probe.fine_mode = saved
            if val < best_val:
                best_val, best_action = val, name

        if fine_near_target and engaged_fine and not env.probe.fine_mode:
            best_action = "toggle_fine"  # force the fine-mode engagement once triggered

        obs, reward, terminated, truncated, info = env.step(ACTIONS.index(best_action))
        steps += 1
        if terminated or truncated or steps >= max_steps:
            alpha, d = env._pose_error(target)
            outcome = "timeout_subtask" if terminated else ("timeout_episode" if truncated else "max_steps_cap")
            return dict(success=info["success"], steps=steps, terminal_d_mm=d * 1000,
                        terminal_alpha_deg=np.degrees(alpha), reward=info["reward"], outcome=outcome)


def check1(n=N_PHANTOMS_CHECK1):
    print(f"\n=== CHECK 1: does the success path work? (N={n} phantoms, femur) ===")
    env = UltrasoundProbeEnv(single_target=True, single_target_which=TARGET, seed=1)
    results = []
    for i in range(n):
        env.reset(seed=1000 + i)
        target = env.targets[env.target_idx]
        alpha, d = find_best_pose(env, target)
        within_tol = alpha <= env.alpha_tol and d <= env.d_tol
        obs, reward, terminated, truncated, info = env.step(FREEZE_IDX)
        results.append(dict(i=i, alpha_deg=np.degrees(alpha), d_mm=d * 1000,
                             within_tol=within_tol, success=info["success"], reward=reward))
    n_success = sum(r["success"] for r in results)
    n_within_tol = sum(r["within_tol"] for r in results)
    print(f"  optimizer found within-tolerance pose: {n_within_tol}/{n}")
    print(f"  freeze at that pose registered success: {n_success}/{n}")
    for r in results[:5]:
        print(f"    phantom {r['i']}: alpha={r['alpha_deg']:.2f}deg d={r['d_mm']:.2f}mm "
              f"within_tol={r['within_tol']} success={r['success']} reward={r['reward']:.2f}")
    bugged = n_within_tol > 0 and n_success < n_within_tol
    print(f"  VERDICT: {'BUGGED -- freeze at a known-good pose does NOT register success' if bugged else 'success path OK'}")
    return dict(n=n, n_within_tol=n_within_tol, n_success=n_success, bugged=bugged, results=results)


def check2(n=N_EPISODES_CHECK2):
    print(f"\n=== CHECK 2: can a greedy scripted controller reach d<12mm? (N={n} episodes, femur) ===")
    env = UltrasoundProbeEnv(single_target=True, single_target_which=TARGET, seed=2)
    results = [greedy_episode(env) for _ in range(n)]
    success_rate = np.mean([r["success"] for r in results])
    terminal_d = np.array([r["terminal_d_mm"] for r in results])
    terminal_alpha = np.array([r["terminal_alpha_deg"] for r in results])
    steps_success = [r["steps"] for r in results if r["success"]]
    outcomes = {}
    for r in results:
        outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1

    print(f"  success rate: {success_rate:.4f} ({sum(r['success'] for r in results)}/{n})")
    if steps_success:
        print(f"  median steps-to-success (successes only): {np.median(steps_success):.1f}")
    else:
        print(f"  median steps-to-success: N/A (0 successes)")
    print(f"  terminal d (mm): median={np.median(terminal_d):.2f} mean={np.mean(terminal_d):.2f} "
          f"min={np.min(terminal_d):.2f} max={np.max(terminal_d):.2f} "
          f"p10={np.percentile(terminal_d,10):.2f} p90={np.percentile(terminal_d,90):.2f}")
    print(f"  terminal alpha (deg): median={np.median(terminal_alpha):.2f} mean={np.mean(terminal_alpha):.2f}")
    print(f"  outcome breakdown: {outcomes}")

    frac_under_tol = np.mean(terminal_d <= 12.0)
    print(f"  fraction of episodes with terminal d<=12mm at ANY point they froze/ended: {frac_under_tol:.3f}")

    if success_rate >= 0.5:
        verdict = "SOLVABLE (greedy succeeds reliably) -> LEARNING/REWARD problem"
    elif success_rate > 0.0:
        verdict = "PARTIALLY SOLVABLE (greedy sometimes succeeds) -> ambiguous, lean LEARNING/REWARD"
    else:
        verdict = "NOT SOLVED BY GREEDY -> investigate SETUP (granularity vs coupling) in Checks 3/4"
    print(f"  VERDICT: {verdict}")
    return dict(n=n, success_rate=success_rate, terminal_d=terminal_d, terminal_alpha=terminal_alpha,
                outcomes=outcomes, results=results)


def check3(n_phantoms=5):
    print(f"\n=== CHECK 3: reward-surface readout + alpha-d coupling (N={n_phantoms} phantoms, femur) ===")
    env = UltrasoundProbeEnv(single_target=True, single_target_which=TARGET, seed=3)

    # -- shaping terms at the observed settling point vs the optimum --
    alpha_settle = np.radians(0.3)
    d_settle = 0.035  # 35mm, this experiment's observed median terminal d
    alpha_opt, d_opt = 0.0, 0.0

    alpha_term_settle = POTENTIAL_ALPHA_WEIGHT * np.exp(-alpha_settle / POTENTIAL_ALPHA_SCALE)
    d_term_settle = POTENTIAL_D_WEIGHT * np.exp(-d_settle / POTENTIAL_D_SCALE)
    alpha_term_opt = POTENTIAL_ALPHA_WEIGHT * np.exp(-alpha_opt / POTENTIAL_ALPHA_SCALE)
    d_term_opt = POTENTIAL_D_WEIGHT * np.exp(-d_opt / POTENTIAL_D_SCALE)

    print(f"  at settling point (alpha=0.30deg, d=35mm):")
    print(f"    alpha-term = {alpha_term_settle:.4f} (max={alpha_term_opt:.4f}, "
          f"{100*alpha_term_settle/alpha_term_opt:.1f}% of max already banked)")
    print(f"    d-term     = {d_term_settle:.4f} (max={d_term_opt:.4f}, "
          f"{100*d_term_settle/d_term_opt:.1f}% of max already banked)")
    total_settle = alpha_term_settle + d_term_settle
    total_opt = alpha_term_opt + d_term_opt
    print(f"    total shaping (v=0 term omitted) = {total_settle:.4f} / {total_opt:.4f} = "
          f"{100*total_settle/total_opt:.1f}% of max already banked at the settling point")

    # -- coupling: at the position that minimizes d alone (no tilt), what alpha results? --
    rows = []
    for i in range(n_phantoms):
        env.reset(seed=2000 + i)
        target = env.targets[env.target_idx]
        theta_c, phi_c, alpha_raw, d_min = find_min_d_position(env, target)
        # now tilt-polish AT THAT SAME FIXED POSITION to see how much of alpha_raw can be corrected
        alpha_full, d_full = find_best_pose(env, target)  # full joint optimum for reference
        env.probe.theta, env.probe.phi = theta_c, phi_c
        bounds = env._actuator_pose_bounds()
        tilt_bounds = bounds[2:]
        rng = np.random.default_rng(30000 + i)

        def tilt_only_objective(rpy_params):
            env.probe.roll, env.probe.pitch, env.probe.yaw = rpy_params
            a, _ = env._pose_error(target)
            return a
        best_alpha_here = np.inf
        for _ in range(3):
            x0 = np.array([rng.uniform(lo, hi) for lo, hi in tilt_bounds])
            res = minimize(tilt_only_objective, x0, method="Nelder-Mead", bounds=tilt_bounds,
                            options=dict(xatol=1e-3, fatol=1e-3, maxiter=200, maxfev=200))
            best_alpha_here = min(best_alpha_here, res.fun)

        rows.append(dict(i=i, d_min_mm=d_min * 1000, alpha_raw_deg=np.degrees(alpha_raw),
                          alpha_corrected_deg=np.degrees(best_alpha_here),
                          joint_opt_alpha_deg=np.degrees(alpha_full), joint_opt_d_mm=d_full * 1000))
        print(f"  phantom {i}: at min-d position (d={d_min*1000:.2f}mm) -- "
              f"alpha with NO tilt correction={np.degrees(alpha_raw):.2f}deg, "
              f"alpha with tilt-polish AT THAT SAME POSITION={np.degrees(best_alpha_here):.2f}deg "
              f"(joint optimum: alpha={np.degrees(alpha_full):.2f}deg, d={d_full*1000:.2f}mm)")

    mean_raw = np.mean([r["alpha_raw_deg"] for r in rows])
    mean_corrected = np.mean([r["alpha_corrected_deg"] for r in rows])
    print(f"  mean alpha with no tilt correction at min-d position: {mean_raw:.2f}deg")
    print(f"  mean alpha AFTER tilt-polish at that SAME position: {mean_corrected:.2f}deg")
    print(f"  -> coupling exists at the raw (theta,phi)-only level (large alpha with roll=pitch=yaw=0), "
          f"but is {'FULLY' if mean_corrected <= np.degrees(env.alpha_tol) else 'NOT fully'} "
          f"correctable by tilt alone at that same fixed position.")
    return dict(alpha_term_settle=alpha_term_settle, d_term_settle=d_term_settle,
                pct_banked=100*total_settle/total_opt, rows=rows,
                mean_raw_alpha_deg=mean_raw, mean_corrected_alpha_deg=mean_corrected)


def check4(n=N_EPISODES_CHECK2, fine_trigger_mult=3.0, label=""):
    print(f"\n=== CHECK 4{label}: does toggle_fine close the gap? "
          f"(N={n} episodes, femur, greedy+fine, trigger_mult={fine_trigger_mult}) ===")
    env = UltrasoundProbeEnv(single_target=True, single_target_which=TARGET, seed=4)
    results = [greedy_episode(env, fine_near_target=True, fine_trigger_mult=fine_trigger_mult) for _ in range(n)]
    success_rate = np.mean([r["success"] for r in results])
    terminal_d = np.array([r["terminal_d_mm"] for r in results])
    terminal_alpha = np.array([r["terminal_alpha_deg"] for r in results])
    steps_success = [r["steps"] for r in results if r["success"]]
    outcomes = {}
    for r in results:
        outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1
    print(f"  success rate WITH fine-mode engagement near target: {success_rate:.4f} "
          f"({sum(r['success'] for r in results)}/{n})")
    if steps_success:
        print(f"  median steps-to-success (successes only): {np.median(steps_success):.1f}")
    print(f"  terminal d (mm): median={np.median(terminal_d):.2f} mean={np.mean(terminal_d):.2f}")
    print(f"  terminal alpha (deg): median={np.median(terminal_alpha):.2f} mean={np.mean(terminal_alpha):.2f}")
    print(f"  outcome breakdown: {outcomes}")
    return dict(n=n, success_rate=success_rate, terminal_d=terminal_d, terminal_alpha=terminal_alpha,
                outcomes=outcomes, results=results)


def main():
    r1 = check1()
    r2 = check2()
    r3 = check3()
    r4 = None
    if r2["success_rate"] < 0.5:
        r4 = check4(fine_trigger_mult=3.0, label=" (a)")
        r4b = check4(fine_trigger_mult=1.3, label=" (b, tighter trigger)")
    else:
        print("\nCheck 2 already shows greedy succeeding reliably -- skipping Check 4 (only run if SETUP suspected).")

    print("\n\n=== LOCALIZATION VERDICT ===")
    print(f"1. Success path: {'BUGGED' if r1['bugged'] else 'WORKS'} "
          f"({r1['n_success']}/{r1['n_within_tol']} freezes-at-known-good-pose registered success)")
    print(f"2. Solvability (greedy, real actions): success_rate={r2['success_rate']:.4f}, "
          f"median terminal d={np.median(r2['terminal_d']):.2f}mm")
    if r4 is not None:
        print(f"   with fine-mode engagement: success_rate={r4['success_rate']:.4f}, "
              f"median terminal d={np.median(r4['terminal_d']):.2f}mm")
    print(f"3. Settling-point shaping already banked: {r3['pct_banked']:.1f}% of max")
    print(f"   coupling: no-tilt alpha at min-d position={r3['mean_raw_alpha_deg']:.2f}deg -> "
          f"tilt-corrected={r3['mean_corrected_alpha_deg']:.2f}deg")


if __name__ == "__main__":
    main()
