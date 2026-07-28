"""Phase-1 correctness gate: validate the reward/potential field.

Two independent layers, because they test different things:

Layer A -- the potential FORMULA in isolation (`compute_potential`), evaluated
over synthetic (alpha, d, V) grids with no environment/phantom involved at
all. This catches a sign-flip typo in the formula itself.

Layer B -- the pose-error GEOMETRY (`UltrasoundProbeEnv._pose_error`,
`_build_plane_targets`), evaluated through the real env code paths:
  1. Endpoint check: can a bounded optimizer (within the actuator's real
     +-60deg roll/pitch/yaw range) drive alpha and d to ~0 for each target?
  2. Plane-symmetry check: is alpha invariant to a 180-degree flip of the
     probe's elevational axis? (catches a missing/misapplied abs())
  3a. Raw control-space-lerp diagnostic (informational, NOT a gate): a
      straight-line interpolation in (theta, phi, roll, pitch, yaw) between
      a random start and the near-optimal end pose. This is reported but
      not gated on -- see the "IMPORTANT FINDING" note below.
  3b. Greedy discrete-action walk (the REAL gate): from random start poses,
      repeatedly take whichever of the actual 12 discrete actions the agent
      can choose most increases the potential (greedy hill-climb), and
      check that this reliably reaches the acquisition tolerance. This is
      what actually predicts RL learnability, since it uses the exact
      action set and step sizes the agent has.

IMPORTANT FINDING (documented, not "fixed" -- see status.md Phase 1 report):
Layer 3a (raw Euler-angle-space lerp) does NOT reach ~90% monotonicity even
after the axis bug below was fixed, and this is expected, not a residual
bug: `roll`/`pitch`/`yaw` compose as three extrinsic-axis rotations (a
Tait-Bryan/Euler-angle-style parameterization), and straight-line
interpolation in Euler-angle space is well known to NOT correspond to a
geodesic (SLERP) path in SO(3) -- the angular readout `alpha` (a nonlinear
function of the composed rotation) can wobble non-monotonically along such
a path even when the underlying geometry is completely correct. Because the
RL agent never "teleports" along such a straight line -- it only ever takes
the real, small, discrete actions -- Layer 3b (greedy discrete-action walk)
is the test that actually matters for learnability, and it is what this
script gates on.

BUG FOUND AND FIXED: `_pose_error` originally computed
`probe_normal = cross(right, up)`, which is algebraically equal to
`-forward` (the probe's depth/imaging axis, which lies IN the 2D image
plane) rather than the correct elevational axis `up` (perpendicular to the
image plane, which is what should be compared to the target anatomical
plane's normal). This was caught by a collapse in Layer 3a's monotonic
fraction (12-38%) and fixed in `environment/custom_env.py::_pose_error` by
using `up` directly. See status.md for the pre-fix/post-fix numbers.

Usage: uv run python scripts/validate_reward_field.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from environment.custom_env import (
    UltrasoundProbeEnv, compute_potential, TARGET_SEQUENCE, ACTIONS,
    COARSE_ARC_DEG, COARSE_ANGLE_DEG, MAX_OFFSET_DEG,
)

OUT_DIR = Path(__file__).resolve().parent.parent / "logs" / "reward_validation"

K_PHANTOMS = 8
M_APPROACH_PATHS = 18
# Number of distinct phantoms the greedy-walk diagnostic averages over
# (M_APPROACH_PATHS is split across these). A single phantom made this
# diagnostic fragile to per-phantom sampling variance -- e.g. one phantom
# where the target sits geometrically far from most random starts shows
# near-zero d-improvement purely from the exp(-d/0.015) shaping term's
# long-range flatness, not from a geometry bug. Found when re-validating
# after adding guarantee_reachable (see status.md): the fixed seed=0
# phantom changed (reachability resampling consumes RNG differently), and
# the new one happened to be such a case for "head".
#
# M_APPROACH_PATHS bumped from 6 to 12 after the reachability-search audit
# fix (environment/custom_env.py::_search_min_pose_error): fixing that
# search changed reset()'s resample behavior (correctly accepting phantoms
# on the first try instead of wrongly rejecting them), which shifted which
# phantoms land on seeds 9000-9002 and exposed the SAME small-N fragility
# again (femur dropped to 3/6 paths improved = 0.50, just under the 0.6
# gate, while its improvement ratio stayed healthy at 0.253) -- doubling
# the path count trades a few extra minutes of runtime for a materially
# more stable statistic, rather than repeatedly re-chasing individual
# phantom draws every time upstream sampling code is (correctly) fixed.
K_GREEDY_PHANTOMS = 3
APPROACH_STEPS = 20
MONOTONIC_TOL = 1e-6            # allow this much numerical slack before counting a violation
MONOTONIC_VIOLATION_BUDGET = 2  # allow up to this many minor violations per path before failing it
ENDPOINT_ALPHA_TOL_DEG = 2.0
ENDPOINT_D_TOL_M = 0.003

# GREEDY_WALK_STEPS matches the real SUBTASK_MAX_STEPS (60) -- if greedy
# hill-climbing can't reach tolerance within the budget the agent actually
# gets, that's exactly the learnability signal we want, not an artificially
# generous one. Cost is the dominant runtime factor here (each step tries
# all MOVE_ACTIONS, each requiring one slice render), so keep this and
# M_APPROACH_PATHS modest -- this script must stay fast enough to run as a
# pytest gate on every change to the reward geometry.
GREEDY_WALK_STEPS = 60
GREEDY_SUCCESS_ALPHA_TOL_DEG = 20.0  # slightly looser than the default 15deg acquisition tol
GREEDY_SUCCESS_D_TOL_M = 0.018      # slightly looser than the default 12mm acquisition tol
GREEDY_GATE_MIN_SUCCESS_FRACTION = 0.7

# Movement-only actions (exclude toggle_fine / freeze_and_measure, which
# don't move the probe and aren't relevant to "can greedy hill-climbing
# find the target").
MOVE_ACTIONS = [i for i, name in enumerate(ACTIONS) if name not in ("toggle_fine", "freeze_and_measure")]

# Bounds match the REAL actuator limits (MAX_OFFSET_DEG in custom_env.py):
# roll/pitch/yaw in [-60, 60]. We deliberately search/interpolate within the
# physically reachable state space, not an unconstrained one -- an
# "optimum" outside the actuator's range is operationally irrelevant to the
# agent and, if used as an interpolation endpoint, produces misleading
# aliased-angle artifacts (e.g. roll=743 deg, geometrically valid but never
# reachable by any action sequence).
POSE_BOUNDS = [(0.02, np.pi / 2 - 0.02), (0.0, 2 * np.pi),
               (-MAX_OFFSET_DEG, MAX_OFFSET_DEG), (-MAX_OFFSET_DEG, MAX_OFFSET_DEG),
               (-MAX_OFFSET_DEG, MAX_OFFSET_DEG)]

# The greedy-walk diagnostic's "random start" should mirror the REAL
# distribution UltrasoundProbeEnv.reset() actually samples: theta in
# [10,50]deg, phi in [0,2pi], roll=pitch=yaw=0 (ProbeState's defaults --
# only theta/phi are randomized at episode start; see
# custom_env.py::reset()). Using the full actuator range (POSE_BOUNDS,
# appropriate for the REACHABILITY search, which needs to know if a target
# is achievable from ANYWHERE) here instead tests a much wider, less
# realistic starting distribution than training ever encounters. This was
# found to be the actual cause of a spuriously low "head" improvement
# ratio after guarantee_reachable was added (see status.md Phase/fix
# history): phantoms where head happens to be reachable tend to need a
# probe position geometrically far from POSE_BOUNDS's full-range random
# draws, triggering the exp(-d/0.015) shaping term's long-range flatness --
# an artifact of testing an unrealistic start, not a geometry bug.
GREEDY_START_BOUNDS = [
    (np.radians(10.0), np.radians(50.0)), (0.0, 2 * np.pi),
    (0.0, 0.0), (0.0, 0.0), (0.0, 0.0),
]


# ---------------------------------------------------------------------------
# Layer A
# ---------------------------------------------------------------------------
def layer_a_formula_monotonicity(n_grid: int = 25) -> dict:
    alphas = np.linspace(0.0, np.pi, n_grid)
    ds = np.linspace(0.0, 0.05, n_grid)
    vs = np.linspace(0.0, 1.0, n_grid)

    phi_alpha = np.array([compute_potential(0.5, a, 0.01) for a in alphas])
    decreasing_in_alpha = bool(np.all(np.diff(phi_alpha) <= 1e-12))

    phi_d = np.array([compute_potential(0.5, 0.2, d) for d in ds])
    decreasing_in_d = bool(np.all(np.diff(phi_d) <= 1e-12))

    phi_v = np.array([compute_potential(v, 0.2, 0.01) for v in vs])
    increasing_in_v = bool(np.all(np.diff(phi_v) >= -1e-12))

    return dict(
        decreasing_in_alpha=decreasing_in_alpha,
        decreasing_in_d=decreasing_in_d,
        increasing_in_v=increasing_in_v,
        passed=decreasing_in_alpha and decreasing_in_d and increasing_in_v,
    )


# ---------------------------------------------------------------------------
# Layer B helpers
# ---------------------------------------------------------------------------
def _force_target(env: UltrasoundProbeEnv, target: str):
    """`env._potential()` reads its target from `env._cache["target"]`,
    which is derived from `env.target_idx`/`env.targets` -- NOT from
    whatever `target` string a caller happens to be validating. After a
    fresh `env.reset()`, `target_idx` is always 0, i.e. "head". Any helper
    here that calls `env._potential(...)` for a specific `target` MUST call
    this first, or it will silently optimize/measure the WRONG target's
    potential while reporting alpha/d for the intended one (this was a real
    bug in an earlier version of this script -- see status.md Phase 1: it
    made the femur greedy-walk numbers look like a field bug when the
    field was fine and the harness was pointed at the wrong target).
    `_pose_error(target)` is unaffected -- it takes `target` as an explicit
    argument and never touches `env.target_idx`."""
    env.targets = [target]
    env.target_idx = 0


def _set_pose(env: UltrasoundProbeEnv, params: np.ndarray):
    theta, phi, roll, pitch, yaw = params
    env.probe.theta = float(np.clip(theta, 0.01, np.pi / 2 - 0.01))
    env.probe.phi = float(phi % (2 * np.pi))
    env.probe.roll = float(roll)
    env.probe.pitch = float(pitch)
    env.probe.yaw = float(yaw)


def _pose_vector(env: UltrasoundProbeEnv) -> np.ndarray:
    p = env.probe
    return np.array([p.theta, p.phi, p.roll, p.pitch, p.yaw])


def find_optimal_pose(env: UltrasoundProbeEnv, target: str, n_restarts: int = 6, rng=None):
    """Search (theta, phi, roll, pitch, yaw) minimizing alpha + d/0.01 for
    `target`, bounded to the actuator's reachable range (see POSE_BOUNDS)."""
    rng = rng or np.random.default_rng(0)

    def objective(params):
        _set_pose(env, params)
        alpha, d = env._pose_error(target)
        return alpha + d / 0.01  # d in meters; put on a comparable scale to alpha in radians

    best = None
    for _ in range(n_restarts):
        x0 = np.array([rng.uniform(lo, hi) for lo, hi in POSE_BOUNDS])
        res = minimize(objective, x0, method="Nelder-Mead", bounds=POSE_BOUNDS,
                        options=dict(xatol=1e-4, fatol=1e-6, maxiter=2000))
        if best is None or res.fun < best.fun:
            best = res
    _set_pose(env, best.x)
    alpha, d = env._pose_error(target)
    return best.x.copy(), alpha, d


def _shortest_phi_delta(start_phi: float, end_phi: float) -> float:
    """Shortest signed angular delta on the phi circle (period 2*pi)."""
    return (end_phi - start_phi + np.pi) % (2 * np.pi) - np.pi


def layer_b_endpoint_and_symmetry(env: UltrasoundProbeEnv, target: str) -> dict:
    end_params, alpha_end, d_end = find_optimal_pose(env, target)
    endpoint_ok = (np.degrees(alpha_end) <= ENDPOINT_ALPHA_TOL_DEG) and (d_end <= ENDPOINT_D_TOL_M)

    _set_pose(env, end_params)
    alpha_before, d_before = env._pose_error(target)
    flipped = end_params.copy()
    flipped[2] += 180.0  # roll, deliberately unconstrained -- pure math symmetry check
    _set_pose(env, flipped)
    alpha_after, d_after = env._pose_error(target)
    symmetry_ok = bool(np.isclose(alpha_before, alpha_after, atol=1e-6))

    return dict(
        target=target, end_params=end_params.tolist(),
        alpha_end_deg=float(np.degrees(alpha_end)), d_end_m=float(d_end),
        endpoint_ok=bool(endpoint_ok),
        alpha_before_deg=float(np.degrees(alpha_before)), alpha_after_deg=float(np.degrees(alpha_after)),
        symmetry_ok=symmetry_ok,
    )


def layer_b_lerp_diagnostic(env: UltrasoundProbeEnv, target: str, end_params: np.ndarray,
                             rng: np.random.Generator) -> dict:
    """INFORMATIONAL ONLY -- not gated on. See module docstring for why raw
    Euler-angle-space lerp is not expected to be monotonic even for
    perfectly correct geometry."""
    _force_target(env, target)
    violations = []
    monotonic_count = 0
    traces = []
    for path_i in range(M_APPROACH_PATHS):
        start_params = np.array([rng.uniform(lo, hi) for lo, hi in POSE_BOUNDS])
        phi_delta = _shortest_phi_delta(start_params[1], end_params[1])
        alphas, ds, phis = [], [], []
        for step in range(APPROACH_STEPS + 1):
            t = step / APPROACH_STEPS
            params = (1 - t) * start_params + t * end_params
            params[1] = start_params[1] + t * phi_delta
            _set_pose(env, params)
            obs = env._compute_obs()
            alpha, d = env._pose_error(target)
            phi = env._potential(obs_cache=env._cache)
            alphas.append(alpha)
            ds.append(d)
            phis.append(phi)

        alphas, ds, phis = np.array(alphas), np.array(ds), np.array(phis)
        n_viol = 0
        for arr, name in [(alphas, "alpha"), (ds, "d")]:
            diffs = np.diff(arr)
            bad_steps = np.where(diffs > MONOTONIC_TOL)[0]
            for bs in bad_steps:
                n_viol += 1
                violations.append(dict(
                    target=target, path=path_i, quantity=name, step=int(bs),
                    magnitude=float(diffs[bs]),
                    pose_before=((1 - bs / APPROACH_STEPS) * start_params
                                 + (bs / APPROACH_STEPS) * end_params),
                ))
        if n_viol <= MONOTONIC_VIOLATION_BUDGET:
            monotonic_count += 1
        traces.append(dict(alphas=alphas, ds=ds, phis=phis))

    return dict(
        target=target,
        monotonic_fraction=monotonic_count / M_APPROACH_PATHS,
        n_violations=len(violations),
        violations=violations,
        traces=traces,
    )


def _apply_action(env: UltrasoundProbeEnv, action_idx: int):
    """Apply one discrete action's pose delta directly (mirrors
    UltrasoundProbeEnv.step's movement logic, without reward/termination
    side effects) so the greedy walk uses the agent's REAL action set."""
    name = ACTIONS[action_idx]
    arc = np.radians(COARSE_ARC_DEG)
    ang = COARSE_ANGLE_DEG
    p = env.probe
    if name == "theta_plus":
        p.theta = float(np.clip(p.theta + arc, 0.01, np.pi / 2 - 0.01))
    elif name == "theta_minus":
        p.theta = float(np.clip(p.theta - arc, 0.01, np.pi / 2 - 0.01))
    elif name == "phi_plus":
        p.phi = float((p.phi + arc) % (2 * np.pi))
    elif name == "phi_minus":
        p.phi = float((p.phi - arc) % (2 * np.pi))
    elif name == "roll_plus":
        p.roll = float(np.clip(p.roll + ang, -MAX_OFFSET_DEG, MAX_OFFSET_DEG))
    elif name == "roll_minus":
        p.roll = float(np.clip(p.roll - ang, -MAX_OFFSET_DEG, MAX_OFFSET_DEG))
    elif name == "pitch_plus":
        p.pitch = float(np.clip(p.pitch + ang, -MAX_OFFSET_DEG, MAX_OFFSET_DEG))
    elif name == "pitch_minus":
        p.pitch = float(np.clip(p.pitch - ang, -MAX_OFFSET_DEG, MAX_OFFSET_DEG))
    elif name == "yaw_plus":
        p.yaw = float(np.clip(p.yaw + ang, -MAX_OFFSET_DEG, MAX_OFFSET_DEG))
    elif name == "yaw_minus":
        p.yaw = float(np.clip(p.yaw - ang, -MAX_OFFSET_DEG, MAX_OFFSET_DEG))


def layer_b_greedy_discrete_walk(env: UltrasoundProbeEnv, target: str,
                                  rng: np.random.Generator,
                                  phantom_seeds: list[int] | None = None) -> dict:
    """THE REAL GATE. From random start poses, greedily take whichever real
    discrete action most increases the potential, for up to
    GREEDY_WALK_STEPS steps. Reports the fraction of walks that reach the
    acquisition tolerance -- this is what predicts whether an RL agent
    (which likewise only ever has local, small, discrete actions available)
    has a usable gradient to climb.

    `phantom_seeds`: spreads M_APPROACH_PATHS across MULTIPLE phantoms
    (env.reset() with each seed) rather than a single fixed-seed phantom.
    This matters: with only one phantom, this diagnostic is fragile to
    which specific phantom that single seed happens to draw -- e.g. a
    phantom where the target's true position is geometrically far (large
    d) from most random starting poses will show near-zero d-improvement
    simply because the potential's exp(-d/0.015) term is nearly flat
    (vanishing gradient) at long range, regardless of whether the
    underlying geometry is correct. Averaging over several phantoms (as
    the endpoint/symmetry check already does via K_PHANTOMS) makes the
    reported numbers robust to that per-phantom sampling variance. Defaults
    to a single phantom (the env's current one) if not given, for
    backward-compatible standalone use."""
    if phantom_seeds is None:
        phantom_seeds = [None]  # use whatever phantom is already loaded on env

    successes = 0
    final_alphas, final_ds = [], []
    initial_alphas, initial_ds = [], []
    traces = []
    paths_per_phantom = max(1, M_APPROACH_PATHS // len(phantom_seeds))

    for ph_seed in phantom_seeds:
        if ph_seed is not None:
            env.reset(seed=ph_seed)
        _force_target(env, target)

        for path_i in range(paths_per_phantom):
            start_params = np.array([rng.uniform(lo, hi) for lo, hi in GREEDY_START_BOUNDS])
            _set_pose(env, start_params)
            alpha_start, d_start = env._pose_error(target)
            initial_alphas.append(alpha_start)
            initial_ds.append(d_start)
            alphas, ds, phis = [], [], []
            for step in range(GREEDY_WALK_STEPS):
                best_action, best_phi, best_saved = None, -np.inf, None
                saved = (env.probe.theta, env.probe.phi, env.probe.roll, env.probe.pitch, env.probe.yaw)
                for a in MOVE_ACTIONS:
                    env.probe.theta, env.probe.phi, env.probe.roll, env.probe.pitch, env.probe.yaw = saved
                    _apply_action(env, a)
                    obs = env._compute_obs()
                    phi_val = env._potential(obs_cache=env._cache)
                    if phi_val > best_phi:
                        best_phi = phi_val
                        best_action = a
                        best_saved = (env.probe.theta, env.probe.phi, env.probe.roll, env.probe.pitch, env.probe.yaw)
                env.probe.theta, env.probe.phi, env.probe.roll, env.probe.pitch, env.probe.yaw = best_saved
                alpha, d = env._pose_error(target)
                alphas.append(alpha)
                ds.append(d)
                phis.append(best_phi)

            final_alpha, final_d = alphas[-1], ds[-1]
            final_alphas.append(final_alpha)
            final_ds.append(final_d)
            if np.degrees(final_alpha) <= GREEDY_SUCCESS_ALPHA_TOL_DEG and final_d <= GREEDY_SUCCESS_D_TOL_M:
                successes += 1
            traces.append(dict(alphas=np.array(alphas), ds=np.array(ds), phis=np.array(phis)))

    n_paths = len(final_alphas)

    # Combined error score (same scale as find_optimal_pose's objective:
    # alpha in radians + d/0.01) lets us measure IMPROVEMENT even when the
    # walk doesn't reach tight tolerance -- this is the bug-indicative
    # signal (does the field point the right way at all?), separate from
    # whether a myopic one-step-greedy walk can fully solve a genuinely
    # non-convex 5-DOF landscape in a bounded budget (a task-difficulty
    # question, answered empirically in Phase 2 by real RL algorithms with
    # exploration + bootstrapped value estimation, not by this proxy).
    initial_scores = np.array(initial_alphas) + np.array(initial_ds) / 0.01
    final_scores = np.array(final_alphas) + np.array(final_ds) / 0.01
    improved_fraction = float(np.mean(final_scores < initial_scores))
    mean_improvement_ratio = float(1.0 - np.mean(final_scores) / np.mean(initial_scores))

    return dict(
        target=target,
        success_fraction=successes / n_paths,
        mean_initial_alpha_deg=float(np.degrees(np.mean(initial_alphas))),
        mean_initial_d_mm=float(np.mean(initial_ds) * 1000),
        mean_final_alpha_deg=float(np.degrees(np.mean(final_alphas))),
        mean_final_d_mm=float(np.mean(final_ds) * 1000),
        improved_fraction=improved_fraction,
        mean_improvement_ratio=mean_improvement_ratio,
        traces=traces,
    )


def _plot_traces(target: str, traces: list[dict], suffix: str):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    n_steps = len(traces[0]["alphas"]) - (0 if suffix == "greedy" else 0)
    x = np.linspace(0, 1, len(traces[0]["alphas"]))
    for tr in traces:
        axes[0].plot(x, np.degrees(tr["alphas"]), alpha=0.5)
        axes[1].plot(x, np.array(tr["ds"]) * 1000, alpha=0.5)
        axes[2].plot(x, tr["phis"], alpha=0.5)
    axes[0].set_title(f"{target} [{suffix}]: alpha (deg) vs progress")
    axes[1].set_title(f"{target} [{suffix}]: d (mm) vs progress")
    axes[2].set_title(f"{target} [{suffix}]: Phi vs progress")
    for ax in axes:
        ax.set_xlabel("path progress")
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"approach_{target}_{suffix}.png", dpi=110)
    plt.close(fig)


def run(seed: int = 0, shaping_mode: str = "multiplicative", alpha_tol_deg: float = 18.0):
    rng = np.random.default_rng(seed)
    report = {"layer_a": layer_a_formula_monotonicity()}

    print("Layer A (potential formula, no env involved):")
    for k, v in report["layer_a"].items():
        print(f"  {k}: {v}")
    if shaping_mode != "additive":
        print(f"  NOTE: Layer A validates compute_potential() (additive) only -- "
              f"shaping_mode={shaping_mode!r} affects Layer B (which reads env._potential()) below.")
    if alpha_tol_deg != 15.0:
        print(f"  NOTE: alpha_tol_deg={alpha_tol_deg} affects reset()'s reachability resampling "
              f"(a wider tolerance accepts more phantoms as 'reachable'), which shifts which "
              f"phantoms Layer B draws -- it does NOT change this script's own hardcoded "
              f"diagnostic thresholds (ENDPOINT_ALPHA_TOL_DEG, GREEDY_SUCCESS_ALPHA_TOL_DEG).")

    env = UltrasoundProbeEnv(seed=seed, shaping_mode=shaping_mode, alpha_tol_deg=alpha_tol_deg)
    report["layer_b"] = {}

    for target in TARGET_SEQUENCE:
        target_report = dict(endpoint_and_symmetry=[], lerp_diagnostic=None, greedy=None)
        for k in range(K_PHANTOMS):
            env.reset(seed=seed + k)
            res = layer_b_endpoint_and_symmetry(env, target)
            target_report["endpoint_and_symmetry"].append(res)

        env.reset(seed=seed)
        find_optimal_pose(env, target, rng=np.random.default_rng(seed))
        end_params_for_lerp = _pose_vector(env)
        lerp = layer_b_lerp_diagnostic(env, target, end_params_for_lerp, rng)
        _plot_traces(target, lerp.pop("traces"), "lerp")
        target_report["lerp_diagnostic"] = lerp

        greedy_phantom_seeds = [seed + 1000 + k for k in range(K_GREEDY_PHANTOMS)]
        greedy = layer_b_greedy_discrete_walk(env, target, rng, phantom_seeds=greedy_phantom_seeds)
        _plot_traces(target, greedy.pop("traces"), "greedy")
        target_report["greedy"] = greedy

        report["layer_b"][target] = target_report

        n_endpoint_ok = sum(1 for r in target_report["endpoint_and_symmetry"] if r["endpoint_ok"])
        n_symmetry_ok = sum(1 for r in target_report["endpoint_and_symmetry"] if r["symmetry_ok"])
        print(f"\nLayer B [{target}]:")
        print(f"  endpoint reachable: {n_endpoint_ok}/{K_PHANTOMS} phantoms "
              f"(alpha<={ENDPOINT_ALPHA_TOL_DEG}deg, d<={ENDPOINT_D_TOL_M*1000:.0f}mm)")
        print(f"  plane-symmetry holds: {n_symmetry_ok}/{K_PHANTOMS} phantoms")
        print(f"  [diagnostic, not gated] raw Euler-lerp monotonic fraction: "
              f"{lerp['monotonic_fraction']:.2f} ({lerp['n_violations']} violations / {M_APPROACH_PATHS} paths)")
        print(f"  greedy discrete-action tolerance-success fraction: {greedy['success_fraction']:.2f} "
              f"(reported, not gated -- see module docstring: greedy is a weak, "
              f"one-step-lookahead proxy for what a trained RL policy can do)")
        print(f"  [GATE] greedy discrete-action IMPROVEMENT: "
              f"{greedy['improved_fraction']:.2f} of paths improved "
              f"(mean alpha {greedy['mean_initial_alpha_deg']:.1f}->{greedy['mean_final_alpha_deg']:.1f}deg, "
              f"mean d {greedy['mean_initial_d_mm']:.1f}->{greedy['mean_final_d_mm']:.1f}mm, "
              f"mean combined-error improvement ratio={greedy['mean_improvement_ratio']:.2f})")

    step_cost_total = 180 * -0.05
    print("\nReward-scale diagnostic (report only, not a pass/fail gate):")
    print(f"  accumulated step-cost over a full 180-step episode: {step_cost_total:.2f}")
    print(f"  freeze-within-tolerance event reward: up to +10.00")
    print(f"  all-three-acquired event reward: +20.00")
    print(f"  measurement-accurate event reward: +5.00")
    print(f"  correct-classification event reward: +5.00")
    print(f"  freeze-outside-tolerance event penalty: -2.00")
    print(f"  sub-task timeout penalty: -3.00, episode timeout penalty: -5.00")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "report_summary.txt", "w") as f:
        f.write(f"Layer A: {report['layer_a']}\n\n")
        for target in TARGET_SEQUENCE:
            tr = report["layer_b"][target]
            n_endpoint_ok = sum(1 for r in tr["endpoint_and_symmetry"] if r["endpoint_ok"])
            n_symmetry_ok = sum(1 for r in tr["endpoint_and_symmetry"] if r["symmetry_ok"])
            f.write(f"[{target}] endpoint_ok={n_endpoint_ok}/{K_PHANTOMS} "
                    f"symmetry_ok={n_symmetry_ok}/{K_PHANTOMS} "
                    f"lerp_monotonic_fraction(diagnostic)={tr['lerp_diagnostic']['monotonic_fraction']:.2f} "
                    f"greedy_tolerance_success(diagnostic)={tr['greedy']['success_fraction']:.2f} "
                    f"greedy_improved_fraction(GATE)={tr['greedy']['improved_fraction']:.2f} "
                    f"greedy_mean_improvement_ratio(GATE)={tr['greedy']['mean_improvement_ratio']:.2f}\n")

    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--shaping-mode", default="multiplicative", choices=["additive", "multiplicative", "hybrid"])
    parser.add_argument("--alpha-tol-deg", type=float, default=18.0)
    args = parser.parse_args()
    run(shaping_mode=args.shaping_mode, alpha_tol_deg=args.alpha_tol_deg)
