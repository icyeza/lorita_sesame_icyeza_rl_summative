"""Gymnasium environment: obstetric ultrasound probe-guidance task.

The agent controls a 5-DOF virtual probe constrained to the maternal
abdominal surface and must locate three standard fetal biometry planes in
sequence (head -> abdomen -> femur), freeze on each, then a clinical flag
(AGA / SGA) is derived from the resulting (simulated) biometry.

The target plane's pose is NEVER included in the observation -- the agent
must infer its position from image-derived features + its own
proprioception. See README.md and the project brief for full design intent.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from scipy.optimize import minimize

from environment import clinical_constants as cc
from environment.phantom import FetalPhantom, build_phantom, MATERNAL_ABDOMEN_SEMI_AXES
from environment.slicer import cast_slice, structure_visibility
from environment.features import extract_image_features, N_IMAGE_FEATURES

TARGET_SEQUENCE = ["head", "abdomen", "femur"]
REQUIRED_STRUCTURES = {
    "head": ["skull", "falx", "thalamus_L", "thalamus_R", "csp"],
    "abdomen": ["stomach", "umbilical_vein", "spine_abdo"],
    "femur": ["femur"],
}
EXCLUDER_STRUCTURES = {
    "head": ["cerebellum"],
    "abdomen": ["heart", "kidney_L", "kidney_R"],
    "femur": [],
}

N_PROPRIO_FEATURES = 12
N_CONTEXT_FEATURES = 5
OBS_DIM = N_IMAGE_FEATURES + N_PROPRIO_FEATURES + N_CONTEXT_FEATURES

# DEFAULT subtask step budget. Now also a per-instance constructor argument
# (`subtask_max_steps`, see UltrasoundProbeEnv.__init__) -- this module-level
# constant remains ONLY the default value (and is what scripts/validate_
# reward_field.py's GREEDY_WALK_STEPS deliberately mirrors, since that gate
# is checking against the real default budget). See the "two-lever fix"
# experiment (status.md) for why 60 was found to be the binding constraint:
# even a ground-truth ORACLE greedy controller needed a median 55 of 60
# subtask steps to succeed, and still timed out 65% of the time -- the
# budget was set at the oracle's own ceiling, leaving no room for a
# still-improving (not yet optimal) learner to ever succeed, so RL had no
# achievable target to bootstrap a gradient from.
SUBTASK_MAX_STEPS = 60
EPISODE_MAX_STEPS = 180

COARSE_ARC_DEG = 2.0
COARSE_ANGLE_DEG = 3.0

# DEFAULT actuator tilt-cone half-angle (degrees) for roll/pitch/yaw offset
# from the local surface normal. This is now a per-instance constructor
# argument (`actuator_limit_deg`, see UltrasoundProbeEnv.__init__) -- this
# module-level constant is ONLY the default value and the value
# `ACTUATOR_POSE_BOUNDS` (below) is built from for backward-compatible
# standalone use (e.g. scripts that don't have a live env instance yet).
# Any code with a live `env` should use `env.actuator_limit_deg` /
# `env._actuator_pose_bounds()` instead of these module-level values, so it
# reflects whatever cone width that specific env was constructed with.
MAX_OFFSET_DEG = 60.0
ACTUATOR_LIMIT_DEG_CAP = 80.0  # hard cap: beyond this is not physically
# defensible for a probe against an abdomen -- UltrasoundProbeEnv.__init__
# refuses to construct with actuator_limit_deg above this.

ACTIONS = [
    "theta_plus", "theta_minus", "phi_plus", "phi_minus",
    "roll_plus", "roll_minus", "pitch_plus", "pitch_minus",
    "yaw_plus", "yaw_minus", "toggle_fine", "freeze_and_measure",
]


def _actuator_pose_bounds(actuator_limit_deg: float = MAX_OFFSET_DEG):
    """(theta, phi, roll_deg, pitch_deg, yaw_deg) bounds matching the REAL
    actuator limits for a given cone half-angle -- used by the
    reachability search in `UltrasoundProbeEnv._is_target_reachable` (see
    below). Prefer `env._actuator_pose_bounds()` when a live env instance
    is available; this free function exists for standalone/module-level
    use (e.g. the module-level `ACTUATOR_POSE_BOUNDS` default below)."""
    return [
        (0.02, np.pi / 2 - 0.02), (0.0, 2 * np.pi),
        (-actuator_limit_deg, actuator_limit_deg),
        (-actuator_limit_deg, actuator_limit_deg),
        (-actuator_limit_deg, actuator_limit_deg),
    ]


# Backward-compatible module-level default (cone = MAX_OFFSET_DEG = 60).
ACTUATOR_POSE_BOUNDS = _actuator_pose_bounds(MAX_OFFSET_DEG)

# Cap on resample attempts in reset() when guarantee_reachable=True -- see
# UltrasoundProbeEnv.__init__'s `max_reachability_attempts`.
DEFAULT_MAX_REACHABILITY_ATTEMPTS = 50
# Restarts for the bounded Nelder-Mead reachability search per target --
# see `_is_target_reachable`. _pose_error involves no rendering, so this is
# cheap enough to run at every reset().
DEFAULT_REACHABILITY_SEARCH_RESTARTS = 3


POTENTIAL_V_WEIGHT = 1.0
POTENTIAL_ALPHA_WEIGHT = 2.0
POTENTIAL_ALPHA_SCALE = 0.3
POTENTIAL_D_WEIGHT = 2.0
# Distance-shaping length scale (meters). WAS 0.015 (15mm): exp(-d/0.015) is
# essentially flat (~0.02) across the 40-90mm range where episodes actually
# operate, so there was almost no gradient pulling the agent's position
# toward the target plane until it was already within ~15mm -- which it
# never reached. This caused a real training wall (see status.md
# "single_target PPO calibration wall"): a calibration run drove orientation
# to near-perfect (median terminal alpha 0.30deg) while parking distance at
# median 59mm (tol=12mm) and NEVER attempting freeze (2/3336 episodes
# succeeded), because there was no reward gradient rewarding further
# progress on distance once alpha was already exploited.
#
# Raised to 0.05 (50mm): exp(-59/50) ~= 0.31 -- a live gradient across the
# real operating range, while still rising toward 1.0 as d->0. This changes
# ONLY the shaping's reach; d_tol (still 12mm, unchanged) still defines what
# counts as an acquisition -- this makes the PATH to that tolerance visible
# to the reward, it does not relax the tolerance itself.
POTENTIAL_D_SCALE = 0.05


def compute_potential(v: float, alpha: float, d: float) -> float:
    """Pure potential-shaping formula (Ng et al. 1999 potential-based shaping):

        Phi(s) = POTENTIAL_V_WEIGHT*v
                 + POTENTIAL_ALPHA_WEIGHT*exp(-alpha/POTENTIAL_ALPHA_SCALE)
                 + POTENTIAL_D_WEIGHT*exp(-d/POTENTIAL_D_SCALE)

    Single source of truth for the potential formula -- `UltrasoundProbeEnv._potential`
    calls this directly rather than re-implementing the formula inline (an
    earlier version of `_potential` had `0.3`/`0.015` hardcoded inline,
    completely bypassing these constants; validating this standalone
    function was therefore checking a different formula instance than what
    training actually used, even though the literal values happened to
    match at the time). Kept standalone (no env/phantom dependency) so it
    can also be validated in isolation -- see
    `scripts/validate_reward_field.py` Layer A -- from the env's pose-error
    geometry, which is validated separately in Layer B.
    """
    return float(
        POTENTIAL_V_WEIGHT * v
        + POTENTIAL_ALPHA_WEIGHT * np.exp(-alpha / POTENTIAL_ALPHA_SCALE)
        + POTENTIAL_D_WEIGHT * np.exp(-d / POTENTIAL_D_SCALE)
    )


# Combined weight for the multiplicative alpha*d coupling term below --
# chosen so the new potential's max value (v=1, alpha=d=0 -> V_WEIGHT +
# COUPLE_WEIGHT = 1 + 4 = 5) matches the additive formula's max (V_WEIGHT +
# ALPHA_WEIGHT + D_WEIGHT = 1 + 2 + 2 = 5), so per-step reward magnitudes
# (and their relationship to the fixed -0.05 step cost and +10/+20 event
# bonuses) stay comparable -- this experiment is testing the ADDITIVE-vs-
# MULTIPLICATIVE combination, not also silently rescaling the whole reward.
POTENTIAL_COUPLE_WEIGHT = POTENTIAL_ALPHA_WEIGHT + POTENTIAL_D_WEIGHT


def compute_potential_multiplicative(v: float, alpha: float, d: float) -> float:
    """EXPERIMENTAL alternative to `compute_potential`: gates the alpha and
    d terms MULTIPLICATIVELY instead of additively.

        Phi(s) = POTENTIAL_V_WEIGHT*v
                 + POTENTIAL_COUPLE_WEIGHT * f(alpha) * g(d)
        where f(alpha) = exp(-alpha/POTENTIAL_ALPHA_SCALE) in [0,1] (->1 at alpha=0)
              g(d)     = exp(-d/POTENTIAL_D_SCALE)     in [0,1] (->1 at d=0)

    Motivation (see status.md "two-lever fix" experiment): under the
    additive formula, perfecting alpha alone banks nearly all of that
    term's reward (f(alpha)->1) regardless of how bad d is, so an agent can
    "cash out" on the alpha axis and have little further incentive to
    improve d. Multiplicative gating means the alpha-term's contribution is
    scaled BY g(d) -- perfect alpha with poor d yields only
    `COUPLE_WEIGHT * 1 * g(d)`, not the full COUPLE_WEIGHT -- so there is no
    axis to bank alone; both must improve together to capture the reward.
    This preserves the potential-based shaping FORM (Phi is still a pure
    function of state, so `gamma*Phi(s')-Phi(s)` is still policy-invariant
    per Ng et al. 1999 -- multiplicative composition does not break
    telescoping, only the additive assumption some intuition about
    "axis-independent credit" relied on). Selected via
    `UltrasoundProbeEnv(shaping_mode="multiplicative")` -- default remains
    "additive" (`compute_potential`), unchanged, until a human commits this.
    """
    f_alpha = np.exp(-alpha / POTENTIAL_ALPHA_SCALE)
    g_d = np.exp(-d / POTENTIAL_D_SCALE)
    return float(POTENTIAL_V_WEIGHT * v + POTENTIAL_COUPLE_WEIGHT * f_alpha * g_d)


DEFAULT_HYBRID_WEIGHT = 0.2


def compute_potential_hybrid(v: float, alpha: float, d: float, hybrid_weight: float = DEFAULT_HYBRID_WEIGHT) -> float:
    """EXPERIMENTAL alternative to `compute_potential_multiplicative`:
    keeps the multiplicative coupling term (so the near-goal fix from that
    experiment is preserved) but adds back a WEAK additive breadcrumb term
    so the far field is not flat.

        Phi(s) = POTENTIAL_V_WEIGHT*v
                 + POTENTIAL_COUPLE_WEIGHT * f(alpha) * g(d)          <- multiplicative term (unchanged)
                 + hybrid_weight * POTENTIAL_COUPLE_WEIGHT * (f(alpha) + g(d))   <- weak additive term
        where f(alpha), g(d) as in compute_potential_multiplicative.

    Motivation (see status.md "three-arm exploration fix" experiment): a
    40k-step PPO smoke train under pure multiplicative shaping showed
    terminal alpha drifting AWAY from tolerance (67deg -> 74deg) while
    terminal d barely moved and success stayed at exactly 0% -- consistent
    with the product f(alpha)*g(d) going nearly flat in the far field (when
    both factors are small, their product is smaller still, with almost no
    usable gradient), so a policy starting far from the goal has no signal
    pointing toward it. The additive term `f(alpha)+g(d)` is never flat --
    improving EITHER axis alone always increases it a little -- giving a
    weak breadcrumb trail from anywhere in the state space, while
    `hybrid_weight` (default 0.2, i.e. this term maxes at 40% of the
    multiplicative term's max) keeps it small enough that, near the goal,
    the multiplicative term's axis-coupling behavior (the reason it fixed
    the original bank-alpha-alone problem) still dominates. Still a pure
    function of state -- potential-based telescoping is unaffected.
    Selected via `UltrasoundProbeEnv(shaping_mode="hybrid")` -- default
    remains "additive", unchanged.
    """
    f_alpha = np.exp(-alpha / POTENTIAL_ALPHA_SCALE)
    g_d = np.exp(-d / POTENTIAL_D_SCALE)
    mult_term = POTENTIAL_COUPLE_WEIGHT * f_alpha * g_d
    breadcrumb_term = hybrid_weight * POTENTIAL_COUPLE_WEIGHT * (f_alpha + g_d)
    return float(POTENTIAL_V_WEIGHT * v + mult_term + breadcrumb_term)


def _rot_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / (np.linalg.norm(axis) + 1e-9)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


@dataclass
class ProbeState:
    theta: float  # polar angle on abdominal ellipsoid, [0, pi/2)
    phi: float  # azimuthal angle, [0, 2pi)
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    fine_mode: bool = False


class UltrasoundProbeEnv(gym.Env):
    """Custom Gymnasium env for fetal biometry plane acquisition."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, alpha_tol_deg: float = 18.0, d_tol_m: float = 0.012,
                 single_target: bool = True, allow_transverse: bool = False,
                 force_iugr: bool | None = None, render_mode: str | None = None,
                 seed: int | None = None, guarantee_reachable: bool = True,
                 max_reachability_attempts: int = DEFAULT_MAX_REACHABILITY_ATTEMPTS,
                 reachability_search_restarts: int = DEFAULT_REACHABILITY_SEARCH_RESTARTS,
                 actuator_limit_deg: float = MAX_OFFSET_DEG,
                 single_target_which: str | None = None,
                 freeze_miss_penalty: float = -2.0,
                 freeze_reward_mode: str = "cliff",
                 freeze_grade_sigma_m: float = 0.03,
                 freeze_attempt_cost: float = 0.3,
                 subtask_max_steps: int = SUBTASK_MAX_STEPS,
                 shaping_mode: str = "multiplicative",
                 tilt_step_deg: float = COARSE_ANGLE_DEG,
                 hybrid_weight: float = DEFAULT_HYBRID_WEIGHT,
                 start_curriculum: bool = True,
                 start_curriculum_max_random_steps: int = 8,
                 start_curriculum_push_prob: float = 0.0,
                 start_curriculum_push_max_iters: int = 30):
        """
        single_target: **LOCKED default True** (see status.md "make
            single_target the environment default" pass) -- one randomly-
            drawn target (head/abdomen/femur) per episode, no clinical
            classification. Changed from the original False (the full
            head->abdomen->femur->classification sequence) after every
            piece of real evidence gathered across this project's fix/
            calibration passes (the reward-field gate, the oracle
            diagnostics, the locked curriculum's 92-98% success, and the
            headline run's 0% full-task vs 98% single-target result) was
            single-target. The sequential 3-target task remains fully
            supported (`single_target=False`) but is no longer the
            active default -- callers that specifically need it (e.g.
            `evaluation/generalization.py`'s classification/held-out
            comparison) now pass it explicitly rather than relying on
            what used to be the default.
        single_target_which: EXPERIMENTAL, sampling-only lever (see
            scripts/experiment_freeze_wall.py). If set (one of
            TARGET_SEQUENCE) and `single_target=True`, fixes every episode's
            target to this one instead of drawing randomly from
            TARGET_SEQUENCE -- used to isolate a single (e.g. fully
            "recovered") target for an experiment without head/abdomen as
            confounders. Touches ONLY which target gets sampled in
            `reset()`; no reward/geometry effect. None (default) preserves
            the original uniform-random single_target behavior.
        freeze_miss_penalty: EXPERIMENTAL lever (see
            scripts/experiment_freeze_wall.py) on the event reward for a
            freeze OUTSIDE tolerance, used when `freeze_reward_mode="cliff"`
            (the default). Default -2.0 is the frozen/committed
            environment's actual value -- this argument exists so an
            experiment script can test alternative values (e.g. 0.0, -0.3,
            -0.5) WITHOUT editing this file. Changing the default here is a
            separate, deliberate decision for later, not something this
            parameter's mere existence implies.
        freeze_reward_mode: EXPERIMENTAL, one of "cliff" (default) or
            "graded" (see scripts/experiment_freeze_reward.py). "cliff" is
            the original/committed behavior: a flat `freeze_miss_penalty`
            for ANY freeze outside tolerance, regardless of how close it
            was -- a freeze at 13mm and a freeze at 90mm are rewarded
            identically, giving the agent no gradient toward better
            placement once it decides to commit. "graded" replaces that
            flat penalty with a smooth function of `d`
            (`10*exp(-d/freeze_grade_sigma_m) - freeze_attempt_cost`) so a
            near-miss is rewarded close to (but always less than) a hit,
            and reward decays smoothly as `d` grows -- while still
            subtracting a small fixed `freeze_attempt_cost` so spamming
            freeze from far away remains net-negative. The INSIDE-TOLERANCE
            success branch (`alpha<=alpha_tol and d<=d_tol` -> +10 scaled by
            alpha, acquisition, termination) is IDENTICAL in both modes --
            "graded" only changes what happens on a MISS, it does not
            relax what counts as a successful acquisition.
        freeze_grade_sigma_m: decay length scale (meters) for the "graded"
            freeze-miss reward. Only used when `freeze_reward_mode="graded"`.
        freeze_attempt_cost: fixed cost subtracted from every missed-freeze
            reward in "graded" mode (the anti-spam term). Only used when
            `freeze_reward_mode="graded"`.
        subtask_max_steps: EXPERIMENTAL lever (see status.md "two-lever
            fix" experiment) -- per-subtask step budget before a timeout
            (-3.0 penalty, subtask force-advanced). Default
            `SUBTASK_MAX_STEPS` (60) is the frozen/committed value. A
            scripted-oracle diagnostic found 60 is set at roughly the
            ORACLE's own ceiling (median 55/60 steps needed with
            ground-truth pose knowledge and a one-step-lookahead greedy
            controller) -- leaving no headroom for a still-improving,
            not-yet-optimal learner to ever succeed within budget. Raising
            this gives a learner room to succeed slowly while still
            improving, which is what RL needs to have a climbable gradient
            at all.
        shaping_mode: one of "additive" (`compute_potential`) or
            "multiplicative" (default, `compute_potential_multiplicative`)
            -- see that function's docstring for the full rationale.
            **LOCKED default as of the environment lock** (see status.md
            "lock the environment" pass): multiplicative shaping couples
            the alpha and d terms so perfecting one does not "bank" reward
            independent of the other -- measured to fix the reward-field
            gate's greedy-improvement ratio across all three targets
            (head 0.07->0.78, abdomen 0.16->0.92, femur held at 0.83).
            "additive" remains available for comparison/regression checks
            but is no longer the default.
        tilt_step_deg: the roll/pitch/yaw per-action increment (halved
            when `fine_mode` is on). Default 3.0deg (`COARSE_ANGLE_DEG`,
            unchanged) -- co-calibrated with `alpha_tol_deg`'s locked
            18deg value (see status.md "action/tolerance co-calibration"):
            a scripted-oracle diagnostic found ~30% of femur starts got
            stuck oscillating FOREVER just outside a 15deg tolerance
            regardless of step budget (a 3deg step overshoots a 15deg-wide
            window from certain approach angles); widening the tolerance
            to 18deg (keeping this 3deg step) resolved it (oracle success
            70%->92%) more reliably than shrinking the step did.
        hybrid_weight: only used when `shaping_mode="hybrid"` -- see
            `compute_potential_hybrid`'s docstring. Default 0.2. "hybrid"
            was tested and NOT selected for the lock (it regressed the
            reward-field gate's head/abdomen improvement ratios vs
            "multiplicative": 0.78->0.24, 0.92->0.31 -- see status.md
            "three-arm exploration fix").
        start_curriculum: if True (default, **LOCKED** as of the
            environment lock), `reset()` teleports the probe to a
            near-optimal pose for the episode's first target (reusing the
            same fine-search optimizer as the reachability check,
            `_search_min_pose_error`), then applies a random number (0 to
            `start_curriculum_max_random_steps`) of REAL discrete actions
            (movement only, no freeze) drawn uniformly at random, so every
            curriculum start is reachable by construction (it IS a real
            action sequence away from a known-good pose) rather than an
            invented alpha/d offset. Locked in at radius 8 after a
            twice-reproduced, deterministic 92% success measurement (see
            status.md "generalization check" pass) -- set False to restore
            the original uniform-random (theta, phi) start with
            roll=pitch=yaw=0, e.g. for a from-scratch far-field comparison
            (measured 0% success under the locked reward/tolerance/step
            config at 40k-120k steps; an explicitly documented open
            limitation, not silently hidden -- see that same pass).
        start_curriculum_max_random_steps: upper bound (inclusive) on the
            number of random real actions applied after teleporting near
            the optimum, when `start_curriculum=True`. Default 8 (LOCKED
            -- the only radius with a clean, reproduced success number;
            wider radii were tried in a widening-schedule train and
            caused catastrophic forgetting, see status.md).
        start_curriculum_push_prob: EXPERIMENTAL lever (see status.md
            "navigation-skill gap" pass), default 0.0 (no effect unless
            explicitly set -- fully backward compatible). Problem this
            addresses: at the default random-walk curriculum above, a
            deterministic check found ~95% of episodes land back INSIDE
            alpha_tol/d_tol regardless of `start_curriculum_max_random_steps`
            (a symmetric +/-  random walk mostly cancels out, even at
            radius 25) -- so the trained PPO headline model saw almost no
            genuine "close a real gap" episodes during training, and
            scored 0/9 (0%) on the rare ones it did encounter, always
            timing out at `subtask_max_steps`, vs 100% on trivially-already-
            solved starts (measured this session, N=200). When
            `self._rng.random() < start_curriculum_push_prob` (checked once
            per `reset()`), the curriculum instead calls
            `_apply_directed_push`: 2 FIXED movement directions applied
            repeatedly (not resampled each step, so displacement
            accumulates instead of cancelling) until the pose is genuinely
            outside alpha_tol/d_tol or `start_curriculum_push_max_iters` is
            hit. Still only real env actions -- never an invented alpha/d
            offset. Default 0.0 preserves the exact original curriculum;
            training scripts ramp this up via `set_start_curriculum`'s
            `push_prob` argument on a staged, success-gated schedule (NOT
            a blind timestep trigger like the earlier widening-schedule
            attempt) -- see scripts/fix_navigation_gap.py.
        start_curriculum_push_max_iters: cap on the directed-push loop
            above, so a phantom where the push direction happens to be
            degenerate can't hang `reset()`. Default 30 (empirically:
            ~87% of pushes escape tolerance well before this cap).
        guarantee_reachable: if True (default), `reset()` resamples the
            fetal pose until all of this episode's target plane(s) --
            just the active one in `single_target` mode, all three
            otherwise -- are reachable within the actuator's real
            roll/pitch/yaw tilt cone (`actuator_limit_deg`) AND this env's
            own alpha_tol/d_tol (so tightening tolerances tightens
            reachability too, no separate hardcoded criterion). Some
            randomly-sampled fetal poses place a target plane's required
            orientation outside the actuator cone, making that episode
            physically unwinnable regardless of policy -- see
            `scripts/measure_reachability.py` for the measured
            unfiltered reachable fraction. Set False to restore the old
            unfiltered behavior (e.g. for the generalization holdout,
            or to reproduce pre-fix numbers for comparison).
        max_reachability_attempts: resample cap per reset() when
            guarantee_reachable=True. If hit, the last-sampled (possibly
            unreachable) pose is used as a fallback and
            `self.reachability_capped` is set True with a warning --
            repeated cap-hits would mean the reachable fraction is
            unexpectedly low for the configured tolerances.
        reachability_search_restarts: Nelder-Mead random restarts used by
            the reachability search per target per attempt. Cheap (no
            rendering), so this runs at every reset() when enabled.
        actuator_limit_deg: the roll/pitch/yaw tilt-cone half-angle
            (degrees) from the local surface normal. Single source of
            truth for the actuator clamp -- used by `step()`'s per-action
            clamps, `_probe_frame()`'s contact-validity check, and the
            reachability search's bounds (`_actuator_pose_bounds()`).
            Per-action tilt INCREMENTS (`COARSE_ANGLE_DEG` = 3deg) are
            unchanged by this -- only the clamp RANGE moves. Capped at
            `ACTUATOR_LIMIT_DEG_CAP` (80deg): beyond that is not
            physically defensible for a probe against an abdomen.
        """
        super().__init__()
        if actuator_limit_deg > ACTUATOR_LIMIT_DEG_CAP:
            raise ValueError(
                f"actuator_limit_deg={actuator_limit_deg} exceeds the hard cap of "
                f"{ACTUATOR_LIMIT_DEG_CAP}deg -- not physically defensible for a probe "
                f"against an abdomen. See environment/custom_env.py."
            )
        self.alpha_tol = np.radians(alpha_tol_deg)
        self.d_tol = d_tol_m
        self.single_target = single_target
        self.allow_transverse = allow_transverse
        self.force_iugr = force_iugr
        self.render_mode = render_mode
        self.guarantee_reachable = guarantee_reachable
        self.max_reachability_attempts = max_reachability_attempts
        self.actuator_limit_deg = actuator_limit_deg
        self.reachability_search_restarts = reachability_search_restarts
        if single_target_which is not None and single_target_which not in TARGET_SEQUENCE:
            raise ValueError(f"single_target_which={single_target_which!r} must be one of {TARGET_SEQUENCE}")
        self.single_target_which = single_target_which
        self.freeze_miss_penalty = freeze_miss_penalty
        if freeze_reward_mode not in ("cliff", "graded"):
            raise ValueError(f"freeze_reward_mode={freeze_reward_mode!r} must be 'cliff' or 'graded'")
        self.freeze_reward_mode = freeze_reward_mode
        self.freeze_grade_sigma_m = freeze_grade_sigma_m
        self.freeze_attempt_cost = freeze_attempt_cost
        self.subtask_max_steps = subtask_max_steps
        if shaping_mode not in ("additive", "multiplicative", "hybrid"):
            raise ValueError(f"shaping_mode={shaping_mode!r} must be 'additive', 'multiplicative', or 'hybrid'")
        self.shaping_mode = shaping_mode
        self.tilt_step_deg = tilt_step_deg
        self.hybrid_weight = hybrid_weight
        self.start_curriculum = start_curriculum
        self.start_curriculum_max_random_steps = start_curriculum_max_random_steps
        if not 0.0 <= start_curriculum_push_prob <= 1.0:
            raise ValueError(f"start_curriculum_push_prob={start_curriculum_push_prob} must be in [0, 1]")
        self.start_curriculum_push_prob = start_curriculum_push_prob
        self.start_curriculum_push_max_iters = start_curriculum_push_max_iters
        self._last_search_pose = None

        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.Box(low=-5.0, high=5.0, shape=(OBS_DIM,), dtype=np.float32)

        self._rng = np.random.default_rng(seed)
        self.phantom: FetalPhantom | None = None
        self.probe = ProbeState(theta=np.radians(20), phi=0.0)
        self.targets: list[str] = []
        self.target_idx = 0
        self.acquired: dict[str, dict] = {}
        self.steps_in_subtask = 0
        self.total_steps = 0
        self._prev_potential: float | None = None
        self._last_image = None
        self._last_reward_info: dict = {}
        self._last_terminated = False
        self._last_truncated = False
        self.reachability_capped = False
        self.reachability_attempts_used = 0

    def seed(self, seed=None):
        self._rng = np.random.default_rng(seed)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Target(s) are chosen ONCE, before any resampling, and held fixed
        # across resample attempts -- only the fetal POSE is resampled.
        # This keeps the target distribution exactly uniform in
        # single_target mode; resampling the target jointly with the
        # phantom would skew the accepted-episode target distribution
        # toward whichever target happens to be more often reachable.
        if self.single_target:
            targets = [self.single_target_which] if self.single_target_which is not None \
                else [str(self._rng.choice(TARGET_SEQUENCE))]
        else:
            targets = list(TARGET_SEQUENCE)

        self.reachability_capped = False
        self.reachability_attempts_used = 0

        if self.guarantee_reachable:
            for attempt in range(1, self.max_reachability_attempts + 1):
                self.reachability_attempts_used = attempt
                self.phantom = build_phantom(self._rng, allow_transverse=self.allow_transverse,
                                              force_iugr=self.force_iugr)
                if all(self._is_target_reachable(t) for t in targets):
                    break
            else:
                self.reachability_capped = True
                warnings.warn(
                    f"UltrasoundProbeEnv.reset(): hit max_reachability_attempts="
                    f"{self.max_reachability_attempts} without finding a fetal pose "
                    f"reaching all of targets={targets} within tolerance -- falling "
                    f"back to the last-sampled (possibly unreachable) pose. Repeated "
                    f"cap-hits mean the reachable fraction is unexpectedly low for "
                    f"the configured alpha_tol/d_tol; see "
                    f"scripts/measure_reachability.py.",
                    RuntimeWarning,
                )
        else:
            self.phantom = build_phantom(self._rng, allow_transverse=self.allow_transverse,
                                          force_iugr=self.force_iugr)

        if self.start_curriculum:
            target0 = targets[0]
            self._search_min_pose_error(target0)  # populates self._last_search_pose
            theta0, phi0, roll0, pitch0, yaw0 = self._last_search_pose
            self.probe = ProbeState(theta=theta0, phi=phi0, roll=roll0, pitch=pitch0, yaw=yaw0, fine_mode=False)
            move_names = [a for a in ACTIONS if a != "freeze_and_measure"]
            if self.start_curriculum_push_prob > 0.0 and self._rng.random() < self.start_curriculum_push_prob:
                self._apply_directed_push(target0, move_names)
            else:
                n_random = int(self._rng.integers(0, self.start_curriculum_max_random_steps + 1))
                for _ in range(n_random):
                    self._apply_movement(str(self._rng.choice(move_names)))
        else:
            self.probe = ProbeState(theta=float(self._rng.uniform(np.radians(10), np.radians(50))),
                                     phi=float(self._rng.uniform(0, 2 * np.pi)))
        self.targets = targets
        self.target_idx = 0
        self.acquired = {}
        self.steps_in_subtask = 0
        self.total_steps = 0
        self._prev_potential = None
        self._episode_freeze_attempted = False
        self._last_terminated = False
        self._last_truncated = False
        info = {}
        obs = self._compute_obs()
        self._prev_potential = self._potential(obs_cache=self._cache)
        return obs, info

    # ------------------------------------------------------------------
    # Reachability (guarantee_reachable=True support)
    # ------------------------------------------------------------------
    def _actuator_pose_bounds(self):
        """(theta, phi, roll_deg, pitch_deg, yaw_deg) bounds for THIS
        instance's actual `actuator_limit_deg` -- use this (not the
        module-level `ACTUATOR_POSE_BOUNDS`, which is frozen at the
        default 60deg) whenever a live env instance is available, so
        reachability search bounds always track whatever cone width this
        env was constructed with."""
        return _actuator_pose_bounds(self.actuator_limit_deg)

    def _search_min_pose_error(self, target: str) -> tuple[float, float]:
        """Search over (theta, phi, roll, pitch, yaw) within the actuator's
        real limits (`self._actuator_pose_bounds()`), reusing `_pose_error`
        -- the exact reward criterion, not a separate approximation -- to
        find the best achievable (alpha, d) for `target` on the CURRENT
        `self.phantom`. Temporarily mutates `self.probe` and restores it
        afterward. No rendering is involved (`_pose_error` only needs
        `_probe_frame` + the phantom's plane targets), so this is cheap
        enough to run at every reset().

        MEASUREMENT BUG FOUND AND FIXED (scripts/audit_reachability_search.py):
        a pure 5D-joint Nelder-Mead (blind random restarts over all of
        theta/phi/roll/pitch/yaw at once) badly under-reported reachability
        for targets like "head", where alpha~0 is achievable from a large,
        flat manifold of (theta, phi) positions -- the blind joint search
        would frequently land in a bad local basin and report e.g.
        alpha=45deg when a fine-grid audit at a good d~0 position found
        alpha=0.3deg was achievable there (confirmed on 20/20 sampled
        "unreachable" head cases -- ALL were actually reachable). This also
        explained an observed non-monotonicity bug (widening the actuator
        cone sometimes made the reported optimum WORSE, which is
        mathematically impossible for a true optimum over a strictly larger
        search space -- only possible for an unreliable optimizer).

        Fix: in addition to the original blind 5D joint restarts (kept, for
        diversity / as a fallback), also run `reachability_search_restarts`
        rounds of a POSITION-FIRST strategy that mirrors how the audit's
        fine grid found good solutions: (1) search (theta, phi) ALONE
        (roll=pitch=yaw=0) to find a low-d position, then (2) polish
        (roll, pitch, yaw) alone at that FIXED position to minimize alpha.
        This directly encodes the problem's structure (many d~0 positions
        exist; a small tilt search at most of them reaches alpha~0) instead
        of hoping blind random restarts stumble into the right basin, and
        is still cheap (small 2D and 3D sub-searches, no rendering).

        Restart points come from a FRESH, fixed-seed RNG created inside
        this call (not `self._rng`) specifically so `_is_target_reachable`
        is a deterministic, repeatable function of `self.phantom` alone --
        calling it twice on the same phantom always searches the same
        restart points and gives the same answer. Drawing restarts from
        `self._rng` instead (an earlier version of this code did) made the
        search's outcome depend on how many other draws had happened
        first, so `reset()` could accept a pose as reachable and a later,
        independent call could then report the same target unreachable on
        the same phantom -- caught by test flakiness in
        tests/test_reachability.py, not by design."""
        saved = (self.probe.theta, self.probe.phi, self.probe.roll, self.probe.pitch, self.probe.yaw)
        restart_rng = np.random.default_rng(20260726)  # fixed seed: same points every call
        bounds = self._actuator_pose_bounds()
        pos_bounds, tilt_bounds = bounds[:2], bounds[2:]

        def objective(params):
            self.probe.theta, self.probe.phi, self.probe.roll, self.probe.pitch, self.probe.yaw = params
            alpha, d = self._pose_error(target)
            # normalize by the env's OWN tolerances so both terms are
            # comparable and the objective's scale tracks alpha_tol/d_tol
            return alpha / self.alpha_tol + d / max(self.d_tol, 1e-9)

        def position_only_objective(pos_params):
            self.probe.theta, self.probe.phi = pos_params
            self.probe.roll = self.probe.pitch = self.probe.yaw = 0.0
            _, d = self._pose_error(target)
            return d

        def tilt_only_objective(rpy_params, theta_fixed, phi_fixed):
            self.probe.theta, self.probe.phi = theta_fixed, phi_fixed
            self.probe.roll, self.probe.pitch, self.probe.yaw = rpy_params
            alpha, d = self._pose_error(target)
            return alpha / self.alpha_tol + d / max(self.d_tol, 1e-9)

        best_x, best_val = None, np.inf

        # (1) blind 5D joint restarts (original strategy, kept as a fallback
        # / for diversity against cases the position-first strategy below
        # doesn't suit).
        for _ in range(self.reachability_search_restarts):
            x0 = np.array([restart_rng.uniform(lo, hi) for lo, hi in bounds])
            res = minimize(objective, x0, method="Nelder-Mead", bounds=bounds,
                            options=dict(xatol=1e-3, fatol=1e-3, maxiter=200, maxfev=200))
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x

        # (2) position-first, then tilt-polish (the fix).
        for _ in range(self.reachability_search_restarts):
            pos_x0 = np.array([restart_rng.uniform(lo, hi) for lo, hi in pos_bounds])
            pos_res = minimize(position_only_objective, pos_x0, method="Nelder-Mead", bounds=pos_bounds,
                                options=dict(xatol=1e-4, fatol=1e-6, maxiter=200, maxfev=200))
            theta_c, phi_c = pos_res.x
            tilt_x0 = np.array([restart_rng.uniform(lo, hi) for lo, hi in tilt_bounds])
            tilt_res = minimize(tilt_only_objective, tilt_x0, method="Nelder-Mead", bounds=tilt_bounds,
                                 args=(theta_c, phi_c),
                                 options=dict(xatol=1e-3, fatol=1e-3, maxiter=200, maxfev=200))
            candidate_x = np.array([theta_c, phi_c, *tilt_res.x])
            val = objective(candidate_x)
            if val < best_val:
                best_val = val
                best_x = candidate_x

        self.probe.theta, self.probe.phi, self.probe.roll, self.probe.pitch, self.probe.yaw = best_x
        alpha, d = self._pose_error(target)
        # Stashed (not returned, to keep this method's signature/behavior
        # unchanged for its original reachability-check callers) so
        # reset()'s start_curriculum lever can teleport there -- see
        # `start_curriculum`'s docstring above.
        self._last_search_pose = tuple(best_x)
        self.probe.theta, self.probe.phi, self.probe.roll, self.probe.pitch, self.probe.yaw = saved
        return alpha, d

    def _is_target_reachable(self, target: str) -> bool:
        """True iff some probe configuration within the actuator's real
        limits brings `target` within THIS env's own alpha_tol/d_tol --
        i.e. reachable by the actual acquisition criterion `step()` uses,
        so tightening alpha_tol_deg/d_tol_m tightens this check too."""
        alpha, d = self._search_min_pose_error(target)
        return alpha <= self.alpha_tol and d <= self.d_tol

    def set_start_curriculum(self, start_curriculum: bool, max_random_steps: int | None = None,
                              push_prob: float | None = None):
        """Runtime setter for the `start_curriculum`/
        `start_curriculum_max_random_steps`/`start_curriculum_push_prob`
        levers (see status.md "generalization check" and "navigation-skill
        gap" experiments). `SubprocVecEnv` workers can't have constructor
        arguments changed after creation, so a training-time schedule
        needs a live setter it can call mid-training via
        `VecEnv.env_method("set_start_curriculum", ...)` -- see the
        widening-schedule callback in `scripts/generalization_check.py`
        (drives `max_random_steps`) and the push-schedule callback in
        `scripts/fix_navigation_gap.py` (drives `push_prob`). Takes effect
        from the NEXT `reset()` onward; does not affect the current
        episode in progress."""
        self.start_curriculum = start_curriculum
        if max_random_steps is not None:
            self.start_curriculum_max_random_steps = max_random_steps
        if push_prob is not None:
            self.start_curriculum_push_prob = push_prob

    # ------------------------------------------------------------------
    # Probe geometry
    # ------------------------------------------------------------------
    def _probe_frame(self):
        a, b, c = MATERNAL_ABDOMEN_SEMI_AXES
        th, ph = self.probe.theta, self.probe.phi
        pos = np.array([a * np.sin(th) * np.cos(ph),
                         b * np.sin(th) * np.sin(ph),
                         c * np.cos(th)])
        normal = np.array([pos[0] / a ** 2, pos[1] / b ** 2, pos[2] / c ** 2])
        normal /= np.linalg.norm(normal) + 1e-9

        world_up = np.array([0.0, 1.0, 0.0])
        tangent_up = world_up - np.dot(world_up, normal) * normal
        if np.linalg.norm(tangent_up) < 1e-6:
            tangent_up = np.array([1.0, 0.0, 0.0])
        tangent_up /= np.linalg.norm(tangent_up)
        right = np.cross(tangent_up, normal)
        right /= np.linalg.norm(right) + 1e-9
        up = np.cross(normal, right)

        forward = -normal  # ray direction points into the body

        offset_angle = np.radians(np.sqrt(self.probe.roll ** 2 + self.probe.pitch ** 2 + self.probe.yaw ** 2))
        contact_valid = offset_angle <= np.radians(self.actuator_limit_deg)

        Rr = _rot_from_axis_angle(forward, np.radians(self.probe.roll))
        Rp = _rot_from_axis_angle(right, np.radians(self.probe.pitch))
        Ry = _rot_from_axis_angle(up, np.radians(self.probe.yaw))
        R = Ry @ Rp @ Rr
        forward = R @ forward
        right = R @ right
        up = R @ up

        return pos, forward, right, up, contact_valid

    def _apply_movement(self, name: str):
        """The probe-pose delta for one non-freeze action -- single source
        of truth for `step()` AND `reset()`'s start-state curriculum
        (`start_curriculum=True`), which applies a random sequence of REAL
        actions from a near-optimal pose rather than inventing an alpha/d
        offset directly. `freeze_and_measure` and unrecognized names are a
        no-op here (freeze has no pose effect; its reward/termination
        handling lives in `step()`)."""
        arc_deg = COARSE_ARC_DEG / 2.0 if self.probe.fine_mode else COARSE_ARC_DEG
        ang_deg = self.tilt_step_deg / 2.0 if self.probe.fine_mode else self.tilt_step_deg
        arc = np.radians(arc_deg)
        if name == "theta_plus":
            self.probe.theta = float(np.clip(self.probe.theta + arc, 0.01, np.pi / 2 - 0.01))
        elif name == "theta_minus":
            self.probe.theta = float(np.clip(self.probe.theta - arc, 0.01, np.pi / 2 - 0.01))
        elif name == "phi_plus":
            self.probe.phi = float((self.probe.phi + arc) % (2 * np.pi))
        elif name == "phi_minus":
            self.probe.phi = float((self.probe.phi - arc) % (2 * np.pi))
        elif name == "roll_plus":
            self.probe.roll = float(np.clip(self.probe.roll + ang_deg, -self.actuator_limit_deg, self.actuator_limit_deg))
        elif name == "roll_minus":
            self.probe.roll = float(np.clip(self.probe.roll - ang_deg, -self.actuator_limit_deg, self.actuator_limit_deg))
        elif name == "pitch_plus":
            self.probe.pitch = float(np.clip(self.probe.pitch + ang_deg, -self.actuator_limit_deg, self.actuator_limit_deg))
        elif name == "pitch_minus":
            self.probe.pitch = float(np.clip(self.probe.pitch - ang_deg, -self.actuator_limit_deg, self.actuator_limit_deg))
        elif name == "yaw_plus":
            self.probe.yaw = float(np.clip(self.probe.yaw + ang_deg, -self.actuator_limit_deg, self.actuator_limit_deg))
        elif name == "yaw_minus":
            self.probe.yaw = float(np.clip(self.probe.yaw - ang_deg, -self.actuator_limit_deg, self.actuator_limit_deg))
        elif name == "toggle_fine":
            self.probe.fine_mode = not self.probe.fine_mode

    def _apply_directed_push(self, target: str, move_names: list[str]):
        """`reset()`'s `start_curriculum_push_prob` lever: picks 2 FIXED
        movement directions (not resampled each iteration, unlike the
        default curriculum's per-step-random walk) and applies them
        repeatedly so displacement accumulates instead of averaging back
        out -- a symmetric random walk mostly cancels itself out relative
        to the generous alpha_tol/d_tol window (measured: even radius=25
        pure-random-walk starts left ~90% still inside tolerance), so it
        essentially never produced genuinely non-trivial training episodes.
        Stops as soon as the pose is genuinely outside tolerance, or after
        `start_curriculum_push_max_iters` iterations (2 actions each) if
        the chosen directions happen to be degenerate for this phantom --
        empirically ~87% of pushes escape well before the cap. Only ever
        applies real `_apply_movement` actions, same as the rest of the
        curriculum -- never an invented alpha/d offset."""
        push_dirs = [str(self._rng.choice(move_names)) for _ in range(2)]
        alpha, d = self._pose_error(target)
        iters = 0
        while alpha <= self.alpha_tol and d <= self.d_tol and iters < self.start_curriculum_push_max_iters:
            for name in push_dirs:
                self._apply_movement(name)
            alpha, d = self._pose_error(target)
            iters += 1

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(self, action: int):
        assert self.action_space.contains(action)
        name = ACTIONS[action]

        did_freeze = False
        if name == "freeze_and_measure":
            did_freeze = True
            self._episode_freeze_attempted = True
        else:
            self._apply_movement(name)

        self.steps_in_subtask += 1
        self.total_steps += 1

        obs = self._compute_obs()
        current_target = self.targets[self.target_idx]
        alpha, d = self._pose_error(current_target)
        potential = self._potential(obs_cache=self._cache)
        prev_potential = self._prev_potential if self._prev_potential is not None else potential
        reward = 0.99 * potential - prev_potential
        self._prev_potential = potential
        reward -= 0.05  # per-step cost

        terminated = False
        truncated = False
        flag = None

        if did_freeze:
            if alpha <= self.alpha_tol and d <= self.d_tol:
                reward += 10.0 * (1.0 - alpha / max(self.alpha_tol, 1e-6))
                self.acquired[current_target] = self._measure(current_target)
                self.target_idx += 1
                self.steps_in_subtask = 0
                if self.target_idx >= len(self.targets):
                    terminated = True
                    if not self.single_target and len(self.acquired) == 3:
                        reward += 20.0
                        err_ok, flag = self._finalize_clinical()
                        if err_ok:
                            reward += 5.0
                        if flag is not None:
                            reward += 5.0
            else:
                if self.freeze_reward_mode == "graded":
                    # Smooth function of d: near +10 close to the tolerance
                    # boundary, decaying smoothly as d grows, minus a fixed
                    # anti-spam cost. Never touches the inside-tolerance
                    # success branch above -- only reshapes what happens on
                    # a miss. See freeze_reward_mode's docstring.
                    reward += 10.0 * np.exp(-d / self.freeze_grade_sigma_m) - self.freeze_attempt_cost
                else:
                    reward += self.freeze_miss_penalty

        if not terminated and self.steps_in_subtask >= self.subtask_max_steps:
            reward -= 3.0
            self.target_idx += 1
            self.steps_in_subtask = 0
            if self.target_idx >= len(self.targets):
                terminated = True

        if not terminated and self.total_steps >= EPISODE_MAX_STEPS:
            reward -= 5.0
            truncated = True

        info = {
            "target": current_target if self.target_idx < len(self.targets) else None,
            "alpha_deg": np.degrees(alpha), "d_m": d,
            "acquired": list(self.acquired.keys()), "flag": flag,
            "reward": float(reward),
            # Diagnostic-only fields (do not affect reward/geometry): let
            # SB3's Monitor(info_keywords=...) log these directly per
            # episode, rather than needing checkpoint re-evaluation to
            # reconstruct them after the fact -- see
            # scripts/run_freeze_placement_confirmation.py.
            "success": bool(len(self.acquired) > 0) if terminated else False,
            "freeze_attempted": self._episode_freeze_attempted,
            # Diagnostic-only, added for the headline-run report (status.md
            # "headline run" pass): "success" above is >=1 acquisition
            # (meaningful in single_target mode); "full_task_success" is
            # the stricter all-3-acquired signal needed to report a
            # genuine full-task success rate. "true_is_iugr" is the
            # phantom's ground-truth growth-restriction label, needed to
            # score `flag` (AGA/SGA) classification ACCURACY against
            # ground truth, not just report the predicted flag itself.
            # Neither affects reward/geometry/classification logic.
            "full_task_success": bool(len(self.acquired) == 3) if terminated else False,
            "true_is_iugr": bool(self.phantom.is_iugr) if self.phantom is not None else None,
        }
        self._last_reward_info = info
        # Diagnostic-only, cached the same way as _last_reward_info: lets
        # VizBridge.publish() (which only receives `env`, not step()'s
        # direct return values) tell whether the episode ended on THIS
        # step, to show an episode-end indicator in the live viz. No
        # reward/geometry effect.
        self._last_terminated = terminated
        self._last_truncated = truncated
        return obs, float(reward), terminated, truncated, info

    # ------------------------------------------------------------------
    # Observation / reward helpers
    # ------------------------------------------------------------------
    def _compute_obs(self) -> np.ndarray:
        pos, forward, right, up, contact_valid = self._probe_frame()
        image, hit_masks, points, tissue_id = cast_slice(self.phantom, pos, forward, right, up, self._rng)
        self._last_image = image
        img_feats = extract_image_features(image)

        target = self.targets[self.target_idx] if self.target_idx < len(self.targets) else self.targets[-1]

        proprio = np.array([
            self.probe.theta / (np.pi / 2), self.probe.phi / (2 * np.pi),
            np.sin(np.radians(self.probe.roll)), np.cos(np.radians(self.probe.roll)),
            np.sin(np.radians(self.probe.pitch)), np.cos(np.radians(self.probe.pitch)),
            np.sin(np.radians(self.probe.yaw)), np.cos(np.radians(self.probe.yaw)),
            float(self.probe.fine_mode), float(contact_valid),
            self.total_steps / EPISODE_MAX_STEPS,
            1.0 - self.steps_in_subtask / self.subtask_max_steps,
        ], dtype=np.float32)

        onehot = np.zeros(3, dtype=np.float32)
        if target in TARGET_SEQUENCE:
            onehot[TARGET_SEQUENCE.index(target)] = 1.0
        context = np.concatenate([
            onehot,
            np.array([len(self.acquired) / 3.0], dtype=np.float32),
            np.array([(self.phantom.ga_weeks - cc.GA_MIN_WEEKS) / (cc.GA_MAX_WEEKS - cc.GA_MIN_WEEKS)], dtype=np.float32),
        ])

        self._cache = dict(hit_masks=hit_masks, target=target, image=image)
        obs = np.concatenate([img_feats, proprio, context]).astype(np.float32)
        return obs

    def _pose_error(self, target: str):
        """alpha = angle between the image slice-plane's NORMAL (the probe's
        elevational axis, i.e. `up` -- perpendicular to the 2D image plane
        spanned by `forward` and `right`) and the target anatomical plane's
        normal. d = perpendicular distance from the probe to that plane.

        NOTE: `cross(right, up)` is algebraically `-forward` (the probe's
        depth/imaging axis, which lies IN the image plane, not
        perpendicular to it) -- using that here was a bug caught by
        `scripts/validate_reward_field.py` Layer B (approach-monotonicity
        collapsed to 12-38% because the wrong axis was being compared).
        `up` is already the correct elevational axis and is unit-length by
        construction in `_probe_frame`.
        """
        pos, forward, right, up, _ = self._probe_frame()
        pt = self.phantom.plane_targets[target]
        probe_elevational = up / (np.linalg.norm(up) + 1e-9)
        cos_a = np.clip(abs(np.dot(probe_elevational, pt.normal)), -1.0, 1.0)
        alpha = float(np.arccos(cos_a))
        d = float(abs(np.dot(pos - pt.point, pt.normal)))
        return alpha, d

    def _potential(self, obs_cache: dict) -> float:
        target = obs_cache["target"]
        hit_masks = obs_cache["hit_masks"]
        required = REQUIRED_STRUCTURES.get(target, [])
        v = np.mean([structure_visibility(hit_masks, s) for s in required]) if required else 0.0
        v = min(v * 20.0, 1.0)  # sector coverage of a thin structure is naturally small
        alpha, d = self._pose_error(target)
        if self.shaping_mode == "multiplicative":
            return compute_potential_multiplicative(v, alpha, d)
        if self.shaping_mode == "hybrid":
            return compute_potential_hybrid(v, alpha, d, self.hybrid_weight)
        return compute_potential(v, alpha, d)

    def _measure(self, target: str) -> dict:
        b = self.phantom.biometry_mm
        noise = lambda v: v * float(self._rng.normal(1.0, 0.02))
        if target == "head":
            return {"BPD": noise(b["BPD"]), "HC": noise(b["HC"])}
        if target == "abdomen":
            return {"AC": noise(b["AC"])}
        if target == "femur":
            return {"FL": noise(b["FL"])}
        return {}

    def _finalize_clinical(self):
        bpd = self.acquired.get("head", {}).get("BPD", self.phantom.biometry_mm["BPD"])
        hc = self.acquired.get("head", {}).get("HC", self.phantom.biometry_mm["HC"])
        ac = self.acquired.get("abdomen", {}).get("AC", self.phantom.biometry_mm["AC"])
        fl = self.acquired.get("femur", {}).get("FL", self.phantom.biometry_mm["FL"])
        true = self.phantom.biometry_mm
        errs = [abs(bpd - true["BPD"]) / true["BPD"], abs(hc - true["HC"]) / true["HC"],
                abs(ac - true["AC"]) / true["AC"], abs(fl - true["FL"]) / true["FL"]]
        err_ok = all(e < 0.05 for e in errs)
        flag = cc.classify_growth(bpd, hc, ac, fl, self.phantom.ga_weeks)
        return err_ok, flag

    # ------------------------------------------------------------------
    def render(self):
        from environment.rendering import render_matplotlib
        return render_matplotlib(self)

    def close(self):
        pass
