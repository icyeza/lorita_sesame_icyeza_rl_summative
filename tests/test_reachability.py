"""Tests for the guarantee_reachable resample-until-reachable fix.

`UltrasoundProbeEnv(guarantee_reachable=True)` (the default) resamples the
fetal pose in `reset()` until every active target is reachable by the exact
same criterion `step()` uses (`_pose_error` against the env's own
alpha_tol/d_tol). These tests only exercise pose sampling -- they do not
touch phantom geometry, features, or reward math.

HISTORY: an earlier pass found the raw (unfiltered) reachable fraction was
only ~45% (head ~55%), which `guarantee_reachable=True` resampled around.
A later audit (`scripts/audit_reachability_search.py`) proved that number
was a MEASUREMENT ARTIFACT: `_search_min_pose_error`'s blind 5D-joint
Nelder-Mead restarts frequently failed to find solutions that provably
exist (confirmed by a fine-grid sweep finding 20/20 sampled "unreachable"
head cases were actually reachable). The search was fixed (position-first,
then tilt-polish -- see that method's docstring), and the TRUE unfiltered
reachable fraction, re-measured at N=500 across four actuator cone widths,
is 1.000 for every target. So `guarantee_reachable=True` now almost never
needs to resample at all -- see `logs/reachability/cone_sweep_summary.json`
for the real numbers behind this.
"""
import numpy as np

from environment.custom_env import UltrasoundProbeEnv, TARGET_SEQUENCE

N_FULL_TASK_EPISODES = 15
N_SINGLE_TARGET_EPISODES = 15
N_UNFILTERED_SAMPLES = 40


def test_guarantee_reachable_true_yields_all_reachable_full_task():
    env = UltrasoundProbeEnv(seed=0, guarantee_reachable=True)
    for i in range(N_FULL_TASK_EPISODES):
        env.reset(seed=100 + i)
        assert not env.reachability_capped, f"episode {i}: hit the resample cap unexpectedly"
        for target in env.targets:
            assert env._is_target_reachable(target), (
                f"episode {i}: target {target!r} not reachable despite guarantee_reachable=True"
            )


def test_guarantee_reachable_true_yields_all_reachable_single_target():
    env = UltrasoundProbeEnv(seed=0, guarantee_reachable=True, single_target=True)
    for i in range(N_SINGLE_TARGET_EPISODES):
        env.reset(seed=200 + i)
        assert len(env.targets) == 1
        assert not env.reachability_capped
        assert env._is_target_reachable(env.targets[0])


def test_reachability_predicate_is_not_a_noop():
    """Guards against a no-op predicate that trivially returns True always.
    Can no longer rely on the natural (unfiltered) distribution containing
    failures -- the corrected search measures that fraction at 1.000 (see
    module docstring), which is itself the point: that's real, not a no-op
    artifact, since a genuinely-impossible tolerance below DOES still fail
    (proving the predicate can and does return False when warranted)."""
    env = UltrasoundProbeEnv(seed=0, guarantee_reachable=False,
                              alpha_tol_deg=0.001, d_tol_m=0.0001)
    any_failure = False
    for i in range(N_UNFILTERED_SAMPLES):
        env.reset(seed=300 + i)
        for target in TARGET_SEQUENCE:
            if not env._is_target_reachable(target):
                any_failure = True
                break
        if any_failure:
            break
    assert any_failure, (
        "the reachability predicate reported EVERY target reachable even under a "
        "near-impossible (0.001deg, 0.1mm) tolerance -- it appears to be a no-op "
        "that always returns True"
    )


def test_reachability_is_near_universal_on_the_real_distribution():
    """The corrected finding (replaces the old ~45%/55% numbers): on the
    REAL unfiltered distribution with the env's default tolerances, nearly
    every sampled (phantom, target) is reachable. A modest N here (this
    test needs to stay fast); the authoritative N=500 x 4-cone-width sweep
    lives in logs/reachability/cone_sweep_summary.json (all 1.000)."""
    env = UltrasoundProbeEnv(seed=0, guarantee_reachable=False)
    n_reachable = 0
    n_total = 0
    for i in range(N_UNFILTERED_SAMPLES):
        env.reset(seed=300 + i)
        for target in TARGET_SEQUENCE:
            n_total += 1
            if env._is_target_reachable(target):
                n_reachable += 1
    fraction = n_reachable / n_total
    assert fraction >= 0.9, (
        f"reachable fraction {fraction:.3f} on {n_total} (phantom, target) samples is well "
        f"below the ~1.0 the corrected search measures at N=500 -- possible regression"
    )


def test_reachability_filtering_is_reproducible():
    """Same seed -> identical sampled (phantom pose, probe start pose,
    targets) with guarantee_reachable=True, including identical resample
    behavior (same number of attempts)."""
    env_a = UltrasoundProbeEnv(seed=0, guarantee_reachable=True)
    obs_a, _ = env_a.reset(seed=777)
    state_a = (env_a.phantom.ga_weeks, env_a.phantom.fetal_position.tolist(),
               env_a.probe.theta, env_a.probe.phi, tuple(env_a.targets),
               env_a.reachability_attempts_used)

    env_b = UltrasoundProbeEnv(seed=0, guarantee_reachable=True)
    obs_b, _ = env_b.reset(seed=777)
    state_b = (env_b.phantom.ga_weeks, env_b.phantom.fetal_position.tolist(),
               env_b.probe.theta, env_b.probe.phi, tuple(env_b.targets),
               env_b.reachability_attempts_used)

    assert state_a == state_b
    assert np.allclose(obs_a, obs_b)


def test_single_target_mode_only_gates_on_active_target():
    """In single_target mode, only the ONE active target needs to be
    reachable -- the other two targets on the same accepted phantom are
    allowed to be unreachable. This test sets a tiny resample cap and a
    single fixed target across several seeds; if the predicate incorrectly
    required all three targets to be reachable, single_target episodes
    would hit the resample cap far more often than full-task episodes at
    the same cap (since requiring 3-of-3 is a strictly harder condition
    than 1-of-1)."""
    single_capped = 0
    full_capped = 0
    trials = 20
    tiny_cap = 3

    for i in range(trials):
        env_single = UltrasoundProbeEnv(seed=0, guarantee_reachable=True, single_target=True,
                                         max_reachability_attempts=tiny_cap)
        env_single.reset(seed=400 + i)
        single_capped += int(env_single.reachability_capped)

        env_full = UltrasoundProbeEnv(seed=0, guarantee_reachable=True, single_target=False,
                                       max_reachability_attempts=tiny_cap)
        env_full.reset(seed=400 + i)
        full_capped += int(env_full.reachability_capped)

    assert single_capped <= full_capped, (
        f"single_target hit the resample cap {single_capped}/{trials} times vs "
        f"full_task's {full_capped}/{trials} -- single_target should be at least as "
        f"easy to satisfy (1 target) as full_task (3 targets), suggesting "
        f"single_target mode isn't actually gating on just the active target"
    )


CONE_MONOTONICITY_WIDTHS_DEG = [60.0, 70.0, 80.0]
N_MONOTONICITY_SAMPLES = 100
# Widening the cone strictly enlarges the search space `_is_target_reachable`
# optimizes over, so reachability is mathematically monotonic non-decreasing
# in cone width. In practice the search uses a fixed-seed restart-point set
# that itself shifts slightly as the bounds widen (see
# `_search_min_pose_error`), so a tiny bit of optimizer noise is possible --
# this tolerance is for THAT, not for a real regression.
MONOTONICITY_TOLERANCE = 0.03


def test_reachable_fraction_monotonically_nondecreasing_with_cone_width():
    """Sanity guard on the actuator_limit_deg parameterization (see
    scripts/sweep_actuator_cone.py): widening the cone must not make
    reachability WORSE. Uses head (the target most sensitive to cone width)
    at a modest sample size to stay fast; the authoritative N=500 sweep
    numbers live in logs/reachability/cone_sweep_summary.json."""
    fractions = []
    for cone in CONE_MONOTONICITY_WIDTHS_DEG:
        env = UltrasoundProbeEnv(seed=0, guarantee_reachable=False, actuator_limit_deg=cone)
        n_reachable = 0
        for i in range(N_MONOTONICITY_SAMPLES):
            env.reset(seed=500 + i)
            if env._is_target_reachable("head"):
                n_reachable += 1
        fractions.append(n_reachable / N_MONOTONICITY_SAMPLES)

    for i in range(1, len(fractions)):
        assert fractions[i] >= fractions[i - 1] - MONOTONICITY_TOLERANCE, (
            f"reachable fraction DECREASED from {fractions[i-1]:.3f} at "
            f"{CONE_MONOTONICITY_WIDTHS_DEG[i-1]}deg to {fractions[i]:.3f} at "
            f"{CONE_MONOTONICITY_WIDTHS_DEG[i]}deg (beyond the {MONOTONICITY_TOLERANCE} "
            f"optimizer-noise tolerance) -- widening the cone should never hurt "
            f"reachability; this suggests actuator_limit_deg isn't correctly wired "
            f"into the reachability search bounds"
        )
