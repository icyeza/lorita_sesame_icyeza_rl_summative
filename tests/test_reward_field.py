"""Phase-1 correctness gate, as an assertable test.

Wraps `scripts/validate_reward_field.py` (see its module docstring for the
full rationale, including the axis bug it caught and fixed, and why the
gate is calibrated on IMPROVEMENT rather than tolerance-hit-rate for the
greedy discrete-action check).

Gates:
  - Layer A: potential formula must be exactly decreasing in alpha/d and
    increasing in V (pure math, no tolerance).
  - Layer B endpoint: a bounded (theta, phi, roll, pitch, yaw) optimizer
    must reach near-zero alpha/d for most sampled phantoms per target
    (some phantoms are legitimately unreachable within the +-60deg
    actuator range for a given target -- see status.md -- so this is a
    majority-fraction check, not 100%).
  - Layer B symmetry: alpha must be EXACTLY invariant to a 180-degree flip
    of the probe's elevational axis, for every sampled phantom (this one is
    pure math via abs() and should never fail once correct).
  - Layer B greedy-improvement: greedy discrete-action hill-climbing from a
    random start must make the combined (alpha, d) error meaningfully
    smaller on average, and improve on most individual paths -- this is
    the bug-indicative signal (does the field point the right way at all),
    distinct from whether naive one-step-greedy fully SOLVES the task
    (it often doesn't -- see status.md; that's a task-difficulty finding
    for Phase 2 to test empirically with real RL algorithms, not a bug).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from environment.custom_env import UltrasoundProbeEnv, TARGET_SEQUENCE
from scripts.validate_reward_field import (
    layer_a_formula_monotonicity, layer_b_endpoint_and_symmetry,
    layer_b_greedy_discrete_walk, K_PHANTOMS, K_GREEDY_PHANTOMS,
)

MIN_ENDPOINT_REACHABLE_FRACTION = 0.6
# improved_fraction is the primary, more reliable bug-catching signal here
# (see the per-target ratio thresholds below for why the ratio's magnitude
# varies a lot by target for legitimate reasons). A true sign-flip/wrong-
# axis bug drives this toward 0; a healthy field keeps it high regardless
# of target-specific landscape shape.
MIN_GREEDY_IMPROVED_FRACTION = 0.6

# --- Distance-shaping fix (environment/custom_env.py::POTENTIAL_D_SCALE
# raised 0.015 -> 0.05) and what it changed here ---
#
# HISTORY: an earlier pass found head/abdomen's improvement ratios weak
# (~0.06-0.10) under `POTENTIAL_D_SCALE=0.015` (15mm) and RE-LOWERED this
# test's threshold to 0.03 to accommodate it, reasoning it was "a genuine,
# pre-existing property of the fixed exponential scales... out of scope to
# change." That reasoning was WRONG in a consequential way: the same
# 15mm-flat-gradient property went on to wall an actual PPO calibration
# run at reward ~-6 (median terminal alpha 0.30deg -- solved -- but median
# terminal d 59mm vs a 12mm tolerance -- never solved -- and the agent
# never once attempted `freeze`, 2/3336 episodes succeeded). Lowering this
# test's threshold papered over a bug that then broke real training.
#
# THIS TIME: the fix is to the shaping SCALE itself (0.015 -> 0.05, see
# custom_env.py), not the gate. The pass condition is RATIO RECOVERY
# toward the healthy band the field's other well-behaved cases already
# showed (~0.33-0.39), not merely clearing a lowered number. Measured
# after the fix (real runs, phantom_seeds=[9000,9001,9002], not
# fabricated):
#   femur:   0.48 -> 0.72   RECOVERED (comfortably exceeds the healthy band)
#   abdomen: 0.10 -> 0.19   PARTIALLY RECOVERED (better, still below 0.33-0.39)
#   head:    0.06 -> 0.06   NOT RECOVERED
#
# head was ALSO tested at sigma=0.07 and sigma=0.10 (informational, not
# committed) to check whether a larger distance scale would help further --
# it made head WORSE, not better (ratio 0.106 -> 0.054 -> 0.040 as sigma
# rose from 0.05 to 0.10, using a different rng draw than the table above,
# so magnitudes differ but the direction is consistent and clear). This
# rules out "just raise sigma more" for head. head's ratio therefore
# remains a genuine OPEN FINDING, not swept under a lowered gate: given its
# ~90-100mm typical operating distance is simply farther than abdomen's
# (~30-50mm) or femur's (~50mm) and a single global sigma cannot serve all
# three ranges at once, the likely next step is a genuinely PER-TARGET
# distance scale (or a different mechanism entirely) -- future work, not
# assumed here.
#
# --- Environment LOCK (status.md "lock the environment" pass): shaping_mode
# default flipped additive -> "multiplicative" -- this test constructs
# UltrasoundProbeEnv(seed=0) with no override, so it now exercises the
# LOCKED (multiplicative) field, not the additive one the numbers above
# were measured under. head's "documented open issue" status is RESOLVED
# by this lock, not just papered over: multiplicative coupling was
# measured (real runs, phantom_seeds=[9000,9001,9002]) to fix all three
# targets' improvement ratios, most dramatically the two that additive
# shaping never recovered:
#   head:    0.06 (additive, NOT RECOVERED) -> 0.78 (multiplicative)
#   abdomen: 0.19 (additive, PARTIALLY RECOVERED) -> 0.92 (multiplicative)
#   femur:   0.72 (additive, RECOVERED) -> 0.83 (multiplicative, held/improved)
# Thresholds below are raised accordingly -- comfortably below the
# measured multiplicative values (so real per-phantom sampling variance
# doesn't flake the gate) but tight enough to catch a real regression back
# toward additive-era numbers. head no longer gets a separately-labeled,
# lower "known open issue" bar -- it clears the same bar abdomen/femur do.
MIN_GREEDY_IMPROVEMENT_RATIO_RECOVERED = 0.6  # all three targets, under locked multiplicative shaping


def test_layer_a_potential_formula_monotonicity():
    result = layer_a_formula_monotonicity()
    assert result["decreasing_in_alpha"]
    assert result["decreasing_in_d"]
    assert result["increasing_in_v"]
    assert result["passed"]


def test_layer_b_endpoint_and_symmetry():
    env = UltrasoundProbeEnv(seed=0)
    for target in TARGET_SEQUENCE:
        results = []
        for k in range(K_PHANTOMS):
            env.reset(seed=100 + k)
            results.append(layer_b_endpoint_and_symmetry(env, target))

        endpoint_fraction = sum(r["endpoint_ok"] for r in results) / K_PHANTOMS
        symmetry_fraction = sum(r["symmetry_ok"] for r in results) / K_PHANTOMS

        assert endpoint_fraction >= MIN_ENDPOINT_REACHABLE_FRACTION, (
            f"{target}: only {endpoint_fraction:.2f} of phantoms had a reachable "
            f"near-zero-error pose within the actuator's bounds"
        )
        assert symmetry_fraction == 1.0, (
            f"{target}: plane-symmetry (alpha invariant to 180deg elevational-axis "
            f"flip) failed for {1 - symmetry_fraction:.0%} of phantoms -- likely a "
            f"missing/misapplied abs() in _pose_error"
        )


def test_layer_b_greedy_walk_shows_real_improvement():
    """Averages over K_GREEDY_PHANTOMS distinct phantoms (not one fixed
    phantom) -- a single phantom made this fragile to per-phantom sampling
    variance. See scripts/validate_reward_field.py's K_GREEDY_PHANTOMS
    docstring note, and this file's module-level comment for why the
    LOCKED multiplicative shaping now clears a single, shared threshold
    for all three targets (head is no longer a documented open issue --
    see the lock pass in status.md)."""
    env = UltrasoundProbeEnv(seed=0)
    rng = np.random.default_rng(7)
    phantom_seeds = [9000 + k for k in range(K_GREEDY_PHANTOMS)]
    for target in TARGET_SEQUENCE:
        result = layer_b_greedy_discrete_walk(env, target, rng, phantom_seeds=phantom_seeds)
        assert result["improved_fraction"] >= MIN_GREEDY_IMPROVED_FRACTION, (
            f"{target}: only {result['improved_fraction']:.2f} of greedy walks improved "
            f"the combined pose error -- the potential field may point the wrong way"
        )
        assert result["mean_improvement_ratio"] >= MIN_GREEDY_IMPROVEMENT_RATIO_RECOVERED, (
            f"{target}: mean combined-error improvement ratio "
            f"{result['mean_improvement_ratio']:.2f} below {MIN_GREEDY_IMPROVEMENT_RATIO_RECOVERED} "
            f"-- a regression from the locked multiplicative-shaping numbers in status.md"
        )
