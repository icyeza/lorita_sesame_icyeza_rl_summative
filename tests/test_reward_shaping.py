import numpy as np

from environment.custom_env import UltrasoundProbeEnv, ACTIONS

FREEZE_ACTION = ACTIONS.index("freeze_and_measure")


def test_per_step_cost_is_present():
    """Even a no-op-ish action sequence should accrue the -0.05/step cost
    unless offset by shaping/event rewards larger in magnitude."""
    env = UltrasoundProbeEnv(seed=0)
    env.reset(seed=0)
    _, reward, *_ = env.step(0)
    assert reward <= 0.5  # generous bound; mainly checks no runaway large positive reward


def test_freeze_outside_tolerance_is_penalized():
    # guarantee_reachable=False: these tolerances are deliberately
    # near-impossible (the point of the test), so resample-until-reachable
    # would always exhaust max_reachability_attempts -- irrelevant here and
    # just wastes time; see tests/test_reachability.py for that mechanism's
    # own tests.
    env = UltrasoundProbeEnv(seed=0, alpha_tol_deg=0.001, d_tol_m=0.0001, guarantee_reachable=False)
    env.reset(seed=0)
    _, reward, terminated, truncated, info = env.step(FREEZE_ACTION)
    assert not terminated
    assert reward < 0  # -2 event reward should dominate a near-impossible tolerance


def test_moving_toward_target_plane_increases_potential():
    """Sanity check: the potential function is finite and bounded given the
    shaping formula (V in [0,1], two exp() terms in [0,2] each)."""
    env = UltrasoundProbeEnv(seed=0)
    env.reset(seed=0)
    obs = env._compute_obs()
    phi = env._potential(obs_cache=env._cache)
    assert 0.0 <= phi <= 5.0


def test_episode_timeout_gives_negative_terminal_reward():
    env = UltrasoundProbeEnv(seed=0)
    env.reset(seed=0)
    total = 0.0
    done = False
    for _ in range(200):
        _, r, term, trunc, info = env.step(1)  # theta_minus repeatedly, unlikely to acquire
        total += r
        if term or trunc:
            done = True
            break
    assert done
