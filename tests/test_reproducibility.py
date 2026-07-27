import numpy as np

from environment.custom_env import UltrasoundProbeEnv


def _rollout(seed: int, n_steps: int = 15):
    env = UltrasoundProbeEnv(seed=seed)
    obs, _ = env.reset(seed=seed)
    action_rng = np.random.default_rng(seed)
    trace = [obs.copy()]
    rewards = []
    for _ in range(n_steps):
        a = int(action_rng.integers(0, env.action_space.n))
        obs, r, term, trunc, info = env.step(a)
        trace.append(obs.copy())
        rewards.append(r)
        if term or trunc:
            break
    return trace, rewards


def test_same_seed_gives_identical_rollout():
    trace_a, rewards_a = _rollout(123)
    trace_b, rewards_b = _rollout(123)
    assert len(trace_a) == len(trace_b)
    for oa, ob in zip(trace_a, trace_b):
        assert np.allclose(oa, ob)
    assert np.allclose(rewards_a, rewards_b)


def test_different_seed_gives_different_rollout():
    trace_a, _ = _rollout(1)
    trace_b, _ = _rollout(2)
    assert not all(np.allclose(oa, ob) for oa, ob in zip(trace_a, trace_b))
