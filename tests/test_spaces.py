from gymnasium import spaces

from environment.custom_env import UltrasoundProbeEnv, ACTIONS


def test_action_space_is_discrete_12():
    env = UltrasoundProbeEnv(seed=0)
    assert isinstance(env.action_space, spaces.Discrete)
    assert env.action_space.n == 12
    assert len(ACTIONS) == 12


def test_observation_space_is_box_and_excludes_target_pose():
    env = UltrasoundProbeEnv(seed=0)
    assert isinstance(env.observation_space, spaces.Box)
    obs, _ = env.reset(seed=0)
    assert obs.shape[0] == env.observation_space.shape[0]
    # the ground-truth target plane pose must never leak into the observation
    assert env.phantom.plane_targets["head"].point.shape == (3,)
    assert not any(
        abs(v) > 4.99 for v in env.phantom.plane_targets["head"].point
    ), "sanity: target pose values should not trivially appear at obs bounds"


def test_all_actions_are_applicable():
    env = UltrasoundProbeEnv(seed=0)
    env.reset(seed=0)
    for a in range(env.action_space.n):
        env.reset(seed=0)
        obs, reward, term, trunc, info = env.step(a)
        assert env.observation_space.contains(obs)
