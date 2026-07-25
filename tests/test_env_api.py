import numpy as np

from environment.custom_env import UltrasoundProbeEnv


def test_reset_returns_valid_obs():
    env = UltrasoundProbeEnv(seed=0)
    obs, info = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    assert env.observation_space.contains(obs)
    assert isinstance(info, dict)


def test_step_runs_full_episode_without_crashing():
    env = UltrasoundProbeEnv(seed=1)
    env.reset(seed=1)
    for _ in range(180):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)
        assert np.isfinite(reward)
        if terminated or truncated:
            break
    assert terminated or truncated


def test_single_target_mode_terminates_on_one_acquisition():
    env = UltrasoundProbeEnv(seed=2, single_target=True)
    env.reset(seed=2)
    assert len(env.targets) == 1


def test_render_rgb_array():
    env = UltrasoundProbeEnv(seed=3, render_mode="rgb_array")
    env.reset(seed=3)
    env.step(env.action_space.sample())
    frame = env.render()
    assert frame is not None
    assert frame.ndim == 3 and frame.shape[2] == 3
