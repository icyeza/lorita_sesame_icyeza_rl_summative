from gymnasium.envs.registration import register

register(
    id="UltrasoundProbe-v0",
    entry_point="environment.custom_env:UltrasoundProbeEnv",
    max_episode_steps=180,
)
