from gymnasium.envs.registration import register

register(
    id="Shepherding-v0",
    entry_point="shepherding.envs:ShepherdingEnv",
    max_episode_steps=2000,
)

register(
    id="Shepherding-v1",
    entry_point="shepherding.envs:ShepherdingEnvNew",
    max_episode_steps=2000,
)