from gym.envs.registration import registry, register, make, spec

register(
    id='Swimmer-v2',
    entry_point='envs.mujoco.swimmer_env:SwimmerEnv_rllab',
    max_episode_steps=1000,
    reward_threshold=360.0,
)

register(
    id='Hopper-v2',
    entry_point='envs.mujoco.swimmer_env:HopperEnv_rllab',
    max_episode_steps=1000,
    reward_threshold=3800.0,
)

register(
    id='Ant-v2',
    entry_point='envs.mujoco:AntEnv_rllab',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

register(
    id='InvertedPendulum-v2',
    entry_point='envs.mujoco:InvertedPendulumEnv_rllab',
    max_episode_steps=1000,
    reward_threshold=950.0,
)

register(
    id='Walker2d-v2',
    max_episode_steps=1000,
    entry_point='envs.mujoco:Walker2dEnv_rllab',
)
