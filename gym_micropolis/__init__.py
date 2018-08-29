from gym.envs.registration import register

register(
    id='MicropolisEnv-v0',
    entry_point='gym_micropolis.envs:MicropolisEnv',
)

