from gym.envs.registration import register

register(
    id='MicropolisEnv-v0',
    entry_point='gym_micropolis.envs:MicropolisEnv',
)

register(
    id='MicropolisWalkEnv-v0',
    entry_point='gym_micropolis.envs:MicroWalkEnv',
)

register(
    id='MicropolisArcadeEnv-v0',
    entry_point='gym_micropolis.envs:MicroArcadeEnv',
)
