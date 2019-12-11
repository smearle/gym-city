from gym.envs.registration import register

register(
    id='MicropolisEnv-v0',
    entry_point='gym_micropolis.envs:MicropolisEnv',
    kwargs={'MAP_X':14, 'MAP_Y':14}
)
register(
    id='MicropolisPaintEnv-v0',
    entry_point='gym_micropolis.envs:MicropolisPaintEnv',
    kwargs={'MAP_X':14, 'MAP_Y':14}
)
register(
    id='MicropolisWalkEnv-v0',
    entry_point='gym_micropolis.envs:MicroWalkEnv',
)

register(
    id='MicropolisArcadeEnv-v0',
    entry_point='gym_micropolis.envs:MicroArcadeEnv',
)
