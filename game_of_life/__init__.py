from gym.envs.registration import register

register(
    id='GameOfLifeEnv-v0',
    entry_point='game_of_life.envs:GameOfLifeEnv'
)

register(
    id='GoLMultiEnv-v0',
    entry_point='game_of_life.envs:GoLMultiEnv'
)
