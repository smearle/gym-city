from gym.envs.registration import register
import gym_micropolis.envs.tilemap
import gym_micropolis.envs.corecontrol
import gym_micropolis.envs.env

register(
    id='MicropolisEnv-v0',
    entry_point='gym_micropolis.envs:MicropolisEnv',
)

