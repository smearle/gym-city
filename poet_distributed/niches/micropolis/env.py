# The following code is modified from hardmaru/estool (https://github.com/hardmaru/estool/) under the MIT License.

# Modifications Copyright (c) 2019 Uber Technologies, Inc.


from collections import namedtuple
# import gym
from .micropolis_custom import MicropolisCustom, Env_config


def make_env(env_name, seed, render_mode=False, env_config=None):
    if env_name.startswith("MicropolisEnv-v0"):
        env = MicropolisCustom(env_config)
    else:
        # env = gym.make(env_name)
        raise Exception('Got env_name {}'.format(env_name))
    if render_mode and not env_name.startswith("Roboschool"):
        env.render("human")
    if (seed >= 0):
        env.seed(seed)

    # print("environment details")
    # print("env.action_space", env.action_space)
    # print("high, low", env.action_space.high, env.action_space.low)
    # print("environment details")
    # print("env.observation_space", env.observation_space)
    # print("high, low", env.observation_space.high, env.observation_space.low)
    # assert False

    return env


Game = namedtuple('Game', ['env_name', 'num_inputs', 'map_width',
                    'num_actions', 'rule', 'in_w', 'in_h', 'out_w', 'out_h',
                    'n_recs'])

micropolis_custom = Game(env_name='MicropolisEnv-v0',
                        num_inputs=35,
                        map_width=16,
                        num_actions=19,
                        rule='extend',
                        n_recs=5,
                        in_w=1, in_h=1, out_w=1, out_h=1)
