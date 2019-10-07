# The following code is modified from openai/gym (https://github.com/openai/gym) under the MIT License.

# Modifications Copyright (c) 2019 Uber Technologies, Inc.


import sys
import math
import numpy as np

import gym_micropolis
from gym_micropolis.envs.env import MicropolisEnv

import gym
from gym import spaces
from gym.utils import colorize, seeding
from collections import namedtuple

# This is simple 4-joints walker robot environment.
#
# There are two versions:
#
# - Normal, with slightly uneven terrain.
#
# - Hardcore with ladders, stumps, pitfalls.
#
# Reward is given for moving forward, total 300+ points up to the far end. If the robot falls,
# it gets -100. Applying motor torque costs a small amount of points, more optimal agent
# will get better score.
#
# Heuristic is provided for testing, it's also useful to get demonstrations to
# learn from. To run heuristic:
#
# python gym/envs/box2d/bipedal_walker.py
#
# State consists of hull angle speed, angular velocity, horizontal speed, vertical speed,
# position of joints and joints angular speed, legs contact with ground, and 10 lidar
# rangefinder measurements to help to deal with the hardcore version. There's no coordinates
# in the state vector. Lidar is less useful in normal version, but it works.
#
# To solve the game you need to get 300 points in 1600 time steps.
#
# To solve hardcore version you need 300 points in 2000 time steps.
#
# Created by Oleg Klimov. Licensed on the same terms as the rest of OpenAI Gym.

Env_config = namedtuple('Env_config', [
    'name',
    'res_pop', 'com_pop', 'ind_pop',
    'traffic', 'num_plants', 'mayor_rating'
])

class MicropolisCustom(MicropolisEnv):
    metadata = {}

    def __init__(self, env_config):
        super(MicropolisCustom, self).__init__()
        self.set_env_config(env_config)


    def set_env_config(self, env_config):
        self.config = env_config
        self.set_city_trgs(env_config)


    def save_env_def(self, filename):
        import json
        a = {'config': self.config._asdict(), 'seed': self.env_seed}
        with open(filename, 'w') as f:
            json.dump(a, f)
