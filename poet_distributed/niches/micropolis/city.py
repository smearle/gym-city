# Copyright (c) 2019 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from ..core import Niche
from .model import FractalModel, simulate
from .env import micropolis_custom, Env_config
from collections import OrderedDict

DEFAULT_ENV = Env_config(
        name='default_env',
        res_pop = 1,
        com_pop = 0,
        ind_pop = 0,
        traffic = 0,
        num_plants = 0,
        mayor_rating = 0)

class CityNiche(Niche):
    def __init__(self, env_configs, seed, init='random', stochastic=False,
            args=None, render=False):
        self.model = FractalModel(micropolis_custom)
        if not isinstance(env_configs, list):
            env_configs = [env_configs]
        self.env_configs = OrderedDict()
        for env in env_configs:
            self.env_configs[env.name] = env
        self.seed = seed
        self.stochastic = stochastic
        self.model.make_env(seed=seed, env_config=DEFAULT_ENV, args=args,
                render_mode=render)
        self.init = init

    def render(self):
        pass
       #self.model.env.render_gui=True

    def add_env(self, env):
        env_name = env.name
        assert env_name not in self.env_configs.keys()
        self.env_configs[env_name] = env

    def delete_env(self, env_name):
        assert env_name in self.env_configs.keys()
        self.env_configs.pop(env_name)

    def initial_theta(self):
        if self.init == 'random':
            return self.model.get_random_model_params()
       #elif self.init == 'zeros':
       #    import numpy as np
       #    return np.zeros(self.model.param_count)
        else:
            raise NotImplementedError(
                'Undefined initialization scheme `{}`'.format(self.init))

    def rollout(self, theta, random_state, eval=False):
       #print(len(theta))
        self.model.set_model_params(theta)
        total_returns = 0
        total_length = 0
        if self.stochastic:
            seed = random_state.randint(1000000)
        else:
            seed = self.seed
        for env_config in self.env_configs.values():
            returns, lengths = simulate(
                self.model, seed=seed, train_mode=not eval, num_episode=1, env_config_this_sim=env_config)
            total_returns += returns[0]
            total_length += lengths[0]
        return total_returns / len(self.env_configs), total_length
