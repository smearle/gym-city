import numpy as np
import gym
from baselines.common.vec_env import VecEnv

class DummyVecEnv(VecEnv):
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space)

        obs_spaces = self.observation_space.spaces if isinstance(self.observation_space, gym.spaces.Tuple) else (self.observation_space,)
        self.buf_obs = [np.zeros((self.num_envs,) + tuple(s.shape), s.dtype) for s in obs_spaces]
        self.buf_dones = np.zeros((self.num_envs,), dtype=np.bool)
        self.buf_rews  = np.zeros((self.num_envs,), dtype=np.float32)
        self.buf_infos = [{} for _ in range(self.num_envs)]
        self.actions = None

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        for i in range(self.num_envs):
            obs_tuple, self.buf_rews[i], self.buf_dones[i], self.buf_infos[i] = self.envs[i].step(self.actions[i])
            if isinstance(obs_tuple, (tuple, list)):
                for t,x in enumerate(obs_tuple):
                    self.buf_obs[t][i] = x
            else:
                self.buf_obs[0][i] = obs_tuple
        return self.buf_obs, self.buf_rews, self.buf_dones, self.buf_infos

    def reset(self):
        for i in range(self.num_envs):
            obs_tuple = self.envs[i].reset()
            if isinstance(obs_tuple, (tuple, list)):
                for t,x in enumerate(obs_tuple):
                    self.buf_obs[t][i] = x
            else:
                self.buf_obs[0][i] = obs_tuple
        return self.buf_obs


    def get_param_bounds(self):
        return self.envs[0].get_param_bounds()


    def set_param_bounds(self, bounds):
        for env in self.envs:
            return env.set_param_bounds(bounds)


    def set_params(self, params):
        for env in self.envs:
            return env.set_params(params)

    def setMapSize(self, size, **kwargs):
        for env in self.envs:
            env.setMapSize(size, **kwargs)

    def set_extinction_type(self, ext_type, ext_prob):
        for env in self.envs:
            env.set_extinction_type(ext_type, ext_prob)

    def reset_episodes(self):
        for env in self.envs:
            env.reset_episodes()


    def close(self):
        return
