import numpy as np
import gym
from stable_baselines3.common.vec_env import DummyVecEnv

class DDummyVecEnv(DummyVecEnv):
    '''Deals with multiple environments sequentially. With extras.'''
    #TODO: test with multiple envs
    def __init__(self, env_fns):
        DummyVecEnv.__init__(self, env_fns)
        print('obs and act spaces on DDummyVecEnv init: {}, {}'.format(self.observation_space, self.action_space))
        self.init_storage()

    def init_storage(self):
        obs_spaces = self.observation_space.spaces if isinstance(self.observation_space, gym.spaces.Tuple) else (self.observation_space,)
        self.buf_obs = [np.zeros((self.num_envs,) + tuple(s.shape), s.dtype) for s in obs_spaces]
        self.buf_dones = np.zeros((self.num_envs,), dtype=np.bool)
        self.buf_rews  = np.zeros((self.num_envs,), dtype=np.float32)
        self.buf_infos = [{} for _ in range(self.num_envs)]
        self.actions = None

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        if self.buf_dones[0]:
            self.reset()
        for i in range(self.num_envs):
            if isinstance(self.actions, dict):
                actions = {}
                for act_name, action in self.actions.items():
                    actions[act_name] = action[i]
            else:
                actions = self.actions[i]
            obs_tuple, self.buf_rews[i], self.buf_dones[i], self.buf_infos[i] = self.envs[i].step(actions)
            if isinstance(obs_tuple, (tuple, list)):
                for t,x in enumerate(obs_tuple):
                    self.buf_obs[t][i] = x
            else:
                self.buf_obs[0][i] = obs_tuple
        return self.buf_obs, self.buf_rews, self.buf_dones, self.buf_infos

    def reset(self):
        for i in range(self.num_envs):
            obs_tuple = self.envs[i].reset()[0]
            if isinstance(obs_tuple, (tuple, list)):
                for t,x in enumerate(obs_tuple):
                    self.buf_obs[t][i] = x
            else:
                try:
                    self.buf_obs[0][i] = obs_tuple
                # Hack to dunamically change map size
                except:
                    map_width = self.envs[0].width
                    self.buf_obs = [np.zeros((self.num_envs,) + (self.envs[0].num_obs_channels, map_width,
                        map_width))]
                self.buf_obs[0][i] = obs_tuple
        return self.buf_obs

    def get_spaces(self):
        for env in self.envs:
            return env.get_spaces()

    def render(self, mode=None):
        for env in self.envs:
            env.render(mode=mode)

    def get_param_bounds(self):
        return self.envs[0].get_param_bounds()

    def set_active_agent(self, n_agent):
        for env in self.envs:
            env.set_active_agent(n_agent)

    def set_param_bounds(self, bounds):
        for env in self.envs:
            env.set_param_bounds(bounds)
        return len(bounds)

    def set_trgs(self, trgs):
        for env in self.envs:
            env.set_trgs(trgs)

    def get_param_trgs(self):
        return self.envs[0].get_param_trgs()

    def configure(self, **kwargs):
        print('configurin dummy')
        for env in self.envs:
            env.configure(**kwargs)

    def set_save_dir(self, save_dir):
        for env in self.envs:
            envs.set_save_dir(save_dir)

    def set_map(self, map):
        for env in self.envs:
            env.set_map(map)

    def set_extinction_type(self, *args):
        for env in self.envs:
            env.set_extinction_type(*args)

    def reset_episodes(self, im_log_dir):
        for env in self.envs:
            env.reset_episodes(im_log_dir)

    def set_log_dir(self, log_dir):
        for env in self.envs:
            env.im_log_dir = log_dir

    def close(self):
        return
