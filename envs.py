import os
import sys

import gym
import numpy as np
import torch
from gym.spaces.box import Box

from baselines import bench
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.common.vec_env import VecEnvWrapper
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize as VecNormalize_

import csv
class MicropolisMonitor(bench.Monitor):
    def __init__(self, env, filename, allow_early_resets=True, reset_keywords=(), info_keywords=()):
        append_log = False # are we merging to an existing log file after pause in training?
        logfile = filename + '.monitor.csv'
        if os.path.exists(logfile):
            append_log = True
            old_log = '{}_old'.format(logfile)
            os.rename(logfile, old_log)
        super(MicropolisMonitor, self).__init__(env, filename, allow_early_resets=allow_early_resets, reset_keywords=reset_keywords, info_keywords=info_keywords)
        if append_log:
            with open(old_log, newline='') as old:
                reader = csv.DictReader(old, fieldnames=('r', 'l', 't'))
                h = 0
                for row in reader:
                    if h > 1:
                        row['t'] = 0.0001 * h # HACK: false times for past logs to maintain order
                        self.logger.writerow(row)
                    h += 1
            os.remove(old_log)


try:
    import dm_control2gym
except ImportError:
    pass

try:
    import roboschool
except ImportError:
    pass

try:
    import pybullet_envs
except ImportError:
    pass


def make_env(env_id, seed, rank, log_dir, add_timestep, allow_early_resets, map_width=20, render_gui=False, print_map=False, parallel_py2gui=False, noreward=False, max_step=None, simple_reward=False, args=None):
    def _thunk():
        if env_id.startswith("dm"):
            _, domain, task = env_id.split('.')
            env = dm_control2gym.make(domain_name=domain, task_name=task)
        else:
            env = gym.make(env_id)
            if 'gameoflife' in env_id.lower():
               #print("ENV RANK: ", rank)
                power_puzzle = False
               #if args.power_puzzle:
               #    power_puzzle = True
                if rank == 0:
                    render = render_gui
                else:render = False
                env.configure(map_width=map_width, render=render,
                        prob_life = args.prob_life)

            if 'micropolis' in env_id.lower():
                print("ENV RANK: ", rank)
                power_puzzle = False
                if args.power_puzzle:
                    power_puzzle = True
                if rank == 0:
                    render = render_gui
                else:render = False
                env.setMapSize(map_width, print_map=print_map, render_gui=render, empty_start=not args.random_terrain, max_step=max_step, rank=rank, simple_reward=simple_reward, power_puzzle=power_puzzle)

        is_atari = hasattr(gym.envs, 'atari') and isinstance(
            env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
        if is_atari:
            env = make_atari(env_id)
        env.seed(seed + rank)

        obs_shape = env.observation_space.shape

        if add_timestep and len(
                obs_shape) == 1 and str(env).find('TimeLimit') > -1:
            env = AddTimestep(env)

        if log_dir is not None:
            env = MicropolisMonitor(env, os.path.join(log_dir, str(rank)),
                                allow_early_resets=True)
            
           #print(log_dir)
           # 
           #print(type(env))
           #print(dir(env))
           #raise Exception

        if is_atari and len(env.observation_space.shape) == 3:
            env = wrap_deepmind(env)

        # If the input has shape (W,H,3), wrap for PyTorch convolutions
        obs_shape = env.observation_space.shape
        if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
            env = TransposeImage(env)

        return env

    return _thunk

def make_vec_envs(env_name, seed, num_processes, gamma, log_dir, add_timestep,
                  device, allow_early_resets, num_frame_stack=None, 
                  args=None):
    envs = [make_env(env_name, seed, i, log_dir, add_timestep, 
        allow_early_resets, map_width=args.map_width, render_gui=args.render, 
        print_map=args.print_map, noreward=args.no_reward, max_step=args.max_step, simple_reward=args.simple_reward, args=args)
            for i in range(num_processes)]

    if len(envs) > 1:
        envs = SubprocVecEnv(envs)
    else:
        if sys.version[0] =='2':
            envs = DummyVecEnv('DummyVecEnv', (), {1:envs})
        else:
            envs = DummyVecEnv(envs)

    if len(envs.observation_space.shape) == 1:
        if gamma is None:
            envs = VecNormalize(envs, ret=False)
        else:
            envs = VecNormalize(envs, gamma=gamma)

    envs = VecPyTorch(envs, device)

    if num_frame_stack is not None:
        print('stacking {} frames'.format(num_frame_stack))
        envs = VecPyTorchFrameStack(envs, num_frame_stack, device)
    elif len(envs.observation_space.shape) == 3:
        envs = VecPyTorchFrameStack(envs, 1, device)

    return envs


# Can be used to test recurrent policies for Reacher-v2
class MaskGoal(gym.ObservationWrapper):
    def observation(self, observation):
        if self.env._elapsed_steps > 0:
            observation[-2:0] = 0
        return observation


class AddTimestep(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(AddTimestep, self).__init__(env)
        self.observation_space = Box(
            self.observation_space.low[0],
            self.observation_space.high[0],
            [self.observation_space.shape[0] + 1],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        return np.concatenate((observation, [self.env._elapsed_steps]))


class TransposeImage(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(TransposeImage, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[0], obs_shape[1]],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        return observation.transpose(2, 0, 1)


class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        self.device = device
        # TODO: Fix data types

    def reset(self):
        obs = self.venv.reset()
        ### micropolis ###
        obs = np.array(obs)
        ### ########## ###
        obs = torch.from_numpy(obs).int().to(self.device)
        return obs

    def step_async(self, actions):
        actions_async = actions.squeeze(1).cpu().numpy()
        self.venv.step_async(actions_async)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        ### micropolis ###
        obs = np.array(obs)
        ### ########## ###
        obs = torch.from_numpy(obs).int().to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info


class VecNormalize(VecNormalize_):

    def __init__(self, *args, **kwargs):
        super(VecNormalize, self).__init__(*args, **kwargs)
        self.training = True

    def _obfilt(self, obs):
        if self.ob_rms:
            if self.training:
                self.ob_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def train(self):
        self.training = True

    def eval(self):
        self.training = False


# Derived from
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_frame_stack.py
class VecPyTorchFrameStack(VecEnvWrapper):
    def __init__(self, venv, nstack, device=None):
        self.venv = venv
        self.nstack = nstack

        wos = venv.observation_space  # wrapped ob space
        self.shape_dim0 = wos.shape[0]

        low = np.repeat(wos.low, self.nstack, axis=0)
        high = np.repeat(wos.high, self.nstack, axis=0)

        if device is None:
            device = torch.device('cpu')
        self.stacked_obs = torch.zeros((venv.num_envs,) + low.shape).to(device)

        observation_space = gym.spaces.Box(
            low=low, high=high, dtype=venv.observation_space.dtype)
        VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        self.stacked_obs[:, :-self.shape_dim0] = \
            self.stacked_obs[:, self.shape_dim0:]
        for (i, new) in enumerate(news):
            if new:
                self.stacked_obs[i] = 0
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs, rews, news, infos


    def reset(self):
        obs = self.venv.reset()
        self.stacked_obs.zero_()
       #print(self.stacked_obs.shape, obs.shape)
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs

    def close(self):
        self.venv.close()
