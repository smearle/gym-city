import csv
import os
import sys
import time

import gym
import numpy as np
import torch
from pathlib import Path
from gym.spaces.box import Box

from baselines import bench
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.common.vec_env import VecEnvWrapper
from baselines.common.vec_env.vec_normalize import \
    VecNormalize as VecNormalize_
#from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from dummy_vec_env import DDummyVecEnv as DummyVecEnv
#from gym_city.wrappers import ImRenderMicropolis
#from gym_pcgrl.envs.play_pcgrl_env import PlayPcgrlEnv
#from gym_pcgrl.wrappers import ActionMapImagePCGRLWrapper, MaxStep
#from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from subproc_vec_env import SubprocVecEnv
from wrappers import ParamRewMulti, ParamRew, ExtinguisherMulti, Extinguisher, ImRenderMulti, ImRender #NoiseyTargets, 
#Griddly



class MicropolisMonitor(bench.Monitor):
    def __init__(self, env, filename, allow_early_resets=False, reset_keywords=(), info_keywords=()):
        self.env = env
       #self.width = self.unwrapped.width
        self.dist_entropy = 0
        append_log = False # are we merging to an existing log file after pause in training?
        logfile = filename + '.monitor.csv'
        curr_dir = os.curdir
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        try:
            try:
                path = Path(filename).parent.parent.parent
                os.mkdir(path)
            except:
                pass
            path = Path(filename).parent.parent
            os.mkdir(path)
            path = Path(filename).parent
            os.mkdir(path)
        except:
            pass

        if os.path.exists(logfile):
            append_log = True
            old_log = '{}_old'.format(logfile)
            os.rename(logfile, old_log)
        else:
            print('no old logfile {}'.format(logfile))
           #raise Exception
        info_keywords = (*info_keywords, 'e', 'p')
        super(MicropolisMonitor, self).__init__(
            env, filename, allow_early_resets=allow_early_resets, reset_keywords=reset_keywords,
            info_keywords=info_keywords)

        if append_log:
            with open(old_log, newline='') as old:
                reader = csv.DictReader(old, fieldnames=('r', 'l', 't','e', 'p'))
                h = 0

                for row in reader:
                    if h > 1:
                        row['t'] = 0.0001 * h # HACK: false times for past logs to maintain order
                        # TODO: logger or results_writer, what's going on here?

                        if hasattr(self, 'logger'):
                            self.logger.writerow(row)
                            self.f.flush()
                        else:
                            assert hasattr(self, 'results_writer')
                            self.results_writer.write_row(row)
                           #self.results_writer.flush()
                    h += 1
            os.remove(old_log)
            os.chdir(curr_dir)

    def step(self, action):
        if self.needs_reset:
            raise RuntimeError("Tried to step environment that needs reset")
        ob, rew, done, info = self.env.step(action)
        self.rewards.append(rew)

        if done:
            self.needs_reset = True
            eprew = float(sum(self.rewards))
            eplen = len(self.rewards)
            dist_entropy = self.dist_entropy
            if dist_entropy is None:
                dist_entropy = 0
            if eprew is None:
                eprew = 0
            epinfo = {"r": round(eprew, 6), "l": eplen, "t": round(time.time() - self.tstart, 6),
                    "e": round(dist_entropy, 6)}

            if "p" in epinfo.keys():
                epinfo["p"] = round(self.curr_param_vals[0].item(), 6)

            for k in self.info_keywords:
                if False and k != 'e' and k!= 'p':
                    epinfo[k] = info[k]
            self.episode_rewards.append(eprew)
            self.episode_lengths.append(eplen)
            self.episode_times.append(time.time() - self.tstart)
            epinfo.update(self.current_reset_info)

            if hasattr(self, 'logger'):
                self.logger.writerow(epinfo)
                self.f.flush()
            else:
                assert hasattr(self, 'results_writer')
                self.results_writer.write_row(epinfo)
               #self.results_writer.flush()
            info['episode'] = epinfo
        self.total_steps += 1
       #print('dones: {}'.format(done))
        return (ob, rew, done, info)

    def setRewardWeights(self):
        return self.env.setRewardWeights()


    def reset(self):
        obs = super().reset()
        self.needs_reset = False
#       obs = self.env.reset()

        return obs


class MultiMonitor(MicropolisMonitor):
    def __init__(self, env, filename, allow_early_resets=False, reset_keywords=(), info_keywords=()):
        super(MultiMonitor, self).__init__(env, filename, allow_early_resets=allow_early_resets, reset_keywords=reset_keywords, info_keywords=info_keywords)
        ''' For GoLMultiEnv'''

    def step(self, action):
        if self.needs_reset:
            raise RuntimeError("Tried to step environment that needs reset")
        ob, rew, done, info = self.env.step(action)
        self.rewards.append(rew.sum())

        if done.all():
            self.needs_reset = True
            eprew = float(sum(self.rewards))
            eplen = len(self.rewards) * self.env.num_proc
            dist_entropy = self.dist_entropy
            if dist_entropy is None:
                dist_entropy = 0
            if eprew is None:
                eprew = 0
            epinfo = {"r": round(eprew, 6), "l": eplen, "t": round(time.time() - self.tstart, 6),
                    "e": round(dist_entropy, 6),
                    "p": round(self.trg_param_vals[0, 0].item(), 6)}

            for k in self.info_keywords:
                if k != 'e' and k!= 'p':
                    epinfo[k] = info[k]
            self.episode_rewards.append(eprew)
            self.episode_lengths.append(eplen)
            self.episode_times.append(time.time() - self.tstart)
            epinfo.update(self.current_reset_info)
            ob = self.reset()
            self.needs_reset = False

            if hasattr(self, 'logger'):
                self.logger.writerow(epinfo)
                self.f.flush()
            else:
                assert hasattr(self, 'results_writer')
                self.results_writer.write_row(epinfo)
            info[0]['episode'] = epinfo
        self.total_steps += 1

        return (ob, rew, done, info)

try:
    import dm_control2gym
except ImportError:
    pass

try:
    import roboschool
except ImportError:
    pass

#try:
#    import pybullet_envs
#except ImportError:
#

class MapDims(gym.Wrapper):
    """
    Wrapper to render environments of a particular rank.
    """
    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        # FIXME: only supports square maps
        self.MAP_X = self.width


class Render(gym.Wrapper):
    """
    Wrapper to render environments of a particular rank.
    """
    def __init__(self, env, rank, **kwargs):
        super().__init__(env)
        self.rank = rank
        self.render_gui = kwargs.get('render', False)
        self.render_rank = kwargs.get('render_rank', [0,2,4])
        if isinstance(self.render_rank, int):
            self.render_rank = [self.render_rank]

    def step(self, action):
        if self.render_gui and self.rank in self.render_rank:
            self.render()

        return super().step(action)


class ToPytorchOrder(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=self.observation_space.high[0, 0, 0],
                shape=(self.observation_space.shape[-1], self.shape[1], self.shape[0]))
               #shape=(self.observation_space.shape[0], self.shape[1], self.shape[2]))

    def reset(self):
        obs = self.env.reset()
        obs = obs.swapaxes(0, 2)

        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = obs.swapaxes(0, 2)

        return obs, reward, done, info

def make_env(env_id, seed, rank, log_dir, add_timestep, allow_early_resets, map_width=20, render_gui=False,
        print_map=False, parallel_py2gui=False, noreward=False, max_step=None, param_rew=False,
        env_params=[],
        args=None):
    ''' return a function which starts the environment'''
    def _thunk():
        record = args.record
        extinction = args.extinction_type is not None

        if env_id.startswith("dm"):
            _, domain, task = env_id.split('.')
            env = dm_control2gym.make(domain_name=domain, task_name=task)
        else:

            if 'rct' in env_id.lower():
                from micro_rct.gym_envs.rct_env import RCT

            if 'GDY' in env_id:
                # it's a griddly environment
                from griddly import GymWrapperFactory, gd
                wrapper = GymWrapperFactory()
                # we assume it's a particular environment:
                wrapper.build_gym_from_yaml(
                        'Sokoban-Adv',
                        'Single-Player/GVGAI/sokoban.yaml',
                        player_observer_type=gd.ObserverType.VECTOR,
                        level=2
                    )
                env = gym.make('GDY-Sokoban-Adv-v0')
                # need to reset to initialize env.observation_space
                env.reset()
                env = Griddly(env)

            else:
                print('render gui', render_gui)
                env = gym.make(env_id, 
                        render_gui=render_gui, rank=rank, max_step=max_step, map_width=map_width)

            if record:
                record_dir = log_dir
            else:
                record_dir = None




            if 'gameoflife' in env_id.lower():
                if rank == 0:
                    render = render_gui
                else: render = False
                env.configure(map_width=map_width, render=render,
                        prob_life = args.prob_life, record=record_dir,
                        max_step=max_step)

            if 'golmulti' in env_id.lower():
                multi_env = True
                env.configure(map_width=map_width, render=render_gui,
                        prob_life=args.prob_life, record=record_dir,
                        max_step=max_step, cuda=args.cuda,
                        num_proc=args.num_processes)
                env = ParamRewMulti(env)
                if extinction:
                    env = ExtinguisherMulti(env, args.extinction_type, args.extinction_prob)
                if args.im_render:
                    env = ImRenderMulti(env, log_dir, rank)
            else:
                multi_env = False


            #FIXME: make this recognize pcgrl environments in general

            if '-wide' in env_id:
                env = ActionMapImagePCGRLWrapper(env_id)
                env.adjust_param(width=args.map_width, heigh=args.map_width)
                env = ToPytorchOrder(env)
                env = MaxStep(env, args.max_step)
                env = Render(env, rank, render = render_gui, render_rank=0)
                env = MapDims(env)
                if True or param_rew: 
                    print('pcgrl reward teacher')
                    assert len(env_params) != 0
                    env = ParamRew(env, env_params)
                    env.configure(map_width=args.map_width, max_step=args.max_step)
                if False:
                    env = NoiseyTargets(env)

            if 'rct' in env_id.lower():
                if param_rew:
                    assert len(env_params) != 0
                    rand_params = rank < args.n_rand_envs
                    env = ParamRew(env, env_params, rand_params=rand_params)

                env.configure()

            if 'micropolis' in env_id.lower():
                power_puzzle = False

                if args.power_puzzle:
                    power_puzzle = True

                if rank in [0]:
                    print_map = args.print_map
                    render = render_gui
                else:
                    print_map = False
                   #render = render_gui
                    render = False

                if extinction:
                    ages = True
                else:
                    ages = False

                if param_rew:
                    assert len(env_params) != 0
                    env = ParamRew(env, env_params)
                env.configure(
                        map_width=map_width,
                        max_step=max_step,
                        rank=rank,
                        print_map=print_map,
                        render_gui=render,
                        empty_start=not args.random_terrain,
                        power_puzzle=power_puzzle,
                        record=record,
                        random_builds=args.random_builds,
                        poet=args.poet,
                        ages=ages,
                        )
                if False:
                    env = NoiseyTargets(env)
                if extinction:
                    env = Extinguisher(env, args.extinction_type, args.extinction_prob)

        is_atari = hasattr(gym.envs, 'atari') and isinstance(
            env.unwrapped, gym.envs.atari.atari_env.AtariEnv)

        if is_atari:
            env = make_atari(env_id)
        env.seed(seed + rank)

        obs_shape = env.observation_space.shape

        if add_timestep and len(
                obs_shape) == 1 and str(env).find('TimeLimit') > -1:
            env = AddTimestep(env)

        if multi_env:
            env = MultiMonitor(env, os.path.join(log_dir, str(rank)),
                            allow_early_resets=True)
        else:
            print(log_dir, rank)

            if args.vis:
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


        if args.im_render and not multi_env:
            print('wrapping in imrender, rank {}'.format(rank))
#           if 'micropolis' in args.env_name.lower():
#               env = ImRenderMicropolis(env, log_dir, rank)

        assert env is not None

        return env

    return _thunk

def make_vec_envs(env_name, seed, num_processes, gamma, log_dir, add_timestep,
                  device, allow_early_resets, num_frame_stack=None, param_rew=False,
                  env_params=[], args=None):

    print('make vec envs random map? {}'.format(args.random_terrain))
    if 'golmultienv' in env_name.lower():
        num_processes=1 # smuggle in real num_proc in args so we can run them as one NN
   #assert num_env_params != 0
    envs = [make_env(env_name, seed, i, log_dir, add_timestep,
        allow_early_resets, map_width=args.map_width, render_gui=args.render,
        print_map=args.print_map, noreward=args.no_reward, max_step=args.max_step, param_rew=param_rew,
        env_params=env_params, args=args)
            for i in range(num_processes)]

    if 'golmultienv' in env_name.lower():
        return envs[0]()

    if len(envs) > 1:
       #print(envs)
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
            [obs_shape[2], obs_shape[1], obs_shape[0]],
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
        obs = np.array(obs[0])
        ### ########## ###
        obs = torch.from_numpy(obs).int().to(self.device)

        return obs

    def step_async(self, actions):
        actions_async = {}
        if isinstance(actions, dict):
            for act_name, action in actions.items():
                actions_async[act_name] = action.squeeze(1).cpu().numpy()
        else:
            actions_async = actions.squeeze(1).cpu().numpy()
        self.venv.step_async(actions_async)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        ### micropolis ###
        obs = np.array(obs)
        ### ########## ###
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()

        return obs, reward, done, info

    def get_param_bounds(self):
        return self.venv.get_param_bounds()


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
        self.device = device

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
        try:
            self.stacked_obs[:, -self.shape_dim0:] = obs
        # hack to dynamically change map size
        except RuntimeError:
            wos, _ = self.venv.get_spaces()  # wrapped ob space
            self.shape_dim0 = wos.shape[0]
            low = np.repeat(wos.low, self.nstack, axis=0)
            self.stacked_obs = torch.zeros((self.venv.num_envs,) + low.shape).to(self.device)
        self.stacked_obs[:, -self.shape_dim0:] = obs

        return self.stacked_obs

    def close(self):
        self.venv.close()

    def get_param_bounds(self):
        return self.venv.get_param_bounds()

    def set_param_bounds(self, bounds):
        return self.venv.venv.set_param_bounds(bounds)

    def set_params(self,params):
        return self.venv.venv.set_params(params)
