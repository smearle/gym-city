'''Deal with communication between training loop and a bunch of agents.
'''
from multiprocessing import Pipe, Process

import numpy as np
from baselines.common.vec_env import CloudpickleWrapper, VecEnv


def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()

    while True:
        cmd, data = remote.recv()

        if cmd == 'step':
            ob, reward, done, info = env.step(data)

            if done:
                ob = env.reset()
            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset()
            remote.send(ob)
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'close':
            remote.close()

            break
        elif cmd == 'get_spaces':
            print('getting spaces envs.py')
            spaces = env.observation_space, env.action_space, env.unwrapped.player_action_space
            remote.send(spaces)
        elif cmd == 'get_param_bounds':
            param_bounds = env.get_param_bounds()
            remote.send(param_bounds)
        elif cmd == 'set_params':
            env.set_params(data)
            remote.send(None)
        elif cmd == 'set_param_bounds':
            env.set_param_bounds(data)
            remote.send(None)
        elif cmd == 'render':
            env.render()
            remote.send(None)
        elif cmd == 'set_map':
            remote.send(env.unwrapped.set_map(data))
        elif hasattr(env, cmd):
            cmd_fn = getattr(env, cmd)

            if isinstance(data, dict):
                ret_val = cmd_fn(**data)
            elif isinstance(data, list) or isinstance(data, tuple):
                ret_val = cmd_fn(*data)
            else:
                ret_val = data
            remote.send(ret_val)
        else:
            print('invalid command, data: {}, {}'.format(cmd, data))
            raise NotImplementedError

class VecEnvPlayer(VecEnv):
    def __init__(self, num_envs, observation_space, action_space, player_action_space):
        VecEnv.__init__(self, num_envs, observation_space, action_space)
        self.player_action_space = player_action_space


class SubprocVecEnv(VecEnv):
    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        # TODO: subclass and expand this
        self.playable_map = None

        for p in self.ps:
            p.daemon = True # if the main process crashes, we should not cause things to hang
            p.start()
        self.remotes[0].send(('get_spaces', None))
        spaces = self.remotes[0].recv()

        for remote in self.work_remotes:
            remote.close()

        if len(spaces) == 2:
            observation_space, action_space = spaces
            VecEnv.__init__(self, len(env_fns), observation_space, action_space)

        if len(spaces) == 3:
            observation_space, action_space, player_action_space = spaces
            VecEnvPlayer.__init__(self, len(env_fns), observation_space, action_space, player_action_space)
        print('spaces found by SubprocVecEnv: obs {}, act {}'.format(observation_space, action_space))
        print('obs space in SubProcVec init: {}'.format(self.observation_space))

    def init_storage(self):
        pass

    def get_param_bounds(self):
        worker = self.remotes[0]
        worker.send(('get_param_bounds', None))
        param_bounds = worker.recv()

        return param_bounds

    def get_param_trgs(self):
        worker = self.remotes[0]
        worker.send(('get_param_trgs', None))
        param_trgs = worker.recv()

        return param_trgs

    def reset_episodes(self, im_log_dir):
        for remote in self.remotes:
            remote.send(('reset_episodes', [im_log_dir]))
            remote.recv()

        return

    def configure(self, **kwargs):
        for remote in self.remotes:
            remote.send(('configure', kwargs))
            remote.recv()

        return

    def set_extinction_type(self, *args):
        for remote in self.remotes:
            remote.send(('set_extinction_type', args))
            remote.recv()

        return

    def set_log_dir(self, log_dir):
        for remote in self.remotes:
            remote.send(('set_log_dir', log_dir))
            remote.recv()

    def set_params(self, params):
        for remote in self.remotes:
            remote.send(('set_params', params))
            remote.recv()

        return

    def set_active_agent(self, n_agent):
        for remote in self.remotes:
            remote.send(('set_active_agent', [n_agent]))
            remote.recv()

        return

    def set_save_dir(self, save_dir):
        for remote in self.remotes:
            remote.send(('set_save_dir', [save_dir]))
            remote.recv()

        return

    def set_map(self, map):
        for remote in self.remotes:
            remote.send(('set_map', [map]))
            remote.recv()

        return

    def set_param_bounds(self, param_bounds):
        worker = self.remotes[0]
        worker.send(('set_param_bounds', param_bounds))
        num_params = worker.recv()
        print('{} env params'.format(num_params))

        return num_params

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)

        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))

        return np.stack([remote.recv() for remote in self.remotes])

    def render(self, mode=None):
        remote = self.remotes[0]
        remote.send(('render', None))

        return remote.recv()

       #return np.stack([remote.recv() for remote in self.remotes])

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))

        return np.stack([remote.recv() for remote in self.remotes])

    def get_spaces(self):
        for remote in self.remotes:
            remote.send(('get_spaces', None))
            space_list = remote.recv()
            print('space list', space_list)

            return space_list

    def close(self):
        if self.closed:
            return

        if self.waiting:
            for remote in self.remotes:
                remote.recv()

        for remote in self.remotes:
            remote.send(('close', None))

        for p in self.ps:
            p.join()
        self.closed = True

