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
            remote.send((env.observation_space, env.action_space))
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
        elif hasattr(env, cmd):
            cmd_fn = getattr(env, cmd)
            print(data)
            if isinstance(data, dict):
                ret_val = cmd_fn(**data)
            else:
                ret_val = cmd_fn(*data)
            remote.send(ret_val)
        else:
            print('invalid command, data: {}, {}'.format(cmd, data))
            raise NotImplementedError



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

        for p in self.ps:
            p.daemon = True # if the main process crashes, we should not cause things to hang
            p.start()

        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def get_param_bounds(self):
        worker = self.remotes[0]
        worker.send(('get_param_bounds', None))
        param_bounds = worker.recv()

        return param_bounds

    def reset_episodes(self):
        for remote in self.remotes:
            remote.send(('reset_episodes', {}))
            remote.recv()

        return

    def setMapSize(self, map_size):
        for remote in self.remotes:
            remote.send(('setMapSize', [map_size]))
            remote.recv()

        return

    def set_extinction_type(self, ext_type, ext_prob):
        for remote in self.remotes:
            remote.send(('set_extinction_type', [ext_type, ext_prob]))
            remote.recv()

        return

    def set_params(self, params):
        for remote in self.remotes:
            remote.send(('set_params', params))
            remote.recv()

        return

    def set_param_bounds(self, param_bounds):
        worker = self.remotes[0]
        worker.send(('set_param_bounds', param_bounds))
        worker.recv()

        return

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

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))

        return np.stack([remote.recv() for remote in self.remotes])

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
