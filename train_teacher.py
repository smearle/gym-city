import copy
import glob
import os
import time
from collections import deque, OrderedDict

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import csv


import algo
from arguments import get_args
from envs import make_vec_envs
from model import Policy
from storage import RolloutStorage, CuriosityRolloutStorage
from utils import get_vec_normalize
from visualize import Plotter
from shutil import copyfile
from teachDRL.teachers.algos.alp_gmm import ALPGMM
from train import init_agent, Trainer




class Teacher(Trainer):
    def get_fieldnames(self):
        return ['r','l','t','e','p']

    def get_save_dict(self):
        d = super().get_save_dict()
        d['alp_gmm'] = self.alp_gmm
        d['param_hist'] = self.param_hist
        return d

    def __init__(self):
        # have to do above before call to parent to inirialize Evaluator correctly
        super(Teacher, self).__init__()
        # dictionary of param names to target histories as set by alp_gmm
        self.param_hist = {}
        envs = self.envs
        args = self.args
        env_param_bounds = envs.get_param_bounds()
        # in case we want to change this dynamically in the future (e.g., we may
        # not know how much traffic the agent can possibly produce in Micropolis)
        envs.set_param_bounds(env_param_bounds) # start with default bounds
        env_param_bounds = env_param_bounds
        num_env_params = 4
        env_param_ranges = []
        env_param_lw_bounds = []
        env_param_hi_bounds = []
        i = 0
        for k, v in env_param_bounds.items():
            if i < num_env_params:
                env_param_ranges += [abs(v[1] - v[0])]
                env_param_lw_bounds += [v[0]]
                env_param_hi_bounds += [v[1]]
                i += 1
            else:
                break
        alp_gmm = None
        if self.checkpoint:
            alp_gmm = self.checkpoint['alp_gmm']
        if alp_gmm is None:
            alp_gmm = ALPGMM(env_param_lw_bounds, env_param_hi_bounds)
        params_vec = alp_gmm.sample_task()
        self.alp_gmm = alp_gmm

        params = OrderedDict()
        print('\n env_param_bounds', env_param_bounds)
        print(params_vec)
        trial_remaining = args.max_step
        trial_reward = 0

        self.env_param_bounds = env_param_bounds
        self.num_env_params = num_env_params
        self.env_param_ranges = env_param_ranges
        self.params_vec = params_vec
        self.params = params
        self.trial_remaining = args.max_step
        self.trial_reward = trial_reward

    def check_params(self):
        trial_remaining = self.trial_remaining
        params = self.params
        trial_reward = self.trial_reward
        params_vec = self.params_vec
        args = self.args
        alp_gmm = self.alp_gmm
        num_env_params = self.num_env_params
        env_param_bounds = self.env_param_bounds

        if trial_remaining == 0:
            trial_reward = trial_reward / args.num_processes
            alp_gmm.update(params_vec, trial_reward)
            trial_reward = 0
            trial_remaining = args.max_step
            # sample random environment parameters
            params_vec = alp_gmm.sample_task()
            prm_i = 0
            for k, v in env_param_bounds.items():
                if prm_i < num_env_params:
                    params[k] = params_vec[prm_i]
                    prm_i += 1
                else:
                    break
            self.envs.set_params(params)
        trial_remaining -= args.num_steps

        self.trial_remaining = trial_remaining

    def plot_trg_params(self):
        for param in self.params:
           #print('plotting param. {}'.format(param))
            pass


    def main(self):
        for self.n_train in range(self.updates_remaining):
            self.check_params()
            self.plot_trg_params()
            self.train()


if __name__ == "__main__":
    teacher = Teacher()
    teacher.main()
