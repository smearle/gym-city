import copy
import csv
import glob
import os
import time
from collections import OrderedDict, deque
from shutil import copyfile

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import algo
from arguments import get_args
from envs import make_vec_envs
from model import Policy
from storage import CuriosityRolloutStorage, RolloutStorage
from teachDRL.teachers.algos.alp_gmm import ALPGMM
from train import Trainer, init_agent
from utils import get_vec_normalize
from visualize import Plotter

class Teacher(Trainer):
    def get_fieldnames(self):
        return ['r','l','t','e','p']

    def get_save_dict(self):
        d = super().get_save_dict()
        d['alp_gmm'] = self.alp_gmm
        d['param_hist'] = self.param_hist

        return d 

    def __init__(self, args=None):
        if args is None:
            args = get_args()
        device = torch.device("cuda:0" if args.cuda else "cpu")
        self.device = device
        # have to do above before call to parent to initialize Evaluator correctly
        # ^^ huh?
        args.param_rew = True
        envs = self.make_vec_envs(args)
        super(Teacher, self).__init__(args=args, envs=envs)
        # dictionary of param names to target histories as set by alp_gmm
        self.env_params = self.args.env_params
        self.param_hist = {}
        envs = self.envs
        args = self.args
        env_param_bounds = envs.get_param_bounds()
        # in case we want to change this dynamically in the future (e.g., we may
        # not know how much traffic the agent can possibly produce in Micropolis)
        envs.set_param_bounds(env_param_bounds) # start with default bounds
        env_param_ranges = []
        env_param_lw_bounds = []
        env_param_hi_bounds = []
        i = 0

        for k in self.env_params:
            v = env_param_bounds[k]
            env_param_ranges += [abs(v[1] - v[0])]
            env_param_lw_bounds += [v[0]]
            env_param_hi_bounds += [v[1]]
            i += 1
        alp_gmm = None

        if self.checkpoint:
            if 'alp_gmm' in self.checkpoint:
                alp_gmm = self.checkpoint['alp_gmm']

        if alp_gmm is None:
            alp_gmm = ALPGMM(env_param_lw_bounds, env_param_hi_bounds)
        params_vec = alp_gmm.sample_task()
        self.alp_gmm = alp_gmm

        params = OrderedDict()
        print('\n env_param_bounds', env_param_bounds)
        trial_reward = 0

        self.env_param_bounds = env_param_bounds
        self.env_param_ranges = env_param_ranges
        self.params_vec = params_vec
        self.params = params
        self.len_trial = args.max_step * 1
        self.trial_remaining = self.len_trial
        self.trial_reward = trial_reward

    def check_params(self):
        trial_remaining = self.trial_remaining
        params = self.params
        trial_reward = self.trial_reward
        params_vec = self.params_vec
        args = self.args
        alp_gmm = self.alp_gmm
        env_param_bounds = self.env_param_bounds

        if trial_remaining <= 0:
            trial_reward = trial_reward / args.num_processes
            alp_gmm.update(params_vec, trial_reward)
            trial_reward = 0
            trial_remaining = self.len_trial
            # sample random environment parameters
            params_vec = alp_gmm.sample_task()
            prm_i = 0

#           print(params_vec)
            for k in self.env_params:
                params[k] = params_vec[prm_i]
                prm_i += 1
            self.envs.set_trgs(params)
            print('setting params: {}'.format(params))
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
