import copy
import glob
import os
import time
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from shutil import copyfile

from arguments import get_args
from envs import make_vec_envs
from model import Policy
from storage import RolloutStorage, CuriosityRolloutStorage
from utils import get_vec_normalize
from visualize import Plotter
import algo

import csv


class Evaluator(object):
    ''' Runs inference on a bunch of envs'''
    def __init__(self, args, actor_critic, device, envs=None, vec_norm=None,
            frozen=False, fieldnames=['r', 'l', 't']):
        ''' frozen: we are not in the main training loop, but evaluating frozen model separately'''
        if frozen:
            self.win_eval = None
            past_frames = args.past_frames
        self.frozen = frozen
       #eval_args.render = True
        self.device = device
       #if args.model == 'fractal':
       #    for i in range(-1, args.n_recs):
       #        eval_log_dir = args.log_dir + "_eval_col_{}".format(i)
       #        try:
       #            os.makedirs(eval_log_dir)
       #        except OSError:
       #            files = glob.glob(os.path.join(eval_log_dir,  '*.monitor.csv'))
       #            for f in files:
       #                os.remove(f)
       #        setattr(self, 'eval_log_dir_col_{}'.format(i), eval_log_dir)
        if frozen:
            if 'GameOfLife' in args.env_name:
                self.eval_log_dir = args.log_dir + "/eval_{}-frames_w{}_{}rec_{}s_{}pl".format(past_frames,
                        args.map_width, args.n_recs, args.max_step, args.prob_life, '.1f')
            else:
                self.eval_log_dir = args.log_dir + "/eval_{}-frames_w{}_{}rec_{}f".format(past_frames,
                        args.map_width, args.n_recs, args.max_step, '.1f')
            merge_col_logs = True
        else:
            self.eval_log_dir = args.log_dir + "_eval"
            merge_col_logs = False
        try:
            os.makedirs(self.eval_log_dir)
        except OSError:
            files = glob.glob(os.path.join(self.eval_log_dir,  '*.monitor.csv'))
            files += glob.glob(os.path.join(self.eval_log_dir, '*_eval.csv'))
            if args.overwrite:
                for f in files:
                    os.remove(f)
            elif files:
                merge_col_logs = True

        self.args = args
        self.actor_critic = actor_critic
        self.num_eval_processes = 1
        if envs and False:
            self.eval_envs = envs
            self.vec_norm = vec_norm
            self.num_eval_processes = args.num_processes
        else:

           #print('making envs in Evaluator: ', self.args.env_name, self.args.seed + self.num_eval_processes, self.num_eval_processes,
           #            self.args.gamma, self.eval_log_dir, self.args.add_timestep, self.device, True, self.args)
            eval_args = copy.deepcopy(args)
            eval_args.render = args.render
            self.eval_envs = make_vec_envs(
                        self.args.env_name, self.args.seed + self.num_eval_processes, self.num_eval_processes,
                        self.args.gamma, self.eval_log_dir, self.args.add_timestep, self.device, False, args=eval_args)
            self.vec_norm = get_vec_normalize(self.eval_envs)
        if self.vec_norm is not None:
            self.vec_norm.eval()
            self.vec_norm.ob_rms = get_vec_normalize(self.eval_envs).ob_rms
        self.tstart = time.time()
        model = actor_critic.base
        if args.model == 'FractalNet':
            n_cols = model.n_cols
        else:
            n_cols = 0
        self.plotter = Plotter(n_cols, self.eval_log_dir, self.num_eval_processes, max_steps=self.args.max_step)
        eval_cols = range(-1, n_cols)
        if args.model == 'fixed' and model.RAND:
            eval_cols = model.eval_recs
        if eval_cols is not None:
            for i in eval_cols:
                log_file = '{}/col_{}_eval.csv'.format(self.eval_log_dir, i)
                if merge_col_logs and os.path.exists(log_file):
                    merge_col_log = True
                else:
                    merge_col_log = False
                if merge_col_log:
                    if len(eval_cols) > 1 and i == eval_cols[-2] and self.args.auto_expand: # problem if we saved model after auto-expanding, without first evaluating!
                        # for the newly added column, we duplicate the last col.'s records
                        new_col_log_file = '{}/col_{}_eval.csv'.format(self.eval_log_dir, i + 1)
                        copyfile(log_file, new_col_log_file)
                    old_log = '{}_old'.format(log_file)
                    os.rename(log_file, old_log)
                log_file_col = open(log_file, mode='w')
                setattr(self, 'log_file_col_{}'.format(i), log_file_col)
                writer_col = csv.DictWriter(log_file_col, fieldnames=fieldnames)
                setattr(self, 'writer_col_{}'.format(i), writer_col)
                if merge_col_log:
                    with open(old_log, newline='') as old:
                        reader = csv.DictReader(old, fieldnames=fieldnames)
                        h = 0
                        try: # in case of null bytes resulting from interrupted logging
                            for row in reader:
                                if h > 1:
                                    row['t'] = 0.0001 * h # HACK: false times for past logs to maintain order
                                    writer_col.writerow(row)
                                h += 1
                        except csv.Error: # I guess this error happens at most once then?
                            h_i = 0
                            for row in reader:
                                if h_i > h:
                                    row['t'] = 0.0001 * h_i # HACK: false times for past logs to maintain order
                                    writer_col.writerow(row)
                                h_i += 1
                    os.remove(old_log)

                else:
                    writer_col.writeheader()
                    log_file_col.flush()



    def evaluate(self, column=None, num_recursions=None):
        model = self.actor_critic.base
        if num_recursions is not None:
            model.num_recursions = num_recursions
        if column is not None and self.args.model == 'FractalNet':
            model.set_active_column(column)
        self.actor_critic.visualize_net()
        eval_episode_rewards = []
        obs = self.eval_envs.reset()
        if 'LSTM' in self.args.model:
            recurrent_hidden_state_size = self.actor_critic.base.get_recurrent_state_size()
            eval_recurrent_hidden_states = torch.zeros(2, self.num_eval_processes,
                             *recurrent_hidden_state_size, device=self.device)
        else:
            recurrent_hidden_state_size = self.actor_critic.recurrent_hidden_state_size
            eval_recurrent_hidden_states = torch.zeros(self.num_eval_processes,
                            recurrent_hidden_state_size, device=self.device)
            eval_masks = torch.zeros(self.num_eval_processes, 1, device=self.device)

        i = 0
        done = np.array([False])
        while not (done.all() or i > self.args.max_step):
       #while len(eval_episode_rewards) < self.num_eval_processes:
       #while i < self.args.max_step:
            with torch.no_grad():
                _, action, eval_recurrent_hidden_states, _ = self.actor_critic.act(
                    obs, eval_recurrent_hidden_states, eval_masks, deterministic=True)

            # Observe reward and next obs
            obs, reward, done, infos = self.eval_envs.step(action)
            if self.args.render:
                if self.args.num_processes == 1:
                    if not ('Micropolis' in self.args.env_name or 'GameOfLife' in self.args.env_name or 'GoL' in self.args.env_name):
                        self.eval_envs.venv.venv.render()
                    else:
                        pass
                       #self.eval_envs.venv.venv.envs[0].render()
                else:
                    if not ('Micropolis' in self.args.env_name or 'GameOfLife' in self.args.env_name or 'GoL' in self.args.env_name):
                        self.eval_envs.venv.venv.render()
                    else:
                        pass
                       #self.eval_envs.venv.venv.remotes[0].send(('render', None))
                       #self.eval_envs.venv.venv.remotes[0].recv()

            eval_masks = torch.FloatTensor([[0.0] if done_ else [1.0]
                                            for done_ in done])
            for info in infos:
                if 'episode' in info.keys():
                    eval_episode_rewards.append(info['episode']['r'])
            i += 1

        self.eval_envs.reset()
       #self.eval_envs.close()
        eprew = np.mean(eval_episode_rewards)
        args = self.args
        if not self.frozen:
            # note: eval interval given in terms of updates consisting of num_steps each
            n_frame = args.num_steps * args.num_processes * args.eval_interval # relative to training session
        else:
            n_frame = args.max_step * args.num_processes

        if num_recursions is not None:
            column = num_recursions
        if column is not None:
            print(" Column {}".format(column))
            log_info = {'r': round(eprew, 6),  'l': n_frame, 't': round(time.time() - self.tstart, 6)}
            writer, log_file = getattr(self, 'writer_col_{}'.format(column)),\
                               getattr(self, 'log_file_col_{}'.format(column))
            writer.writerow(log_info)
            log_file.flush()
            print(" Evaluation using {} episodes: mean reward {:.5f}\n".
                format(len(eval_episode_rewards),
                   eprew))

        if self.frozen:
            if args.vis:
                from visdom import Visdom
                viz = Visdom(port=args.port)
                self.win_eval = self.plotter.bar_plot(viz, self.win_eval, self.eval_log_dir, self.eval_log_dir.split('/')[-1],
                                  args.algo, args.num_frames, n_cols=model.n_cols)


