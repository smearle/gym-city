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

import algo
from arguments import get_args
from envs import make_vec_envs
from model import Policy
from storage import RolloutStorage, CuriosityRolloutStorage
from utils import get_vec_normalize
from visualize import visdom_plot

import csv

import random
import gym_micropolis
import game_of_life

args = get_args()
args.log_dir = args.save_dir + '/logs'
assert args.algo in ['a2c', 'ppo', 'acktr']
if args.recurrent_policy:
    assert args.algo in ['a2c', 'ppo'], \
        'Recurrent policy is not implemented for ACKTR'

num_updates = int(args.num_frames) // args.num_steps // args.num_processes

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

graph_name = args.save_dir.replace('trained_models/', '').replace('/', ' ')

def main():

    actor_critic = False
    agent = False
    past_steps = 0
    try:
        os.makedirs(args.log_dir)
    except OSError:
        files = glob.glob(os.path.join(args.log_dir, '*.monitor.csv'))
        for f in files:
            if args.overwrite:
                os.remove(f)
            else:
                pass
    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    if args.vis:
        from visdom import Visdom
        viz = Visdom(port=args.port)
        win = None
        win_eval = None
    if 'GameOfLife' in args.env_name:
        print('env name: {}'.format(args.env_name))
        num_actions = 1
    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                        args.gamma, args.log_dir, args.add_timestep, device, False, None,

                        args=args)

    if isinstance(envs.observation_space, gym.spaces.Discrete):
        num_inputs = envs.observation_space.n
    elif isinstance(envs.observation_space, gym.spaces.Box):
        if len(envs.observation_space.shape) == 3:
            in_w = envs.observation_space.shape[1]
            in_h = envs.observation_space.shape[2]
        else:
            in_w = 1
            in_h = 1
        num_inputs = envs.observation_space.shape[0]
    if isinstance(envs.action_space, gym.spaces.Discrete):
        out_w = 1
        out_h = 1
        num_actions = envs.action_space.n
    elif isinstance(envs.action_space, gym.spaces.Box):
        if len(envs.action_space.shape) == 3:
            out_w = envs.action_space.shape[1]
            out_h = envs.action_space.shape[2]
        elif len(envs.action_space.shape) == 1:
            out_w = 1
            out_h = 1
        num_actions = envs.action_space.shape[-1]

    actor_critic = Policy(envs.observation_space.shape, envs.action_space,
        base_kwargs={'map_width': args.map_width, 'num_actions': num_actions,
            'recurrent': args.recurrent_policy,
            'in_w': in_w, 'in_h': in_h, 'num_inputs': num_inputs,
            'out_w': out_w, 'out_h': out_h},
                     curiosity=args.curiosity, algo=args.algo,
                     model=args.model, args=args)
   #args.n_recs = args.n_recs + 1

    evaluator = None

    if not agent:
        if args.algo == 'a2c':
            agent = algo.A2C_ACKTR_NOREWARD(actor_critic, args.value_loss_coef,
                                   args.entropy_coef, lr=args.lr,
                                   eps=args.eps, alpha=args.alpha,
                                   max_grad_norm=args.max_grad_norm,
                                   curiosity=args.curiosity, args=args)
        elif args.algo == 'ppo':
            agent = algo.PPO(actor_critic, args.clip_param, args.ppo_epoch, args.num_mini_batch, args.value_loss_coef, args.entropy_coef, lr=args.lr,
                                   eps=args.eps,
                                   max_grad_norm=args.max_grad_norm)
        elif args.algo == 'acktr':
            agent = algo.A2C_ACKTR_NOREWARD(actor_critic, args.value_loss_coef,
                                   args.entropy_coef, lr=args.lr,
                                   eps=args.eps, alpha=args.alpha,
                                   max_grad_norm=args.max_grad_norm,
                                   acktr=True,
                                   curiosity=args.curiosity, args=args)

   #saved_model = os.path.join(args.save_dir, args.env_name + '.pt')
    saved_model = os.path.join(args.save_dir, args.env_name + '.tar')
    if os.path.exists(saved_model) and not args.overwrite:
        checkpoint = torch.load(saved_model)
        actor_critic.load_state_dict(checkpoint['model_state_dict'])
        actor_critic.to(device)
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        past_steps = checkpoint['past_steps']
        ob_rms = checkpoint['ob_rms']
        saved_args = checkpoint['args']
        new_recs = args.n_recs - saved_args.n_recs
        for nr in range(new_recs):
            actor_critic.base.auto_expand()
        past_steps = next(iter(agent.optimizer.state_dict()['state'].values()))['step']
        if saved_args.n_recs > args.n_recs:
            print('applying {} fractal expansions to network'.format(saved_args.n_recs - args.n_recs))
        print('Resuming from step {}'.format(past_steps))

       #print(type(next(iter((torch.load(saved_model))))))
       #actor_critic, ob_rms = \
       #        torch.load(saved_model)
       #agent = \
       #    torch.load(os.path.join(args.save_dir, args.env_name + '_agent.pt'))
       #if not agent.optimizer.state_dict()['state'].values():
       #    past_steps = 0
       #else:

       #    raise Exception

        vec_norm = get_vec_normalize(envs)
        if vec_norm is not None:
            vec_norm.eval()
            vec_norm.ob_rms = ob_rms
    actor_critic.to(device)

    if 'LSTM' in args.model:
        recurrent_hidden_state_size = actor_critic.base.get_recurrent_state_size()
    else:
        recurrent_hidden_state_size = actor_critic.recurrent_hidden_state_size
    if args.curiosity:
        rollouts = CuriosityRolloutStorage(args.num_steps, args.num_processes,
                            envs.observation_space.shape, envs.action_space,
                            recurrent_hidden_state_size, actor_critic.base.feature_state_size(), args=args)
    else:
        rollouts = RolloutStorage(args.num_steps, args.num_processes,
                            envs.observation_space.shape, envs.action_space,
                            recurrent_hidden_state_size, args=args)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)

    start = time.time()
    model = actor_critic.base
    for j in range(past_steps, num_updates):
        if args.model == 'fractal' and args.drop_path:
            model.set_drop_path()
        if args.model == 'fixed' and model.RAND:
            model.num_recursions = random.randint(1, model.map_width * 2)

        player_act = None
        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                if args.render:
                    envs.venv.venv.remotes[0].send(('render', None))
                value, action, action_log_probs, recurrent_hidden_states = actor_critic.act(
                        rollouts.obs[step],
                        rollouts.recurrent_hidden_states[step],
                        rollouts.masks[step],
                        player_act=player_act,
                        icm_enabled=args.curiosity,
                        deterministic=False)

            # Observe reward and next obs
            obs, reward, done, infos = envs.step(action)

            player_act = None
            if args.render:
                if infos[0]:
                    if 'player_move' in infos[0].keys():
                        player_act = infos[0]['player_move']
            

            if args.curiosity:
                # run icm
                with torch.no_grad():


                    feature_state, feature_state_pred, action_dist_pred = actor_critic.icm_act(
                            (rollouts.obs[step], obs, action_bin)
                            )

                intrinsic_reward = args.eta * ((feature_state - feature_state_pred).pow(2)).sum() / 2.
                if args.no_reward:
                    reward = 0
                reward += intrinsic_reward.cpu()

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0]
                                       for done_ in done])
            if args.curiosity:
                rollouts.insert(obs, recurrent_hidden_states, action, action_log_probs, value, reward, masks,
                                feature_state, feature_state_pred, action_bin, action_dist_pred)
            else:
                rollouts.insert(obs, recurrent_hidden_states, action, action_log_probs, value, reward, masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(rollouts.obs[-1],
                                                rollouts.recurrent_hidden_states[-1],
                                                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau)
        if args.curiosity:
            value_loss, action_loss, dist_entropy, fwd_loss, inv_loss = agent.update(rollouts)
        else:
            value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()



        total_num_steps = (j + 1) * args.num_processes * args.num_steps

        if not dist_entropy:
            dist_entropy = 0
        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            end = time.time()
            print("Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n \
dist entropy {:.1f}, val/act loss {:.1f}/{:.1f},".
                format(j, total_num_steps,
                       int((total_num_steps - past_steps * args.num_processes * args.num_steps) / (end - start)),
                       len(episode_rewards),
                       np.mean(episode_rewards),
                       np.median(episode_rewards),
                       np.min(episode_rewards),
                       np.max(episode_rewards), dist_entropy,
                       value_loss, action_loss))
            if args.curiosity:
                print("fwd/inv icm loss {:.1f}/{:.1f}\n".
                format(
                       fwd_loss, inv_loss))

        if (args.eval_interval is not None and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            if evaluator is None:
                evaluator = Evaluator(args, actor_critic, device)


            model = evaluator.actor_critic.base
            if args.model == 'fractal':
                n_cols = model.n_cols
                if args.rule == 'wide1' and args.n_recs > 3:
                    col_step = 3
                else:
                    col_step = 1
            else:
                n_cols = 0
                col_step = 1
            col_idx = [-1, *range(0, n_cols, col_step)]
            for i in col_idx:
                evaluator.evaluate(column=i)
           #num_eval_frames = (args.num_frames // (args.num_steps * args.eval_interval * args.num_processes)) * args.num_processes *  args.max_step
           # making sure the evaluator plots the '-1'st column (the overall net)

            if args.vis and j % args.vis_interval == 0:
                try:
                    # Sometimes monitor doesn't properly flush the outputs
                    win_eval = visdom_plot(viz, win_eval, evaluator.eval_log_dir, graph_name,
                                  args.algo, args.num_frames, n_graphs= col_idx)
                except IOError:
                    pass
           #elif args.model == 'fixed' and model.RAND:
           #    for i in model.eval_recs:
           #        evaluator.evaluate(num_recursions=i)
           #    win_eval = visdom_plot(viz, win_eval, evaluator.eval_log_dir, graph_name,
           #                           args.algo, args.num_frames, n_graphs=model.eval_recs)
           #else:
           #    evaluator.evaluate(column=-1)
           #    win_eval = visdom_plot(viz, win_eval, evaluator.eval_log_dir, graph_name,
           #                  args.algo, args.num_frames)

        if j % args.save_interval == 0 and args.save_dir != "":
            save_path = os.path.join(args.save_dir)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            # A really ugly way to save a model to CPU
            save_model = actor_critic
            ob_rms = getattr(get_vec_normalize(envs), 'ob_rms', None)
            save_model = copy.deepcopy(actor_critic)
            save_agent = copy.deepcopy(agent)
            if args.cuda:
                save_model.cpu()
            optim_save = save_agent.optimizer.state_dict()

            # experimental:
            torch.save({
                'past_steps': step,
                'model_state_dict': save_model.state_dict(),
                'optimizer_state_dict': optim_save,
                'ob_rms': ob_rms,
                'args': args
                }, os.path.join(save_path, args.env_name + ".tar"))

           #save_model = [save_model,
           #              getattr(get_vec_normalize(envs), 'ob_rms', None)]

           #torch.save(save_model, os.path.join(save_path, args.env_name + ".pt"))
           #save_agent = copy.deepcopy(agent)

           #torch.save(save_agent, os.path.join(save_path, args.env_name + '_agent.pt'))
           #torch.save(actor_critic.state_dict(), os.path.join(save_path, args.env_name + "_weights.pt"))

        if args.vis and j % args.vis_interval == 0:
            try:
                # Sometimes monitor doesn't properly flush the outputs
                win = visdom_plot(viz, win, args.log_dir, graph_name,
                                  args.algo, args.num_frames)
            except IOError:
                pass


class Evaluator(object):
    ''' Manages environments used for evaluation during main training loop.'''
    def __init__(self, args, actor_critic, device):
        eval_args = args
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
                
        self.eval_log_dir = args.log_dir + "_eval"
        merge_col_logs = False
        try:
            os.makedirs(self.eval_log_dir)
        except OSError:
            files = glob.glob(os.path.join(self.eval_log_dir,  '*.monitor.csv'))
            if args.overwrite:
                for f in files:
                    os.remove(f)
            elif files:
                merge_col_logs = True

        self.num_eval_processes = 20
        self.eval_envs = make_vec_envs(
                    eval_args.env_name, eval_args.seed + self.num_eval_processes, self.num_eval_processes,
                    eval_args.gamma, self.eval_log_dir, eval_args.add_timestep, self.device, True, args=eval_args)
        self.vec_norm = get_vec_normalize(self.eval_envs)
        if self.vec_norm is not None:
            self.vec_norm.eval()
            self.vec_norm.ob_rms = get_vec_normalize(self.eval_envs).ob_rms
        self.actor_critic = actor_critic
        self.tstart = time.time()
        fieldnames = ['r', 'l', 't']
        model = actor_critic.base
        if args.model == 'fractal':
            n_cols = model.n_cols
        else:
            n_cols = 0
        eval_cols = range(-1, n_cols)
        if args.model == 'fixed' and model.RAND:
            eval_cols = model.eval_recs
        if eval_cols is not None:
            for i in eval_cols:
                log_file = '{}/col_{}_eval.csv'.format(self.eval_log_dir, i)
                if merge_col_logs:
                    old_log = '{}_old'.format(log_file)
                    os.rename(log_file, old_log)
                log_file_col = open(log_file, mode='w')
                setattr(self, 'log_file_col_{}'.format(i), log_file_col)
                writer_col = csv.DictWriter(log_file_col, fieldnames=fieldnames)
                setattr(self, 'writer_col_{}'.format(i), writer_col)
                if merge_col_logs:
                    with open(old_log, newline='') as old:
                        reader = csv.DictReader(old, fieldnames=('r', 'l', 't'))
                        h = 0
                        try: # in case of null bytes resulting from interrupted logging
                            for row in reader:
                                if h > 1:
                                    row['t'] = 0.0001 * h # HACK: false times for past logs to maintain order
                                    writer_col.writerow(row)
                                h += 1
                        except csv.Error:
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
        self.args = eval_args


    def evaluate(self, column=None, num_recursions=None):

        model = self.actor_critic.base
        if num_recursions is not None:
            model.num_recursions = num_recursions
        if column is not None and self.args.model == 'fractal':
            model.set_active_column(column)

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

        while len(eval_episode_rewards) < self.num_eval_processes:
            with torch.no_grad():
                _, action, eval_recurrent_hidden_states, _ = self.actor_critic.act(
                    obs, eval_recurrent_hidden_states, eval_masks, deterministic=True)

            # Observe reward and next obs
            obs, reward, done, infos = self.eval_envs.step(action)
            if self.args.render:
                self.eval_envs.venv.venv.remotes[0].send(('render', None))

            eval_masks = torch.FloatTensor([[0.0] if done_ else [1.0]
                                            for done_ in done])
            for info in infos:
                if 'episode' in info.keys():
                    eval_episode_rewards.append(info['episode']['r'])

        self.eval_envs.reset() 
       #eval_envs.close()
        eprew = np.mean(eval_episode_rewards)
        args = self.args
        n_frame = args.num_steps * args.num_processes * args.eval_interval # relative to training session

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


if __name__ == "__main__":
    main()
