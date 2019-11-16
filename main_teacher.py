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

import algo
from arguments import get_args
from envs import make_vec_envs
from model import Policy
from storage import RolloutStorage, CuriosityRolloutStorage
from utils import get_vec_normalize
from visualize import Plotter
from shutil import copyfile

from teachDRL.teachers.algos.alp_gmm import ALPGMM

import csv




def init_agent(actor_critic, args):
    if args.algo == 'a2c':
        agent = algo.A2C(actor_critic, args.value_loss_coef,
                               args.entropy_coef, lr=args.lr,
                               eps=args.eps, alpha=args.alpha,
                               max_grad_norm=args.max_grad_norm,
                               curiosity=args.curiosity, args=args)
    elif args.algo == 'ppo':
        agent = algo.PPO(actor_critic, args.clip_param, args.ppo_epoch, args.num_mini_batch, args.value_loss_coef, args.entropy_coef, lr=args.lr,
                               eps=args.eps,
                               max_grad_norm=args.max_grad_norm)
    elif args.algo == 'acktr':
        agent = algo.A2C(actor_critic, args.value_loss_coef,
                               args.entropy_coef, lr=args.lr,
                               eps=args.eps, alpha=args.alpha,
                               max_grad_norm=args.max_grad_norm,
                               acktr=True,
                               curiosity=args.curiosity, args=args)
    return agent

#class Teacher():
#    def __init__(self, env_param_bounds):
#        '''
#            - list of tuples corresponding to boundaries of parameters
#        '''
#        self.env_param_bounds = env_param_bounds
#        self.num_env_params = len(env_param_bounds)
#        env_param_ranges = [abs(env_param_bounds[i][1] - env_param_bounds[i][0])
#                                for i in range(self.num_env_params)]
#        self.prm_rew = {}
#        self.prm_alp = {}
#        # one episode per trial
#        self.num_trials = args.num_frames / (args.max_step * args.num_processes)
#
#
#    def teach(self):
#        env_param_lw_bounds = [b for a, b in self.env_param_bounds]
#        for i in range(self.num_trials):
#            env_params = np.random(self.num_env_params) * env_param_ranges + env_param_lw_bounds
#            epi_rew = self.student.train()
#
#
#
#class Student():
#    def __init__(self):
#        pass


def main():
    import random
    import gym_micropolis
    import game_of_life

    args = get_args()
    args.log_dir = args.save_dir + '/logs'
    assert args.algo in ['a2c', 'ppo', 'acktr']
    if args.recurrent_policy:
        assert args.algo in ['a2c', 'ppo'], \
            'Recurrent policy is not implemented for ACKTR'
    args.poet = True # hacky

    num_updates = int(args.num_frames) // args.num_steps // args.num_processes

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    graph_name = args.save_dir.split('trained_models/')[1].replace('/', ' ')

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
        if 'golmulti' in args.env_name.lower():
            observation_space_shape = envs.observation_space.shape[1:]
        else:
            observation_space_shape = envs.observation_space.shape
        if len(envs.observation_space.shape) == 3:
            in_w = observation_space_shape[1]
            in_h = observation_space_shape[2]
        else:
            in_w = 1
            in_h = 1
        num_inputs = observation_space_shape[0]
    if isinstance(envs.action_space, gym.spaces.Discrete):
        out_w = 1
        out_h = 1
        if 'Micropolis' in args.env_name: #otherwise it's set
            if args.power_puzzle:
                num_actions = 1
            else:
                num_actions = 19 # TODO: have this already from env
        elif 'GameOfLife' in args.env_name:
            num_actions = 1
        else:
            num_actions = envs.action_space.n
    elif isinstance(envs.action_space, gym.spaces.Box):
        if len(envs.action_space.shape) == 3:
            out_w = envs.action_space.shape[1]
            out_h = envs.action_space.shape[2]
        elif len(envs.action_space.shape) == 1:
            out_w = 1
            out_h = 1
        num_actions = envs.action_space.shape[-1]
    print('num actions {}'.format(num_actions))

    if args.auto_expand:
        args.n_recs -= 1
    actor_critic = Policy(envs.observation_space.shape, envs.action_space,
        base_kwargs={'map_width': args.map_width, 'num_actions': num_actions,
            'recurrent': args.recurrent_policy, 'prebuild': args.prebuild,
            'in_w': in_w, 'in_h': in_h, 'num_inputs': num_inputs,
            'out_w': out_w, 'out_h': out_h},
                     curiosity=args.curiosity, algo=args.algo,
                     model=args.model, args=args)
    if args.auto_expand:
        args.n_recs += 1

    evaluator = None

    if not agent:
        agent = init_agent(actor_critic, args)

   #saved_model = os.path.join(args.save_dir, args.env_name + '.pt')
    if args.load_dir:
        saved_model = os.path.join(args.load_dir, args.env_name + '.tar')
    else:
        saved_model = os.path.join(args.save_dir, args.env_name + '.tar')
    vec_norm = get_vec_normalize(envs)
    alp_gmm = None
    if os.path.exists(saved_model) and not args.overwrite:
        checkpoint = torch.load(saved_model)
        saved_args = checkpoint['args']
        actor_critic.load_state_dict(checkpoint['model_state_dict'])
        actor_critic.to(device)
        actor_critic.cuda()
       #agent = init_agent(actor_critic, saved_args)
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if args.auto_expand:
            if not args.n_recs - saved_args.n_recs == 1:
                print('can expand by 1 rec only from saved model, not {}'.format(args.n_recs - saved_args.n_recs))
                raise Exception
            actor_critic.base.auto_expand()
            print('expanded net: \n{}'.format(actor_critic.base))
        past_steps = checkpoint['past_steps']
        ob_rms = checkpoint['ob_rms']

        past_steps = next(iter(agent.optimizer.state_dict()['state'].values()))['step']
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
        if 'alp_gmm' in checkpoint.keys():
            alp_gmm = checkpoint['alp_gmm']

        if vec_norm is not None:
            vec_norm.eval()
            vec_norm.ob_rms = ob_rms
        saved_args.num_frames = args.num_frames
        saved_args.vis_interval = args.vis_interval
        saved_args.eval_interval = args.eval_interval
        saved_args.overwrite = args.overwrite
        saved_args.n_recs = args.n_recs
        saved_args.intra_shr = args.intra_shr
        saved_args.inter_shr = args.inter_shr
        saved_args.map_width = args.map_width
        saved_args.render = args.render
        saved_args.print_map = args.print_map
        saved_args.load_dir = args.load_dir
        saved_args.experiment_name = args.experiment_name
        saved_args.log_dir = args.log_dir
       #saved_args.save_dir = args.save_dir
        saved_args.num_processes = args.num_processes
        saved_args.n_chan = args.n_chan
        saved_args.prebuild = args.prebuild
        saved_args.log_interval = args.log_interval
        args = saved_args
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
    print('rollout obs shape:{} \nobs shape: {}'.format(rollouts.obs[0].shape, obs.shape))
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)

    start = time.time()
    model = actor_critic.base
    reset_eval = False
    plotter = None
    if args.model == 'FractalNet' or args.model == 'fractal':
        n_cols = model.n_cols
        if args.rule == 'wide1' and args.n_recs > 3:
            col_step = 3
        else:
            col_step = 1
    else:
        n_cols = 0
        col_step = 1
    env_param_bounds = envs.get_param_bounds()
    # in case we want to change this dynamically in the future (e.g., we may
    # not know how much traffic the agent can possibly produce in Micropolis)
    envs.set_param_bounds(env_param_bounds) # start with default bounds
    #num_env_params = len(env_param_bounds)
    num_env_params = 1
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
    if alp_gmm is None:
        alp_gmm = ALPGMM(env_param_lw_bounds, env_param_hi_bounds)
    params_vec = alp_gmm.sample_task()
    params = OrderedDict()
    print('\n env_param_bounds', env_param_bounds)
    print(params_vec)
    trial_remaining = args.max_step
    trial_reward = 0
    for j in range(past_steps, num_updates):
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
            envs.set_params(params)
        trial_remaining -= args.num_steps
        if reset_eval:
            print('post eval reset')
            obs = envs.reset()
            rollouts.obs[0].copy_(obs)
            rollouts.to(device)
            reset_eval = False
       #if np.random.rand(1) < 0.1:
       #    envs.venv.venv.remotes[1].send(('setRewardWeights', None))
        if args.model == 'FractalNet' and args.drop_path:
           #if args.intra_shr and args.inter_shr:
           #    n_recs = np.randint
           #    model.set_n_recs()
            model.set_drop_path()
        if args.model == 'fixed' and model.RAND:
            model.num_recursions = random.randint(1, model.map_width * 2)
        player_act = None
        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                if args.render:
                    if args.num_processes == 1:
                        if not ('Micropolis' in args.env_name or 'GameOfLife' in args.env_name
                                or 'GoL' in args.env_name):
                            envs.venv.venv.render()
                        else:
                            pass
                    else:
                        if not ('Micropolis' in args.env_name or 'GameOfLife' in args.env_name
                                or 'GoL' in args.env_name):
                            envs.render()
                            envs.venv.venv.render()
                        else:
                            pass
                           #envs.venv.venv.remotes[0].send(('render', None))
                           #envs.venv.venv.remotes[0].recv()
                value, action, action_log_probs, recurrent_hidden_states = actor_critic.act(
                        rollouts.obs[step],
                        rollouts.recurrent_hidden_states[step],
                        rollouts.masks[step],
                        player_act=player_act,
                        icm_enabled=args.curiosity,
                        deterministic=False)

            # Observe reward and next obs
            obs, reward, done, infos = envs.step(action)
            if done.all(): # usually this is done elsewhere...
                obs = envs.reset()

            player_act = None
            if args.render:
                if type(infos) is list:
                    info = infos[0]
                else:
                    info = infos # when we're dealing with a multi-env
                if info:
                    if 'player_move' in info.keys():
                        player_act = info['player_move']
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
                    epi_reward = info['episode']['r']
                    episode_rewards.append(epi_reward)
                    trial_reward += epi_reward

            # If done then clean the history of observations.
           #print('done shape: {}'.format(done.shape))
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
        envs.dist_entropy = dist_entropy

        rollouts.after_update()



        total_num_steps = (j + 1) * args.num_processes * args.num_steps

        if not dist_entropy:
            dist_entropy = 0
        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            end = time.time()
            print("Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.5f}/{:.5f}, min/max reward {:.5f}/{:.5f}\n \
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
                evaluator = Evaluator(args, actor_critic, device, envs=envs, vec_norm=vec_norm)


            model = evaluator.actor_critic.base

            col_idx = [-1, *range(0, n_cols, col_step)]
            for i in col_idx:
                evaluator.evaluate(column=i)
           #num_eval_frames = (args.num_frames // (args.num_steps * args.eval_interval * args.num_processes)) * args.num_processes *  args.max_step
           # making sure the evaluator plots the '-1'st column (the overall net)

            if args.vis: #and j % args.vis_interval == 0:
                try:
                    # Sometimes monitor doesn't properly flush the outputs
                    win_eval = evaluator.plotter.visdom_plot(viz, win_eval, evaluator.eval_log_dir, graph_name,
                                  args.algo, args.num_frames, n_graphs= col_idx, header='r')
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
            reset_eval = True

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
                'past_steps': next(iter(agent.optimizer.state_dict()['state'].values()))['step'],
                'model_state_dict': save_model.state_dict(),
                'optimizer_state_dict': optim_save,
                'ob_rms': ob_rms,
                'args': args,
                'alp_gmm': alp_gmm
                }, os.path.join(save_path, args.env_name + ".tar"))

           #save_model = [save_model,
           #              getattr(get_vec_normalize(envs), 'ob_rms', None)]

           #torch.save(save_model, os.path.join(save_path, args.env_name + ".pt"))
           #save_agent = copy.deepcopy(agent)

           #torch.save(save_agent, os.path.join(save_path, args.env_name + '_agent.pt'))
           #torch.save(actor_critic.state_dict(), os.path.join(save_path, args.env_name + "_weights.pt"))

        if args.vis and j % args.vis_interval == 0:
            if plotter is None:
                plotter = Plotter(n_cols, args.log_dir, args.num_processes)
            try:
                # Sometimes monitor doesn't properly flush the outputs
                win_e = plotter.visdom_plot(viz, win, args.log_dir, graph_name,
                                  args.algo, args.num_frames, header='e')
                win_r = plotter.visdom_plot(viz, win, args.log_dir, graph_name,
                                  args.algo, args.num_frames, header='r')
            except IOError:
                pass


class Evaluator(object):
    ''' Manages environments used for evaluation during main training loop.'''
    def __init__(self, args, actor_critic, device, envs=None, vec_norm=None,
            frozen=False):
        ''' frozen: we are not in the main training loop, but evaluating frozen model separately'''
        if frozen:
            self.win_eval = None
            past_steps = args.past_steps
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
                self.eval_log_dir = args.log_dir + "/eval_{}-steps_w{}_{}rec_{}s_{}pl".format(past_steps,
                        args.map_width, args.n_recs, args.max_step, args.prob_life, '.1f')
            else:
                self.eval_log_dir = args.log_dir + "/eval_{}-steps_w{}_{}rec_{}s".format(past_steps,
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
        if envs and 'GoLMulti' in args.env_name:
            self.num_eval_processes = args.num_processes
            self.eval_envs = envs
            self.vec_norm = vec_norm
        else:
            self.num_eval_processes = 20
            print('making envs in Evaluator: ', self.args.env_name, self.args.seed + self.num_eval_processes, self.num_eval_processes,
                        self.args.gamma, self.eval_log_dir, self.args.add_timestep, self.device, True, self.args)
            self.eval_envs = make_vec_envs(
                        self.args.env_name, self.args.seed + self.num_eval_processes, self.num_eval_processes,
                        self.args.gamma, self.eval_log_dir, self.args.add_timestep, self.device, False, args=self.args)
            self.vec_norm = get_vec_normalize(self.eval_envs)
        if self.vec_norm is not None:
            self.vec_norm.eval()
            self.vec_norm.ob_rms = get_vec_normalize(self.eval_envs).ob_rms
        self.tstart = time.time()
        fieldnames = ['r', 'l', 't', 'e']
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
                    # cor the newly added column, we duplicate the last col.'s records
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
                        reader = csv.DictReader(old, fieldnames=('r', 'l', 't', 'e'))
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
        # We let all of our eval_envs to play one episode. Does not support
        # early reset.
        while not done.all():
            with torch.no_grad():
                _, action, eval_recurrent_hidden_states, _ = self.actor_critic.act(
                    obs, eval_recurrent_hidden_states, eval_masks, deterministic=True)

            # Observe reward and next obs
            obs, reward, done, infos = self.eval_envs.step(action)
            if done.all(): # usually this is done elsewhere...
                obs = self.eval_envs.reset()
            if self.args.render:
                if self.args.num_processes == 1:
                    if not ('Micropolis' in self.args.env_name or 'GameOfLife' in self.args.env_name or 'GoL' in self.args.env_name):
                        self.eval_envs.venv.venv.render()
                        self.eval_envs.venv.venv.rcv()
                    else:
                        pass
                       #self.eval_envs.venv.venv.envs[0].render()
                else:
                    if not ('Micropolis' in self.args.env_name or 'GameOfLife' in self.args.env_name or 'GoL in self.args.env_name'):
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

if __name__ == "__main__":
    main()
