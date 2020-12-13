from pdb import set_trace as T
import copy
import math
import csv
import glob
import os
import random
import time
from collections import deque
from shutil import copyfile

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import algo
#import game_of_life
#import gym_city
#import gym_pcgrl
from arguments import get_args
from envs import make_vec_envs
from evaluate import Evaluator
from model import Policy
from storage import CuriosityRolloutStorage, RolloutStorage
from utils import get_space_dims, get_vec_normalize
from visualize import Plotter


def main():
    trainer = Trainer()
    trainer.main()

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

class Teacher():
    def __init__(self):
        pass

class Trainer():
    def get_fieldnames(self):
        return ['r','l','t','e']

    def __init__(self, envs=None, args=None):
        if args is None:
            args = get_args()
        self.n_train = 0
        self.fieldnames = self.get_fieldnames()
        self.n_frames = 0
        self.best_eval_score = float('-inf')


        assert args.algo in ['a2c', 'ppo', 'acktr']

        if args.recurrent_policy:
            assert args.algo in ['a2c', 'ppo'], \
                    'Recurrent policy is not implemented for ACKTR'


        torch.manual_seed(args.seed)

        if args.cuda:
            print('CUDA ENABLED')
            torch.cuda.manual_seed(args.seed)

        graph_name = args.save_dir.split('trained_models/')[1].replace('/', ' ')
        self.graph_name = graph_name

        actor_critic = False
        agent = False
        past_frames = 0
        try:
            if hasattr(args, 'log_dir'):
                os.makedirs(args.log_dir)
        except OSError:
            files = glob.glob(os.path.join(args.log_dir, '*.monitor.csv'))

            for f in files:
                if args.overwrite:
                    os.remove(f)
                else:
                    if not args.load:
                        raise Exception('experiment exists, not overwriting')
        torch.set_num_threads(torch.get_num_threads())
        device = torch.device("cuda:0" if args.cuda else "cpu")
        self.device = device

        if args.vis:
            from visdom import Visdom
            viz = Visdom(port=args.port)
            self.viz = viz
            win = None
            self.win = win
            win_eval = None
            self.win_eval = win_eval
        print('env name: {}'.format(args.env_name))

        if envs is None:
            envs = self.make_vec_envs(args)
        self.in_w, self.in_h, self.num_inputs, self.out_w, self.out_h, self.num_actions = get_space_dims(envs, args)
        self.envs = envs

        if args.auto_expand:
            args.n_recs -= 1
        actor_critic = self.init_policy(envs, args)

        if not agent:
            agent = init_agent(actor_critic, args)

        if args.auto_expand:
            args.n_recs += 1

        evaluator = None
        self.evaluator = evaluator
        vec_norm = get_vec_normalize(envs)
        self.vec_norm = vec_norm
       #saved_model = os.path.join(args.save_dir, args.env_name + '.pt')
        if args.load_dir:
            saved_model = os.path.join(args.load_dir, args.env_name + '.tar')
        else:
            saved_model = os.path.join(args.save_dir, args.env_name + '.tar')
        self.checkpoint = None

        if os.path.exists(saved_model) and not args.overwrite:
           #print('current actor_critic params: {}'.format(actor_critic.parameters()))
            checkpoint = torch.load(saved_model)
            self.checkpoint = checkpoint
            saved_args = checkpoint['args']
            actor_critic.load_state_dict(checkpoint['model_state_dict'])
            opt = agent.optimizer.state_dict()
            opt_load = checkpoint['optimizer_state_dict']

            for o, l in zip(opt, opt_load):
               #print(o, l)
                param = opt[o]
                param_load = opt_load[l]
               #print('current: {}'.format(param), 'load: {}'.format(param_load))
               #print(param_load.keys())
               #params = param_load[0]['params']
               #param[0]['params'] = params
               #for m, n in zip(param, param_load):
               #    for p, q in zip(m, n):
               #        print(p, q)
               #        if type(m[p]) == list:
               #            print(len(m[p]), len(n[q]))
               #            agent.optimizer[m][p] = m[q]

           #print(agent.optimizer.state_dict()['param_groups'])
           #print('\n')
           #print(checkpoint['model_state_dict'])
            actor_critic.to(self.device)
           #actor_critic.cuda()
           #agent = init_agent(actor_critic, saved_args)
            agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            if args.auto_expand:
                if not args.n_recs - saved_args.n_recs == 1:
                    print('can expand by 1 rec only from saved model, not {}'.format(args.n_recs - saved_args.n_recs))
                    raise Exception
                actor_critic.base.auto_expand()
                print('expanded net: \n{}'.format(actor_critic.base))
                # TODO: Are we losing something crucial here? Probably not ideal.
                agent = init_agent(actor_critic, args)

            past_frames = checkpoint['n_frames']
            ob_rms = checkpoint['ob_rms']
           #past_steps = next(iter(agent.optimizer.state_dict()['state'].values()))['step']
            print('Resuming from frame {}'.format(past_frames))

           #print(type(next(iter((torch.load(saved_model))))))
           #actor_critic, ob_rms = \
           #        torch.load(saved_model)
           #agent = \
           #    torch.load(os.path.join(args.save_dir, args.env_name + '_agent.pt'))
           #if not agent.optimizer.state_dict()['state'].values():
           #    past_steps = 0
           #else:

           #    raise Exception

            if vec_norm is not None:
                vec_norm.eval()
                vec_norm.ob_rms = ob_rms
           #saved_args.num_frames = args.num_frames
           #saved_args.vis_interval = args.vis_interval
           #saved_args.eval_interval = args.eval_interval
           #saved_args.overwrite = args.overwrite
           #saved_args.n_recs = args.n_recs
           #saved_args.intra_shr = args.intra_shr
           #saved_args.inter_shr = args.inter_shr
           #saved_args.map_width = args.map_width
           #saved_args.render = args.render
           #saved_args.print_map = args.print_map
           #saved_args.load_dir = args.load_dir
           #saved_args.experiment_name = args.experiment_name
           #saved_args.log_dir = args.log_dir
           #saved_args.save_dir = args.save_dir
           #saved_args.num_processes = args.num_processes
           #saved_args.n_chan = args.n_chan
           #saved_args.prebuild = args.prebuild
           #args = saved_args
           #args.eval_interval = saved_args.eval_interval
            args.save_dir = saved_args.save_dir
            args.model = saved_args.model
            args.env_name = saved_args.env_name
           #args.poet = saved_args.poet
        actor_critic.to(device)

        updates_remaining = int(args.num_frames - past_frames) // (args.num_steps * args.num_processes)
        self.n_frames = self.n_frames + past_frames
        self.past_frames = past_frames

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
        self.model = model = actor_critic.base
        self.reset_eval = False
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
        self.col_step = col_step
        self.updates_remaining = updates_remaining
        self.envs = envs
        self.start = start
        self.rollouts = rollouts
        self.args = args
        self.actor_critic = actor_critic
        self.plotter = plotter
        self.agent = agent
        self.episode_rewards = episode_rewards
        self.n_cols = n_cols
        self.multi_env = 'golmulti' in args.env_name.lower()

    def make_vec_envs(self, args):
        if args is None:
            args = get_args()
        args.log_dir = args.save_dir + '/logs'
        args.log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.log_dir)
        if not hasattr(args, 'param_rew'):
            args.param_rew = False
        try:
            envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                    args.gamma, args.log_dir, args.add_timestep, self.device, False, None,
                    param_rew=args.param_rew, env_params=args.env_params, args=args)
        except gym.error.UnregisteredEnv as err:
            print(err, '\n available envs: \n{}'.format(gym.envs.registry.all()))
            raise err

        return envs

    def init_policy(self, envs, args):
        actor_critic = Policy(envs.observation_space.shape, envs.action_space,
                              base_kwargs={
                                  'map_width': args.map_width,
                                  'recurrent': args.recurrent_policy,
                                  'prebuild': args.prebuild,
                                  'in_w': self.in_w,
                                  'in_h': self.in_h,
                                  'num_inputs': self.num_inputs,
                                  'out_w': self.out_w,
                                  'out_h': self.out_h,
                                  'num_actions': self.num_actions,
                                  },
                              curiosity=args.curiosity,
                              algo=args.algo,
                              model=args.model,
                              args=args
                            )

        return actor_critic


    def main(self):

        # Main training loop

        for self.n_train in range(self.updates_remaining):
            self.train()

    def render(self):
        args = self.args
        envs = self.envs
        multi_env = self.multi_env

        if self.args.num_processes == 1:
            if not ('Micropolis' in args.env_name or 'GameOfLife' in args.env_name or multi_env):
                envs.venv.venv.render()
            else:
                pass
        else:
            pass
           #if not ('Micropolis' in args.env_name or 'GameOfLife' in args.env_name or multi_env):
           #    envs.render()
           #    envs.venv.venv.render()
           #else:
           #    pass
           #   #envs.venv.venv.remotes[0].send(('render', None))
           #   #envs.venv.venv.remotes[0].recv()

    def step(self):
        player_act = self.player_act
        n_step = self.n_step
        args = self.args
        episode_rewards = self.episode_rewards
        actor_critic = self.actor_critic
        envs = self.envs
        rollouts = self.rollouts
        with torch.no_grad():
            if args.render:
                self.render()

            value, action, action_log_probs, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[n_step],
                    rollouts.recurrent_hidden_states[n_step],
                    rollouts.masks[n_step],
                    player_act=player_act,
                    icm_enabled=args.curiosity,
                    deterministic=False)

        # Observe reward and next obs
        obs, reward, done, infos = envs.step(action)
#       print(obs[0])

        player_act = None

        if args.render:
            pass
           #print('infos be: {}'.format(infos))
           #if infos[0]:
           #    if 'player_move' in infos[0].keys():
           #        player_act = infos[0]['player_move']

        if args.curiosity:
            # run icm
            with torch.no_grad():


                feature_state, feature_state_pred, action_dist_pred = actor_critic.icm_act(
                        (rollouts.obs[n_step], obs, action_bin)
                        )

            intrinsic_reward = args.eta * ((feature_state - feature_state_pred).pow(2)).sum() / 2.

            if args.no_reward:
                reward = 0
            reward += intrinsic_reward.cpu()

        if type(infos) is dict:
            infos = [infos]

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
        self.n_frames += self.args.num_processes
        reward = reward.squeeze(-1)
       #print('step rew shape {}'.format(reward.shape))

        return obs, reward, done, infos



    def train(self):
        evaluator = self.evaluator
        episode_rewards = self.episode_rewards
        args = self.args
        actor_critic = self.actor_critic
        rollouts = self.rollouts
        agent = self.agent
        envs = self.envs
        plotter = self.plotter
        n_train = self.n_train
        start = self.start
        plotter = self.plotter
        n_cols = self.n_cols
        model = self.model
        device = self.device
        vec_norm = self.vec_norm
        n_frames = self.n_frames

        if self.reset_eval:
            obs = envs.reset()
            rollouts.obs[0].copy_(obs)
            rollouts.to(device)
            self.reset_eval = False

        if args.model == 'FractalNet' and args.drop_path:
            model.set_drop_path()

        if args.model == 'fixed' and model.RAND:
            model.num_recursions = random.randint(1, model.map_width * 2)
        self.player_act = None

        cum_rews = torch.zeros(self.args.num_processes)

        for self.n_step in range(args.num_steps):
            # Sample actions
            _, rewards, dones, infos = self.step()
           #print('rews', rewards.shape)
            rewards = rewards.squeeze(-1)
           #print('cum_rews', cum_rews.shape)
            cum_rews += rewards

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



       #total_num_steps = (n_train + 1) * args.num_processes * args.num_steps

        if not dist_entropy:
            dist_entropy = 0
       #print(episode_rewards)
       #if torch.max(rollouts.rewards) > 0:
       #    print(rollouts.rewards)
        if args.log and n_train % args.log_interval == 0  and len(episode_rewards) > 1:
            end = time.time()
            print("Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.6f}/{:.6f}, min/max reward {:.6f}/{:.6f}\n \
dist entropy {:.6f}, val/act loss {:.6f}/{:.6f},".
                      format(n_train, # TODO: deal with reloading
                       self.n_frames,
                       int((self.n_frames - self.past_frames) / (end - start)),
                       len(episode_rewards),
                       round(np.mean(episode_rewards), 6),
                       round(np.median(episode_rewards), 6),
                       round(np.min(episode_rewards), 6),
                       round(np.max(episode_rewards), 6), round(dist_entropy, 6),
                       round(value_loss, 6), round(action_loss, 6)))

            if args.curiosity:
                print("fwd/inv icm loss {:.1f}/{:.1f}\n".
                format(
                       fwd_loss, inv_loss))

        if (args.eval_interval and args.eval_interval != -1 and len(episode_rewards) > 1
                and n_train % args.eval_interval == 0):

            if evaluator is None:
                evaluator = Evaluator(args, actor_critic, device, envs=envs, vec_norm=vec_norm,
                        fieldnames=self.fieldnames)
                self.evaluator = evaluator

            col_idx = [-1, *[i for i in range(0, n_cols, self.col_step)]]

            for i in col_idx:
                print('evaluating column {}'.format(i))
                eval_score = evaluator.evaluate(column=i)
                if eval_score > self.best_eval_score:
                    self.best_eval_score = eval_score
           #num_eval_frames = (args.num_frames // (args.num_steps * args.eval_interval * args.num_processes)) * args.num_processes *  args.max_step
           # making sure the evaluator plots the '-1'st column (the overall net)
            viz = self.viz
            win_eval = self.win_eval
            graph_name = self.graph_name

            if args.vis: #and n_train % args.vis_interval == 0:
                try:
                    # Sometimes monitor doesn't properly flush the outputs
                    win_eval = evaluator.plotter.visdom_plot(viz, win_eval, evaluator.eval_log_dir, graph_name,
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
            self.reset_eval = True

        if args.save and n_train % args.save_interval == 0 and args.save_dir != "":
            self.save(args, actor_critic, envs, agent)
        if args.vis and self.n_train % args.vis_interval == 0:
            self.visualize(plotter)
            print('visualize train')


        return cum_rews, dones, infos

    def save(self, args, actor_critic, envs, agent):
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
        self.agent = agent
        self.save_model = save_model
        self.optim_save = optim_save
        self.args = args
        self.ob_rms = ob_rms
        if self.n_train % args.checkpoint_interval == 0:
            save_path = os.path.join(save_path, 'checkpoint_{}'.format(self.n_frames))
            try:
                os.mkdir(save_path)
            except FileExistsError:
                pass
        torch.save(self.get_save_dict(), os.path.join(save_path, args.env_name + ".tar"))

       #save_model = [save_model,
       #              getattr(get_vec_normalize(envs), 'ob_rms', None)]

       #torch.save(save_model, os.path.join(save_path, args.env_name + ".pt"))
       #save_agent = copy.deepcopy(agent)

       #torch.save(save_agent, os.path.join(save_path, args.env_name + '_agent.pt'))
       #torch.save(actor_critic.state_dict(), os.path.join(save_path, args.env_name + "_weights.pt"))

        print('model saved at {}'.format(save_path))



    def visualize(self, plotter=None, log_dir=None):
        n_cols = self.n_cols
        args = self.args

        if log_dir == None:
            log_dir = args.log_dir

        if plotter is None:
            plotter = Plotter(n_cols, args.log_dir, args.num_processes)
        self.plotter = plotter

        try:
            # Sometimes monitor doesn't properly flush the outputs
            viz = self.viz
            win = self.win
            graph_name = self.graph_name
            if not args.n_rand_envs:
                n_plot_frames = self.n_frames
            else:
                n_plot_frames = math.ceil(self.n_frames * (args.n_rand_envs / args.num_processes)) + 10000
            win = plotter.visdom_plot(viz, win, args.log_dir, graph_name,
                              args.algo, n_plot_frames, max_env_id=args.n_rand_envs)
        except IOError:
            pass

    def get_save_dict(self):
        agent = self.agent
        save_model = self.save_model
        optim_save = self.optim_save
        ob_rms = self.ob_rms
        args = self.args
        # experimental:
        d = {
           #'past_steps': next(iter(agent.optimizer.state_dict()['state'].values()))['step'],
            'n_frames': self.n_frames,
            'model_state_dict': save_model.state_dict(),
            'optimizer_state_dict': optim_save,
            'ob_rms': ob_rms,
            'args': args,
            }

        return d


if __name__ == "__main__":
    main()
