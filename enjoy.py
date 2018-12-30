import argparse
import os

import numpy as np
import torch

from model import Policy
from envs import VecPyTorch, make_vec_envs
from utils import get_render_func, get_vec_normalize


parser = argparse.ArgumentParser(description='RL')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10,
                    help='log interval, one log per n updates (default: 10)')
parser.add_argument('--env-name', default='MicropolisEnv-v0',
                    help='environment to train on (default: MicropolisEnv-v0)')
parser.add_argument('--load-dir', default='./trained_models/a2c',
                    help='directory to save agent logs (default: ./trained_models/)')
parser.add_argument('--add-timestep', action='store_true', default=False,
                    help='add timestep to observations')
parser.add_argument('--non-det', action='store_true', default=False,
                    help='whether to use a non-deterministic policy')
parser.add_argument('--map-width', type=int, default=50,
                    help='whether to use a non-deterministic policy')
parser.add_argument('--print-map', action='store_true', default=False)
parser.add_argument('--no-render', action='store_true', default=False)
parser.add_argument('--max-step', type=int, default=None)
parser.add_argument('--model', default=None)
args = parser.parse_args()

args.det = not args.non_det

import gym
import gym_micropolis
env = make_vec_envs(args.env_name, args.seed + 1000, 1,
                            None, None, args.add_timestep, device='cpu',
                            allow_early_resets=False,
                            map_width=args.map_width,
                            print_map=args.print_map, render_gui=not args.no_render, parallel_py2gui=False, max_step = args.max_step)

# Get a render function
# render_func = get_render_func(env)

# We need to use the same statistics for normalization as used in training
actor_critic = Policy(env.observation_space.shape, env.action_space, args=args)
#dict1 = torch.load(os.path.join(args.load_dir, args.env_name + "_weights.pt"))
#dict2 = {}
#for s in dict1.keys():
#    s2 = ''.join(''.join(s.split('.module')).split('.add_bias')).replace('._bias', '.bias')
#    dict2[s2] = dict1[s]
#    if len(dict1[s].shape) == 2 and dict1[s].shape[1] == 1:
#        dict2[s2] = dict2[s2].view((dict2[s2].shape)).view((-1))
#    if len(dict2[s2].shape) == 4 and dict2[s2].shape[1] == 15:
##       dict2[s2] = dict2[s2].view((1,) + dict2[s2].shape)
#        print(dict1[s].shape)
##actor_critic.load_state_dict(dict2)

actor_critic, ob_rms = \
            torch.load(os.path.join(args.load_dir, args.env_name + ".pt"))

vec_norm = get_vec_normalize(env)
if vec_norm is not None:
    vec_norm.eval()
    vec_norm.ob_rms = ob_rms

recurrent_hidden_states = torch.zeros(1, actor_critic.recurrent_hidden_state_size)
masks = torch.zeros(1, 1)

#if render_func is not None:
#    render_func('human')

obs = env.reset()
obs = torch.Tensor(obs)

if args.env_name.find('Bullet') > -1:
    import pybullet as p

    torsoId = -1
    for i in range(p.getNumBodies()):
        if (p.getBodyInfo(i)[0].decode() == "torso"):
            torsoId = i
num_step = 0
player_act = None
while True:
    with torch.no_grad():
        value, action, _, _, recurrent_hidden_states = actor_critic.act(
            obs, recurrent_hidden_states, masks, deterministic=args.det,
            player_act=player_act)

    if num_step >= 5000:
        env.reset()
        num_step = 0
    # Obser reward and next obs
    obs, reward, done, infos = env.step(action)

    player_act = None
    if infos[0]:
        if 'player_move' in infos[0].keys():
            player_act = infos[0]['player_move']

    num_step += 1

    masks.fill_(0.0 if done else 1.0)

    if args.env_name.find('Bullet') > -1:
        if torsoId > -1:
            distance = 5
            yaw = 0
            humanPos, humanOrn = p.getBasePositionAndOrientation(torsoId)
            p.resetDebugVisualizerCamera(distance, yaw, -20, humanPos)

#   if render_func is not None:
#       render_func('human')
