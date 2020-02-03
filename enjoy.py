import argparse
import os
import time

import numpy as np
import torch

from model import Policy
from envs import VecPyTorch, make_vec_envs
from utils import get_render_func, get_vec_normalize

from arguments import get_parser
from train import Evaluator


parser = get_parser()
parser.add_argument('--non-det', action='store_true', default=False,
                    help='whether to use a non-deterministic policy')
parser.add_argument('--active-column', default=None, type=int, help='Run only one vertical column of a fractal model to see what it has learnt independently')
parser.add_argument('--evaluate', action='store_true', default=False, help= 'record trained network\'s performance')

args = parser.parse_args()
args.render = True

args.det = not args.non_det


import gym
import gym_city
import game_of_life

env_name = args.load_dir.split('/')[-1].split('_')[0]
if torch.cuda.is_available() and not args.no_cuda:
    args.cuda = True
    device = torch.device('cuda')
    map_location = torch.device('cuda')
else:
    args.cuda = False
    device = torch.device('cpu')
    map_location = torch.device('cpu')
try:
    checkpoint = torch.load(os.path.join(args.load_dir, env_name + '.tar'),
                            map_location=map_location)
except FileNotFoundError:
    print('load-dir does not start with valid gym environment id, using command line args')
    env_name = args.env_name
    checkpoint = torch.load(os.path.join(args.load_dir, env_name + '.tar'),
                        map_location=map_location)
saved_args = checkpoint['args']
#past_steps = checkpoint['past_steps']
#args.past_steps = past_steps
env_name = saved_args.env_name

if 'Micropolis' in env_name:
    args.power_puzzle = saved_args.power_puzzle

if not args.evaluate and not 'GoLMulti' in env_name:
    # assume we just want to observe/interact w/ a single env.
    args.num_proc = 1
dummy_args = args
env = make_vec_envs(env_name, args.seed + 1000, 1,
                    None, args.load_dir, args.add_timestep, device=device,
                    allow_early_resets=False,
                    args=dummy_args)
print(args.load_dir)

# Get a render function
# render_func = get_render_func(env)

if isinstance(env.observation_space, gym.spaces.Discrete):
    in_width = 1
    num_inputs = env.observation_space.n
elif isinstance(env.observation_space, gym.spaces.Box):
    if len(env.observation_space.shape) == 3:
        in_w = env.observation_space.shape[1]
        in_h = env.observation_space.shape[2]
    else:
        in_w = 1
        in_h = 1
    num_inputs = env.observation_space.shape[0]
if isinstance(env.action_space, gym.spaces.Discrete):
    out_w = 1
    out_h = 1
    if 'Micropolis' in env_name:
        num_actions = env.venv.venv.envs[0].num_tools
    elif 'GameOfLife' in env_name:
        num_actions = 1
    else:
        num_actions = env.action_space.n
elif isinstance(env.action_space, gym.spaces.Box):
    out_w = env.action_space.shape[0]
    out_h = env.action_space.shape[1]
    num_actions = env.action_space.shape[-1]

#actor_critic, ob_rms = \
#            torch.load(os.path.join(args.load_dir, args.env_name + ".pt"))

if saved_args.model == 'fractal':
    saved_args.model = 'FractalNet'
# We need to use the same statistics for normalization as used in training
saved_args.n_chan = args.n_chan
saved_args.val_kern = int(args.val_kern)
actor_critic = Policy(env.observation_space.shape, env.action_space,
        base_kwargs={'map_width': args.map_width,
                     'recurrent': args.recurrent_policy,
                    'in_w': in_w, 'in_h': in_h, 'num_inputs': num_inputs,
            'out_w': out_w, 'out_h': out_h },
                     curiosity=args.curiosity, algo=saved_args.algo,
                     model=saved_args.model, args=saved_args)
actor_critic.to(device)
torch.nn.Module.dump_patches = True
actor_critic.load_state_dict(checkpoint['model_state_dict'])
ob_rms = checkpoint['ob_rms']
if 'fractal' in args.model.lower():
    new_recs = args.n_recs - saved_args.n_recs
    for nr in range(new_recs):
        actor_critic.base.auto_expand()
    print('expanded network:\n', actor_critic.base)
    if args.active_column is not None \
            and hasattr(actor_critic.base, 'set_active_column'):
        actor_critic.base.set_active_column(args.active_column)
vec_norm = get_vec_normalize(env)
if vec_norm is not None:
    vec_norm.eval()
    vec_norm.ob_rms = ob_rms

recurrent_hidden_states = torch.zeros(1, actor_critic.recurrent_hidden_state_size)
masks = torch.zeros(1, 1)

#if render_func is not None:
#    render_func('human')



if args.env_name.find('Bullet') > -1:
    import pybullet as p

    torsoId = -1
    for i in range(p.getNumBodies()):
        if (p.getBodyInfo(i)[0].decode() == "torso"):
            torsoId = i

# You may need to fiddle with this when loading old models
if args.evaluate:
    env.close() # only needed it to probe obs/act space shape
   #saved_args.num_processes = args.num_processes
   #saved_args.vis_interval = args.vis_interval
   #saved_args.render = args.render
   #saved_args.prob_life = args.prob_life
   #saved_args.record = args.record
   #args.poet = saved_args.poet
    args.env_name = saved_args.env_name
    args.log_dir = args.load_dir
    args.model = saved_args.model
    args.rule = saved_args.rule
   #args.n_recs = saved_args.n_recs
    args.intra_shr = saved_args.intra_shr
    args.inter_shr = saved_args.inter_shr
    args.n_chan = saved_args.n_chan
    args.val_kern = saved_args.val_kern
    print('steps: ', saved_args.max_step, '\n')
    evaluator = Evaluator(args, actor_critic, device, frozen=True)
    while True:
        if hasattr(actor_critic.base, 'n_cols'):
            for i in range(-1, actor_critic.base.n_cols):
                evaluator.evaluate(column=i)

obs = env.reset()
#obs = torch.Tensor(obs)
num_step = 0
player_act = None
env_done = False
while True:
    with torch.no_grad():
        value, action, _, recurrent_hidden_states = actor_critic.act(
            obs, recurrent_hidden_states, masks, deterministic=args.det,
            player_act=player_act)

    if env_done:
        num_step = 0
    # Observe reward and next obs
    obs, reward, done, infos = env.step(action)
    env_done = done[0] # assume we have only one env.
   #env.venv.venv.envs[0].render()
   #time.sleep(0.08)

    player_act = None
    if infos[0]:
        if 'player_move' in infos[0].keys():
            player_act = infos[0]['player_move']

    num_step += 1

   #masks.fill_(0.0 if done else 1.0)

    if args.env_name.find('Bullet') > -1:
        if torsoId > -1:
            distance = 5
            yaw = 0
            humanPos, humanOrn = p.getBasePositionAndOrientation(torsoId)
            p.resetDebugVisualizerCamera(distance, yaw, -20, humanPos)





