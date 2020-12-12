import argparse
import os
import time

import gym
import numpy as np
import torch

import game_of_life
import gym_city
from arguments import get_parser
from envs import VecPyTorch, make_vec_envs
from model import Policy
from train import Evaluator
from utils import get_render_func, get_space_dims, get_vec_normalize
from opensimplex import OpenSimplex
simplex = OpenSimplex(seed=1)

parser = get_parser()
parser.add_argument('--non-det', action='store_true', default=False,
                    help='whether to use a non-deterministic policy')
parser.add_argument('--active-column', default=None, type=int, help='Run only one vertical column of a fractal model to see what it has learnt independently')
parser.add_argument('--evaluate', action='store_true', default=False, help= 'record trained network\'s performance')

args = parser.parse_args()
args.render = True

args.det = not args.non_det



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
   #checkpoint = torch.load(os.path.join(args.load_dir, env_name + '.tar'),
    checkpoint = torch.load(args.load_dir,
                            map_location=map_location)
except FileNotFoundError:
    print('load-dir does not start with valid gym environment id, using command line args')
    env_name = args.env_name
   #checkpoint = torch.load(os.path.join(args.load_dir, env_name + '.tar'),
    checkpoint = torch.load(args.load_dir,
                        map_location=map_location)
args.load_dir = '/'.join(args.load_dir.split('/')[:-1])
saved_args = checkpoint['args']
if not hasattr(saved_args, 'poet'):
    pass
else:
    args.poet = saved_args.poet
#past_steps = checkpoint['past_steps']
#args.past_steps = past_steps
env_name = saved_args.env_name

if 'Micropolis' in env_name:
    args.power_puzzle = saved_args.power_puzzle

if not args.evaluate and not 'GoLMulti' in env_name:
    # assume we just want to observe/interact w/ a single env.
    args.num_proc = 1
dummy_args = args
param_rew = len(args.env_params) > 0
env = make_vec_envs(env_name, args.seed + 1000, 1,
                    None, args.load_dir, args.add_timestep, device=device,
                    allow_early_resets=False,
                    param_rew=param_rew, env_params=args.env_params,
                    args=dummy_args)
print(args.load_dir)

# Get a render function
# render_func = get_render_func(env)

in_w, in_h, num_inputs, out_w, out_h, num_actions = get_space_dims(env, args)

print('num_actions: {}'.format(num_actions))

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
                    'out_w': out_w, 'out_h': out_h, 'num_actions':num_actions},
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
actor_critic.to(device)

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
    args.env_params = saved_args.env_params
   #args.past_frames = 0
   #args.param_rew = False
    print('steps: ', saved_args.max_step, '\n')
    evaluator = Evaluator(args, actor_critic, device, frozen=True)

    while True:
        if hasattr(actor_critic.base, 'n_cols'):
            for i in range(-1, actor_critic.base.n_cols):
                eval_score = evaluator.evaluate(column=i)
        else:
            eval_score = evaluator.evaluate()
        print('score: {}'.format(eval_score))

obs = None
#obs = torch.Tensor(obs)
num_step = 0
player_act = None
env_done = False

last_rew = 0
MODULATE_TRGS = True
while True:
    if obs is None:
        obs = env.reset()
    if args.det:
        if np.random.random() < -0.1:
            deterministic = False
        else: deterministic = True
    else:
        deterministic = args.det
    with torch.no_grad():
        value, action, _, recurrent_hidden_states = actor_critic.act(
            obs, recurrent_hidden_states, masks, 
            deterministic=deterministic,
            player_act=player_act)

    if env_done:
        last_rew = 0
        num_step = 0
        print('metrics, end of episode:\n', env.venv.venv.envs[0].metrics)
    # Observe reward and next obs
    obs, reward, done, infos = env.step(action)
   #print('reward: ', reward)
    env_done = done[0] # assume we have only one env.
    last_rew = reward
    env.venv.venv.envs[0].render()
   #time.sleep(0.08)

    player_act = None

    if infos[0]:
        if 'player_move' in infos[0].keys():
            player_act = infos[0]['player_move']

    num_step += 1

   #masks.fill_(0.0 if done else 1.0)

#   if args.env_name.find('Bullet') > -1:
#       if torsoId > -1:
#           distance = 5
#           yaw = 0
#           humanPos, humanOrn = p.getBasePositionAndOrientation(torsoId)
#           p.resetDebugVisualizerCamera(distance, yaw, -20, humanPos)
