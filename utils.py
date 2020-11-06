import gym
import torch
import torch.nn as nn

from envs import VecNormalize


def get_space_dims(envs, args):
    if isinstance(envs.observation_space, gym.spaces.Discrete):
        num_inputs = envs.observation_space.n
    elif isinstance(envs.observation_space, gym.spaces.Box):
        if 'golmulti' in args.env_name.lower():
           #observation_space_shape = envs.observation_space.shape[1:]
            observation_space_shape = envs.observation_space.shape
        else:
            observation_space_shape = envs.observation_space.shape
           #if '-wide-' in args.env_name:
           #    observation_space_shape = (observation_space_shape[1],
           #            observation_space_shape[2], observation_space_shape[0])

        if len(observation_space_shape) == 3:
           #print(observation_space_shape)
           #raise Exception
            in_w = observation_space_shape[1]
            in_h = observation_space_shape[2]
        else:
            in_w = 1
            in_h = 1
        num_inputs = observation_space_shape[0]

    if isinstance(envs.action_space, gym.spaces.Dict):
        if 'position' in envs.action_space.spaces:
            out_w = int(envs.action_space.spaces['position'].high[0])
            out_h = int(envs.action_space.spaces['position'].high[1])
            num_actions = envs.action_space.spaces['build'].n
        elif 'map' in envs.action_space.spaces:
            out_w = args.map_width
            out_h = args.map_width
            num_actions = envs.action_space.spaces['act'].n + envs.action_space.spaces['rotation'].n
        else:
            out_w = args.map_width
            out_h = args.map_width
            num_actions = envs.action_space.spaces['act'].n + envs.action_space.spaces['rotation'].n

    if isinstance(envs.action_space, gym.spaces.Discrete) or\
        isinstance(envs.action_space, gym.spaces.Box):
        out_w = args.map_width
        out_h = args.map_width
        num_actions = int(envs.action_space.n // (out_w * out_h))

        #if 'Micropolis' in args.env_name: #otherwise it's set
        #    if args.power_puzzle:
        #        num_actions = 1
        #    else:
        #        num_actions = 19 # TODO: have this already from env
        #elif 'GameOfLife' in args.env_name:
        #    num_actions = 1
        ## for PCGRL

        #if '-wide' in args.env_name:
        #    #TODO: should be done like this for all envs!
        #    print('obs space shape: {}'.format(envs.observation_space.shape))
        #    map_width = envs.observation_space.shape[1]
        #    map_height = envs.observation_space.shape[2]
        #    out_w = map_width
        #    out_h = map_height
        #    print(envs.action_space.n)
        #    num_actions = envs.action_space.n / (map_width * map_height)
        #    num_actions = int(num_actions)
    elif isinstance(envs.action_space, gym.spaces.Box):
        if len(envs.action_space.shape) == 3:
            out_w = envs.action_space.shape[1]
            out_h = envs.action_space.shape[2]
        elif len(envs.action_space.shape) == 1:
            out_w = 1
            out_h = 1
        num_actions = envs.action_space.shape[-1]
    # for PCGRL
    elif isinstance(envs.action_space, gym.spaces.MultiDiscrete):
       #out_w = envs.action_space.nvec[0]
       #out_h = envs.action_space.nvec[1]
       #num_actions = envs.action_space.nvec[2]
        num_actions = sum([n for n in envs.action_space.nvec])
        out_w = 1
        out_h = 1
    print('envs.action_space: {}'.format(envs.action_space))
    print('observation space {}'.format(observation_space_shape))
    print('out w, out_h, num actions {}, {}, {}'.format(out_w, out_h, num_actions))

    return in_w, in_h, num_inputs, out_w, out_h, num_actions




# Get a render function
def get_render_func(venv):
    if hasattr(venv, 'envs'):
        return venv.envs[0].render
    elif hasattr(venv, 'venv'):
        return get_render_func(venv.venv)
    elif hasattr(venv, 'env'):
        return get_render_func(venv.env)

    return None


def get_vec_normalize(venv):
    if isinstance(venv, VecNormalize):
        return venv
    elif hasattr(venv, 'venv'):
        return get_vec_normalize(venv.venv)

    return None


# Necessary for my KFAC implementation.
class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data)
    bias_init(module.bias.data)

    return module


# https://github.com/openai/baselines/blob/master/baselines/common/tf_util.py#L87
def init_normc_(weight, gain=1):
    weight.normal_(0, 1)
    weight *= gain / torch.sqrt(weight.pow(2).sum(1, keepdim=True))
