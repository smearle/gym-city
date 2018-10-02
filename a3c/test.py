from __future__ import division
from setproctitle import setproctitle as ptitle
import torch
from environment import atari_env, micropolis_env
from utils import setup_logger
import model
from player_util import Agent
from torch.autograd import Variable
import time
import logging


def test(args, shared_model, env_conf):
#   print('IN TEST')
    ptitle('Test Agent')
    gpu_id = args.gpu_ids[-1]
    log = {}
    setup_logger('{}_log'.format(args.env), r'{0}{1}_log'.format(
        args.log_dir, args.env))
    log['{}_log'.format(args.env)] = logging.getLogger('{}_log'.format(
        args.env))
    setup_logger('{}_map_log'.format(args.env), r'{0}{1}_map_log'.format(
        args.log_dir, args.env))
    log['{}_map_log'.format(args.env)] = logging.getLogger('{}_map_log'.format(
        args.env))

    d_args = vars(args)
    for k in d_args.keys():
        log['{}_log'.format(args.env)].info('{0}: {1}'.format(k, d_args[k]))

    torch.manual_seed(args.seed)
    if gpu_id >= 0:
        torch.cuda.manual_seed(args.seed)
    if 'micropolis' in args.env.lower():
        import gym_micropolis
        env = micropolis_env(args.env, env_conf, args)
    else:
 #      print('using atari env for test')
        env = atari_env(args.env, env_conf, args)
    reward_sum = 0
    entropy_sum = 0
    start_time = time.time()
    num_tests = 0
    reward_total_sum = 0
    player = Agent(None, env, args, None)
    player.gpu_id = gpu_id
    if 'micropolis' in args.env.lower():
        modelInit = getattr(model, args.design_head)
        player.model = modelInit(player.env.observation_space.shape[0],
                                     player.env.action_space, player.env.env.env.MAP_X)
        player.lstm_sizes = player.model.getMemorySizes()
        if not 'arcade' in args.env.lower():
            player.lstm_size = (1, 16, 
                    player.env.env.env.MAP_X, env.env.env.MAP_Y)
    else:
        player.model = A3Clstm(player.env.observation_space.shape[0],
                           player.env.action_space)

    player.state = player.env.reset()
    player.eps_len += 2
    player.state = torch.from_numpy(player.state).float()
    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            player.model = player.model.cuda()
            player.state = player.state.cuda()
    flag = True
    max_score = 0
    while True:
        
        if flag:
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.model.load_state_dict(shared_model.state_dict())
            else:
                player.model.load_state_dict(shared_model.state_dict())
            player.model.eval()
            flag = False

        player.action_test()
        reward_sum += player.reward
        entropy_sum += player.entropy.data.item()

        if player.done and not player.info:
            state = player.env.reset()
            player.eps_len += 2
            player.state = torch.from_numpy(state).float()
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.state = player.state.cuda()
        elif player.info:
            flag = True
            num_tests += 1
            reward_total_sum += reward_sum
            reward_mean = reward_total_sum / num_tests
            log['{}_log'.format(args.env)].info(
                    "Time {0}, episode reward {1:1.5e}, entropy {4:1.5e} episode length {2}, reward mean {3:1.5e}".
                format(
                    time.strftime("%Hh %Mm %Ss",
                                  time.gmtime(time.time() - start_time)),
                    reward_sum, player.eps_len, reward_mean, entropy_sum))    
            import numpy as np
            np.set_printoptions(threshold=400)
            log['{}_map_log'.format(args.env)].info(
                    '\n{}'.format(np.array2string(np.add(player.env.env.env.micro.map.zoneMap[-1],
                                   np.full((player.env.env.env.MAP_X, 
                                            player.env.env.env.MAP_Y), 2))).replace('\n ', '').replace('][',']\n[')
                                    .replace('[[','[').replace(']]',']')))

            if args.save_max and reward_sum >= max_score:
                max_score = reward_sum
                if gpu_id >= 0:
                    with torch.cuda.device(gpu_id):
                        state_to_save = player.model.state_dict()
                        torch.save(state_to_save, '{0}{1}.dat'.format(
                            args.save_model_dir, args.env))
                else:
                    state_to_save = player.model.state_dict()
                    torch.save(state_to_save, '{0}{1}.dat'.format(
                        args.save_model_dir, args.env))

            reward_sum = 0
            entropy_sum = 0
            player.eps_len = 0
            state = player.env.reset()
            player.eps_len += 2
            time.sleep(10)
            player.state = torch.from_numpy(state).float()
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.state = player.state.cuda()
