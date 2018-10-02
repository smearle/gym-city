from __future__ import division
from setproctitle import setproctitle as ptitle
import torch
import torch.optim as optim
from environment import atari_env, micropolis_env
from utils import ensure_shared_grads, setup_logger
import model
from player_util import Agent
from torch.autograd import Variable
import numpy as np
import logging
import time


def train(rank, args, shared_model, optimizer, env_conf):
    start_time = time.time()
    ptitle('Training Agent: {}'.format(rank))
   #log = {}

   #setup_logger('{}_train_log'.format(args.env), r'{0}{1}_train_log'.format(
   #    args.log_dir, args.env))
   #log['{}_train_log'.format(args.env)] = logging.getLogger(
   #        '{}_train_log'.format(args.env))

    gpu_id = args.gpu_ids[rank % len(args.gpu_ids)]
    torch.manual_seed(args.seed + rank)
    if gpu_id >= 0:
        torch.cuda.manual_seed(args.seed + rank)
    if 'micropolis' in args.env.lower():
        env = micropolis_env(args.env, env_conf, args)
    if optimizer is None:
        if args.optimizer == 'RMSprop':
            optimizer = optim.RMSprop(shared_model.parameters(), lr=args.lr)
        if args.optimizer == 'Adam':
            optimizer = optim.Adam(
                shared_model.parameters(), lr=args.lr, amsgrad=args.amsgrad)
    env.seed(args.seed + rank)
    player = Agent(None, env, args, None)
    player.gpu_id = gpu_id
    if 'micropolis' in args.env.lower():
        modelInit = getattr(model, args.design_head)
        player.model = modelInit(player.env.observation_space.shape[0],
                                     player.env.action_space, player.env.env.env.MAP_X)
        player.lstm_sizes = player.model.getMemorySizes()
    else:
        player.model = A3Clstm(player.env.observation_space.shape[0],
                               player.env.action_space)
    lstm_size = 512
    if 'micropolis' in args.env.lower():
        if 'arcade' not in args.env.lower():
            lstm_size = (1, 16, 
                    env.env.env.MAP_X, env.env.env.MAP_Y)
    player.lstm_size = lstm_size
    player.state = player.env.reset()
    player.state = torch.from_numpy(player.state).float()
    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            player.state = player.state.cuda()
            player.model = player.model.cuda()
    player.model.train()
    player.eps_len += 2
    log_counter = 0
    while True:
        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                player.model.load_state_dict(shared_model.state_dict())
        else:
            player.model.load_state_dict(shared_model.state_dict())
        num_lstm_layers = len(player.lstm_sizes)
        if player.done:
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.cx = [Variable(torch.zeros(player.lstm_sizes[i]).cuda())
                            for i in range(num_lstm_layers)] 
                    player.hx = [Variable(torch.zeros(player.lstm_sizes[i]).cuda())
                            for i in range(num_lstm_layers)]
            else:
                player.cx = [Variable(torch.zeros(lstm_sizes[i]))
                            for i in range(num_lstm_layers)]
                player.hx = [Variable(torch.zeros(lstm_sizes[i]))
                            for i in range(num_lstm_layers)]
        else:
            player.cx = [Variable(player.cx[i].data) for i in range(num_lstm_layers)]
            player.hx = [Variable(player.hx[i].data) for i in range(num_lstm_layers)] 


        for step in range(args.num_steps):
            player.action_train()
            if player.done:
                break
        if player.done:
            state = player.env.reset()
            player.state = torch.from_numpy(state).float()
            if args.randomize_exploration:
                player.certainty = np.random.uniform(0.5, 1.5)
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.state = player.state.cuda()

        R = torch.zeros(1, 1)
        if not player.done:
            values, logit, _ = player.model((Variable(player.state.unsqueeze(0)),
                                        (player.hx, player.cx)))
            if values.size()[1] == 1:
                value = values
            else:
                prob = torch.nn.functional.softmax(logit, dim=1)
                action = prob.multinomial(1).data
                value = values[0][action]

            R = value.data

        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                gae = torch.zeros(1, 1).cuda()
                R = Variable(R).cuda()
        else:
            R = Variable(R)
        player.values.append(R)
        policy_loss = 0
        value_loss = 0

        for i in reversed(range(len(player.rewards))):
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.rewards[i] = torch.Tensor([player.rewards[i]]).cuda()
            R = args.gamma * R + player.rewards[i]
            advantage = R - player.values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimataion
            delta_t = player.rewards[i] + args.gamma * \
                player.values[i + 1].data - player.values[i].data
            gae = gae * args.gamma * args.tau + delta_t
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    gae = Variable(gae.cuda())
            else:
                gae = Variable(gae)
            policy_loss = policy_loss - \
                player.log_probs[i] * Variable(gae) - 0.01 * player.entropies[i]

       #if log_counter % 10 == 0:
       #    log['{}_train_log'.format(args.env)].info(
       #            "Time {0}, reward {1}, policy loss {2}, value loss {3}, entropy {4}".
       #            format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time)),
       #                '{:9.2e}'.format(float(sum(player.rewards) / len(player.rewards))),
       #                '{:9.2e}'.format(float(policy_loss.data.item())), 
       #                '{:9.2e}'.format(float(value_loss.data.item())),
       #                 '{:10.8e}'.format(float(sum(player.entropies)))))
       #log_counter += 1


        optimizer.zero_grad()
        a3c = args.lmbda * (policy_loss + 0.5 * value_loss)
        a3c.backward()

        torch.nn.utils.clip_grad_norm_(player.model.parameters(), 40)
        ensure_shared_grads(player.model, shared_model, gpu=gpu_id >= 0)
        optimizer.step()

        player.clear_actions()
