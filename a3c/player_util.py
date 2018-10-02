from __future__ import division
import torch
import torch.nn.functional as F
from torch.autograd import Variable


class Agent(object):
    def __init__(self, model, env, args, state):
        self.model = model
        self.env = env
        self.state = state
        self.hx = None
        self.cx = None
        self.eps_len = 0
        self.args = args
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        self.done = True
        self.info = None
        self.reward = 0
        self.gpu_id = -1
        
        self.lstm_sizes = None
        self.certainty = 1

        self.inverses = []
        self.forwards = []
        self.actions = []
        self.vec_st1s = []
        self.s_t1s = []

    def action_train(self):
        values, logit, (self.hx, self.cx) = self.model(
                (Variable(self.state.unsqueeze(0)), (self.hx, self.cx)) 
                )
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                self.state = self.state.cuda()
        s_t = self.state
        prob = F.softmax(logit, dim=1)
        log_prob = F.log_softmax(logit, dim=1)
        entropy = -(log_prob * prob).sum(1)
        self.entropies.append(entropy)
      # action = prob.max(0)[1]
        action = (self.certainty * prob).multinomial(1).data
 
        if values.size()[1] == 1:
            value = values
        else:
            value = values[0][action]

        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                a_t = Variable(torch.zeros(self.env.env.env.num_tools, 
                      self.env.env.env.MAP_X, self.env.env.env.MAP_Y).cuda())
        else:

            a_t = Variable(torch.zeros(self.env.env.env.num_tools, 
                  self.env.env.env.MAP_X, self.env.env.env.MAP_Y))
        a_pos = self.env.env.env.intsToActions[action.item()]
        a_t[a_pos[0], a_pos[1], a_pos[2]] = 1
        oh_action = torch.Tensor(1, self.env.env.env.action_space.n)
        oh_action.zero_()
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                oh_action = oh_action.cuda()
        oh_action.scatter_(1, action, 1)
        oh_action = Variable(oh_action)
        # represent action as boolean on build-grid so we can concatenate with
        # previous state, allowing icm predictor to use convolutions


        log_prob = log_prob.gather(1, Variable(action)).view(-1)
        action0 = action.item()
#       print(int(action0.cpu().numpy()))
        state, self.reward, self.done, self.info = self.env.step(action0)
        self.state = torch.from_numpy(state).float()
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                self.state = self.state.cuda()
        ### DO NOT CLIP REWARD FOR MICROPOLIS
#       self.reward = max(min(self.reward, 1), -1)
        s_t1 = self.state

        self.values.append(value)
        self.log_probs.append(log_prob)
        self.rewards.append(self.reward)
        return self

    def action_test(self):
       #self.env.render()
       #self.env.render()

        with torch.no_grad():
            num_lstm_layers = len(self.lstm_sizes)
            if self.done:
                if self.gpu_id >= 0:
                    with torch.cuda.device(self.gpu_id):
                        self.cx = [Variable(
                            torch.zeros(self.lstm_sizes[i]).cuda())
                            for i in range(num_lstm_layers)]
                        self.hx = [Variable(
                            torch.zeros(self.lstm_sizes[i]).cuda())
                            for i in range(num_lstm_layers)]
                else:
                    self.cx = [Variable(torch.zeros(self.lstm_sizes[i]))
                            for i in range(num_lstm_layers)]

                    self.hx = [Variable(torch.zeros(self.lstm_sizes[i]))
                            for i in range(num_lstm_layers)]

            else:
                self.cx = [Variable(self.cx[i].data) 
                            for i in range(num_lstm_layers)]
                self.hx = [Variable(self.hx[i].data) 
                            for i in range(num_lstm_layers)]
            value, logit, (self.hx, self.cx) = self.model((Variable(
                self.state.unsqueeze(0)), (self.hx, self.cx)))
        prob = F.softmax(logit, dim=1)
        log_prob = F.log_softmax(logit, dim=1)
        entropy = -(log_prob * prob).sum(1)
        self.entropy = entropy

        action = prob.multinomial(1).data.cpu().numpy()
       #action = prob.max(1)[1]
        state, self.reward, self.done, self.info = self.env.step(action.item())
        self.state = torch.from_numpy(state).float()
#       self.reward = torch.from_numpy(reward).float()
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                self.state = self.state.cuda()
        self.eps_len += 1
        import numpy as np
        np.set_printoptions(threshold=1600)
        print('\n{}'.format(np.add(self.env.env.env.micro.map.zoneMap[-1],
                                   np.full((self.env.env.env.MAP_X, 
                                       self.env.env.env.MAP_Y), 2)))
                                    .replace('\n ', '').replace('][',']\n[')
                                    .replace('[[','[').replace(']]',']')
                                   + '{}'.format(self.reward))
        return self

    def clear_actions(self):
        
       #print('reward: {}'.format((sum(self.rewards) / len(self.rewards)).item()))
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
#       self.value_losses = []
#       self.policy_losses = []
#       self.inverses = []
#       self.forwards = []
#       self.actions = []
#       self.vec_st1s = []
        return self
