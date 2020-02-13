import torch
import gym
import copy

class ParamRewMulti(gym.Wrapper):
    def __init__(self, env):
        super(ParamRewMulti, self).__init__(env)
        if self.cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def step(self, action):
        self.last_metrics = copy.deepcopy(self.metrics)
        ob, rew, done, info = super().step(action)
        rew = self.get_reward()
        return ob, rew, done, info

    def get_reward(self):
        reward = torch.zeros(self.num_proc)
        reward = reward.to(self.device)
        for metric, trg in self.metric_trgs.items():
            last_val = self.last_metrics[metric].to(self.device)
            trg_change = trg - last_val
            val = self.metrics[metric]
            change = val - last_val
            metric_rew = torch.zeros(self.num_proc)
            metric_rew = metric_rew.to(self.device)
            same_sign = (change < 0) == (trg_change < 0)
            # changed in wrong direction
            metric_rew += (same_sign == False) * -abs(change)
            less_change = abs(change) < abs(trg_change)
            # changed a little, in the right direction
            metric_rew += (same_sign) * (less_change) * abs(change)
            # overshot
            metric_rew += (same_sign) * (less_change == False) * (abs(trg_change) - abs(trg_change - change))
            reward += metric_rew * self.metric_weights[metric]
        reward = reward.unsqueeze(-1)
        reward = reward.to(torch.device('cpu'))
       #print(reward)
        return reward
