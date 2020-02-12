import numpy as np
import gym
import copy

class ParamRew(gym.Wrapper):
    def __init__(self, env):
        super(ParamRew, self).__init__(env)

    def step(self, action):
        ob, rew, done, info = super().step(action)
        self.last_metrics = copy.deepcopy(self.metrics)
        rew = self.get_reward()
        return ob, rew, done, info

    def get_reward(self):
        for metric, trg in self.metric_trgs.items():
            last_val = self.last_metrics[metric]
            trg_change = trg - last_val
            val = self.metrics[metric]
            change = val - last_val
            print(trg_change, change)
            if np.sign(change) != np.sign(trg_change):
                metric_rew = -abs(change)
            elif abs(change) < abs(trg_change):
                metric_rew = abs(change)
            else:
                metric_rew = abs(trg_change) - abs(trg_change - change)
            reward += metric_rew * self.weights[metric]
        return reward
