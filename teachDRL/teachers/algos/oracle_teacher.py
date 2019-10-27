import numpy as np
from gym.spaces import Box
from collections import deque

class OracleTeacher():
    def __init__(self, mins, maxs, window_step_vector,
                 seed=None, reward_thr=230, step_rate=50):
        self.seed = seed
        if not seed:
            self.seed = np.random.randint(42,424242)
        np.random.seed(self.seed)

        self.mins = np.array(mins, dtype=np.float32)
        self.maxs = np.array(maxs, dtype=np.float32)
        self.window_step_vector = window_step_vector
        self.reward_thr = reward_thr
        self.step_rate = step_rate
        self.window_range = (self.maxs - self.mins) / 6
        self.window_pos = np.zeros(len(self.mins), dtype=np.float32)  # stores bottom left point of window
        for i, step in enumerate(self.window_step_vector):
            if step > 0: # if step is positive, we go from min to max (and thus start at min)
                self.window_pos[i] = self.mins[i]
            else: # if step is negative, we go from max to min (and thus start at max - window_range)
                self.window_pos[i] = self.maxs[i] - self.window_range[i]

        self.train_rewards = []
        print("window range:{} \n position:{}\n step:{}\n"
              .format(self.window_range, self.window_pos, self.window_step_vector))

    def update(self, task, reward):
        self.train_rewards.append(reward)
        if len(self.train_rewards) == self.step_rate:
            mean_reward = np.mean(self.train_rewards)
            self.train_rewards = []
            if mean_reward > self.reward_thr:
                for i,step in enumerate(self.window_step_vector):
                    if step > 0:  # check if not stepping over max
                        self.window_pos[i] = min(self.window_pos[i] + step, self.maxs[i] - self.window_range[i])
                    elif step <= 0: # check if not stepping below min
                        self.window_pos[i] = max(self.window_pos[i] + step, self.mins[i])
                print('mut stump: mean_ret:{} window_pos:({})'.format(mean_reward, self.window_pos))

    def sample_task(self):
        task = np.random.uniform(self.window_pos, self.window_pos+self.window_range).astype(np.float32)
        #print(task)
        return task

    def dump(self, dump_dict):
        return dump_dict