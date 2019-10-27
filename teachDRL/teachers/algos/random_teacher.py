import numpy as np
from gym.spaces import Box

class RandomTeacher():
    def __init__(self, mins, maxs, seed=None):
        self.seed = seed
        if not seed:
            self.seed = np.random.randint(42,424242)
        np.random.seed(self.seed)

        self.mins = mins
        self.maxs = maxs

        self.random_task_generator = Box(np.array(mins), np.array(maxs), dtype=np.float32)

    def update(self, task, competence):
        pass

    def sample_task(self):
        return self.random_task_generator.sample()

    def dump(self, dump_dict):
        return dump_dict