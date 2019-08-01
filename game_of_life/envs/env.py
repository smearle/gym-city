import gym
from gym import core, spaces
from gym.utils import seeding
from .gol import utils
import argparse
import itertools

import cv2
import numpy as np
import torch
from torch import ByteTensor, Tensor
from torch.nn import Conv2d, Parameter
from torch.nn.init import zeros_
from .world import World







class GameOfLifeEnv(core.Env):
    def __init__(self):
        self.num_tools = 1 # bring cell to life

    def configure(self, render=False, map_width=16, prob_life=20):
        self.size = size = map_width
        self.observation_space = spaces.Box(low=np.zeros((1, size, size)),
                high=np.ones((1, size, size)), dtype=int)
        self.action_space = spaces.Discrete(self.num_tools * size * size)

        self.step_count = 0
        self.prob_life = prob_life
        self.record_entropy = True
        self.world = World(self.size, self.size, prob_life=20)
        self.state = None
       #self.device = device = torch.device("cpu")
        self.device = device = torch.device(
                 "cuda" if torch.cuda.is_available() else "cpu")
        self.render = render
        self.max_step = 100

        self.entropies = []
        if self.render:
            cv2.namedWindow("Game of Life", cv2.WINDOW_NORMAL)

       #with torch.no_grad():
       #    self.combinations = combinations = 2 ** (3 * 3)
       #    channels = 1
       #    self.state = init_world(size, channels, prob).to(device)
       #    self.get_neighbors = get_neighbors_map(channels).to(device)
       #    self.structure_similarity = get_structure_similarity(combinations, channels)
       #    self.structure_similarity.to(device)
       #    self.i = 0


        self.render = render
        self.size = map_width
        self.intsToActions = [[] for pixel in range(self.num_tools * self.size **2)]
        self.actionsToInts = np.zeros((self.num_tools, self.size, self.size))

        ''' Unrolls the action vector in the same order as the pytorch model
        on its forward pass.'''
        i = 0
        for z in range(self.num_tools):
            for x in range(self.size):
                for y in range(self.size):
                        self.intsToActions[i] = [z, x, y]
                        self.actionsToInts[z, x, y] = i
                        i += 1
       #print('len of intsToActions: {}\n num tools: {}'.format(len(self.intsToActions), self.num_tools))

    def reset(self):
        self.step_count = 0
        self.world.repopulate_cells()
        self.world.prepopulate_neighbours()
        return self.world.state

    def step(self, a):
        z, act_x, act_y = self.intsToActions[a]
        self.world.build_cell(act_x, act_y, alive=True)
        if self.render:
           #print(self.world.render())
            cv2.imshow("Game of Life", np.array(self.world.state.transpose(1, 2, 0) * 255, dtype=np.uint8))
            cv2.waitKey(1)
        self.world._tick()
        terminal = self.step_count == self.max_step
        self.step_count += 1
        if self.render:
           #print(self.world.render())
            cv2.imshow("Game of Life", np.array(self.world.state.transpose(1, 2, 0) * 255, dtype=np.uint8))
            cv2.waitKey(1)

        reward = self.world.state.sum() / self.max_step
        infos = {}
        return (self.world.state, reward, terminal, infos)



    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        seed2 = seeding.hash_seed(seed1 + 1) % 2**31
        np.random.seed(seed)
        return [seed1, seed2]

cv2.destroyAllWindows()
