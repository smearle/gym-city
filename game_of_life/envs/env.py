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
from .im2gif import GifWriter
import os
import shutil






class GameOfLifeEnv(core.Env):
    def __init__(self):
        self.num_tools = 1 # bring cell to life

    def configure(self, render=False, map_width=16, prob_life=20,
            record=None, max_step=None):
        self.size = size = map_width
        self.record = record
        if render and record:
            self.gif_writer = GifWriter()
            try:
                os.mkdir('{}/gifs/im/'.format(record)) # in case we are continuing eval
            except FileNotFoundError: pass
            except FileExistsError: pass
            try:
                os.mkdir('{}/gifs/'.format(record)) # in case we are starting a new eval
                os.mkdir('{}/gifs/im/'.format(record))
                print('madedatfoldo \n')
            except FileExistsError:
                pass
        self.observation_space = spaces.Box(low=np.zeros((1, size, size)),
                high=np.ones((1, size, size)), dtype=int)
        self.action_space = spaces.Discrete(self.num_tools * size * size)
        self.view_agent = render
        if self.view_agent:
            self.agent_builds = np.zeros(shape=(map_width, map_width), dtype=np.uint8)
        self.gif_ep_count = 0
        self.step_count = 0
        self.record_entropy = True
        self.world = World(self.size, self.size, prob_life=prob_life, env=self) 
        self.state = None
       #self.device = device = torch.device("cpu")
       #self.device = device = torch.device(
       #         "cuda" if torch.cuda.is_available() else "cpu")
        self.render_gui = render
        self.max_step = max_step

        self.entropies = []
        if self.render_gui:
            #TODO: render function should deal with this somehow
            cv2.namedWindow("Game of Life", cv2.WINDOW_NORMAL)

       #with torch.no_grad():
       #    self.combinations = combinations = 2 ** (3 * 3)
       #    channels = 1
       #    self.state = init_world(size, channels, prob).to(device)
       #    self.get_neighbors = get_neighbors_map(channels).to(device)
       #    self.structure_similarity = get_structure_similarity(combinations, channels)
       #    self.structure_similarity.to(device)
       #    self.i = 0


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
        if self.view_agent:
            self.agent_builds.fill(0)
        return self.world.state

    def render(self, mode=None):
        rend_arr = np.array(self.world.state, dtype=np.uint8)
        rend_arr = np.vstack((rend_arr * 255, rend_arr * 255, rend_arr * 255))
        if self.view_agent:
            rend_arr[1] = rend_arr[0] = rend_arr[1] - self.agent_builds * 255
        rend_arr = rend_arr.transpose(1, 2, 0)
        cv2.imshow("Game of Life", rend_arr)
        if self.record and not self.gif_writer.done:
            gif_dir = ('{}/gifs/'.format(self.record))
            im_dir = os.path.join(gif_dir, 'im')
            im_path = os.path.join(im_dir, 'e{:02d}_s{:04d}.png'.format(self.gif_ep_count, self.step_count))
           #print('saving frame at {}'.format(im_path))
            cv2.imwrite(im_path, rend_arr)
            if self.gif_ep_count == 0 and self.step_count == self.max_step:
                print('DATBOI \n \n')
                self.gif_writer.create_gif(im_dir, gif_dir, 0, 0, 0)
                self.gif_ep_count = 0
        cv2.waitKey(1)

    def step(self, a):
        if self.step_count == self.max_step:
            self.gif_ep_count += 1
        z, act_x, act_y = self.intsToActions[a]
        self.world.build_cell(act_x, act_y, alive=True)
        if self.view_agent:
            self.agent_builds[act_x, act_y] = 1
        if self.render_gui:
            self.render() # we need to render after moving in case the cell is lost immediately
        self.world._tick()
        reward = self.world.state.sum()
        terminal = (self.step_count == self.max_step) #or\
                    #reward < 2 # impossible situation for agent
        reward = reward / self.max_step
        self.step_count += 1
        if self.render_gui:
           #pass # leave this one to main loop
            self.render() # deal with rendering now
        infos = {}
        return (self.world.state, reward, terminal, infos)



    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        seed2 = seeding.hash_seed(seed1 + 1) % 2**31
        np.random.seed(seed)
        self.world.seed(seed)
        return [seed1, seed2]

cv2.destroyAllWindows()
