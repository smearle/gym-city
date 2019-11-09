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
        self.player_step = False
        self.player_builds = []

    def configure(self, render=False, map_width=16, prob_life=20,
            record=None, max_step=None):
        self.prebuild = False
        self.prebuild_steps = 50
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
            cv2.setMouseCallback("Game of Life", self.player_build)

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
       #print(self.intsToActions)
       #print(self.actionsToInts)
        print('render: ', self.render_gui)

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
        rend_arr = rend_arr.transpose(2, 1, 0)
        cv2.imshow("Game of Life", rend_arr)
        if self.record and not self.gif_writer.done:
            gif_dir = ('{}/gifs/'.format(self.record))
            im_dir = os.path.join(gif_dir, 'im')
            im_path = os.path.join(im_dir, 'e{:02d}_s{:04d}.png'.format(self.gif_ep_count, self.step_count))
           #print('saving frame at {}'.format(im_path))
            cv2.imwrite(im_path, rend_arr)
            if self.gif_ep_count == 0 and self.step_count == self.max_step:
                self.gif_writer.create_gif(im_dir, gif_dir, 0, 0, 0)
                self.gif_ep_count = 0
        cv2.waitKey(1)

    def step(self, a):
        if self.prebuild:
            self.world.set_state(a)
        else:
            if self.player_builds:
                a = self.player_builds[0]
                self.player_builds = self.player_builds[1:]
                self.player_step = True
            else:
                self.player_step = False
            if self.step_count == self.max_step:
                self.gif_ep_count += 1
            if a < 0:
                PLAYER_DEL = True
                a = -a
            else:
                PLAYER_DEL = False
            z, act_x, act_y = self.intsToActions[a]
            if self.player_step:
                print('executing player build at: {}, {}'.format(act_x, act_y))
            if PLAYER_DEL:
                self.world.build_cell(act_x, act_y, alive=False)
        if not self.prebuild or self.step_count < self.prebuild_steps:
            if self.prebuild:
                reward = self.world.state.sum() / self.max_step
               #reward = 0
            else:
                reward = self.world.state.sum()
            if not PLAYER_DEL:
                self.world.build_cell(act_x, act_y, alive=True)
                if self.view_agent:
                    self.agent_builds[act_x, act_y] = 1
            if self.render_gui:
                self.render() # we need to render after moving in case the cell is lost immediately
            terminal = False
        if self.prebuild and self.step_count >= self.prebuild_steps:
            reward = self.world.state.sum()
            for i in range(self.max_step):
                self.world._tick()
                reward_i = self.world.state.sum()
                reward += reward_i
                if reward_i ==0 or i == self.max_step or not self.world.state_changed:
                    break
                if self.render_gui:
                    self.render()
            terminal=True
        if not self.prebuild:
            self.world._tick()
            terminal = (self.step_count == self.max_step)  or\
                         reward < 2 # impossible situation for agent
            reward = reward * 100 / (self.max_step * self.size * self.size)
            if self.render_gui:
               #pass # leave this one to main loop
                self.render() # deal with rendering now
        infos = {}
        reward = float(reward) 
        self.step_count += 1
        return (self.world.state, reward, terminal, infos)

    def player_build(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            a = int(self.actionsToInts[0][x][y])
            self.player_builds += [a]
            print('q\'d player build at: {}, {}'.format(x, y))
        elif event == cv2.EVENT_MBUTTONDOWN:
            a = int(self.actionsToInts[0][x][y])
            self.player_builds += [-a]
            print('q\'d player delete at: {}, {}'.format(x, y))


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

def main():
    env = GameOfLifeEnv()
    env.configure(render=True)
    while True:
        env.step(0)

