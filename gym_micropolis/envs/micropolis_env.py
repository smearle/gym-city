#!/Users/sme/

from gym import core, spaces
from gym.utils import seeding
import numpy as np
from gym_micropolis.envs.micropolis_control import MicropolisControl 
from gym_micropolis.envs.tile_map import TileMap 



class MicropolisEnv(core.Env):

    def __init__(self):
        self.micro = MicropolisControl()
        self.MAP_X = self.micro.MAP_X
        self.num_step = 0
        self.minFunds = 1000
        self.initFunds = 1000000000
        self.MAP_Y = self.micro.MAP_Y
        self.num_tools = self.micro.num_tools
        self.num_zones = self.micro.num_zones
        self.action_space = spaces.Discrete(self.MAP_X * self.MAP_Y * self.num_tools)
        low_obs = np.zeros((self.num_zones, self.MAP_X, self.MAP_Y))
        high_obs = np.full((self.num_zones, self.MAP_X, self.MAP_Y), fill_value=1)
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype = int)
        self.state = None
        self.intsToActions = {}
        self.mapIntsToActions()


    def mapIntsToActions(self):
        i = 0
        for x in range(self.MAP_X):
            for y in range(self.MAP_Y):
                for z in range(self.num_tools):
                    self.intsToActions[i] = [z, x, y]
                    i += 1


     #   self.intsToActions
     #   # we assume a square board
     #   tiles = self.MAP_X * self.MAP_X * self.num_tools
     #   i = 0
     #   w = 1
     #   y = self.MAP_X // 2
     #   x = self.MAP_X // 2
     #   while w < self.MAP_X:
     #       s = (w % 2) * 2 - 1 
     #       for k in range(w):
     #           for z in range(self.num_tools):
     #               self.intsToActions[i] = [x, y, z]
     #               i += 1
     #           x += s * 1
     #       for k in range(w):
     #           for z in range(self.num_tools):
     #               self.intsToActions[i] = [x, y, z]
     #               i += 1
     #           y += s * 1
     #       w += 1
     #   s = (w % 2) * 2 - 1 
     #   for k in range(w ):
     #       for z in range(self.num_tools):
     #           self.intsToActions[i] = [x, y, z]
     #           i += 1
     #       x += s * 1


    def close(self):
        self.micro.close()


    def reset(self):
        self.micro.clearMap()
      # self.micro.layGrid(7,19)
        self.num_step = 0
        self.micro.setFunds(self.initFunds)
        self.state = self.micro.map.getMapState()
        return self.state

    def step(self, a):
        self.micro.takeAction(self.intsToActions[a])
        reward = self.micro.getPopulation()
        terminal = False
        if self.num_step % 10 == 0 and self.micro.getFunds() < self.minFunds:
                terminal = True
        terminal = self.num_step == 25
        self.num_step += 1
        return (self.micro.map.getMapState(), reward, terminal, {})

    def test(self):
        env = MicropolisEnv()
        for i in range(5000):
            env.step(env.action_space.sample())
