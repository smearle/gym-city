from gym import core, spaces
from gym.utils import seeding
import numpy as np
from tilemap import TileMap 
from corecontrol import MicropolisControl 
import gtk


class MicropolisEnv(core.Env):

    def __init__(self):
        self.SHOW_GUI=False

    def setMapSize(self, MAP_X=6, MAP_Y=6):
        self.MAP_X = MAP_X
        self.MAP_Y = MAP_Y
        self.micro = MicropolisControl(MAP_X, MAP_Y)
        self.micro.SHOW_GUI=self.SHOW_GUI
        self.num_step = 0
        self.minFunds = 1000
        self.initFunds = 1000000000
        self.num_tools = self.micro.num_tools
        self.num_zones = self.micro.num_zones
        self.action_space = spaces.Discrete(self.num_tools * self.MAP_X * self.MAP_Y)
        low_obs = np.zeros((self.num_zones, self.MAP_X, self.MAP_Y))
        high_obs = np.full((self.num_zones, self.MAP_X, self.MAP_Y), fill_value=1)
        # TODO: can/should we use Tuples of MultiBinaries instead, for greater efficiency?
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype = bool)
        self.state = None
        self.intsToActions = {}
        self.mapIntsToActions()
        self.last_pop = 0


    def mapIntsToActions(self):
        i = 0
        for x in range(self.MAP_X):
            for y in range(self.MAP_Y):
                for z in range(self.num_tools):
                    self.intsToActions[i] = [z, x, y]
                    i += 1

    def close(self):
        self.micro.close()


    def reset(self):
        self.micro.clearMap()
      # self.micro.layGrid(7,19)
        self.num_step = 0
        self.micro.setFunds(self.initFunds)
        self.state = self.micro.map.getMapState()
        self.last_pop=0
        return self.state

    def step(self, a):
        self.micro.takeAction(self.intsToActions[a])
        curr_pop = self.micro.getResPop() / 8 + self.micro.getComPop() + \
                self.micro.getIndPop()
        reward = curr_pop
     #   pop_diff = curr_pop - self.last_pop
     #   # Reward function
     #   if pop_diff > 0: 
     #       reward = 1
     #   elif pop_diff < 0: 
     #       reward = -1
     #   else: 
     #       reward = 0
     # # print(curr_pop, reward)
     #   self.last_pop = curr_pop
        terminal = False
        if self.num_step % 10 == 0 and self.micro.getFunds() < self.minFunds:
                terminal = True
        terminal = self.num_step == 10000
        self.num_step += 1
        return (self.micro.map.getMapState(), reward, terminal, {})
    
    def render(self, mode='human'):
        gtk.mainiteration()

    def test(self):
        env = MicropolisEnv()
        for i in range(5000):
            env.step(env.action_space.sample())



