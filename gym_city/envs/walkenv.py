from gym import core, spaces
from gym.utils import seeding
import numpy as np
from . tilemap import TileMap 
from . walkcontrol import MicroWalkControl 
import gtk
from time import sleep

class MicroWalkEnv(core.Env):

    def __init__(self):
        self.SHOW_GUI=False

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31. * 2
        seed2 = seeding.hash_seed(seed1 + 1)
        # Empirically, we need to seed before loading the ROM (ignoring this for now in our case).
      # return [seed1, seed2]

    def setMapSize(self, MAP_X=12, MAP_Y=12, walk_dist=6):
        self.MAP_X = MAP_X
        self.MAP_Y = MAP_Y
        self.walk_dist = walk_dist
        self.micro = MicroWalkControl(MAP_X, MAP_Y)
        self.win1 = self.micro.win1
        self.micro.SHOW_GUI=self.SHOW_GUI
        self.num_step = 0
        self.minFunds = 1000
        self.initFunds = 1000000000
        self.num_tools = self.micro.num_tools
        self.num_zones = self.micro.num_zones
        self.action_space = spaces.Discrete(self.num_tools * (self.walk_dist * 2 + 1) * (self.walk_dist * 2 + 1))
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
        for x in range(self.walk_dist * 2 + 1):
            for y in range(self.walk_dist * 2 + 1):
                for z in range(self.num_tools):
                    self.intsToActions[i] = [z, x, y]
                    i += 1


    def close(self):
        self.micro.close()


    def randomStart(self):
        i = np.random.randint(0, (self.MAP_X * self.MAP_Y / 3))
        a = (np.random.randint(0, self.num_tools, i), np.random.randint(0, self.MAP_X, i), np.random.randint(0, self.MAP_Y, i))
        for j in range(i):
            self.micro.takeSetupAction((a[0][j], a[1][j], a[2][j]))


    def reset(self):
        self.micro.clearMap()
    #   self.randomStart()
        self.num_step = 0
        self.micro.setFunds(self.initFunds)
        self.state = self.micro.map.getMapState()
        self.last_pop=0
        return self.state

    def step(self, a):
        a = self.intsToActions[a]
        xstep = a[1] - self.walk_dist
        ystep = a[2] - self.walk_dist
        tool = a[0]

        self.micro.takeAction(xstep, ystep, tool)
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
        terminal = self.num_step == 1000
        self.num_step += 1
        return (self.micro.map.getMapState(), reward, terminal, {})
    
    def render(self, mode='human'):
        gtk.mainiteration()

    def test(self):
        env = MicropolisEnv()
        for i in range(5000):
            env.step(env.action_space.sample())



