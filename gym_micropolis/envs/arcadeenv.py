from gym import core, spaces
from gym.utils import seeding
import numpy as np
import sys
if sys.version_info[0] >= 3:
    from gi.repository import Gtk as gtk
    from .tilemap import TileMap 
    from .walkcontrol import MicroWalkControl
else:
    import gtk
    from tilemap import TileMap 
    from walkcontrol import MicroWalkControl

from time import sleep

class MicroArcadeEnv(core.Env):

    def __init__(self, MAP_X=10, MAP_Y=10):
        self.SHOW_GUI=False
        self.MAP_X = MAP_X
        self.MAP_Y = MAP_Y
        self.micro = MicroWalkControl(MAP_X, MAP_Y)
        self.win1 = self.micro.win1
        self.micro.SHOW_GUI=self.SHOW_GUI
        self.num_step = 0
        self.minFunds = 1000
        self.initFunds = 1000000000
        self.num_tools = self.micro.num_tools
        self.num_zones = self.micro.num_zones
        # move l, r, up, down, play tool, or pass
        self.action_space = spaces.Discrete(self.num_tools + 12)
        low_obs = np.zeros((self.num_zones, self.MAP_X, self.MAP_Y))
        high_obs = np.full((self.num_zones, self.MAP_X, self.MAP_Y), fill_value=1)

        # TODO: can/should we use Tuples of MultiBinaries instead, for greater efficiency?
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype = bool)


    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31. * 2
        seed2 = seeding.hash_seed(seed1 + 1)
        # Empirically, we need to seed before loading the ROM (ignoring this for now in our case).
      # return [seed1, seed2]


        self.state = None
        self.last_pop = 0
        assert self.micro.map.getMapState().shape


    def close(self):
        self.micro.close()


    def randomStart(self):
        i = np.random.randint(0, (self.MAP_X * self.MAP_Y))
        a = (np.random.randint(0, self.num_tools, i), np.random.randint(0, self.MAP_X, i), np.random.randint(0, self.MAP_Y, i))
        for j in range(i):
            self.micro.takeSetupAction((a[0][j], a[1][j], a[2][j]))
        for k in range(3):
            self.micro.engine.simTick()


    def reset(self):
        self.micro.clearMap()
   #    self.randomStart()
        self.num_step = 0
        self.micro.setFunds(self.initFunds)
        self.state = self.micro.map.getMapState()
        self.last_pop=0
        return self.state

    def step(self, a):
        if a < self.num_tools:
            toolstr = self.micro.tools[a]
            if toolstr:
                self.micro.doBotTool(self.micro.map.walker_pos[0],\
                    self.micro.map.walker_pos[1], toolstr)
        else:
            a = a - self.num_tools
            self.micro.singleStep(a, 3)
        self.micro.engine.simTick()


        curr_pop = self.micro.getResPop() / 8 + self.micro.getComPop() + \
                self.micro.getIndPop()
        reward = curr_pop / 50
    #   print(self.micro.map.num_empty, self.MAP_X * self.MAP_Y)
        assert self.micro.map.num_empty <= self.MAP_X * self.MAP_Y 
        assert self.micro.map.num_empty >= 0
      # import collections
      # int_zone_map = self.micro.map.zoneMap[-1]
      # zone_counts = collections.Counter(int_zone_map.flatten())
      # num_empty = zone_counts[14] + zone_counts[16]
      # print(num_empty, self.micro.map.num_empty)
      # print(int_zone_map)
      # assert self.micro.map.num_empty == num_empty
      # reward += (self.MAP_X * self.MAP_Y - self.micro.map.num_empty) / (self.MAP_X * self.MAP_Y)
     #  pop_diff = curr_pop - self.last_pop
     #  # Reward function
     #  if pop_diff > 0: 
     #      reward = 1
     #  elif pop_diff < 0: 
     #      reward = -1
     #  else: 
     #      reward = 0
      # print(curr_pop, reward)

      # if self.micro.map.no_change:


        self.last_pop = curr_pop
        terminal = False
      # if self.num_step % 1000 == 0 and self.micro.getFunds() < self.minFunds:
      #         terminal = True
        terminal = self.num_step == 2000
        self.num_step += 1
        return (self.micro.map.getMapState(), reward, terminal, {})
    
    def render(self, mode='human'):
        if sys.version_info[0] >=3:
           gtk.main_iteration()
           gtk.main_iteration()
           pass
        else:
           gtk.mainiteration()
           gtk.mainiteration()
           pass

    def test(self):
        env = MicropolisEnv()
        for i in range(5000):
            env.step(env.action_space.sample())



