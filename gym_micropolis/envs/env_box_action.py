from gym import core, spaces
from gym.utils import seeding
import numpy as np
import sys

if sys.version_info[0] >= 3:
    from gi.repository import Gtk as gtk
    from .tilemap import TileMap
    from .corecontrol import MicropolisControl
else:
    import gtk
    from tilemap import TileMap
    from corecontrol import MicropolisControl
import time
from time import sleep

class MicropolisEnv(core.Env):

    def __init__(self, MAP_X=14, MAP_Y=14, PADDING=0):
        self.SHOW_GUI=False
        self.start_time = time.time()
        self.print_map = False
       #self.setMapSize((MAP_X, MAP_Y), PADDING)

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31. * 2
        seed2 = seeding.hash_seed(seed1 + 1)
        # Empirically, we need to seed before loading the ROM (ignoring this for now in our case).
      # return [seed1, seed2]

    def setMapSize(self, size, print_map=False, PADDING=0, static_builds=True):
        self.MAP_X = size
        self.MAP_Y = size
        self.obs_width = self.MAP_X + PADDING * 2
        self.micro = MicropolisControl(self.MAP_X, self.MAP_Y, PADDING)
        self.static_builds = True
        if self.static_builds:
            self.micro.map.initStaticBuilds()
        self.win1 = self.micro.win1
        self.micro.SHOW_GUI=self.SHOW_GUI
        self.num_step = 0
        self.minFunds = 5000
        self.initFunds = 10000000
        self.num_tools = self.micro.num_tools
        self.num_zones = self.micro.num_zones
        self.num_scalars = 1
        # traffic, power, density
        self.num_obs_channels = self.num_zones + self.num_scalars + 3
        if self.static_builds:
            self.num_obs_channels += 1
        ac_low = np.zeros((3))
        ac_high = np.array([self.num_tools - 1, self.MAP_X - 1, self.MAP_Y - 1])
        self.action_space = spaces.Box(low=ac_low, high=ac_high, dtype=int)
        self.last_state = None
        self.metadata = {'runtime.vectorized': True}
        low_obs = np.zeros((self.num_obs_channels, self.MAP_X, self.MAP_Y))
        high_obs = np.full((self.num_obs_channels, self.MAP_X, self.MAP_Y), fill_value=1)
        # TODO: can/should we use Tuples of MultiBinaries instead, for greater efficiency?
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype = bool)
        self.state = None
#       self.intsToActions = {}
#       self.mapIntsToActions
#       self.mapIntsToActions()
        self.last_pop = 0
        self.last_num_roads = 0
#       self.past_actions = np.full((self.num_tools, self.MAP_X, self.MAP_Y), False)

   #def mapIntsToActionsChunk(self):
   #    ''' Unrolls the action vector into spatial chunks (does this matter empirically?).'''
   #    w0 = 20
   #    w1 = 10
   #    i = 0
   #    for j0 in range(self.MAP_X // w0):
   #        for k0 in range(self.MAP_Y // w0):
   #            for j1 in range(w0 // w1):
   #                for k1 in range(w0 // w1):
   #                    for z in range(self.num_tools):
   #                        for x in range(j0 * w0 + j1*w1,
   #                                j0 * w0 + (j1+1)*w1):
   #                            for y in range(k0 * w0 + k1*w1,
   #                                    k0 * w0 + (k1+1)*w1):
   #                                self.intsToActions[i] = [z, x, y]
   #                                i += 1

   #def mapIntsToActions(self):
   #    ''' Unrolls the action vector in the same order as the pytorch model
   #    on its forward pass.'''
   #    chunk_width = 1
   #    i = 0
   #    for z in range(self.num_tools):
   #        for x in range(self.MAP_X):
   #            for y in range(self.MAP_Y):
   #                    self.intsToActions[i] = [z, x, y]
   #                    i += 1


    def close(self):
        self.micro.close()

    def randomStaticStart(self):
        '''Cannot overwrite itself'''
        half_tiles = self.MAP_X * self.MAP_Y // 2
        r = np.random.randint(1, 5)
        self.micro.setFunds(10000000)
       # self.micr.map.initStaticBuilds
        for i in range(r):
            if self.micro.map.num_empty <= half_tiles:
                break
            else:
                self.step(self.action_space.sample(), static_build=True)

    def randomStart(self):
        r = np.random.randint(0, 100)
        self.micro.setFunds(10000000)
        for i in range(r):
            self.step(self.action_space.sample())
#       i = np.random.randint(0, (self.obs_width * self.obs_width / 3))
#       a = (np.random.randint(0, self.num_tools, i), np.random.randint(0, self.obs_width, i), np.random.randint(0, self.obs_width, i))
#       for j in range(i):
#           self.micro.takeSetupAction((a[0][j], a[1][j], a[2][j]))


    def reset(self):
        self.micro.clearMap()
        self.num_step = 0
       #self.randomStart()
        self.randomStaticStart()
        self.micro.engine.simTick()
        self.micro.setFunds(self.initFunds)
        curr_funds = self.micro.getFunds()
        curr_pop = self.getPop()
        self.state = self.observation([curr_pop])
        self.last_pop=0
        self.micro.num_roads = 0
        self.last_num_roads = 0
       #self.past_actions.fill(False)
        return self.state

    def observation(self, scalars):
        state = self.micro.map.getMapState()
        power = self.micro.getPowerMap()
        pop = self.micro.getPopDensityMap()
        traffic = self.micro.getTrafficDensityMap()
        scalar_layers = np.zeros((len(scalars), self.MAP_X, self.MAP_Y))
        for si in range(len(scalars)):
            scalar_layers[si].fill(scalars[si])
        state = np.concatenate((state, power, pop, traffic, scalar_layers), 0)
        if self.static_builds:
            state = np.concatenate((state, self.micro.map.static_builds), 0)
        return state

    def getPop(self):
        curr_pop = self.micro.getResPop() / 8 + self.micro.getComPop() + \
                self.micro.getIndPop()
        return curr_pop

    def step(self, a, static_build=False):
        reward = 0
#       a = self.intsToActions[a]
        a = list(a)
        print(a)
        self.micro.takeAction(a, static_build)
        self.curr_pop = self.getPop()
        self.state = self.observation([self.curr_pop])
#       reward += (self.micro.total_traffic - self.micro.last_total_traffic) / 50
        # anneal road reward
#       road_diff = self.micro.num_roads - self.last_num_roads
#       road_diff = road_diff * (max(0, 70 - self.micro.num_roads) / 70) * 0.2
#       reward += road_diff  * (max(0, 7200 - abs(time.time() - self.start_time)) / 7200)
        # anneal the following to zero over 1hr
        reward += (self.curr_pop - self.last_pop)#* (max(0, 14400 - abs(time.time() - self.start_time)) / 14400)
        self.last_num_roads = self.micro.num_roads
#       print(self.micro.num_roads)
        self.last_pop = self.curr_pop
        curr_funds = self.micro.getFunds()
        bankrupt = curr_funds < self.minFunds
        terminal = bankrupt or self.num_step >= 100
        if terminal and self.print_map:
            if static_build:
                print('STATIC BUILD')
            self.printMap()
        self.num_step += 1
        return (self.state, reward, terminal, {})

    def printMap(self):
        print('{}\npopulation: {}\ntraffic: {}\n{}\n'.format(np.add(self.micro.map.zoneMap[-1], np.full((self.MAP_X, self.MAP_Y), 2)), self.curr_pop, self.micro.total_traffic, self.micro.map.static_builds))



    def render(self, mode='human'):
        # why does this need to happen twice (or else blank window)?
        gtk.main_iteration()
        gtk.main_iteration()

    def test(self):
        env = MicropolisEnv()
        for i in range(5000):
            env.step(env.action_space.sample())

