from gym import core, spaces
from gym.utils import seeding
import numpy as np
import math
 
import sys
if sys.version_info[0] >= 3:
    import gi
    gi.require_version('Gtk', '3.0')
    from gi.repository import Gtk as gtk
    from .tilemap import TileMap 
    from .corecontrol import MicropolisControl
else:
    import gtk
    from tilemap import TileMap 
    from corecontrol import MicropolisControl
import time

class MicropolisEnv(core.Env):

    def __init__(self, MAP_X=20, MAP_Y=20, PADDING=0):
        self.SHOW_GUI=False
        self.start_time = time.time()
        self.print_map = False
        self.num_episode = 0
        self.max_step = 500
        self.max_static = 0
       #self.setMapSize((MAP_X, MAP_Y), PADDING)

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        seed2 = seeding.hash_seed(seed1 + 1) % 2**31
        np.random.seed(seed)
        return [seed1, seed2]

    def setMapSize(self, size, print_map=False, PADDING=0, static_builds=True, parallel_gui=False, render_gui=False):
        if type(size) == int:
            self.MAP_X = size
            self.MAP_Y = size
        else:
            self.MAP_X = size[0]
            self.MAP_Y = size[1]
        self.obs_width = self.MAP_X + PADDING * 2
        self.micro = MicropolisControl(self.MAP_X, self.MAP_Y, PADDING, parallel_gui=parallel_gui)
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
        self.num_obs_channels = self.micro.map.num_features + self.num_scalars + 3
        if self.static_builds:
            self.num_obs_channels += 1
       #ac_low = np.zeros((3))
       #ac_high = np.array([self.num_tools - 1, self.MAP_X - 1, self.MAP_Y - 1])
       #self.action_space = spaces.Box(low=ac_low, high=ac_high, dtype=int)
        self.action_space = spaces.Discrete(self.num_tools * self.MAP_X * self.MAP_Y)
        self.last_state = None
        self.metadata = {'runtime.vectorized': True}
 
        low_obs = np.zeros((self.num_obs_channels, self.MAP_X, self.MAP_Y))
        high_obs = np.full((self.num_obs_channels, self.MAP_X, self.MAP_Y), fill_value=1)
        # TODO: can/should we use Tuples of MultiBinaries instead, for greater efficiency?
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype = int)
        self.state = None
        self.intsToActions = {}
        self.mapIntsToActions
        self.mapIntsToActions()
        self.last_pop = 0
        self.last_num_roads = 0
#       self.past_actions = np.full((self.num_tools, self.MAP_X, self.MAP_Y), False)
        self.print_map = print_map
        self.render_gui = render_gui

    def mapIntsToActionsChunk(self):
        ''' Unrolls the action vector into spatial chunks (does this matter empirically?).'''
        w0 = 20
        w1 = 10
        i = 0
        for j0 in range(self.MAP_X // w0):
            for k0 in range(self.MAP_Y // w0):
                for j1 in range(w0 // w1):
                    for k1 in range(w0 // w1):
                        for z in range(self.num_tools):
                            for x in range(j0 * w0 + j1*w1, 
                                    j0 * w0 + (j1+1)*w1):
                                for y in range(k0 * w0 + k1*w1, 
                                        k0 * w0 + (k1+1)*w1):
                                    self.intsToActions[i] = [z, x, y]
                                    i += 1
                                
    def mapIntsToActions(self):
        ''' Unrolls the action vector in the same order as the pytorch model
        on its forward pass.'''
        chunk_width = 1
        i = 0
        for z in range(self.num_tools):
            for x in range(self.MAP_X):
                for y in range(self.MAP_Y):
                        self.intsToActions[i] = [z, x, y]
                        i += 1

    def randomStep(self):
        self.step(self.action_space.sample())

    def close(self):
        self.micro.close()

    def randomStaticStart(self):
        '''Cannot overwrite itself'''
        num_static = 50
        lst_epi = 500
#       num_static = math.ceil(((lst_epi - self.num_episode) / lst_epi) * num_static)   
#       num_static = max(0, max_static)
        self.micro.setFunds(10000000)
        if num_static > 0:
            num_static = self.np_random.randint(0, num_static + 1)
        for i in range(num_static):
            self.step(self.action_space.sample(), static_build=True)

    def randomStart(self):
        r = self.np_random.randint(0, 100)
        self.micro.setFunds(10000000)
        for i in range(r):
            self.step(self.action_space.sample())
#       i = np.random.randint(0, (self.obs_width * self.obs_width / 3))
#       a = (np.random.randint(0, self.num_tools, i), np.random.randint(0, self.obs_width, i), np.random.randint(0, self.obs_width, i))
#       for j in range(i):
#           self.micro.takeSetupAction((a[0][j], a[1][j], a[2][j]))


    def reset(self):
        self.micro.newMap()
        self.micro.updateMap()
        self.num_step = 0
#       self.randomStaticStart()
#       self.micro.engine.simTick()
        self.micro.setFunds(self.initFunds)
       #curr_funds = self.micro.getFunds()
        curr_pop = self.getPop()
        self.state = self.observation([curr_pop])
        self.last_pop=0
        self.micro.num_roads = 0
        self.last_num_roads = 0
       #self.past_actions.fill(False)
        self.num_episode += 1
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
        curr_pop = self.micro.getResPop() / 4 + \
                   8 * self.micro.getComPop() + \
                   4 * self.micro.getIndPop()
        return curr_pop

    def step(self, a, static_build=False):
        reward = 0
        a = self.intsToActions[a]
        self.micro.takeAction(a, static_build)
        self.curr_pop = self.getPop()
        self.state = self.observation([self.curr_pop])
        reward += (self.curr_pop - self.last_pop)
#       reward += (self.micro.total_traffic - self.micro.last_total_traffic)
        self.last_pop = self.curr_pop
        curr_funds = self.micro.getFunds()
        bankrupt = curr_funds < self.minFunds
        terminal = bankrupt or self.num_step >= self.max_step
        if True and self.print_map:
            if static_build:
                print('STATIC BUILD')
            self.printMap()
        self.num_step += 1
        if self.render_gui:
            self.micro.render()
        return (self.state, reward, terminal, {})

    def printMap(self, static_builds=False):
            if static_builds:
                static_map = self.micro.map.static_builds
            else:
                static_map = None
            np.set_printoptions(threshold=np.inf)
            zone_map = self.micro.map.zoneMap[-1]
            zone_map = np.array_repr(zone_map).replace(',  ','  ').replace('],\n', ']\n').replace(',\n', ',').replace(', ', ' ').replace('        ',' ').replace('         ','  ')
            print('{}\npopulation: {}, traffic: {}, episode: {}, step: {} \n{}'.format(zone_map, self.curr_pop, self.micro.total_traffic, self.num_episode, self.num_step, static_map))


    
    def render(self, mode='human'):
        # why does this need to happen twice (or else blank window)?
        self.micro.render()

    def test(self):
        env = MicropolisEnv()
        for i in range(5000):
            env.step(env.action_space.sample())

