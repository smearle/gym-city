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
from time import sleep

class MicropolisEnv(core.Env):

    def __init__(self, MAP_X=4, MAP_Y=4, PADDING=0):
        self.SHOW_GUI=False
        self.setMapSize(MAP_X, MAP_Y, PADDING)

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31. * 2
        seed2 = seeding.hash_seed(seed1 + 1)
        # Empirically, we need to seed before loading the ROM (ignoring this for now in our case).
      # return [seed1, seed2]

    def setMapSize(self, MAP_X=4, MAP_Y=4, PADDING=0):
        self.MAP_X = MAP_X
        self.MAP_Y = MAP_Y
        self.obs_width = MAP_X + PADDING * 2
        self.micro = MicropolisControl(MAP_X, MAP_Y, PADDING)
        self.win1 = self.micro.win1
        self.micro.SHOW_GUI=self.SHOW_GUI
        self.num_step = 0
        self.minFunds = 5000
        self.initFunds = 1000000
        self.num_tools = self.micro.num_tools
        self.num_zones = self.micro.num_zones
        self.num_scalars = 2
        self.num_obs_channels = self.num_zones + self.num_scalars
        self.action_space = spaces.Discrete(self.num_tools * self.MAP_X * self.MAP_Y)
        self.last_state = None
        self.metadata = {'runtime.vectorized': True}
 
        low_obs = np.zeros((self.num_obs_channels, self.MAP_X, self.MAP_Y))
        high_obs = np.full((self.num_obs_channels, self.MAP_X, self.MAP_Y), fill_value=1)
        # TODO: can/should we use Tuples of MultiBinaries instead, for greater efficiency?
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype = bool)
        self.state = None
        self.intsToActions = {}
        self.mapIntsToActions
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


    def randomStart(self):
        r = np.random.randint(0, 100)
        self.micro.setFunds(1000000)
        for i in range(r):
            self.step(self.action_space.sample())
#       i = np.random.randint(0, (self.obs_width * self.obs_width / 3))
#       a = (np.random.randint(0, self.num_tools, i), np.random.randint(0, self.obs_width, i), np.random.randint(0, self.obs_width, i))
#       for j in range(i):
#           self.micro.takeSetupAction((a[0][j], a[1][j], a[2][j]))


    def reset(self):
        self.micro.clearMap()
      # self.randomStart()
        self.num_step = 0
        self.micro.setFunds(self.initFunds)
        curr_funds = self.micro.getFunds()
        curr_pop = self.getPop()
        self.state = state =self.observation([curr_funds, curr_pop])
        self.last_pop=0
        return self.state

    def observation(self, scalars):
        state = self.micro.map.getMapState()
        scalar_layers = np.zeros((len(scalars), self.MAP_X, self.MAP_Y))
        for si in range(len(scalars)):
            if si == 0:
                scalar_layers[si].fill(scalars[si])
            elif si == 1:
                scalar_layers[si].fill(scalars[si])
        state = np.concatenate((state, scalar_layers), 0)
        return state

    def getPop(self):
        curr_pop = self.micro.getResPop() / 8 + self.micro.getComPop() + \
                self.micro.getIndPop()
        return curr_pop

    def step(self, a):
     #  print(a)
     #  print(self.num_step)
        a = a
        self.micro.takeAction(self.intsToActions[a])
        curr_pop = self.getPop()
        reward = curr_pop / 2
        curr_funds = self.micro.getFunds()
        bankrupt = curr_funds < self.minFunds
        terminal = bankrupt or self.num_step >= 1000
        state =self.observation([curr_funds, curr_pop])
        self.num_step += 1
        return (state, reward, terminal, {})
    
    def render(self, mode='human'):
        # why does this need to happen twice (or else blank window)?
        gtk.main_iteration()
        gtk.main_iteration()

    def test(self):
        env = MicropolisEnv()
        for i in range(5000):
            env.step(env.action_space.sample())

