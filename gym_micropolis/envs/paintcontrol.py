import sys
import os
import random
import numpy as np
if sys.version_info[0] >= 3:
    from gi.repository import Gtk as gtk
else:
    import gtk
from .corecontrol import MicropolisControl

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
GIT_DIR = os.path.abspath(os.path.join(FILE_DIR, os.pardir, os.pardir))
MICROPOLISCORE_DIR = GIT_DIR + '/gym_micropolis/envs/micropolis/MicropolisCore/src'
if sys.version_info[0] >= 3:
    sys.path.append(MICROPOLISCORE_DIR)
    from .tilemap import TileMap, zoneFromInt
else:
    print('Python 2 not supported')
    sys.path.append(MICROPOLISCORE_DIR)
    from tilemap import TileMap

CURR_DIR = os.getcwd()
# we need to do this so the micropolisgenericengine can access images/micropolisEngine/dataColorMap.png


os.chdir(MICROPOLISCORE_DIR)

#import micropolis
from pyMicropolis.gtkFrontend import main


os.chdir(CURR_DIR)

class MicropolisPaintControl(MicropolisControl):

    def __init__(self, env, **kwargs):
        super(MicropolisPaintControl, self).__init__(env, **kwargs)
        self.env = env
        # have we built on each tile already? No deleting previous builds during
        # single pass over map!
        MAP_W = kwargs['MAP_W']
        MAP_H = kwargs['MAP_H']
        self.acted = np.zeros((MAP_W, MAP_H))


    def takeAction(self, a, static_build=False):
        '''tool int depends on self.tools indexing
         - a: has shape (w, h), provides index of action taken at each tile'''
        reward = 0
        i = 0
        while i in range(self.MAP_X):
            j = 0
            while j in range(self.MAP_Y):
                if j >= self.MAP_Y - 1 or i >= self.MAP_X:
                    break
                t_i = a[i][j]
                tool = self.tools[t_i] # get string
                if self.acted[i, j] == 0:
                    self.doBotTool(i, j, tool, static_build)
                    self.engine.simTick()
                    self.env.render()
                if not tool in ['Nil', 'Clear']:
                    zone_size = self.map.zoneSize[tool]
                else:
                    zone_size = 1
                # increment i and j by (zone_size - 1)
                i_1 = i + zone_size - 1
                j_1 = j + zone_size - 1
                self.acted[i-1:i_1, j-1:j_1] = 1
                j += 1
            i += 1
        self.acted.fill(0)

#       gtk.mainiteration()



