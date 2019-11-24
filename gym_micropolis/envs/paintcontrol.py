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


    def takeAction(self, a, static_build=False):
        '''tool int depends on self.tools indexing
         - a: has shape (w, h), provides index of action taken at each tile'''
        for i in range(self.MAP_X):
            for j in range(self.MAP_Y):
                t = a[i][j]
                tool = self.tools[t] # get string
                self.doBotTool(i, j, tool, static_build)
        self.engine.simTick()
#       gtk.mainiteration()



