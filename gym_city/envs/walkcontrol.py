import sys
import os

if sys.version_info[0] >= 3:
    from gi.repository import Gtk as gtk
else:
    import gtk

import numpy as np
import numpy as np

## assumes you've downloaded the micropolis-4bots repo into the same directory as this (the gym-micropolis) repo.
FILE_DIR = os.path.dirname(os.path.realpath(__file__))
GIT_DIR = os.path.abspath(os.path.join(FILE_DIR, os.pardir, os.pardir))
MICROPOLISCORE_DIR = GIT_DIR + '/gym_city/envs/micropolis/MicropolisCore/src'
sys.path.append(MICROPOLISCORE_DIR)

CURR_DIR = os.getcwd()
# we need to do this so the micropolisgenericengine can access images/micropolisEngine/dataColorMap.png
os.chdir(MICROPOLISCORE_DIR)

from pyMicropolis.gtkFrontend import main

if sys.version_info[0] >= 3:
    from . tilemap import TileMap
else:
    from tilemap import TileMap
os.chdir(CURR_DIR)

class MicroWalkControl():

    def __init__(self, MAP_W=12, MAP_H=12):
        os.chdir(MICROPOLISCORE_DIR)
        self.SHOW_GUI=False
        engine, win1 = main.train()
        os.chdir(CURR_DIR)
        self.engine = engine
      # print(dir(self.engine))
        self.MAP_X = MAP_W
        self.MAP_Y = MAP_H
        self.MAP_XS = 5
        self.MAP_YS = 5
        self.engineTools = ['Residential', 'Commercial', 'Industrial',
                'FireDept',
                'PoliceDept',
                # TODO: implement query (skipped for now by indexing)
               'Query',
               'Wire',
               'Clear',
               'Rail',
               'Road',
                'Stadium',
                'Park',
                 'Seaport',
                'CoalPowerPlant',
                'NuclearPowerPlant',
                'Airport',
                'Net',
                'Water',
                'Land',
                'Forest',
                ]
        self.tools = ['Residential','Commercial','Industrial','Road','Wire','NuclearPowerPlant', 'Park', 'Clear', None]
        # since query is exluded for now:
        self.num_tools = len(self.tools)

        # map with extra channel charting position of walker
        self.map = TileMap(self, self.MAP_X, self.MAP_Y, walker=True)

        self.zones = self.map.zones
        self.num_zones = self.map.num_zones
        # allows building on rubble and forest
        self.engine.autoBulldoze = True
        # for bots
        win1.playCity()
        self.engine.setFunds(1000000)
        engine.setSpeed(3)
        engine.setPasses(500)
        #engine.simSpeed =99
        engine.clearMap()
        self.win1=win1

    def layGrid(self, w, h):

        for i in range(self.MAP_X):
            for j in range(self.MAP_Y):
                gtk.mainiteration()
                self.engine.simTick()
                # vertical road
                if ((i + 4) % w == 0):
                    self.doTool(i, j,'Road')
                    if ((j + 1) % h in [1, h - 1]) and \
                            j not in [0, self.MAP_Y -1]:
                        self.doTool(i, j, 'Wire')
                # horizontal roads
                elif ((j + 1) % h == 0):
                    self.doTool(i, j,'Road')
                    if ((i + 4) % w in [1, w - 1]) and \
                            i not in [0, self.MAP_X - 1]:
                        self.doTool(i, j, 'Wire')
                # random zones
                elif ((i + 2 - (i + 4) // w) % 3) ==0 and \
                     ((j + 2 - (j + 1) // h) % 3) ==0:

                    tool_i = random.randint(0, 3-1)
                    self.doTool(i, j, ['Residential', 'Commercial', 'Industrial'][tool_i])

    def clearMap(self):
        self.engine.clearMap()
        self.map.setEmpty()

    def getFunds(self):
        return self.engine.totalFunds

    def setFunds(self, funds):
        return self.engine.setFunds(funds)

        # called by map module
    def doBulldoze(self, x, y):
        return self.doSimTool(x,y,'Clear')

    def doBotTool(self, x, y, tool):
        '''Takes string for tool'''
        return self.map.addZone(x, y, tool)

    def doTool(self, x, y, tool):
        '''Takes string for tool'''
        return self.map.addZone(x, y, tool)


    def toolDown(self, x, y, tool):
        '''Takes int for tool, depending on engine's index'''

        self.map.addZone(x, y, self.engineTools[tool])

        # called by map module
    def doSimTool(self, x, y, tool):

        x += self.MAP_XS
        y += self.MAP_YS
        x = np.int(x)
        y= np.int(y)
    #   gtk.mainiteration()
        return self.engine.toolDown(self.engineTools.index(tool), x, y)


    def singleStep(self, a, i):
        ''' a is an int between 0 and 4 * i, where i > 0'''
        endpos = self.map.walker_pos
        if a // i == 0:
            endpos[0] += 1
        elif a // i == 1:
            endpos[1] += 1
        elif a // i == 2:
            endpos[0] -= 1
        elif a // i == 3:
            endpos[1] -= 1
        endpos[0] = np.clip(endpos[0], 0, self.MAP_X - 1)
        endpos[1] = np.clip(endpos[1], 0, self.MAP_Y - 1)
        self.map.walker_pos = endpos

    def getResPop(self):
        return self.engine.resPop

    def getComPop(self):
        return self.engine.comPop

    def getIndPop(self):
        return self.engine.indPop

    def getTotPop(self):
        return self.engine.totPop

    def takeSetupAction(self, a):
        tool = self.tools[a[0]]
        x = a[1]
        y = a[2]
        self.doTool(x, y, tool)

    def takeAction(self, xstep, ystep, tool):
        '''tool int depends on self.tools indexing'''
      # print(xstep, ystep, tool)
        tool = self.tools[tool]
        x = np.clip(self.map.walker_pos[0] + xstep, 0, self.MAP_X - 1)
        y = np.clip(self.map.walker_pos[1] + ystep, 0, self.MAP_Y - 1)
        if tool:
            self.doBotTool(x, y, tool)
   #    gtk.mainiteration()
        self.engine.simTick()


    def close(self):
    #   self.engine.doReallyQuit()
        del(self.engine)



