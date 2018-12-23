import sys
import os
import random
import numpy as np
if sys.version_info[0] >= 3:
    from gi.repository import Gtk as gtk
else:
    import gtk

## assumes you've downloaded the micropolis-4bots repo into the same directory as this (the gym-micropolis) repo.
FILE_DIR = os.path.dirname(os.path.realpath(__file__))
GIT_DIR = os.path.abspath(os.path.join(FILE_DIR, os.pardir, os.pardir))
if sys.version_info[0] >= 3:
    MICROPOLISCORE_DIR = GIT_DIR + '/micropolis/MicropolisCore/src'
    sys.path.append(MICROPOLISCORE_DIR)
    from .tilemap import TileMap, zoneFromInt
else:
    MICROPOLISCORE_DIR = GIT_DIR + '/micropolis/MicropolisCore/src'
    sys.path.append(MICROPOLISCORE_DIR)
    from tilemap import TileMap

CURR_DIR = os.getcwd()
# we need to do this so the micropolisgenericengine can access images/micropolisEngine/dataColorMap.png


os.chdir(MICROPOLISCORE_DIR)   

#import micropolis
from pyMicropolis.gtkFrontend import main


os.chdir(CURR_DIR)

class MicropolisPaintControl():

    def __init__(self, MAP_W=12, MAP_H=12, PADDING=13, parallel_gui=False, rank=None):
        self.SHOW_GUI=False
        engine, win1 = main.train(bot=self, rank=rank)
        os.chdir(CURR_DIR)
        self.engine = engine
        self.engine.setGameLevel(2)
        self.MAP_X = MAP_W
        self.MAP_Y = MAP_H
        self.PADDING = PADDING
        # shifts build area to centre of 120 by 100 tile map
       # self.MAP_XS = 59 - self.MAP_X // 2
       # self.MAP_YS = 49 - self.MAP_Y //2
        self.MAP_XS = 5
        self.MAP_YS = 5
        self.num_roads = 0
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
        # Names correspond to those of resultant zones
        self.tools = ['Residential', 'Commercial', 'Industrial', 
                'FireDept', 
                'PoliceDept', 
             # 'Query',
                'Clear',
               'Wire',
              #'Land',
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
        #['Residential','Commercial','Industrial','Road','Wire','NuclearPowerPlant', 'Park', 'Clear']
        # since query is exluded for now:
        self.num_tools = len(self.tools)
        self.map = TileMap(self, self.MAP_X + 2 * PADDING, self.MAP_Y + 2 * PADDING)
        self.zones = self.map.zones
        self.num_zones = self.map.num_zones
        # allows building on rubble and forest
        self.engine.autoBulldoze = True
        # for bots 
        self.land_value = 0
        win1.playCity()
        self.engine.setFunds(1000000)
        engine.setSpeed(3)
        engine.setPasses(500)
        #engine.simSpeed =99
        self.total_traffic = 0
        self.last_total_traffic = 0
#       engine.clearMap()
        self.win1=win1
        self.player_builds = []

    def layGrid(self, w, h):

        for i in range(self.MAP_X):
            for j in range(self.MAP_Y):
            #   gtk.mainiteration()
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
    
    def newMap(self):
        self.engine.generateMap()
        self.updateMap()

    def clearMap(self):
        self.map.setEmpty()
        self.engine.clearMap()
        self.updateMap()

    def clearBotBuilds(self):
        self.map.clearBotBuilds()

    def updateMap(self):
        for i in range(self.MAP_X):
            for j in range(self.MAP_Y):
                tile_int = self.getTile(i,j)
                zone = zoneFromInt(tile_int)
                # assuming there are no zones not built via us, 
                # or else we must find center
                self.map.updateTile(i, j, zone, (i,j))



    def getDensityMaps(self):
       #self.last_pollution = self.pollution
        self.total_traffic = 0
        self.land_value = 0
        density_maps = np.zeros((3, self.MAP_X, self.MAP_Y))
        for i in range (self.MAP_X):
            for j in range(self.MAP_Y):
                im = p_im = t_im = i + self.MAP_XS
                jm = p_jm = t_jm = j + self.MAP_YS
                t_im -= 2
                t_jm -= 4
                t_xy_density = self.engine.getTrafficDensity(im, jm)
                self.total_traffic += t_xy_density
                density_maps[2][i][j] = self.engine.getTrafficDensity(t_im, t_jm)
                p_im -= 2
                p_jm -= 2
                density_maps[1][i][j] = self.engine.getPopulationDensity(p_im, p_jm)
                density_maps[0][i][j] = self.engine.getPowerGrid(im, jm)
                self.land_value += self.engine.getLandValue(im, jm)

        return density_maps

    def getPowerMap(self):
        power_map = np.zeros((1, self.MAP_X, self.MAP_Y))
        for i in range (self.MAP_X):
            for j in range(self.MAP_Y):
                im = i + self.MAP_XS
                jm = j + self.MAP_YS
        return power_map

    def getFunds(self):
        return self.engine.totalFunds

    def render(self):
        while gtk.events_pending():
       #for i in range(2):
            gtk.main_iteration()

    def setFunds(self, funds):
        return self.engine.setFunds(funds)

        # called by map module
    def doBulldoze(self, x, y):
        return self.doSimTool(x,y,'Clear')

    def doLandOver(self, x, y):
        ''' a glitchy replacement to doBulldoze (layered buildings)
        '''

    def doBotTool(self, x, y, tool, static_build=False):
        '''Takes string for tool'''
        return self.map.addZoneBot(x + self.PADDING, y + self.PADDING, tool, static_build=static_build) 

    def doTool(self, x, y, tool):
        '''Takes string for tool'''
        return self.map.addZoneBot(x, y, tool) 

    def playerToolDown(self, tool_int, x, y):
        if not x < self.MAP_X and y < self.MAP_Y:
            return
       #x += self.MAP_XS
       #y += self.MAP_YS
       #tool = self.tools[tool_int]
       #self.map.addZonePlayer(x, y, tool, static_build=True)
        self.player_builds += [(tool_int, x, y)]

    def toolDown(self, x, y, tool):
        '''Takes int for tool, depending on engine's index'''
        self.map.addZoneBot(x, y, self.engineTools[tool])

        # called by map module
    def doSimTool(self, x, y, tool):
        x += self.MAP_XS
        y += self.MAP_YS
        tool = self.engineTools.index(tool)
        return self.doSimToolInt(x, y, tool)

    def getTile(self, x, y):
        x += self.MAP_XS
        y += self.MAP_YS
        return self.engine.getTile(x, y) & 1023

    def doSimToolInt(self, x, y, tool):

        return self.engine.toolDown(tool, x, y)

    def getResPop(self):
        return self.engine.resPop

    def getComPop(self):
        return self.engine.comPop

    def getIndPop(self):
        return self.engine.indPop


    def getTotPop(self):
        return self.engine.totalPop

    def takeSetupAction(self, a):
        tool = self.tools[a[0]]
        x = a[1]
        y = a[2]
        self.doTool(x, y, tool)

    def takeAction(self, a, static_build=False):
        '''tool int depends on self.tools indexing'''
        for i in range(self.MAP_X):
            for j in range(self.MAP_Y):
                for t in range(self.num_tools):
                    if a[t][i][j] == 1:
                        tool = self.tools[t]
                        self.doBotTool(i, j, tool, static_build)
        self.engine.simTick()
#       gtk.mainiteration()

    def printTileMap(self):
        tileMap = np.zeros(shape=(self.MAP_X, self.MAP_Y))
        for i in range(self.MAP_X):
            for j in range(self.MAP_Y):
                tileMap[i][j] = self.getTile(i, j)
        print(tileMap)
 
    def close(self):
    #   self.engine.doReallyQuit()
        del(self.engine)



