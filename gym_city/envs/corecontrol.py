import sys
import os
import random
import numpy as np
if sys.version_info[0] >= 3:
    from gi.repository import Gtk as gtk
    from .tilemap import TileMap, zoneFromInt
else:
    import gtk
import time

## assumes you've downloaded the micropolis-4bots repo into the same directory as this (the gym-micropolis) repo.
FILE_DIR = os.path.dirname(os.path.realpath(__file__))
GIT_DIR = os.path.abspath(os.path.join(FILE_DIR, os.pardir, os.pardir))
#if sys.version_info[0] >= 3:
#    MICROPOLISCORE_DIR = GIT_DIR + '/micropolis/MicropolisCore/src'
#    sys.path.append(MICROPOLISCORE_DIR)
#    from .tilemap import TileMap, zoneFromInt
#else:
#    MICROPOLISCORE_DIR = GIT_DIR + '/micropolis/MicropolisCore/src'
#    sys.path.append(MICROPOLISCORE_DIR)
#    from tilemap import TileMap
#
#CURR_DIR = os.getcwd()
## we need to do this so the micropolisgenericengine can access images/micropolisEngine/dataColorMap.png
#
#
#os.chdir(MICROPOLISCORE_DIR)
sys.path.append(os.path.abspath(os.path.join(FILE_DIR, './micropolis/MicropolisCore/src')))
print(sys.path)
from pyMicropolis.gtkFrontend import main


#os.chdir(CURR_DIR)

class MicropolisControl():

    def __init__(self, env, MAP_W=12, MAP_H=12, PADDING=13, gui=False, rank=None,
            power_puzzle=False, paint=False):
        env.micro = self # attach ourselves to our parent before we start
        # Reuse game engine if we are reinitializing controller (i.e. to change map size)
        if hasattr(self, 'engine'):
            print('REINIT: reuse engine and window')
            engine, win1 = self.engine, self.win1
        else:
            engine, win1 = main.train(env=env, rank=rank, map_x=MAP_W, map_y=MAP_H,
                gui=gui)
       #os.chdir(CURR_DIR)
        self.env = env
        self.engine = engine
        self.engine.setGameLevel(2)
        self.MAP_X = MAP_W
        self.MAP_Y = MAP_H
        self.PADDING = PADDING
        # shifts build area to centre of 120 by 100 tile map
       # self.MAP_XS = 59 - self.MAP_X // 2
       # self.MAP_YS = 49 - self.MAP_Y //2
        self.MAP_XS = 16
        self.MAP_YS = 8
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
        if power_puzzle:
            self.tools = ['Wire']
        else:
            self.tools = [
                'Residential', 'Commercial', 'Industrial',
                'FireDept',
                'PoliceDept',
                # 'Query',
                'Clear',
                'Wire',
                # 'Land',
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
                'Nil' # the agent takes no action
                ]
        #['Residential','Commercial','Industrial','Road','Wire','NuclearPowerPlant', 'Park', 'Clear']
        # since query is exluded for now:
        self.num_tools = len(self.tools)
        # TODO: move age-tracking into wrapper?
        self.map = TileMap(self, self.MAP_X + 2 * PADDING, self.MAP_Y + 2 * PADDING,
                paint=paint)
        self.zones = self.map.zones
        self.num_zones = self.map.num_zones
        # allows building on rubble and forest
        self.engine.autoBulldoze = True
        # for bots
        self.land_value = 0

       #win1.playCity()
        if win1:
           win1.playCity()
        self.engine.resume()
        self.engine.setGameMode('play')

        self.init_funds = 2000000
        self.engine.setFunds(self.init_funds)
        self.engine.setSpeed(3)
        self.engine.setPasses(100)
        #engine.simSpeed =99
        self.total_traffic = 0
        self.last_total_traffic = 0
#       engine.clearMap()
        self.win1=win1
        self.player_builds = []

    def reset_params(self, size):
        '''Change map-size of existing controller object.'''
        # gui is irrelevant here (only passed to micropolis)
        self.__init__(self.env,
                      MAP_W=size, MAP_H=size, PADDING=self.PADDING, gui=False, rank=self.env.rank,
                      power_puzzle=self.env.power_puzzle, paint=False)


    def displayRewardWeights(self, reward_weights):
        self.win1.agentPanel.displayRewardWeights(reward_weights)

    def simTick(self):
       #self.engine.resume()
       #self.engine.cityEvaluation()
        self.engine.tickEngine()
        self.engine.simTick()
       #self.engine.updateHeads()
       #self.engine.updateDate()
       #self.engine.changeCensus()
       #self.engine.simUpdate()
       #self.engine.doTimeStuff()

    def layGrid(self, w, h):

        for i in range(self.MAP_X):
            for j in range(self.MAP_Y):
            #   gtk.mainiteration()
                self.simTick()
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


    def fillDensityMap(self, density_map, i, j, val):
        i = i * 2
        j = j * 2
        density_map[i][j] = val
        density_map[i][j + 1] = val
        density_map[i + 1][j + 1] = val
        density_map[i + 1][j] = val
        return density_map

    def getDensityMaps(self):
       #self.last_pollution = self.pollution
        self.total_traffic = 0
        self.land_value = 0
        density_maps = np.zeros((3, self.MAP_X, self.MAP_Y))
        for i in range (self.MAP_X // 2):
            for j in range(self.MAP_Y // 2):
                im = p_im = t_im = i + self.MAP_YS // 2
                jm = p_jm = t_jm = j + self.MAP_XS // 2
                t_xy_density = self.engine.getTrafficDensity(t_jm, t_im)
                self.total_traffic += t_xy_density
                density_maps[2] = self.fillDensityMap(density_maps[2], i, j, t_xy_density)
                pop_xy_density = self.engine.getPopulationDensity(p_jm, p_im)
                density_maps[1] = self.fillDensityMap(density_maps[1], i, j, pop_xy_density)
        for i in range(self.MAP_X):
            for j in range(self.MAP_Y):
                im = i
                jm = j
                im += self.MAP_YS
                jm += self.MAP_XS
                density_maps[0][i][j] = self.engine.getPowerGrid(jm, im)
               #self.land_value += self.engine.getLandValue(im, jm)
       #if self.total_traffic > 0:
       #    print('TRAFFIC: {}'.format(self.total_traffic))
        return density_maps

    def getPowerMap(self):
        power_map = np.zeros((1, self.MAP_X, self.MAP_Y))
        for i in range (self.MAP_X):
            for j in range(self.MAP_Y):
                im = i + self.MAP_XS
                jm = j + self.MAP_YS
        return power_map

    def getFunds(self):
       #print('getting funds total {}'.format(self.engine.totalFunds))
        return self.engine.totalFunds

    def render(self):
        while gtk.events_pending():
       #for i in range(2):
            gtk.main_iteration()

    def setFunds(self, funds):
       #print('setting funds to {}'.format(funds))
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
        if tool == 'Nil':
            return
        return self.map.addZoneBot(x, y, tool)

    # called from mictopolistool.py in micropoliengine
    def playerToolDown(self, tool_int, x, y):
        if not x < self.MAP_X and y < self.MAP_Y:
            print('build site out of range')
            return
       #x += self.MAP_XS
       #y += self.MAP_YS
        tool_int = self.tools.index(self.engineTools[tool_int])
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
       #print('calling engine doTool {} {} {}'.format(x, y, tool))
        result = self.engine.toolDown(tool, x, y)
       #print('result in SimToolInt: {}'.format(result))
        return result

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
        tool = self.tools[a[0]]
        x = int(a[1])
        y = int(a[2])
       #print('taking action {} {} {}'.format(x + self.MAP_XS, y + self.MAP_YS, tool))
        self.doBotTool(x, y, tool, static_build)
       #gtk.main_iteration() # for observation or recording
       #time.sleep(1/60)
       #self.engine.simTick()
       #time.sleep(1/60)
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



