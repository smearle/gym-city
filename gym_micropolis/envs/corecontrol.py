import sys
import os
import random
import gtk
import tilemap

## depends on the location of the MicropolisCore folder
MICROPOLIS_CORE_DIR = '/home/sme/micropolis/MicropolisCore'
sys.path.append(MICROPOLIS_CORE_DIR + '/src')
os.chdir(MICROPOLIS_CORE_DIR + '/src')

from pyMicropolis.gtkFrontend import main

from tilemap import TileMap

class MicropolisControl():

    def __init__(self, MAP_W=6, MAP_H=6):
        self.SHOW_GUI=False
        engine, win1 = main.train()
        self.engine = engine
      # print(dir(self.engine))
        self.MAP_X = MAP_W
        self.MAP_Y = MAP_H
        # shifts build area to centre of 120 by 100 tile map
       # self.MAP_XS = 59 - self.MAP_X // 2
       # self.MAP_YS = 49 - self.MAP_Y //2
        self.MAP_XS = 1
        self.MAP_YS = 1
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
        self.tools = ['Residential','Commercial','Industrial','Road','Wire','NuclearPowerPlant', 'Park', 'Clear']
        # since query is exluded for now:
        self.num_tools = len(self.tools)

        self.map = TileMap(self)
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
        return self.engine.toolDown(self.engineTools.index(tool), x, y)

    def getResPop(self):
        return self.engine.resPop

    def getComPop(self):
        return self.engine.comPop

    def getIndPop(self):
        return self.engine.indPop

    def getTotPop(self):
        return self.engine.totPop

    def takeAction(self, a):
        '''tool int depends on self.tools indexing'''
     #  if self.SHOW_GUI:
     #      print('show gui')
     #  gtk.mainiteration()
        self.engine.simTick()
        tool = self.tools[a[0]]
        x = a[1] 
        y = a[2]
        self.doTool(x, y, tool)
 
    def close(self):
    #   self.engine.doReallyQuit()
        del(self.engine)


