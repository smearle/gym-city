#!/bin/bash

import pexpect
import numpy as np
import random

from .tile_map import TileMap

class MicropolisControl():

    def __init__(self, MAP_X=10, MAP_Y=10):
        self.bash = pexpect.spawn('micropolis -g -t\n')
        self.expectSim()
        self.bash.send('sim ClearMap\n')
        self.expectSim()
        self.enableAutoBudget()
        self.bash.send('sim Speed 3\n')
        self.expectSim()
        self.bash.send('sim Disasters 0\n')
        self.expectSim()
        self.bash.send('sim Sound 0\n')
        self.expectSim()
        self.bash.send('sim Delay {}\n'.format(1))
        self.expectSim()
        # editor window
        self.e = '.head0.col2.editor1'
        self.v = '.head0.col2.editor1.centerframe.view'
        self.t ='.head0.col1.w3.notice3.top.text'
        # default terrain size is 120 by 100 (x by y)
        self.MAP_X = MAP_X
        self.MAP_Y = MAP_Y
        # shifts build area to centre of 120 by 100 tile map
        self.MAP_XS = 59 - self.MAP_X // 2
        self.MAP_YS = 49 - self.MAP_Y //2
## Might be useful for separate camerafollow object
#       self.terrain_res = (1920, 1600)
#       self.tscale = (self.terrain_res[0] / self.MAP_X,
#               self.terrain_res[1] / self.MAP_Y)
#       self.view_mid = (349, 438)

        self.zonetools =['res','com','ind']
        self.tools = ['Residential', 'Commercial', 'Industrial',
                #'Stadium',
                # 'Seaport',
                #'Airport',
                #'PoliceDept', 'FireDept',
                'Road', 'Park',
                #'Rail',
                'Clear', 'Wire',
                #'CoalPowerPlant',
                'NuclearPowerPlant']
        self.num_tools = len(self.tools)
        self.map = TileMap(self)
        self.zones = self.map.zones
        self.num_zones = self.map.num_zones
        self.bash.send('sim SoundOff\n')
        self.expectSim()
        self.toolOneHot = {}
        self.oneHotTool = {}



        identity = np.identity(len(self.zonetools))
        i = 0
        for t in self.zonetools:
            self.toolOneHot[t] = identity[i]
            self.oneHotTool[i] = t
            i += 1


    def enableAutoBudget(self):
        self.bash.send('sim AutoBudget 1\n')
        self.expectSim()

    def takeAction(self, a):
        self.resume()
        tool = self.tools[a[0]]
        x = a[1]
        y = a[2]
        self.doTool(x, y, tool)

    def close(self):
        self.bash.send('sim ReallyQuit\n')
        self.expectSim()

    def expectSim(self):
        self.bash.expect('sim:')

    def getPopulation(self):
        self.bash.send('sim TotalPop\n')
        self.expectSim()
        result = self.getBuildSuccess()
        return result


    def getYear(self):
        return 2000

    def getResPop(self):
        self.bash.send('sim ResPop\n')
        self.expectSim()
        result = self.getBuildSuccess()
        return result

    def getComPop(self):
        self.bash.send('sim ComPop\n')
        self.expectSim()
        result = self.getBuildSuccess()
        return result

    def getIndPop(self):
        self.bash.send('sim IndPop\n')
        self.expectSim()
        result = self.getBuildSuccess()
        return result


    def getBuildSuccess(self):
        result = int(self.bash.before.split(b'\r\n')[-2])
      # print('Result of build: {}'.format(result))
        return result

    def pause(self):
        self.bash.send('sim Pause\n')
        self.expectSim()

    def resume(self):
        self.bash.send('sim Resume\n')
        self.expectSim()

    def setFunds(self, amount):
        self.bash.send("sim Funds {}\n".format(amount))
        self.bash.expect(b'sim:')
        return self.getBuildSuccess()


    def getFunds(self):
        self.bash.send("sim Funds\n")
        self.expectSim()
        amount = int(self.bash.before.split(b"\n")[-2])
        return amount

    def doTool(self, x, y, tool):
        self.map.addZone(x, y, tool)

    def doSimTool(self, x, y, tool):
        zone = tool
        if tool == 'Clear':
            tool = 'Bulldozer'
        x_center = x + self.MAP_XS
        y_center = y + self.MAP_YS
        sim_str = 'sim {0}Tool {1} {2}\n'.format(tool, x_center, y_center)
      # print(sim_str)
        self.bash.send(sim_str)
        self.expectSim()
        result = self.getBuildSuccess()
        return result



    def doBulldoze(self, x, y):
        x_center = x + self.MAP_XS
        y_center = y + self.MAP_YS
        self.bash.send('sim BulldozerTool {} {}\n'.format(x_center, y_center))
        self.expectSim()
        result = self.getBuildSuccess()
        return result





    def testDiagRoad(self):
        for i in range(100):
            self.buildRoad(i, i)

    def clearMap(self):
        self.bash.send("sim ClearMap\n")
        self.map.setEmpty()
        self.bash.expect("sim:")

    def fillRoad(self):
        for i in range(self.MAP_X):
            for j in range (self.MAP_Y):
                self.buildRoad(i, j)

    def layGrid(self, w, h):
        for i in range(self.MAP_X):
            for j in range(self.MAP_Y):
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

def test():

    micro = MicropolisControl()
    micro.setFunds(9999999)
#   pprint(micro.queryMap())
    micro.layGrid(7, 10)
    for u in range(5):
        micro.bulldoze(random.randint(0,micro.MAP_X), random.randint(0,micro.MAP_Y))
        print(micro.map.zoneCenters)
        print(micro.map.zoneMap)
#   micro.fillRoad()
    print('done test')

