import numpy as np
import gym.spaces

class TileMap(object):
    ''' Map of Micropolis Zones as affected by actions of MicropolisControl object. Also automates bulldozing (always finds deletable tile of larger zones/structures and subsequently removes rubble)'''
    def __init__(self, micro, MAP_X, MAP_Y, walker=False):
        # whether or not the latest call to addZone has any effect on our map
        # no good if we encounter a fixed optimal state / are on a budget
        self.no_change = False
        self.walker = walker
        self.zoneCenters = np.full((MAP_X, MAP_Y), None)

        self.MAP_X = MAP_X
        self.MAP_Y = MAP_Y
        self.num_empty = self.MAP_X * self.MAP_Y
        if self.walker:
            self.walker_pos = [self.MAP_X // 2, self.MAP_Y // 2]

        self.zoneSize = {'Residential': 3, 
                'Commercial' : 3, 
                'Industrial' : 3, 
               #'Seaport' : 4, 
               #'Stadium' : 4, 
               #'PoliceDept' : 3, 
               #'FireDept' : 3, 
               #'Airport' : 6, 
                'NuclearPowerPlant' : 4, 
               #'CoalPowerPlant' : 4, 
                'Road' : 1, 
               #'Rail' : 1, 
                'Park' : 1, 
                'Wire' : 1, 
                'RoadWire': 1, 
                'Clear': None, 
                'Rubble': None 
               #'Net': 1, 
               #'Water': 1, 
               #'Land': 1, 
               #'Forest': 1
               }
                
        self.zones = [z for z in self.zoneSize.keys()]
        self.num_zones = len(self.zones)
        self.zoneInts = {}
        for z in range(self.num_zones):
            self.zoneInts[self.zones[z]] = z
        square_sizes = [3,4,5,6]
        self.zoneSquares = {}
        for z in self.zones:
            z_size = self.zoneSize[z] 
            if z == 'Rubble':
                # make different-size rubble squares
                for s in square_sizes:
                    z0 = 'Rubble_' + str(s)
                    self.zoneSquares[z0] = self.makeZoneSquare(s, self.zoneInts[z])
            elif z_size and z_size > 1:
                self.zoneSquares[z] = self.makeZoneSquare(self.zoneSize[z], self.zoneInts[z])
        # first dimension is one-hot zone encoding followed by zone int
        self.zoneMap = np.zeros((self.num_zones + 1,self.MAP_X, self.MAP_Y), dtype=np.uint8)
        if walker:
            self.walkerZoneMap = np.zeros((self.num_zones, 2 * MAP_X, 2 * MAP_Y))
        self.setEmpty()
        self.micro = micro

    def setEmpty(self):
        self.zoneCenters.fill(None)
        self.zoneMap.fill(0)
        self.zoneMap[-1,:,:] = self.zoneInts['Clear']
        self.zoneMap[self.zoneInts['Clear'],:,:] = 1
        self.num_empty = self.MAP_X * self.MAP_Y
        if self.walker:
#           self.zoneMap[-2, self.walker_pos[0], self.walker_pos[1]] = 0
#           self.zoneMap[-2, self.MAP_X // 2, self.MAP_Y // 2] = 1
            self.walker_pos = [self.MAP_X // 2, self.MAP_Y // 2]


    def makeZoneSquare(self, width, zone_int):
        if self.walker:
            zone_square = np.zeros((self.num_zones + 2, width, width), dtype=int)          
        else:
            zone_square = np.zeros((self.num_zones + 1, width, width), dtype=int)
        zone_square[-1,:,:] = zone_int
        zone_square[zone_int,:,:] = 1
        return zone_square

    def setWalkerPos(self, x, y):
        assert self.walker == True
     #  self.walker_pos[0] = min(x, self.MAP_X - 1)
     #  self.walker_pos[1] = min(y, self.MAP_Y - 1)
#       self.zoneMap[-2, self.walker_pos[0], self.walker_pos[1]] = 0
#       self.zoneMap[-2, x, y] = 1
        self.walker_pos = [x, y]

    def _addZoneInt(self, zone_int, x, y):
        old_zone = self.zoneMap[-1][x][y]
        self.zoneMap[-1][x][y] = zone_int
        self.zoneMap[old_zone][x][y] = 0
        self.zoneMap[zone_int][x][y] = 1


    def addZone(self, x, y, zone):
        ''' (x, y) must be a valid tile in the game Assumes the player has enough money to execute action.'''
    #   if self.walker:
    #       self.setWalkerPos(x, y)
        if not zone:
            return
        old_zone_int = self.zoneMap[-1][x][y]
        zone_int = self.zoneInts[zone]
      # print(zone_int, old_zone_int)
      # print(self.zoneCenters[x][y], (x,y))
        if zone_int == old_zone_int and ((not self.zoneCenters[x][y]) or self.zoneCenters[x][y] == (x,y)):
            self.no_change = True
            return
        # do not punish ineffectual bulldozing
        self.no_change = False
        if zone == 'Clear':
            self.bulldoze(x, y)
            return
        # assumes zones are squares
        zone_size = self.zoneSize[zone]
        clear_int = self.zoneInts['Clear']
        rubble_int = self.zoneInts['Rubble']
        if zone_size == 1:
            if (zone == 'Wire' and old_zone_int == self.zoneInts['Road']) or \
                    (zone == 'Road' and old_zone_int == self.zoneInts['Wire']):
                zone_int = self.zoneInts['RoadWire']
                result = self.micro.doSimTool(x, y, zone)
                if result == 1:
                    self._addZoneInt(zone_int, x, y)
                else: 
         #          print('unexpected (road-wire) build fail: {} at {} {} with code {}'.format(zone, x, y, result))
                    pass
                return
            else:
                if old_zone_int != clear_int:
                    self.bulldoze(x, y)
                    old_zone_int = self.zoneMap[-1][x][y]
                    if zone == 'Park':
                        # remove rubble
                        self.bulldoze(x, y)
                    result = self.micro.doSimTool(x, y, zone)
                    if result != 1:
#                       print('unexpected tile build fail on bulldozed tile: {} at {} {} with code {}'.format(zone, x, y, result))
                        return 
                else:
                    result = self.micro.doSimTool(x, y, zone)
                    if result != 1:
#                       print('unexpected tile build fail on clear tile: {} at {} {} with code {}'.format(zone, x, y, result))
                        return
                if result == 1:
                    self.num_empty -= 1
                    self._addZoneInt(zone_int, x, y)
        else:
            zone_square = self.zoneSquares[zone]
        #   result = self.micro.doSimTool(x, y, zone)
            x0, y0 = max(x - 1, 0), max(y - 1, 0) 
            x1, y1 = min(x + zone_size - 1, self.MAP_X), min(y + zone_size - 1, self.MAP_Y) 
            if True:
                for i in range(x0, x1):
                    for j in range(y0, y1):
                        old_int = self.zoneMap[-1][i][j]
                        if old_int != clear_int and old_int != rubble_int:
                            self.bulldoze(i, j)
                result = self.micro.doSimTool(x, y, zone)
                if result != 1:
#                   print('unexpected (zone-square) build fail: {} at {} {} with code {}'.format(zone, x, y, result))
                    return
            if result == 1:
                self.num_empty -= (x1 - x0) * (y1 - y0)
                self.zoneMap[-1,x0 : x1, y0 : y1] = zone_square[-1, : x1 - x0, : y1 - y0]
                self.zoneMap[:self.num_zones, x0 : x1, y0 : y1] = zone_square[:self.num_zones, : x1 - x0, : y1 - y0]
                centers =  np.empty((x1 - x0, y1 - y0),dtype=object)
                centers.fill((x, y))  
                self.zoneCenters[x0 : x1, y0 : y1] = centers




    def bulldoze(self, x, y):
        clear_int = self.zoneInts['Clear']
        old_zone_int = self.zoneMap[-1, x, y]
        old_zone_size = self.zoneSize[self.zones[old_zone_int]]
        if clear_int == old_zone_int:
            return
        if  ((not old_zone_size) or old_zone_size == 1):
    #       print('deleting tile')
            result = self.micro.doBulldoze(x, y)
            if result == 1:
                self._addZoneInt(clear_int, x, y)
                if old_zone_int != self.zoneInts['Rubble']:
                    self.num_empty += 1
            else:
                pass
    #           print('unexpected bulldoze fail: {} on tile at {} {}'.format(self.zones[old_zone_int], x, y))
        else:
            zone_center = self.zoneCenters[x][y]
            if zone_center:
    #           print('deleting square')
                xc, yc = zone_center
    #           print(x,y)
#               print(width)
                result = self.micro.doBulldoze(xc, yc)
                if result == 1:
                    rubble_square = self.zoneSquares['Rubble_' + str(old_zone_size)]
                    # adjust rubble square to fit in zoneMap, in case zone lies
                    # outside of our map
                    x0, y0 = max(xc - 1, 0), max(yc - 1, 0)
                    x1, y1 = min(xc + old_zone_size - 1, self.MAP_X), min(yc + old_zone_size - 1, self.MAP_Y) 
                    self.zoneMap[:self.micro.num_tools, x0 : x1, y0 : y1] = rubble_square[:self.micro.num_tools, : x1 - x0, : y1 - y0]
                    self.zoneMap[-1:, x0 : x1, y0 : y1] = rubble_square[-1:, : x1 - x0, : y1 - y0]

                    self.num_empty += (x1 - x0) * (y1 - y0)

                    centers =  np.empty((x1 - x0, y1 - y0),dtype=object)
                 #  centers.fill(None)  
                    self.zoneCenters[x0 : x1, y0 : y1] = centers
                else:
#                   print('unexpected zone bulldoze fail at {} {}'.format(x, y))
                    pass

    def getMapState(self):
        if self.walker:
            x0 =  self.walker_pos[0]
            x1 = self.walker_pos[0] + self.MAP_X
            y0 = self.walker_pos[1]
            y1 = self.walker_pos[1] + self.MAP_Y
            self.walkerZoneMap[:, self.MAP_X // 2: self.MAP_X + self.MAP_X // 2, self.MAP_Y // 2: self.MAP_Y + self.MAP_Y // 2] = self.zoneMap[:-1]
            return self.walkerZoneMap[:, x0: x1, y0: y1]
        else:
            return self.zoneMap[:-1,:,:]
