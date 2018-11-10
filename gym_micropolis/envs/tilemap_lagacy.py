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
                'Seaport' : 4, 
                'Stadium' : 4, 
                'PoliceDept' : 3, 
                'FireDept' : 3, 
                'Airport' : 6, 
                'NuclearPowerPlant' : 4, 
                'CoalPowerPlant' : 4, 
                'Road' : 1, 
                'Rail' : 1, 
                'Park' : 1, 
                'Wire' : 1, 
                'Clear': 1, 
                'Rubble': None ,
                'Net': 1, 
                'Water': 1, 
                'Land': 1, 
                'Forest': 1
               }
        # hacky - this way we can exclude them from the one-hot zone map easily
        self.zones = [z for z in self.zoneSize.keys()] + ['RoadWire', 'RailWire']
        composite_zones = {"RoadWire": ["Road", "Wire"], "RailWire": ["Rail", "Wire"]}
        self.zoneSize['RoadWire'] = 1
        self.zoneSize['RailWire'] = 1
        self.num_zones = len(self.zones)
        self.num_features = self.num_zones - len(composite_zones)
        self.zoneInts = {}
        for z in range(self.num_zones):
            self.zoneInts[self.zones[z]] = z
        self.clear_int = self.zoneInts['Clear']

        def makeZoneSquare(width, zone_int, feature_ints=None):
            if feature_ints is None:
                feature_ints = [zone_int]
            if self.walker:
                zone_square = np.zeros((self.num_features + 1, width, width), dtype=int)          
            else:
                zone_square = np.zeros((self.num_features + 1, width, width), dtype=int)
            zone_square[-1,:,:] = zone_int
            for feature_int in feature_ints:
                zone_square[feature_int,:,:] = 1
            return zone_square

        square_sizes = [3,4,5,6]
        self.zoneSquares = {}
        for z in self.zones:
            z_size = self.zoneSize[z] 
            if z == 'Rubble':
                # make different-size rubble squares
                for s in square_sizes:
                    z0 = 'Rubble_' + str(s)
                    self.zoneSquares[z0] = makeZoneSquare(s, self.zoneInts[z])
            else:
                zone_int = self.zoneInts[z]
                if z in composite_zones.keys():
                    feature_ints = [self.zoneInts[z1] for z1 in composite_zones[z]] 
                else:
                    feature_ints = None
                self.zoneSquares[z] = makeZoneSquare(self.zoneSize[z], zone_int, feature_ints)
        # first dimension is one-hot zone encoding (of everything saved RoadWire and RailWire - 
        # which will be represented by concuttent activation of road/rail & wire features) - followed by zone int
        self.zoneMap = np.zeros((self.num_features + 1,self.MAP_X, self.MAP_Y), dtype=np.uint8)
        if walker:
            self.walkerZoneMap = np.zeros((self.num_features, 2 * MAP_X, 2 * MAP_Y))
        self.static_builds = None
        self.setEmpty()
        self.micro = micro

    def initStaticBuilds(self):
        self.static_builds = np.zeros((1, self.MAP_X, self.MAP_Y), dtype=int)

    def setEmpty(self):
        if self.static_builds is not None:
            self.static_builds.fill(0)
        self.zoneCenters.fill(None)
        self.zoneMap.fill(0)
        self.zoneMap[-1,:,:] = self.zoneInts['Clear']
        self.zoneMap[self.zoneInts['Clear'],:,:] = 1
        self.num_empty = self.MAP_X * self.MAP_Y
        if self.walker:
#           self.zoneMap[-2, self.walker_pos[0], self.walker_pos[1]] = 0
#           self.zoneMap[-2, self.MAP_X // 2, self.MAP_Y // 2] = 1
            self.walker_pos = [self.MAP_X // 2, self.MAP_Y // 2]

    def setWalkerPos(self, x, y):
        assert self.walker == True
     #  self.walker_pos[0] = min(x, self.MAP_X - 1)
     #  self.walker_pos[1] = min(y, self.MAP_Y - 1)
#       self.zoneMap[-2, self.walker_pos[0], self.walker_pos[1]] = 0
#       self.zoneMap[-2, x, y] = 1
        self.walker_pos = [x, y]


    ### HUMANS###
    def addZoneSquare(self, zone_int, x, y, static_build=False):
        ''' used only by humans - so we operate on the assumption that this is a static build, 
        potentially overwriting other static builds '''
        zone = self.zones[zone_int]
        old_zone_int = self.zoneMap[-1][x][y]
        if (zone == 'Wire' and old_zone_int == self.zoneInts['Road']) or \
                (zone == 'Road' and old_zone_int == self.zoneInts['Wire']):
            zone_int = self.zoneInts['RoadWire']
            self._addZoneInt(zone_int, x, y, static_build)
            return
        if (zone == 'Wire' and old_zone_int == self.zoneInts['Rail']) or \
                (zone == 'Rail' and old_zone_int == self.zoneInts['Wire']):
            zone_int = self.zoneInts['RailWire']
            self._addZoneInt(zone_int, x, y, static_build)
            return
        zone_size = self.zoneSize[zone]
        zone_square = self.zoneSquares[zone]
        x0, y0 = max(x - 1, 0), max(y - 1, 0) 
        x1, y1 = min(x + zone_size - 1, self.MAP_X), min(y + zone_size - 1, self.MAP_Y) 
        self.num_empty -= (x1 - x0) * (y1 - y0)
        self.zoneMap[-1, x0 : x1, y0 : y1] = zone_square[-1, : x1 - x0, : y1 - y0]
        self.zoneMap[:self.num_features, x0 : x1, y0 : y1] = zone_square[:self.num_features, : x1 - x0, : y1 - y0]
        centers =  np.empty((x1 - x0, y1 - y0),dtype=object)
        centers.fill((x, y))  
        self.zoneCenters[x0 : x1, y0 : y1] = centers
        if static_build and zone_int != self.clear_int:
            self.static_builds[0, x0 : x1, y0 : y1] = np.full((x1 - x0, y1 - y0), 1)
        
    ### BUILDER BOTS ###
    def addZone(self, x, y, zone, static_build=False):
        ''' (x, y) must be a valid tile in the game. Assumes the player has enough money to execute action.'''
    #   if self.walker:
    #       self.setWalkerPos(x, y)
        assert zone
        # do not build over static builds
        if self.static_builds[0][x][y] == 1:
            return
        old_zone_int = self.zoneMap[-1][x][y]
        zone_int = self.zoneInts[zone]
      # print(zone_int, old_zone_int)
      # print(self.zoneCenters[x][y], (x,y))
        if zone_int == old_zone_int and ((not self.zoneCenters[x][y]) or self.zoneCenters[x][y] == (x,y)):
            self.no_change = True
            return
        # do not count ineffectual bulldozing as a no_change move
        self.no_change = False
        if zone_int == self.zoneInts['Road'] or zone_int == self.zoneInts['RoadWire']:
            self.micro.num_roads += 1
        if zone == 'Clear':
            self.bulldoze(x, y)
            return
        # assumes zones are squares
        zone_size = self.zoneSize[zone]
        clear_int = self.zoneInts['Clear']
        rubble_int = self.zoneInts['Rubble']
        if zone_size == 1:
            if self.micro.doSimTool(x, y, zone) == 1:
                self.addZoneSquare(zone_int, x, y)
#           if (zone == 'Wire' and old_zone_int == self.zoneInts['Road']) or \
#                   (zone == 'Road' and old_zone_int == self.zoneInts['Wire']):
#               zone_int = self.zoneInts['RoadWire']
#               result = self.micro.doSimTool(x, y, zone)
#               if result == 1:
#                   self._addZoneInt(zone_int, x, y, static_build)
#               else: 
#        #          print('unexpected (road-wire) build fail: {} at {} {} with code {}'.format(zone, x, y, result))
#                   pass
#               return
#           if (zone == 'Wire' and old_zone_int == self.zoneInts['Rail']) or \
#                   (zone == 'Rail' and old_zone_int == self.zoneInts['Wire']):
#               zone_int = self.zoneInts['RailWire']
#               result = self.micro.doSimTool(x, y, zone)
#               if result == 1:
#                   self._addZoneInt(zone_int, x, y, static_build)
#               else: 
#        #          print('unexpected (road-wire) build fail: {} at {} {} with code {}'.format(zone, x, y, result))
#                   pass
#               return
#           else:
#               if old_zone_int != clear_int:
#                   self.bulldoze(x, y)
#                   old_zone_int = self.zoneMap[-1][x][y]
#                   if zone == 'Park':
#                       # remove rubble
#                       self.bulldoze(x, y)
#                   result = self.micro.doSimTool(x, y, zone)
#                   if result != 1:
#                       print('unexpected tile build fail on bulldozed tile: {} at {} {} with code {}'.format(zone, x, y, result))
#                       return 
#               else:
#                   result = self.micro.doSimTool(x, y, zone)
#                   if result != 1:
#                       print('unexpected tile build fail on clear tile: {} at {} {} with code {}'.format(zone, x, y, result))
#                       return
#               if result == 1:
#                   self.num_empty -= 1
#                   self._addZoneInt(zone_int, x, y, static_build)
        else:
            zone_square = self.zoneSquares[zone]
        #   result = self.micro.doSimTool(x, y, zone)
            x0, y0 = max(x - 1, 0), max(y - 1, 0) 
            x1, y1 = min(x + zone_size - 1, self.MAP_X), min(y + zone_size - 1, self.MAP_Y) 
            if True:
                for i in range(x0, x1):
                    for j in range(y0, y1):
                        # stop bulldozing if we encounter a static build
                        if self.static_builds[0][i][j] == 1:
                            return
                        old_int = self.zoneMap[-1][i][j]
                        if old_int != clear_int and old_int != rubble_int:
                            self.bulldoze(i, j)
                result = self.micro.doSimTool(x, y, zone)
                if result != 1:
#                   print('unexpected (zone-square) build fail: {} at {} {} with code {}'.format(zone, x, y, result))
                    return
            if result == 1:
                # we can just call self._addZoneSquare here, right?
                self.num_empty -= (x1 - x0) * (y1 - y0)
                self.zoneMap[-1,x0 : x1, y0 : y1] = zone_square[-1, : x1 - x0, : y1 - y0]
                self.zoneMap[:self.num_features, x0 : x1, y0 : y1] = zone_square[:self.num_features, : x1 - x0, : y1 - y0]
                centers =  np.empty((x1 - x0, y1 - y0),dtype=object)
                centers.fill((x, y))  
                self.zoneCenters[x0 : x1, y0 : y1] = centers
                if static_build:
                    self.static_builds[0, x0 : x1, y0 : y1] = np.full((x1 - x0, y1 - y0), 1)

    def _addZoneInt(self, zone_int, x, y, static_build=False):
        ''' used only by bots '''
        zone_square = self.zoneSquares[self.zones[zone_int]]
        self.zoneMap[:self.num_features][x][y] = zone_square[:self.num_features][:1][:1]
        self.zoneMap[-1][x][y] = zone_square[-1][:1][:1]
        if static_build and zone_int != self.zoneInts['Clear'] and zone_int != self.zoneInts['Rubble']:
            print("BOT STATIC BUILD******")
            self.static_builds[0][x][y] = 1



    def bulldoze(self, x, y):
        assert self.static_builds[0][x][y] == 0
        clear_int = self.clear_int
        old_zone_int = self.zoneMap[-1, x, y]
        if old_zone_int == self.zoneInts['Road'] or old_zone_int == self.zoneInts['RoadWire']:
            self.micro.num_roads -= 1
        old_zone_size = self.zoneSize[self.zones[old_zone_int]]
        if clear_int == old_zone_int:
            return
#       print("bulldozing {} at ({}, {})".format(self.zones[old_zone_int], x, y))
        if  ((old_zone_size is None) or old_zone_size == 1):
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
                    self.zoneMap[:self.num_features, x0 : x1, y0 : y1] = rubble_square[:self.num_features, : x1 - x0, : y1 - y0]
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
