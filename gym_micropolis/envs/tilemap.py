import numpy as np
import gym.spaces
import bisect

### Ideally, this should depend on the tilecharacter assignment in 
### src/MicropolisEngine/micropolis.h

def zoneFromInt(i):
    zone_bps = [('Land', None), ('Water', 1), ('Forest', 22), ('Rubble', 44), 
            ('Flood', 48), ('Radioactive', 52), ('Trash', 53), ('Fire', 56),
            ('Bridge', 64), ('Road', 66), ('RoadWire', 77), ('Road', 79), 
            ('Wire', 208), ('RailWire', 221), ('Trash', 223), ('Rail', 224),
            ('RoadRail', 237), ('RoadRail', 238), ('Residential', 240), 
            ('Hospital', 405), ('Church', 410), ('Commercial', 423), 
            ('Industrial', 610), ('Seaport', 693), ('Airport', 709), ('Radar', 711), 
            ('Airport', 709), ('CoalPowerPlant', 745), ('FireDept', 761), ('PoliceDept', 770), ('Stadium', 779),
            ('NuclearPowerPlant', 811), ('Lightning', 827), ('Bridge', 828), ('Radar', 832), ('Park', 840),
            ('Net', 844), ('Industrial', 852), ('Rubble', 860), ('Industrial', 888), ('CoalPowerPlant', 916), ('Stadium', 932), ('Bridge', 948), 
            ('NuclearPowerPlant', 952), ('Church', 956)]
    zones = [z[0] for z in zone_bps]
    breakpoints = [z[1] for z in zone_bps[1:]]
    z = bisect.bisect(breakpoints, i)
    return zones[z]

def zoneFromInt_A(i):
    if i == 0: return "Land"
    if i <= 21: return "Water"
    if i <= 43: return "Forest"
    if i <= 47 or 860 <= i <= 864: return "Rubble"
    if i <= 51: return "Flood"
    if i <= 52: return "Radioactive"
    if i <= 55: return "Trash"
    if i <= 63: return "Fire"
    if i <= 65: return "Bridge"
    if i <= 76: return "Road"
    if i <= 78: return "RoadWire"
    if i <= 206: return "Road"
    if 208 <= i <= 220: return "Wire"
    if i <= 222: return "RailWire"
    if i <= 223: return "Trash"
    if i <= 236: return "Rail"
    if i <= 238: return "RoadRail"
    if 240 <= i <= 404: return "Residential"
    if i <= 409: return "Hospital"
    if i <= 422 or 956 <= i <= 1018: return "Church"
    if i <= 609: return "Commercial"
    if 612 <= i <= 692 or 852 <= i <= 855 or 888 <= i <= 915: return "Industrial"
    if i <= 708: return "Seaport"
    if i == 711 or 832 <= i <= 839: return 'Radar'
    if i <= 744 : return "Airport"
    if 745 <= i <= 760 or 916 <= i <= 929 : return "CoalPowerPlant"
    if i <= 769: return "FireDept"
    if i <= 778: return "PoliceDept"
    if 779 <= i <= 810 or 932 <= i <= 948 : return "Stadium"
    if 810 <= i <= 826 or 952 <= i <= 955: return "NuclearPowerPlant"
    if 828 <= i <= 832 or 948 <= i <= 951: return "Bridge"
    if 840 <= i <= 843: return "Park"
    if 844 <= i <= 851: return "Net"
    else: 
        print("TILEMAP KEY ERROR")
        return "???"



class TileMap(object):
    ''' Map of Micropolis Zones as affected by actions of MicropolisControl object. Also automates bulldozing (always finds deletable tile of larger zones/structures and subsequently removes rubble)'''
    def __init__(self, micro, MAP_X, MAP_Y, walker=False):
        # whether or not the latest call to addZone has any effect on our map
        # no good if we encounter a fixed optimal state / are on a budget
        self.no_change = False
        self.walker = walker
        self.centers = np.full((MAP_X, MAP_Y), None)

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
                'Airport' : 5, 
                'NuclearPowerPlant' : 4, 
                'CoalPowerPlant' : 4, 
                'Road' : 1, 
                'Rail' : 1, 
                'Park' : 1, 
                'Wire' : 1, 
                'Rubble': 1,
                'Net': 1, 
                'Water': 1, 
                'Land': 1, 
                'Forest': 1,
                'Church': 3,
                'Hospital': 3,
                'Fire': 1
               }
        # if this feature, then another
        link_features = {
                'Forest' : ['Forest', 'Land'],
                'Church' : ['Church', 'Residential'],
                'Hospital': ['Hospital', 'Residential']
                }
        # a distinct feature expressed as the conjunction of existing features
        composite_zones = {
                "RoadWire": ["Road", "Wire"], 
                "RailWire": ["Rail", "Wire"],
                "Bridge": ["Road", "Water"],
                "RoadRail": ["Road", "Rail"],
                "WaterWire": ["Water", "Wire"],
                "Radar": ["Net", "Airport"],
                "RailBridge": ["Rail", "Water"]
                }
        # a dictionary of zone A to zones B which A cannot overwrite
        zone_compat = {}
        for k in composite_zones:
            l = composite_zones[k]
            for c in l:
                if c in zone_compat:
                    zone_compat[c] += [zone for zone in l if zone not in zone_compat[c]]
                else:
                    zone_compat[c] = l
        self.zone_compat = zone_compat
        print(zone_compat)
        self.zones = list(self.zoneSize.keys()) + list(composite_zones.keys())
        for c in composite_zones.keys():
            self.zoneSize[c] = self.zoneSize[composite_zones[c][0]]
        self.num_zones = len(self.zones)
        self.num_features = self.num_zones - len(composite_zones)
        self.zoneInts = {}
        for z in range(self.num_zones):
            self.zoneInts[self.zones[z]] = z
        self.clear_int = self.zoneInts['Land']


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


        square_sizes = [1, 3,4,5,6]
        self.zoneSquares = {}
        for z in self.zones:
            z_size = self.zoneSize[z] 
            if z == 'Rubble':
                # make different-size rubble squares
                for s in square_sizes:
                    z0 = 'Rubble_' + str(s)
                    self.zoneSquares[z0] = makeZoneSquare(s, self.zoneInts[z])
                self.zoneSquares["Rubble"] = self.zoneSquares["Rubble_1"]
            else:
                zone_int = self.zoneInts[z]
                if z in composite_zones.keys():
                    feature_ints = [self.zoneInts[z1] for z1 in composite_zones[z]]             
                elif z in link_features.keys():
                    feature_ints = [self.zoneInts[z1] for z1 in link_features[z]]
                else:
                    feature_ints = None
                self.zoneSquares[z] = makeZoneSquare(self.zoneSize[z], zone_int, feature_ints)
        # first dimension is binary feature-vector - followed by zone int
        self.zoneMap = np.zeros((self.num_features + 1,self.MAP_X, self.MAP_Y), dtype=np.uint8)
        self.static_builds = None
        self.setEmpty()
        self.micro = micro
        self.zoneCols = {}
        for zone in self.zones:
            self.zoneCols[zone] = self.zoneSquares[zone][:, 0:1, 0:1]


    def initStaticBuilds(self):
        self.static_builds = np.zeros((1, self.MAP_X, self.MAP_Y), dtype=int)


    def setEmpty(self):
        if self.static_builds is not None:
            self.static_builds.fill(0)
        self.centers.fill(None)
        self.zoneMap.fill(0)
        self.zoneMap[-1,:,:] = self.zoneInts['Land']
        self.zoneMap[self.zoneInts['Land'],:,:] = 1
        self.num_empty = self.MAP_X * self.MAP_Y


    def addZonePlayer(self, x, y, tool, static_build=True):
        return self.addZoneBot(x, y, tool, static_build=static_build)


    def addZoneBot(self, x, y, tool, static_build=False):
       #print("BUILD: ", tool, x, y)
        if not static_build and self.static_builds[0][x][y] == 1:
            return
        zone = tool
        if zone == 'Clear': 
            zone = 'Land'
        self.clearPatch(x, y, zone, static_build=static_build)
        result = self.micro.doSimTool(x, y, tool)
        if result == 1:
            self.addZone(x, y, static_build)


    def clearPatch(self, x, y, zone, static_build=False):
        old_zone = self.zones[self.zoneMap[-1][x][y]]
        if zone in self.zone_compat and old_zone in self.zone_compat[zone]:
            return # do not prevent composite build
        zone_size = self.zoneSize[zone]
        if zone_size == 1:
            self.clearTile(x, y, static_build=static_build)
            return
        else:
            x0, x1 = max(x-1, 0), min(x-1+zone_size, self.MAP_X)
            y0, y1 = max(y-1, 0), min(y-1+zone_size, self.MAP_Y)
            for i in range(x0, x1):
                for j in range(y0, y1):
                    if static_build == False and self.static_builds[0][i][j] == 1:
                        return
                    self.clearTile(i, j, static_build=static_build)


    def clearTile(self, x, y, static_build=False):
        ''' This ultimately calls itself until the tile is clear'''
        old_zone = self.zoneMap[-1][x][y]
        if old_zone in ['Land']:
            return
        cnt = self.centers[x][y]
        if cnt[0] >= self.MAP_X or cnt[1] >= self.MAP_Y:
           return
        result = self.micro.doSimTool(cnt[0], cnt[1], 'Clear')
        #assert self.static_builds[0][x][y] == self.static_builds[0][cnt[0]][cnt[1]]
        #assert self.centers[x][y] == self.centers[cnt[0]][cnt[1]]
        if self.static_builds[0][x][y] == 1 and static_build == False:
            return
        self.removeZone(cnt[0], cnt[1])


    def removeZone(self, x, y):
       #print("CLEAR ", x, y)
        old_zone = self.zones[self.zoneMap[-1][x][y]]
        size = self.zoneSize[old_zone]
        if size == 1:
            self.updateTile(x, y, static_build=False)
            return
        if size == 5:
            x -= 1
            y -= 1
        x0, y0 = max(0, x-1), max(0, y-1)
        x1, y1 = min(x-1+size, self.MAP_X,), min(y-1+size, self.MAP_Y)
        for i in range(x0, x1):
            for j in range(y0, y1):
                self.updateTile(i, j, static_build = False)


    def addZone(self, x, y, static_build=False):
        ''' Assume the bot has succeeded in its build (so we can lay the centers) '''
        tile_int = self.micro.getTile(x, y)
        zone = zoneFromInt(tile_int)
        zone_size = self.zoneSize[zone]
        if zone_size == 5:
            center = (x+1, y+1)
        else:
            center = (x, y)
        if zone_size == 1:
            self.updateTile(x, y, zone, center, static_build)
            return
        else:
            x0, y0 = max(0, x - 1), max(0, y - 1)
            x1, y1 = min(x - 1 + zone_size, self.MAP_X), min( y - 1 + zone_size, self.MAP_Y)
            for i in range(x0, x1):
                for j in range(y0, y1):
                    self.updateTile(i, j, zone, center, static_build)

    def updateTile(self, x, y, zone=None, center=None, static_build=None):
        ''' static_build should be None when simply updating from map,
        True when building, and False when deleting '''
        if zone is None:
            tile_int = self.micro.getTile(x, y)
            zone = zoneFromInt(tile_int)
            if self.zoneSize[zone] != 1:
                center = self.centers[x][y]
            else:
                center = (x, y)
            if static_build == False:
                self.static_builds[0][x][y] = 0
      #print(zone, self.micro.getTile(x, y), x, y)
        else: # then we are adding or deleting a zone
           #assert zone == zoneFromInt(self.micro.getTile(x, y))
            if static_build and zone not in ['Land', 'Rubble', 'Water']:
                self.static_builds[0][x][y] = 1
            else:
                self.static_builds[0][x][y] = 0
        zone_col = self.zoneCols[zone]
        self.zoneMap[:, x:x+1, y:y+1] = zone_col
        self.centers[x][y] = center


    


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
