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
    def __init__(self, micro, MAP_X, MAP_Y, walker=False, paint=True):
        self.n_struct_tiles = 0
        self.track_ages = False
        # whether or not the latest call to addZone has any effect on our map
        # no good if we encounter a fixed optimal state / are on a budget
        self.no_change = False
        self.walker = walker
        self.paint = paint
        self.centers = np.full((MAP_X, MAP_Y), None)

        self.MAP_X = MAP_X
        self.MAP_Y = MAP_Y
        self.num_empty = self.MAP_X * self.MAP_Y
        if self.paint:
            self.acted = np.zeros((self.MAP_X, self.MAP_Y))
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
                'Rubble': 1,
                'Net': 1,
                'Water': 1,
                'Land': 1,
                'Forest': 1,
                'Church': 3,
                'Hospital': 3,
               #'Radioactive': 1,
               #'Flood': 1,
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
        self.composite_zones = composite_zones
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
#       print(zone_compat)
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
        self.zoneMap = np.zeros((self.num_features + 1, self.MAP_X, self.MAP_Y), dtype=np.uint8)
        self.static_builds = np.zeros((1, self.MAP_X, self.MAP_Y), dtype=int)
        self.road_int = self.zoneInts['Road']
        self.road_networks = np.zeros((1, self.MAP_X, self.MAP_Y), dtype=int)
        self.setEmpty()
        self.micro = micro
        self.zoneCols = {}
        for zone in self.zones:
            self.zoneCols[zone] = self.zoneSquares[zone][:, 0:1, 0:1]
        self.road_labels = list(range(1, int(self.MAP_X * self.MAP_Y / 2) + 1)) # for potential one-hot encoding, for ex.,
        self.num_road_nets = 0
        self.road_net_sizes = {} # using unique road net numbers as keys
        self.num_roads = 0
        self.num_plants = 0
        self.priority_road_net = None
        # number of surrounding roads for road-tiles, updated on road bld/del
       #self.road_crowding_map = np.zeros(0, self.MAP_X, self.MAP_Y)
        self.numPlants = 0 # power plants

    def init_age_array(self):
        self.track_ages = True
        # track age of build structures
        print('initializing AGES')
        self.age_order = np.zeros((self.MAP_X, self.MAP_Y), dtype=int)
        self.age_order.fill(-1)

    def didRoadBuild(self, x, y):
       #assert self.road_networks[0, x, y] == 0
        x0, y0, x1, y1 = max(0, x-1), max(0, y-1), \
            min(x+1, self.MAP_X-1), min(y+1, self.MAP_Y-1) # get adj. coords
        self.road_networks[0, x, y] = 0
        for xi, yi in [(x0, y), (x1, y), (x, y0), (x, y1)]:
            if xi == x and yi == y:
                pass
            else:
                if self.zoneMap[self.road_int][xi][yi] == 1:
                    net_i = self.road_networks[0, xi, yi]
                   #assert net_i != 0
                    if net_i == 0:
                        print(self.road_networks, x, y, self.road_net_sizes)
                        print('road label is on map, but not in size dict')
                        raise Exception
                    elif self.road_networks[0, x, y] == 0: # set the build road to match a connecting piece
                        self.road_networks[0, x, y] = net_i
                        if net_i in self.road_net_sizes:
                            self.road_net_sizes[net_i] += 1
                        else:
                            print(self.road_networks, x, y, old_net, 'road label is on map, but not in size dict')
                            raise Exception

                    elif self.road_networks[0, x, y] == net_i: # not a conflict
                        pass
                    else: # resolve conflict, assimilate net
                        self.road_labels.append(net_i)
                        self.road_net_sizes.pop(net_i)
                        self.num_road_nets -= 1
                        self.setRoadNet(xi, yi, self.road_networks[0, x, y]) # if we recurse back to (x, y),
                        # setRoadNet will recognize it as friendly and skip

        net_n = self.road_networks[0, x, y]
        if net_n == 0: # this is a new net
            net_n = self.road_labels.pop()
            self.road_networks[0, x, y] = net_n
            self.num_road_nets += 1
            self.road_net_sizes[net_n] = 1
           #assert self.road_net_sizes[net_n] == 1
        else:
           #assert self.road_net_sizes[net_n] > 1
           pass


    def setRoadNet(self, x, y, net_n):
        # switch index of entire road net
        old_net = self.road_networks[0, x, y]
        self.road_networks[0, x, y] = net_n
        self.road_net_sizes[net_n] += 1
        x0, y0, x1, y1 = max(0, x-1), max(0, y-1), min(x+1, self.MAP_X-1), min(y+1, self.MAP_Y-1) # get adj. coords
        for xi, yi in [(x0, y), (x1, y), (x, y0), (x, y1)]:
            if xi == x and yi == y:
                pass
            elif self.zoneMap[self.road_int][xi][yi] == 1:
                net_i = self.road_networks[0, xi, yi]
                if net_i == 0: # in the process of deleting, we have recursed bacck to the piece being deleted
                    pass
                elif net_i == net_n:
                   #assert net_i == net_n
                   pass
                elif net_i == old_net:
                    self.setRoadNet(xi, yi, net_n)
                else:
                    print("found unamissilated net piece while setting it to a new value", self.road_networks, x, y, old_net, net_n, net_i)
                    raise Exception



    def didRoadDelete(self, x, y):
        old_net = self.road_networks[0, x, y]
       #assert self.road_net_sizes[old_net] > 0
        self.road_networks[0, x, y] = 0
        if old_net in self.road_net_sizes:
            self.road_net_sizes[old_net] -= 1
        else:
            print(self.road_networks, self.road_net_sizes, x, y, old_net, "road label not in size dict")
        x0, y0, x1, y1 = max(0, x-1), max(0, y-1), min(x+1, self.MAP_X-1), min(y+1, self.MAP_Y-1) # get adj. coords
        cnx = 0
        cnx_0 = None
        for xi, yi in [(x0, y), (x1, y), (x, y0), (x, y1)]:
            if xi == x and yi == y:
                pass
            elif self.zoneMap[self.road_int][xi][yi] == 1:
               #assert self.road_networks[0, xi, yi] != 0
                cnx += 1
                if self.road_networks[0, xi, yi] != old_net:
                   #assert cnx > 1
                    pass # we have flipped this cnxn while visiting another
                else:
                    # flip all the connected subnets until none belongs to
                    # the former network
                    new_net = self.road_labels.pop()
                    self.num_road_nets += 1
                    self.road_net_sizes[new_net] = 0
                    self.setRoadNet(xi, yi, new_net)
        self.road_labels.append(old_net)
        self.road_net_sizes.pop(old_net)
        self.num_road_nets -= 1
       #try:
       #    assert self.road_net_sizes[old_net] == 0
       #except AssertionError:
       #    print(self.road_networks, x, y, old_net, self.road_net_sizes)
       #    raise Exception("trying to pop a non-zero size net from the size dict")
        if self.priority_road_net == old_net:
            if self.num_roads > 0:
                self.priority_road_net = next(iter(self.road_net_sizes))
            else:
                self.priority_road_net = None

    def clearBotBuilds(self):
        for i in range(self.MAP_X):
            for j in range(self.MAP_Y):
                if self.static_builds[0, i, j] == 0:
                    self.addZoneBot(i, j, 'Land',  static_build=False)

    def setEmpty(self):
        self.num_roads = 0
        self.num_plants = 0
        self.static_builds.fill(0)
        self.road_networks.fill(0)
        self.road_labels = list(range(1, int(self.MAP_X * self.MAP_Y / 2) + 1))
        self.road_net_sizes = {}
        self.priotiry_net_size = 0
        self.priority_road_net = None
        self.num_road_nets = 0
        self.centers.fill(None)
        self.zoneMap.fill(0)
        self.zoneMap[-1,:,:] = self.zoneInts['Land']
        self.zoneMap[self.zoneInts['Land'],:,:] = 1
        self.num_empty = self.MAP_X * self.MAP_Y


    def addZonePlayer(self, x, y, tool, static_build=True):
        return self.addZoneBot(x, y, tool, static_build=static_build)


    def addZoneBot(self, x, y, tool, static_build=False):
       #print("ADDZONEBOT: ", tool, x, y)
        if tool == 'Nil':
            return True
        zone = tool
        old_zone = self.zones[self.zoneMap[-1][x][y]]
        if  (zone in self.zone_compat and (old_zone in self.zone_compat[zone] or (old_zone in self.composite_zones and zone in self.composite_zones[old_zone]))) and zone != 'Water':
            if self.static_builds[0][x][y] == 1:
                static_build = True
        if (not static_build) and self.static_builds[0][x][y] == 1:
            return False

        if zone == 'Clear':
            zone = 'Land'

        result = self.clearForZone(x, y, zone, static_build=static_build)
        if result == 1:
            # call the engine
            result = self.micro.doSimTool(x, y, tool)
            if result == 1:
                result = self.addZone(x, y, zone, static_build)
            else:
                pass
                #TODO: Why is result == 0 here, even when build is successful?
               #print('failed doSimTool w/ result {}'.format(result))
        else:
           #print('failed to clear patch')
            pass
        if result == 1:
            return True
        else:
           #print('failed to add zone {} {} {}'.format(x, y, zone))
            return False




    def clearForZone(self, x, y, zone, static_build=False):
       #print('clearing patch {} {} {}'.format(x, y, zone))
        old_zone = self.zones[self.zoneMap[-1][x][y]]
        if zone in self.zone_compat and (old_zone in self.zone_compat[zone]):
            return 1 # do not prevent composite build
        zone_size = self.zoneSize[zone]
        return self.clearPatch(x, y, patch_size=zone_size, static_build=static_build)

    def clearPatch(self, x, y, patch_size, static_build=False):
        if patch_size == 1 and not ((static_build == False) and self.static_builds[0][x][y] == 1):
            self.clearTile(x, y, static_build=static_build)
            return 1
        else:
            x0, x1 = max(x-1, 0), min(x-1+patch_size, self.MAP_X)
            y0, y1 = max(y-1, 0), min(y-1+patch_size, self.MAP_Y)
            for i in range(x0, x1):
                for j in range(y0, y1):
                    if self.static_builds[0][i][j] == 1:
                       #print('static build encountered - clear failed')
                        return 0
                    if self.paint and self.acted[i][j] == 1:
                       #print('not overwriting build from same turn - clear failed')
                        return 0
                    else:
                        self.clearTile(i, j, static_build=static_build)
            return 1


    def clearTile(self, x, y, static_build=False):
        ''' This ultimately calls itself until the tile is clear'''
       #print('clearing tile {} {}'.format(x, y))
        old_zone = self.zones[self.zoneMap[-1][x][y]]
       #if old_zone in ['Land']:
       #    return
       #if old_zone in ['Water']:
       #    print('Deleting water {} {}'.format(x, y))
        cnt = self.centers[x][y]
        if cnt is None:
            cnt = (x, y)
       #if cnt[0] >= self.MAP_X or cnt[1] >= self.MAP_Y:
       #    return
        if self.static_builds[0][x][y] == 1 and static_build == False:
            result = - 1
            return result

        xc, yc = cnt[0], cnt[1]
        result = self.micro.doSimTool(xc, yc, 'Water')
        result = self.micro.doSimTool(xc, yc, 'Land')
        result = self.micro.doSimTool(xc, yc, 'Clear')
        #assert self.static_builds[0][x][y] == self.static_builds[0][cnt[0]][cnt[1]]
        #assert self.centers[x][y] == self.centers[cnt[0]][cnt[1]]


       #print("CLEAR ", x, y)
        size = self.zoneSize[old_zone]
        if size == 1:
            self.updateTile(xc, yc, static_build=False)
            return
        if size == 6:
            xc -= 1
            yc -= 1
        x0, y0 = max(0, xc-1), max(0, yc-1)
        x1, y1 = min(xc-1+size, self.MAP_X,), min(yc-1+size, self.MAP_Y)
        for i in range(x0, x1):
            for j in range(y0, y1):
                self.updateTile(i, j, static_build = False)


    def addZone(self, x, y, zone, static_build=False):
        ''' Assume the bot has already succeeded in its build (so we can lay the centers) '''
        trg_zone = zone
        tile_int = self.micro.getTile(x, y)
        map_zone = zoneFromInt(tile_int)
       #if map_zone != trg_zone and not (map_zone in self.zone_compat and trg_zone in self.zone_compat[map_zone]):
       #    static_build = False
        zone = map_zone
        zone_size = self.zoneSize[zone]
        if zone_size == 6:
            center = (x + 1, y + 1)
        else:
            center = (x, y)
        if zone_size == 1:
            self.updateTile(x, y, zone, center, static_build)
            if self.paint:
                self.acted[x, y] = 1
            if self.track_ages:
                self.age_order[x][y] = self.micro.env.num_step
            return
        else:
            x0, y0 = max(0, x - 1), max(0, y - 1)
            x1, y1 = min(x - 1 + zone_size, self.MAP_X), min( y - 1 + zone_size, self.MAP_Y)
            for i in range(x0, x1):
                for j in range(y0, y1):
                    self.updateTile(i, j, zone, center, static_build)
                    if self.paint:
                        self.acted[i, j] = 1
                    if self.track_ages:
                        self.age_order[i][j] = self.micro.env.num_step

    def updateTile(self, x, y, zone=None, center=None, static_build=None):
        ''' static_build should be None when simply updating from map,
        True when building, and False when deleting '''
        was_plant = (self.zoneMap[self.zoneInts['NuclearPowerPlant']][x][y] == 1) or (self.zoneMap[self.zoneInts['CoalPowerPlant']][x][y] == 1)
        was_road = self.zoneMap[self.road_int][x][y] == 1
        #TODO: get_natural function
        was_natural = self.zoneMap[self.zoneInts['Land']][x][y] == 1 or \
                     self.zoneMap[self.zoneInts['Water']][x][y] == 1 or \
                     self.zoneMap[self.zoneInts['Rubble']][x][y] == 1 or \
                     self.zoneMap[self.zoneInts['Forest']][x][y] == 1
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
        is_road = self.zoneMap[self.road_int][x][y] == 1
        is_plant = (self.zoneMap[self.zoneInts['NuclearPowerPlant']][x][y] == 1) or (self.zoneMap[self.zoneInts['CoalPowerPlant']][x][y] == 1)

        is_natural = self.zoneMap[self.zoneInts['Land']][x][y] == 1 or \
                     self.zoneMap[self.zoneInts['Water']][x][y] == 1 or \
                     self.zoneMap[self.zoneInts['Rubble']][x][y] == 1 or \
                     self.zoneMap[self.zoneInts['Forest']][x][y] == 1
        if was_natural and not is_natural:
            self.n_struct_tiles += 1
        if not was_natural and is_natural:
            self.n_struct_tiles -= 1
        if self.track_ages:
            if is_natural:
                self.age_order[x][y] = -1



        net = None
        if was_road and not is_road:
            self.num_roads -= 1
            self.didRoadDelete(x, y)
        elif not was_road and is_road:
            self.didRoadBuild(x, y)
            self.num_roads += 1
            if self.priority_road_net is None:
                self.priority_road_net = next(iter(self.road_net_sizes))

        if was_plant and not is_plant:
            assert self.num_plants > 0
            self.num_plants -= 1
        elif not was_plant and is_plant:
            self.num_plants += 1






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
