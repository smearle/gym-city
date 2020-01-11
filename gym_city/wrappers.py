import gym
import numpy as np
import cv2

class ImRender(gym.Wrapper):
    ''' Render micropolis as simple image.
    '''
    def __init__(self, env):
        super(ImRender, self).__init__(env)
        tile_types = {
            'Residential': 'Residential',
            'Commercial' : 'Commercial',
            'Industrial' : 'Industrial',
            'Seaport' : 'Industrial',
            'Stadium' : 'Commercial',
            'PoliceDept' : 'Commercial',
            'FireDept' : 'Commercial',
            'Airport' : 'Commercial',
            'NuclearPowerPlant' : 'Industrial',
            'CoalPowerPlant' : 'Industrial',
            'Road' : 'Infrastructure',
            'Rail' :'Infrastructure',
            'Park' : 'Nature',
            'Wire' : 'Infrastructure',
            'Rubble': 'Disaster',
            'Net': 'Infrastructure',
            'Water': 'Nature',
            'Land': 'Nature',
            'Forest': 'Nature',
            'Church': 'Residential',
            'Hospital': 'Residential',
            'Radioactive': 'Disaster',
            'Flood': 'Disaster',
            'Fire': 'Disaster'}
        type_colors = {
            'Residential': 'Green',
            'Commercial': 'Blue',
            'Industrial': 'Yellow',
            'Nature': 'Brown',
            'Infrastructure': 'Grey',
            'Disaster': 'Red',
                }
        colors = {
                'Green': [1, 0, 0],
                'Blue': [0, 1, 0],
                'Yellow': [0, 0, 1],
                'Brown': [1, 0, 1],
                'Grey': [1, 1, 0],
                'Red': [0, 1, 1],
                }

        self.tile_types = tile_types
        self.type_colors = type_colors
        self.colors = colors
        self.image = np.zeros((self.MAP_X, self.MAP_Y, 3))
        win = cv2.namedWindow('im', cv2.WINDOW_NORMAL)
        cv2.imshow('im', self.image)
        cv2.waitKey(0)

    def step(self, action):
       #if self.unwrapped.render_gui:
       #    self.im_render()
        return super().step(action)

    def im_render(self):
        zone_map = self.unwrapped.micro.map.zoneMap
        zones = self.unwrapped.micro.map.zones
        tile_types = self.tile_types
        type_colors = self.type_colors
        colors = self.colors
        for x in range(self.MAP_X):
            for y in range(self.MAP_Y):
                tile_type = tile_types[zones[zone_map[-1, x, y]]]
                color = colors[type_colors[tile_type]]
                self.image[x][y] = color
                cv2.imshow('im', self.image)
