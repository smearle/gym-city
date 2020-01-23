import os
import shutil
import gzip
import gym
import numpy as np
import cv2

class ImRender(gym.Wrapper):
    ''' Render micropolis as simple image.
    '''
    def __init__(self, env, log_dir):
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
            'NuclearPowerPlant' : 'Power',
            'CoalPowerPlant' : 'Power',
            'Road' : 'Transit',
            'Rail' :'Transit',
            'RoadWire': 'Transit',
            'RoadRail': 'Transit',
            'RailWire': 'Transit',
            'Park' : 'Other',
            'Wire' : 'Power',
            'Rubble': 'Other',
            'Net': 'Power',
            'Water': 'Other',
            'Land': 'Other',
            'Forest': 'Other',
            'Church': 'Residential',
            'Hospital': 'Residential',
            'Radioactive': 'Other',
            'Flood': 'Other',
            'Fire': 'Other'
            }
        type_colors = {
            'Residential': 'Green',
            'Commercial': 'Blue',
            'Industrial': 'Yellow',
            'Transit': 'Red',
            'Power': 'Magenta',
            'Other': 'Cyan',
            }
        colors = {
                # [blue, green, red]
                'Green': [0, 1, 0],
                'Blue': [1, 0, 0],
                'Yellow': [0, 1, 1],
                'Red': [0, 0, 1],
                'Magenta': [1, 0, 1],
                'Cyan': [1, 1, 0],
                }
        self.log_dir = os.path.join(log_dir, 'im_render')
        try:
            os.mkdir(self.log_dir)
        except FileExistsError:
            shutil.rmtree(self.log_dir)
        # save the image at regular intervals
        self.save_interval = 10
        self.n_saved = 0
        self.tile_types = tile_types
        self.type_colors = type_colors
        self.colors = colors
        self.image = np.zeros((self.MAP_X, self.MAP_Y, 3))
        self.image = np.transpose(self.image, (1, 0, 2))
        if self.unwrapped.render_gui and self.unwrapped.rank == 0:
            win = cv2.namedWindow('im', cv2.WINDOW_NORMAL)
            cv2.imshow('im', self.image)

    def step(self, action):
        self.im_render()
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
        self.image = np.transpose(self.image, (1, 0, 2))
        self.image = self.image * 255
        if self.unwrapped.render_gui and self.unwrapped.rank == 0:
            cv2.imshow('im', self.image)
        if self.unwrapped.num_step % self.save_interval == 0:
            log_dir = os.path.join(self.log_dir, '{}.jpg'.format(self.n_saved))
            print(log_dir)
            cv2.imwrite(log_dir, self.image)
            self.n_saved += 1
