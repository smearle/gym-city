'''Wrappers to extend the functionality of the gym_city environment.'''
import sys
import os
import shutil
import math
import gzip
import gym
import numpy as np
import cv2
import wrappers


class ImRenderMicropolis(wrappers.ImRender):
    ''' Render micropolis as simple image.
    '''
    def __init__(self, env, log_dir, rank):
        super(ImRenderMicropolis, self).__init__(env, log_dir, rank)
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
            'Fire': 'Other',
            'Bridge': 'Transit',
            'Radar': 'Industrial',
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

        self.tile_types = tile_types
        self.type_colors = type_colors
        self.colors = colors
        if self.render_gui:
            self.win1.resize(1000, 10000)

    def step(self, action):
       #jpeg_size = self.im_render()
        # save image of map
        if True:
            if self.num_step % 10 == 0 and self.win1.editMapView.buffer is not None:
                self.win1.editMapView.buffer.write_to_png(os.path.join(self.im_log_dir, 'rank {}, episode {}, step {}.png'.format(self.rank, self.n_episode, self.num_step)))
        obs, rew, done, info = super().step(action)
        return obs, rew, done, info

    def reset_episodes(self, im_log_dir):
        if self.MAP_X == 64:
            self.win1.editMapView.changeScale(0.77)
            self.win1.editMapView.centerOnTile(40, 23)
        return super().reset_episodes(im_log_dir)

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
        if self.unwrapped.render_gui:
            cv2.imshow('im', self.image)
        if self.unwrapped.num_step % self.save_interval == 0:
            log_dir = os.path.join(self.im_log_dir, 'rank:{}_epi:{}_step:{}.jpg'.format(
                self.unwrapped.rank, self.n_episode, self.num_step))
           #print(log_dir)
            cv2.imwrite(log_dir, self.image)
            self.n_saved += 1
            size = os.stat(log_dir).st_size
            return size
