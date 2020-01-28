'''Wrappers to extend the functionality of the gym_city environment.'''
import os
import shutil
import gzip
import gym
import numpy as np
import cv2

class Extinguisher(gym.Wrapper):
    '''Trigger intermittent extinction events.'''
    def __init__(self, env, extinction_type=None, extinction_prob=0.1):
        super(Extinguisher, self).__init__(env)
        self.set_extinction_type(extinction_type, extinction_prob)
        print('CREATE EXTI')

    def set_extinction_type(self, extinction_type, extinction_prob):
        '''Set parameters relating to the extinction event.'''
        self.extinction_type = extinction_type
        self.extinction_prob = extinction_prob
        if extinction_prob == 0:
            self.extinction_interval = -1
        else:
            self.extinction_interval = 1 / extinction_prob
        if self.extinction_type == 'age':
            self.unwrapped.micro.map.init_age_array()

    def step(self, a, static_build=False):
        out = self.unwrapped.step(a)
        if self.num_step % self.extinction_interval == 0:
       #if np.random.rand() <= self.extinction_prob:
            self.extinguish(self.extinction_type)
        return out

    def extinguish(self, extinction_type='age'):
        ''' Cause some kind of extinction event to occur.'''
        if extinction_type == 'Monster':
            return self.micro.engine.makeMonster()
        if extinction_type == 'age':
            return self.elderCleanse()
        if extinction_type == 'spatial':
            return self.localWipe()

    def localWipe(self):
        # assume square map
        w = self.MAP_X // 3
        x = np.random.randint(0, self.MAP_X)
        y = np.random.randint(0, self.MAP_Y)
        self.micro.map.clearPatch(x, y, patch_size=w, static_build=False)

    def ranDemolish(self):
        pass
       #for i in range()

    def elderCleanse(self):
        ages = self.micro.map.age_order
        eldest = np.max(ages)
        print('\n AGEIST VIOLENCE')
        ages[ages < 0] = 2*eldest
       #for i in range(20):
        for i in range(30):
       #for i in range((self.MAP_X * self.MAP_Y) // 90):
            ages[ages < 0] = 2*eldest
            xy = np.argmin(ages)
            x = xy // self.MAP_X
            y = xy % self.MAP_X
            x = int(x)
            y = int(y)
           #print('deleting {} {}'.format(x, y))
            result = self.micro.doBotTool(x, y, 'Clear', static_build=True)
            self.render()
           #print('result {}'.format(result))
        ages -= np.min(ages)
        ages[ages>eldest] = -1
        # otherwise it's over!
        self.micro.engine.setFunds(self.micro.init_funds)


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
        self.log_dir = os.path.join(log_dir, 'imRender/None')
        try:
            os.mkdir(self.log_dir)
        except FileExistsError:
            pass

        # save the image at regular intervals
        self.save_interval = 10
        self.n_saved = 0
        self.n_episode = 0
        self.tile_types = tile_types
        self.type_colors = type_colors
        self.colors = colors
        self.image = np.zeros((self.MAP_X, self.MAP_Y, 3))
        self.image = np.transpose(self.image, (1, 0, 2))
        if self.unwrapped.render_gui and self.unwrapped.rank == 0:
            _ = cv2.namedWindow('im', cv2.WINDOW_NORMAL)
            cv2.imshow('im', self.image)

    def step(self, action):
        self.im_render()
        return super().step(action)

    def reset_episodes(self):
        self.n_episode = 0
        self.log_dir = self.log_dir.split('/')[:-1]
        self.log_dir = '/'.join(self.log_dir)
        self.log_dir = os.path.join(self.log_dir, str(self.env.extinction_type))
        try:
            os.mkdir(self.log_dir)
        except FileExistsError:
            pass

    def reset(self):
        self.n_episode += 1
        return super().reset()

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
            log_dir = os.path.join(self.log_dir, 'rank:{}_epi:{}_step:{}.jpg'.format(
                self.unwrapped.rank, self.n_episode, self.num_step))
            print(log_dir)
            cv2.imwrite(log_dir, self.image)
            self.n_saved += 1
