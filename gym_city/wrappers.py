'''Wrappers to extend the functionality of the gym_city environment.'''
import sys
import os
import shutil
import math
import gzip
import gym
import numpy as np
import cv2

class Extinguisher(gym.Wrapper):
    '''Trigger intermittent extinction events.'''
    def __init__(self, env,
                 extinction_type=None,
                 extinction_prob=0.1,
                 xt_dels=25
                 ):
        super(Extinguisher, self).__init__(env)
        self.set_extinction_type(extinction_type, extinction_prob, xt_dels)
       #self.n_dels = int((self.MAP_X * self.MAP_Y) * 0.004)
        print('Wrapping env in Extinguisher')

    def set_extinction_type(self, extinction_type, extinction_prob, extinction_dels):
        '''Set parameters relating to the extinction event.'''
        self.extinction_type = extinction_type
        self.extinction_prob = extinction_prob
        self.n_dels = extinction_dels
        if extinction_prob == 0:
            self.extinction_interval = -1
        else:
            self.extinction_interval = 1 / extinction_prob
        self.unwrapped.micro.map.init_age_array()

    def step(self, a):
        out = self.env.step(a)
        if self.num_step % self.extinction_interval == 0:
       #if np.random.rand() <= self.extinction_prob:
            self.extinguish(self.extinction_type)
        return out

    def extinguish(self, extinction_type='age'):
        ''' Cause some kind of extinction event to occur.'''
        curr_dels = 0
        if extinction_type == 'Monster':
            curr_dels = self.micro.engine.makeMonster()
        if extinction_type == 'age':
            curr_dels = self.elderCleanse()
        if extinction_type == 'spatial':
            curr_dels = self.localWipe()
        if extinction_type == 'random':
            curr_dels = self.ranDemolish()
        print('{} deletions'.format(curr_dels))
        return curr_dels

    def localWipe(self):
        # assume square map
        print('LOCALWIPE')
        curr_dels = 0
        x = np.random.randint(0, self.MAP_X)
        y = np.random.randint(0, self.MAP_Y)
        for r in range(0, max(x, abs(self.MAP_X - x), y, abs(self.MAP_Y - y))):
            curr_dels = self.clear_border(x, y, r, curr_dels)
            if curr_dels >= self.n_dels:
                break
        return curr_dels

    def clear_border(self, x, y, r, curr_dels):
        '''Clears the border r away (Manhattan distance) from a central point, one tile at a time.
        '''
        for x_i in range(x - r, x + r):
            if x_i < 0 or x_i >= self.MAP_X:
                continue
            for y_i in range(y - r, y + r):
                if y_i < 0 or y_i >= self.MAP_X:
                    continue
                ages = self.micro.map.age_order
                if ages[x_i, y_i] > 0:
                   #print(x_i, y_i)
                    self.micro.doBotTool(x_i, y_i, 'Clear', static_build=True)
                    curr_dels += 1
                    if curr_dels == self.n_dels:
                        return curr_dels
        return curr_dels

    def ranDemolish(self):
        # hack this to make it w/o replacement
        print('RANDEMOLISH')
        curr_dels = 0
        for i in range(self.n_dels):
            ages = self.micro.map.age_order
            ages = ages.flatten()
            age_is = np.where(ages > -1)[0]
            if len(age_is) == 0:
                break
           #age_i = np.random.choice(np.where(ages_flat > -1))
            age_i = np.random.choice(age_is)
            x, y = np.unravel_index(age_i, self.micro.map.age_order.shape)
            x, y = int(x), int(y)
            result = self.micro.doBotTool(x, y, 'Clear', static_build=True)
            curr_dels += 1
        return curr_dels

    def elderCleanse(self):
        print('\n AGEIST VIOLENCE')
       #for i in range(20):
        curr_dels = 0
       #np.set_printoptions(threshold=sys.maxsize)
        for i in range(self.n_dels):
       #for i in range((self.MAP_X * self.MAP_Y) // 90):
           #print(str(self.micro.map.age_order).replace('\n ', ' ').replace('] [', '\n'))
            ages = self.micro.map.age_order
            ages = ages.flatten()
            youngest = np.max(ages)
            age_is = np.where(ages > -1)[0]
            if len(age_is) == 0:
                break
            ages = np.copy(ages)
            ages += (ages < 0) * 2 * youngest
            age_i = np.argmin(ages)
            x, y = np.unravel_index(age_i, self.micro.map.age_order.shape)
            x, y = int(x), int(y)
           #print('deleting {} {}'.format(x, y))
           #print('zone {}'.format(self.micro.map.zones[self.micro.map.zoneMap[-1, x, y]]))
            result = self.micro.doBotTool(x, y, 'Clear', static_build=True)
           #self.render()
           #print('result {}'.format(result))
            curr_dels += 1
        # otherwise it's over!
        self.micro.engine.setFunds(self.micro.init_funds)
        return curr_dels

class ImRender(gym.Wrapper):
    ''' Render micropolis as simple image.
    '''
    def __init__(self, env, log_dir, rank):
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
        # TODO: put the extinction-type part of this log_dir in the appropriate wrapper
        im_log_dir = log_dir
       #im_log_dir = os.path.join(log_dir, 'imRender')
       #try:
       #    os.mkdir(im_log_dir)
       #except FileExistsError:
       #    pass
       #im_log_dir = os.path.join(im_log_dir, 'None')
       #try:
       #    os.mkdir(im_log_dir)
       #except FileExistsError:
       #    pass
        self.im_log_dir = im_log_dir
        # save the image at regular intervals
        self.save_interval = 10
        self.n_saved = 0
        self.n_episode = 0
        self.tile_types = tile_types
        self.type_colors = type_colors
        self.colors = colors
        self.image = np.zeros((self.MAP_X, self.MAP_Y, 3))
        self.image = np.transpose(self.image, (1, 0, 2))
        self.rank = rank
        if self.unwrapped.render_gui:
            _ = cv2.namedWindow('im', cv2.WINDOW_NORMAL)
            cv2.imshow('im', self.image)

    def step(self, action):
        self.im_render()
        obs, rew, done, info = self.env.step(action)
        info = {
                **info,
                **self.city_metrics,
                }
        return obs, rew, done, info

    def reset_episodes(self, im_log_dir):
        self.n_episode = 0
        print('reset epis, imrender log dir: {}'.format(self.im_log_dir))
        self.im_log_dir = im_log_dir
       #self.im_log_dir = self.im_log_dir.split('/')[:-1]
       #self.im_log_dir = '/'.join(self.im_log_dir)
       #self.im_log_dir = os.path.join(self.im_log_dir, str(self.env.extinction_type))
        print('reset epis, renamed imrender log dir: {}'.format(self.im_log_dir))
        try:
            os.mkdir(self.im_log_dir)
        except FileExistsError:
            pass

    def reset(self):
        self.n_episode += 1
        return self.env.reset()

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
           #cv2.imwrite(log_dir, self.image)
            self.n_saved += 1
