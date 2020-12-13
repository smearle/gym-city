import random
import copy
import os

import cv2
import gym
import numpy as np
import torch
#from opensimplex import OpenSimplex
import pygame

class Griddly(gym.Wrapper):
    def __init__(self, env):
        super(Griddly, self).__init__(env)
        
    def step(self, a):
        self.render()
        ret = super().step(a)
        return ret

#class NoiseyTargets(gym.Wrapper):
#    '''A bunch of simplex noise instances modulate target metrics.'''
#    def __init__(self, env):
#        super(NoiseyTargets, self).__init__(env)
#        self.param_bounds = self.env.param_bounds
#        self.num_params = self.env.num_params
#        self.noise = OpenSimplex()
#        # Do not reset n_step
#        self.n_step = 0
#    
#    def step(self, a):
#        param_bounds = self.param_bounds
#        trgs = {} 
#        i = 0
#        for k in self.env.usable_metrics:
#            trgs[k] = self.noise.noise2d(x=self.n_step/400, y=i*100)
#            i += 1
#
#        i = 0
#        for k in self.env.usable_metrics:
#            (ub, lb) = param_bounds[k]
#            trgs[k] = ((trgs[k] + 1) / 2 * (ub - lb)) + lb
#            i += 1
#        self.env.set_trgs(trgs)
#        out = self.env.step(a)
#        self.n_step += 1
#        return out
#
#    def reset(self):
#        self.noise = OpenSimplex()
#        return self.env.reset()
        

class Extinguisher(gym.Wrapper):
    '''Trigger intermittent extinction events.'''
    def __init__(self, env,
                 extinction_type=None,
                 extinction_prob=0.1,
                 xt_dels=15
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
        self.init_ages()

    def init_ages(self):
        self.ages = self.unwrapped.micro.map.init_age_array()

    def step(self, a):
        if self.unwrapped.num_step % self.extinction_interval == 0:
       #if np.random.rand() <= self.extinction_prob:
            self.extinguish(self.extinction_type)
       #if self.num_step % 1000 == 0:
           #self.ages = self.ages - np.min(self.ages)
        out = self.env.step(a)

        return out

    def extinguish(self, extinction_type='age'):
        ''' Cause some kind of extinction event to occur.'''
        curr_dels = 0
        if extinction_type == 'Monster':
            curr_dels = self.micro.engine.makeMonster()
        if extinction_type == 'age':
            curr_dels = self.age_event()
        if extinction_type == 'spatial':
            curr_dels = self.localWipe()
        if extinction_type == 'random':
            curr_dels = self.ranDemolish()
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
        ages = self.ages
        for x_i in range(x - r, x + r):
            if x_i < 0 or x_i >= self.MAP_X:
                continue
            for y_i in range(y - r, y + r):
                if y_i < 0 or y_i >= self.MAP_X:
                    continue
                if ages[x_i, y_i] > 0:
                   #self.micro.doBotTool(x_i, y_i, 'Clear', static_build=True)
                    self.delete(x_i, y_i)
                    curr_dels += 1
                    if curr_dels == self.n_dels:
                        return curr_dels
        return curr_dels

    def ranDemolish(self):
        # hack this to make it w/o replacement
       #print('RANDEMOLISH, step {}'.format(self.unwrapped.num_step))
        ages = self.ages
        curr_dels = 0
        for i in range(self.n_dels):
            ages = ages.flatten()
            age_is = np.where(ages > -1)[0]
            if len(age_is) == 0:
                break
           #age_i = np.random.choice(np.where(ages_flat > -1))
            age_i = np.random.choice(age_is)
            x, y = np.unravel_index(age_i, self.unwrapped.micro.map.age_order.shape)
            x, y = int(x), int(y)
           #result = self.micro.doBotTool(x, y, 'Clear', static_build=True)
            self.delete(x, y)
            curr_dels += 1
        return curr_dels

    def age_event(self):
#       print('\n AGEIST VIOLENCE')
       #for i in range(20):
        curr_dels = 0
       #np.set_printoptions(threshold=sys.maxsize)
        ages_arr = self.ages#.cpu().numpy()
        for i in range(self.n_dels):
       #for i in range((self.MAP_X * self.MAP_Y) // 90):
           #print(str(self.micro.map.age_order).replace('\n ', ' ').replace('] [', '\n'))
            ages = ages_arr.flatten()
            youngest = np.max(ages)
            age_is = np.where(ages > -1)[0]
            if len(age_is) == 0:
                break
            ages = np.copy(ages)
            ages += (ages < 0) * 2 * youngest
            age_i = np.argmin(ages)
           #x, y = np.unravel_index(age_i, self.micro.map.age_order.shape)
            x, y = np.unravel_index(age_i, ages_arr.shape)
            x, y = int(x), int(y)
           #print('deleting {} {}'.format(x, y))
           #print('zone {}'.format(self.micro.map.zones[self.micro.map.zoneMap[-1, x, y]]))
            self.delete(x, y)
           #self.render()
           #print('result {}'.format(result))
            curr_dels += 1
        # otherwise it's over!
       #self.micro.engine.setFunds(self.micro.init_funds)
        return curr_dels

    def delete(self, x, y):
        result = self.unwrapped.micro.doBotTool(x, y, 'Clear', static_build=True)
        return result

class ExtinguisherMulti(Extinguisher):
    ''' Handle extinction events, for environments in which 1 env contains many (i.e. GoLMulti).
    '''
    def __init__(self, env, xt_type, xt_probs, xt_dels=15):
        super(ExtinguisherMulti, self).__init__(env, xt_type, xt_probs, xt_dels)
        dels = torch.zeros((self.num_proc, 1, self.map_width, self.map_width), dtype=torch.int8)
        self.dels = dels.to(self.device)
        n_curr_dels = torch.zeros((self.num_proc), dtype=torch.int8)
        self.n_curr_dels = n_curr_dels.to(self.device)
        self.MAP_X = self.MAP_Y = self.map_width

    def set_extinction_type(self, xt_type, xt_prob, xt_dels):
        super().set_extinction_type(xt_type, xt_prob, xt_dels)
        self.n_dels = torch.zeros((self.num_proc), dtype=torch.int8).fill_(xt_dels)
        self.n_dels = self.n_dels.to(self.device)

    def configure(self, map_width, **kwargs):
        self.env.configure(map_width, **kwargs)
        self.MAP_X = self.MAP_Y = self.map_width = map_width
        dels = torch.zeros((self.num_proc, 1, map_width, map_width), dtype=torch.int8)
        self.dels = dels.to(self.device)
        self.init_ages()

    def init_ages(self):
        self.unwrapped.init_ages()

    def localWipe(self):
        # assume square map
        print('LOCALWIPE')
        self.n_curr_dels = self.n_curr_dels.fill_(0)
        x = np.random.randint(0, self.map_width)
        y = np.random.randint(0, self.map_width)

        for r in range(0, self.map_width):
            self.clear_border(x, y, r)
            if (self.n_curr_dels >= self.n_dels).all():
                break
        self.do_dels()
       #print('n_dels: {}'.format(self.n_dels))

    def del_at_tile(self, x, y):
        self.dels[(self.n_curr_dels < self.n_dels), 0, x, y] = 1
        self.dels = self.dels * self.world.state
        self.n_curr_dels = torch.sum(self.dels, dim=[1, 2, 3])
        if (self.n_curr_dels >= self.n_dels).all() or (self.metrics['pop'] == 0).all():
            return False
        return True

    def do_dels(self):
        self.act_tensor(self.dels)
        n_new_dels = torch.sum(self.dels, dim=[1, 2, 3])
       #print('deleted: {}'.format(n_new_dels))
       #print('n_curr_dels: {}'.format(self.n_curr_dels))
       #self.n_curr_dels += n_new_dels.int()
       #self.n_curr_dels += torch.sum(self.dels, dim=[1, 2, 3])
        self.dels = self.dels.fill_(0)
        cv2.waitKey(1)

    def clear_border(self, x, y, r):
        '''Clears the border r away (Manhattan distance) from a central point, one tile at a time.
        '''
       #self.n_curr_dels = n_curr_dels
        for x_i in (x - r, x + r):
            if x_i < 0 or x_i >= self.map_width:
                continue

            for y_i in range(y - r, y + r):
                if y_i < 0 or y_i >= self.map_width:
                    continue
                if not self.del_at_tile(x_i, y_i):
                    return

        for y_i in (y - r, y + r):
            if y_i < 0 or y_i >= self.map_width:
                continue

            for x_i in range(x - r, x + r):
                if x_i < 0 or x_i >= self.map_width:
                    continue
                if not self.del_at_tile(x_i, y_i):
                    return

    def ranDemolish(self):
        self.n_curr_dels = self.n_curr_dels.fill_(0)
        print('RANDEMOLISH, step {}'.format(self.num_step))
       #ages = self.unwrapped.ages.cpu().numpy()
        start_pop = torch.sum(self.world.state)
        n_dels = self.n_dels[0]
        age_dist = (self.ages > 0) * 10000000 + 0.0001
        samples = torch.multinomial(age_dist.view(self.ages.size(0), -1), n_dels)
        proc_ixs = torch.zeros(samples.shape, dtype=int)
        proc_ixs[torch.arange(0, samples.size(0), 1), :] = torch.arange(0, samples.size(0)).unsqueeze(-1).expand(samples.shape)
        xs = samples // self.ages.size(-1)
        ys = samples % self.ages.size(-1)
        self.dels[proc_ixs.flatten(), 0, xs.flatten(), ys.flatten()] = 1
        self.dels = self.dels * self.world.state
        self.do_dels()
        end_pop = torch.sum(self.world.state)
        print('pop diff {}'.format(start_pop - end_pop))
            # sample an index for each process in the batch

        #   del_i = [np.random.choice(len(build_ix[j])) for j in range(self.num_proc)]
        #   del_coords = [build_ix[j][del_i[j]] for j in range(self.num_proc)]
        #  #result = self.micro.doBotTool(x, y, 'Clear', static_build=True)
        #   self.delete(del_coords)
        #   curr_dels += 1
        #   if (self.metrics['pop'] == 0).all():
        #       break

        # return curr_dels

    def delete(self, del_coords):
        dels = self.dels.fill_(0)
        for i, del_xy in enumerate(del_coords):
            dels[i, 0][tuple(del_xy)] = 1
        return self.act_tensor(dels)


    def age_event(self):
        self.n_curr_dels = self.n_curr_dels.fill_(0)
       #print('\n AGEIST VIOLENCE')
       #for i in range(20):
        curr_dels = 0
       #np.set_printoptions(threshold=sys.maxsize)
       #ages_arr = self.ages.cpu().numpy()
        for i in range(self.n_dels[0]):
            ages = self.unwrapped.ages.clone()
            youngest = ages.max()
           #builds = [torch.where(ages[i, 0] > 0) for i in range(self.num_proc)]
           #if len(builds) == 0:
           #    break
           #ages += (ages < 0) * 2 * youngest
           #print(ages.shape)
            ages = ages.view(self.num_proc, -1)
           #print(ages.shape)
            age_i = torch.argmax(ages, dim=1)
           #print(age_i)
           #x, y = np.unravel_index(age_i, self.micro.map.age_order.shape)
            age_i = age_i.unsqueeze(-1)
            x = age_i // self.MAP_Y
            y = age_i % self.MAP_X
            xy = torch.cat((x, y), dim=1)
           #print('deleting {} {}'.format(x, y))
           #print('zone {}'.format(self.micro.map.zones[self.micro.map.zoneMap[-1, x, y]]))
           #result = self.micro.doBotTool(x, y, 'Clear', static_build=True)
            self.delete(xy)
           #self.render()
           #print('result {}'.format(result))
            curr_dels += 1
            if (self.metrics['pop'] == 0).all():
                break
        # otherwise it's over!
       #self.micro.engine.setFunds(self.micro.init_funds)
        return curr_dels

    def step(self, action):
        if self.ages is not None and self.num_step % 1000 == 0:
            self.ages = self.ages - torch.min(self.ages) + 1
        # (reset age to zero in case of new deletions) and increment age of active cells

        self.ages = (self.ages * self.world.state) + self.world.state
        result = super().step(action)
        return result

    def reset(self):
        self.ages = copy.deepcopy(self.world.state.fill_(0))
        self.n_curr_dels = self.n_curr_dels.fill_(0)
        obs = super().reset()

        return obs


class ImRender(gym.Wrapper):
    ''' Render micropolis as simple image.
    '''
    def __init__(self, env, log_dir, rank):
        super(ImRender, self).__init__(env)
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
        self.save_interval = 1
        self.n_saved = 0
        self.n_episode = 0
        self.image = np.zeros((self.MAP_X, self.MAP_Y, 3))
        self.image = np.transpose(self.image, (1, 0, 2))
        self.rank = rank
       #if self.unwrapped.render_gui:
       #    _ = cv2.namedWindow('im', cv2.WINDOW_NORMAL)
       #    cv2.imshow('im', self.image)


    def set_log_dir(self, log_dir):
        self.log_dir = log_dir

    def step(self, action):
        jpeg_size = self.im_render()
        # save image of map
        obs, rew, done, info = self.env.step(action)
        info = {
                **info,
                **self.metrics,
                'jpeg_size': jpeg_size,
                }
        return obs, rew, done, info

    def reset_episodes(self, im_log_dir):
        self.image = np.zeros((self.MAP_X, self.MAP_Y, 3))
        self.image = np.transpose(self.image, (1, 0, 2))
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
        obs = self.env.reset()
        jpeg_size = self.im_render()
        info = {
                'jpeg_size': jpeg_size,
                }
        return obs#, info

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
                self.rank, self.n_episode, self.num_step))
           #print(log_dir)
            cv2.imwrite(log_dir, self.image)
            self.n_saved += 1
            size = os.stat(log_dir).st_size
            return size


class ImRenderMulti(ImRender):
    def __init__(self, env, log_dir, rank):
        super(ImRenderMulti, self).__init__(env, log_dir, rank)

    def step(self, action):
        jpeg_size = self.im_render()
        obs, rew, done, info = self.env.step(action)
        info = info[0]
        info = {
                **info,
                **self.metrics,
                'jpeg_size': jpeg_size,
                }
        return obs, rew, done, [info]

    def reset(self):
        return super().reset()[0]

    def im_render(self):
        state = self.world.state.clone()
        sizes = []
        for i in range(self.num_proc):
            image = state[i].cpu()
            image = np.vstack((image, image, image))
            image = np.transpose(image, (2, 1, 0))
            image = image.astype(np.uint8)
            image = image * 255
            log_dir = os.path.join(self.im_log_dir, 'rank:{}_epi:{}_step:{}.jpg'.format(
                i, self.n_episode, self.num_step))
            cv2.imwrite(log_dir, image)
            size = os.stat(log_dir).st_size
            sizes.append(size)
           #if i == self.rend_idx:
           #    cv2.imshow('im', image)
           #    cv2.waitKey(1)

            self.n_saved += 1
        return sizes
       #zone_map = self.unwrapped.micro.map.zoneMap
       #zones = self.unwrapped.micro.map.zones
       #tile_types = self.tile_types
       #type_colors = self.type_colors
       #colors = self.colors
       #for x in range(self.MAP_X):
       #    for y in range(self.MAP_Y):
       #        tile_type = tile_types[zones[zone_map[-1, x, y]]]
       #        color = colors[type_colors[tile_type]]
       #        self.image[x][y] = color
       #self.image = np.transpose(self.image, (1, 0, 2))
       #self.image = self.image * 255
       #if self.unwrapped.render_gui:
       #    cv2.imshow('im', self.image)
       #if self.unwrapped.num_step % self.save_interval == 0:
       #    log_dir = os.path.join(self.im_log_dir, 'rank:{}_epi:{}_step:{}.jpg'.format(
       #        self.unwrapped.rank, self.n_episode, self.num_step))
       #    print(log_dir)
       #    cv2.imwrite(log_dir, self.image)
       #    self.n_saved += 1



class ParamRew(gym.Wrapper):
    def __init__(self, env, env_params, rand_params=False):

        # Whether to always select random parameters, to stabilize learning multiple objectives 
        # (i.e. prevent overfitting to some subset of objectives)
        self.rand_params = rand_params
        self.env = env
        super().__init__(self.env)
        self.usable_metrics = env_params
        self.num_params = len(env_params)
        self.auto_reset = True

    def configure(self, **kwargs):
        print(kwargs)
        self.unwrapped.configure(**kwargs)
        self.metrics = self.unwrapped.metrics
        # NB: self.metrics needs to be an OrderedDict
        print('usable metrcs:', self.usable_metrics)
        print('unwrapped: {}'.format(self.unwrapped.metrics))
        self.last_metrics = copy.deepcopy(self.metrics)
        self.param_bounds = self.unwrapped.param_bounds
        self.param_ranges = {}
#       self.max_improvement = 0
        for k in self.usable_metrics:
            v = self.param_bounds[k]
            improvement = abs(v[1] - v[0])
            self.param_ranges[k] = improvement
#           self.max_improvement += improvement * self.weights[k]
        self.metric_trgs = self.unwrapped.metric_trgs

        for k in self.usable_metrics:
            v = self.metrics[k]
            self.weights[k] = self.unwrapped.weights[k]
        for k in self.usable_metrics:
            self.param_bounds['{}_weight'.format(k)] = (0, 1)
        self.width = self.unwrapped.width
        self.observation_space = self.env.observation_space
        # FIXME: hack for gym-pcgrl
        print('param orig rew obs space', self.observation_space)
        self.action_space = self.env.action_space
        orig_obs_shape = self.observation_space.shape
        obs_shape = orig_obs_shape[0] + 2 * len(self.usable_metrics), orig_obs_shape[1], orig_obs_shape[2]
        low = self.observation_space.low
        high = self.observation_space.high
        metrics_shape = (2 * len(self.usable_metrics), obs_shape[1], obs_shape[2])
        self.metrics_shape = metrics_shape
        metrics_low = np.full(metrics_shape, fill_value=0)
        metrics_high = np.full(metrics_shape, fill_value=1)
        low = np.vstack((metrics_low, low))
        high = np.vstack((metrics_high, high))
        self.observation_space = gym.spaces.Box(low=low, high=high)
        self.next_trgs = None
        if self.render_gui and True:
            screen_width = 200
            screen_height = 100 * self.num_params
            from wrapper_windows import ParamRewWindow
            win = ParamRewWindow(self, self.metrics, self.metric_trgs, self.param_bounds)
           #win.connect("destroy", Gtk.main_quit)
            win.show_all()
            self.win = win

    def enable_auto_reset(self, button):
        self.auto_reset = button.get_active()


    def getState(self):
        scalars = super().getState()
       #trg_weights = [v for k, v in self.weights.items()]
       #scalars += trg_weights
        print(scalars)
        raise Exception
        return scalars

    def set_trgs(self, trgs):
        if self.rand_params:
            for k in self.usable_metrics:
                min_v, max_v = self.param_bounds[k]
                trgs[k] = random.uniform(min_v, max_v)
        self.next_trgs = trgs

    def do_set_trgs(self):
        trgs = self.next_trgs
        i = 0
        self.init_metrics = copy.deepcopy(self.metrics)

#       self.max_improvement = 0
        for k, trg in trgs.items():
            if k in self.usable_metrics:
                self.metric_trgs[k] = trg
#               self.max_improvement += abs(trg - self.init_metrics[k]) * self.weights[k]

       #for k in self.usable_metrics:
       #    weight_name = '{}_weight'.format(k)
       #    if weight_name in params:
       #        self.weights[k] = params[weight_name]
       #    i += 1

       #for k, _ in self.usable_metrics():
       #    if k in params:
       #        metric_trg = params[k]
       #        self.metric_trgs[k] = metric_trg
       #        self.max_improvement += abs(metric_trg - self.init_metrics[k]) * self.weights[k]
       #    i += 1


        self.display_metric_trgs()

    def reset(self):
        if self.next_trgs:
            self.do_set_trgs()
        ob = super().reset()
        ob = self.observe_metric_trgs(ob)
        self.metrics = self.unwrapped.metrics
        self.last_metrics = copy.deepcopy(self.metrics)
        self.n_step = 0
        return ob

    def observe_metric_trgs(self, obs):
        metrics_ob = np.zeros((self.metrics_shape))
        i = 0
        for k in self.usable_metrics:
            trg = self.metric_trgs[k]
            metric = self.metrics[k]
            if not metric:
                #FIXME: a problem after reset in pcgrl envs
#               print(k, metric, self.metrics)
#               assert self.n_step < 20
                metric = 0
            metrics_ob[i*2, :, :] = trg / self.param_ranges[k]
            metrics_ob[i*2+1, :, :] = metric / self.param_ranges[k]
            i += 1
#       print('param rew obs shape ', obs.shape)
#       print('metric trgs shape ', metrics_ob.shape)
        obs = np.vstack((metrics_ob, obs))
        return obs


    def step(self, action):
        if self.render_gui and True:
            self.win.step()

       #print('unwrapped metrics', self.unwrapped.metrics)
        ob, rew, done, info = super().step(action)
        ob = self.observe_metric_trgs(ob)
        self.metrics = self.unwrapped.metrics
        rew = self.get_reward()
        self.last_metrics = copy.deepcopy(self.metrics)
        self.n_step += 1
        if not self.auto_reset:
            done = False
        return ob, rew, done, info

    def get_param_trgs(self):
        return self.metric_trgs

    def get_param_bounds(self):
        return self.param_bounds

    def set_param_bounds(self, bounds):
        for k, (l, h) in bounds.items():
            self.param_bounds[k] = (l, h)

    def display_metric_trgs(self):
        if self.render_gui:
            self.win.display_metric_trgs()

    def get_reward(self):
        reward = 0

        for metric in self.usable_metrics:
            trg = self.metric_trgs[metric]
            val = self.metrics[metric]
            last_val = self.last_metrics[metric]
            trg_change = trg - last_val
            change = val - last_val
            metric_rew = 0
            same_sign = (change < 0) == (trg_change < 0)
            # changed in wrong direction

            if not same_sign:
                metric_rew -= abs(change)
            else:
                less_change = abs(change) < abs(trg_change)
                # changed not too much, in the right direction

                if less_change:
                    metric_rew += abs(change)
                else:
                    metric_rew += abs(trg_change) - abs(trg_change - change)
            reward += metric_rew * self.weights[metric]
       #assert(reward <= self.max_improvement, 'actual reward {} is less than supposed maximum possible \
       #        improvement toward target vectors of {}'.format(reward, self.max_improvement))
       #if self.max_improvement == 0:
       #    pass
       #else:
       #    reward = 100 * (reward / self.max_improvement)

        return reward


class ParamRewMulti(ParamRew):
    ''' Calculate reward in terms of movement toward vector of target metrics, for environments in
    which 1 env actually contains multiple (i.e. GoLMulti).
    '''
    def __init__(self, env):
        super(ParamRewMulti, self).__init__(env)

        if self.cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def step(self, action):
        self.last_metrics = copy.deepcopy(self.metrics)
        ob, rew, done, info = super().step(action)
        ob = self.get_obs(ob)
        rew = self.get_reward()

        return ob, rew, done, info

    def reset(self):
        obs = super().reset()
        obs = self.get_obs(obs)
        return obs

    def get_obs(self, map_state):
        obs = self.env.get_obs()
        scalar_obs = list(self.metric_trgs.values()) + list(self.weights.values())
        full_obs_shape = list(map_state.shape)
        full_obs_shape[1] += len(scalar_obs)
        full_obs = torch.zeros(full_obs_shape)
        full_obs[:, 0:1, :, :] = map_state
        i = 1
        for s in scalar_obs:
            full_obs[:, i, :, :] = torch.Tensor([s])
            i += 1
        return full_obs

    def get_reward(self):
        reward = torch.zeros(self.num_proc)
        reward = reward.to(self.device)

        for metric, trg in self.metric_trgs.items():
            last_val = self.last_metrics[metric].to(self.device)
            trg_change = trg - last_val
            val = self.metrics[metric]
            val = val.to(self.device)
            change = val - last_val
            metric_rew = torch.zeros(self.num_proc)
            metric_rew = metric_rew.to(self.device)
            same_sign = (change < 0) == (trg_change < 0)
            # changed in wrong direction
            metric_rew += (same_sign == False) * -abs(change)
            less_change = abs(change) < abs(trg_change)
            # changed a little, in the right direction
            metric_rew += (same_sign) * (less_change) * abs(change)
            # overshot
            metric_rew += (same_sign) * (less_change == False) * (abs(trg_change) - abs(trg_change - change))
            reward += metric_rew * self.metric_weights[metric]
        reward = reward.unsqueeze(-1)
        reward = reward.to(torch.device('cpu'))
        return reward
