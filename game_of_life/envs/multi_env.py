from pdb import set_trace as TT
import os
from collections import OrderedDict

import cv2
import numpy as np
import torch
from gym import core, spaces
from gym.utils import seeding

from .im2gif import GifWriter, frames_to_gif
from .world_pytorch import World


class GoLMultiEnv(core.Env):
    ''' Multiple simulations are processed at once by a Neural Network.
    This single environment manages them all.
    '''
    def __init__(self):
        self.num_tools = 1 # bring cell to life
        self.player_step = False
        self.player_builds = []
        self.action_bin = None
        self.render_idx = -1
        self.agent_steps = 1
        self.sim_steps = 1

    def configure(self, map_width, render=False, prob_life=20,
             max_step=200, num_proc=1, record_dir=None, cuda=False,
             poet=False, rand_agent=False, no_agent=False, compare_agents=False):
        self.num_proc = num_proc
        self.prebuild = False
        self.prebuild_steps = 50
        self.map_width = size = map_width
        self.record = record_dir is not None
        self.rand_agent = rand_agent
        self.no_agent = no_agent
        self.compare_agents = compare_agents
        if self.record:
            self.gif_dir = ('{}/gifs/'.format(record_dir))  # WTF
            self.im_paths = []
            self.n_gif_frame = 0
        self.record_dir = record_dir
        self.cuda = cuda

        self.render_gui = render

        if render and self.record:
            self.gif_writer = GifWriter()
            try:
                os.mkdir(os.path.join(self.gif_dir, 'im')) # in case we are continuing eval
            except FileNotFoundError: pass
            except FileExistsError: pass
            try:
                os.mkdir(self.gif_dir) # in case we're starting a new eval
                os.mkdir(os.path.join(self.gif_dir, 'im'))
            except FileExistsError:
                pass
        max_pop = self.map_width * self.map_width
        self.param_bounds = OrderedDict({
                'pop': (0, max_pop)
                })
        self.param_ranges = [abs(ub-lb) for lb, ub in self.param_bounds.values()]
        self.metric_weights = {'pop': 1}
        self.param_ranges = self.param_ranges + [1 for i in self.metric_weights]
       #max_loss = sum(self.param_ranges)
       #self.max_loss = torch.zeros(size=(self.num_proc,)).fill_(max_loss)
        self.metric_trgs = OrderedDict({
               #'pop': 0 # aim for empty board
               #'prob_life':
                'pop': 150,
               #'pop': max_pop,
                # aim for max possible pop (roughly?)
                })
        num_params = len(self.metric_trgs) + len(self.metric_weights)
        # so that we can calculate the loss of each sim separately
        self.trg_param_vals = torch.Tensor([v for v in self.metric_trgs.values()])
        self.trg_param_vals = self.trg_param_vals.unsqueeze(-1).expand(self.num_proc,
                num_params)
        self.curr_param_vals = torch.zeros(self.trg_param_vals.shape)
        self.metrics = {}
        i = 0

        for key, val in self.metric_trgs.items():
            self.metrics[key] = self.curr_param_vals[:,i]
            i += 1
        obs_shape = (1 + 1 * num_params, size, size)
        scalar_obs_shape = (num_proc, num_params, size, size)
        self.num_params = num_params
        slice_shape = (num_proc, 1, size, size)
        action_shape_2D = (num_proc, 1, size, size)
        self.action_shape_2D = action_shape_2D
        low = np.zeros(obs_shape)
        high = np.ones(obs_shape)
       #i = 0 # bounds of scalars in observation space
       ## the scalar parameter channels come first, matched in step()
       #for lb, ub in self.param_bounds.values():
       #    low[:, i:i+1, :, :] = torch.Tensor(size=slice_shape).fill_(lb)
       #    high[:, i:i+1, :, :] = torch.Tensor(size=slice_shape).fill_(ub)
       #    i += 1
        self.observation_space = spaces.Box(low, high, dtype=int)
        self.scalar_obs = torch.Tensor()
        self.action_space = spaces.Discrete(self.num_tools * size * size)
        self.view_agent = render

        self.n_gif_ep = 0
        self.num_step = 0
       #self.record_entropy = True
        self.world = World(self.map_width, self.map_width, prob_life=prob_life,
                           cuda=cuda, num_proc=num_proc, env=self)
        self.state = None
        self.max_step = max_step

        if compare_agents:
            self.compare_idx = 0
            self.init_state = self.world.repopulate_cells().clone()

       #self.entropies = []
        if self.render_gui:
            #TODO: render function should deal with this somehow
            cv2.namedWindow("Game of Life", cv2.WINDOW_NORMAL)
            cv2.setMouseCallback("Game of Life", self.player_build)

        self.intsToActions = [[] for pixel in range(self.num_tools * self.map_width **2)]
        self.actionsToInts = np.zeros((self.num_tools, self.map_width, self.map_width))

        self.terminal = torch.zeros((self.num_proc, 1), dtype=bool)

        ''' Unrolls the action vector in the same order as the pytorch model
        on its forward pass.'''
        i = 0

        for z in range(self.num_tools):
            for x in range(self.map_width):
                for y in range(self.map_width):
                        self.intsToActions[i] = [z, x, y]
                        self.actionsToInts[z, x, y] = i
                        i += 1
        action_bin = torch.zeros(action_shape_2D)
        action_bin = action_bin.byte()
        # indexes of separate envs
        action_ixs = torch.LongTensor(list(range(self.num_proc))).unsqueeze(1)

        if self.cuda:
            action_bin = action_bin.cuda()
            action_ixs = action_ixs.cuda()
        action_bin = action_bin.view(action_bin.shape[0], -1)
        self.action_bin, self.action_ixs = action_bin, action_ixs
        # refill these rather than creating new ones each step
        self.scalar_obs = torch.zeros(scalar_obs_shape)

        if self.cuda:
            self.scalar_obs = self.scalar_obs.cuda()
            self.curr_param_vals = self.curr_param_vals.cuda()
        self.set_params(self.metric_trgs)

    def init_storage(self):
        pass

    def get_spaces(self):
        return self.observation_space, self.action_space

    def get_param_bounds(self):
        return self.param_bounds

    def set_param_bounds(self, bounds):
        self.param_bounds = bounds
        return len(bounds)

    def set_params(self, params):
        print('Updated env targets: {}'.format(params))
        self.metric_trgs = params
        self.trg_param_vals = torch.Tensor([v for v in params.values()])
        self.trg_param_vals = self.trg_param_vals.unsqueeze(0).expand(self.num_proc,
                self.num_params)
        # update our scalar observation
        # TODO: is there a quicker way, that scales to high number of params?
        i = 0
        for v in self.trg_param_vals[0]:
            unit_v = v  / self.param_ranges[i]
            trg_channel = self.scalar_obs[:,i:i+1]
            trg_channel.fill_(unit_v)
            i += 1


    def get_curr_param_vals(self):
        self.curr_pop = self.get_pop()
        self.curr_param_vals[:, 0] = self.curr_pop
        self.metrics['pop'] = self.curr_pop


    def get_pop(self):
        pop = self.world.state.sum(dim=1).sum(1).sum(1)

        return pop

    def init_ages(self):
        self.ages = torch.zeros((self.num_proc, 1, self.map_width, self.map_width))
        if self.cuda:
            self.ages = self.ages.to(torch.device('cuda:0'))

    def step(self, a):
        '''
        a: 1D tensor of integers, corresponding to action indexes (in
            flattened output space)
        '''
        if self.no_agent:
            pass
        else:
            if self.rand_agent:
                a = torch.Tensor([self.action_space.sample() for _ in range(self.num_proc)]).unsqueeze(-1).long()
                if self.cuda:
                    a = a.cuda()
                actions = self.action_idx_to_tensor(a)
            else:
                a = a.long()
               #a.fill_(0)
                actions = self.action_idx_to_tensor(a)

            self.act_tensor(actions)

        if self.num_step % self.agent_steps == 0: # the agent build-turn is over
            if self.render_gui and not self.no_agent:
                self.agent_builds.fill_(0)

            for j in range(self.sim_steps):
                self.world._tick()
                if self.render_gui:
                    self.render(agent=False, n_frame=2)
        self.get_curr_param_vals()
       #loss = abs(self.trg_param_vals - self.curr_param_vals)
       #loss = loss.squeeze(-1)
        # loss in a 1D tensor of losses of individual envs
       #reward = torch.Tensor((100 * 50 / (loss + 50)))
       #reward = reward / (self.max_loss * self.max_step * self.num_proc)
       #reward = torch.Tensor((self.max_loss - loss) * 100 / (self.max_loss * self.max_step * self.num_proc))
        if self.num_step == self.max_step:
            terminal = torch.ones(self.num_proc, dtype=bool)
        else:
            terminal = self.metrics['pop'] == 0
                   # reward < 2 # impossible situation for agent
        #reward: average fullness of board
        #reward = reward * 100 / (self.max_step * self.map_width * self.map_width * self.num_proc)

        if self.record and terminal[self.render_idx]:
            self.render_gif()
            if self.compare_agents:
                #FIXME: resetting twice (so we have to do this here instead of inside reset)
                if self.compare_idx == 2:
                    self.compare_idx = 0
                    self.n_gif_ep += 1
                else:
                    self.compare_idx += 1

        info = [{}]
        self.num_step += 1
        obs = self.get_obs()
        ### OVERRIDE teacher for debuggine
       #reward = self.curr_pop
       #reward = self.curr_pop / (self.max_step * self.num_proc)
       #reward = 256 - self.curr_pop

       #reward = reward.unsqueeze(-1)
        reward = 0
        return (obs, reward, terminal, info)



    def act_tensor(self, actions):
        '''Take a tensor corresponding to the game-map with actions represented as 1s.'''
        acted_state = self.world.state + actions
        new_state = self.world.state.long() ^ actions.long()
        self.agent_dels = self.world.state.long() & actions.long()
        # reset age of deleted tiles to 0
       #self.ages = self.ages - self.ages * self.agent_dels

        if self.render_gui:
            # where cells are already alive
            agent_builds = actions - self.agent_dels
            assert(agent_builds >= 0).all()
           #assert(torch.sum(self.agent_dels + agent_builds) == self.num_proc)

            if not hasattr(self, 'agent_builds') or self.agent_builds.shape != actions.shape:
                self.agent_builds = actions.float()
            else:
                # agent builds accumulate on rendered board during agent's build-turn
                self.agent_builds = torch.clamp(self.agent_builds + actions, 0, 1)
            self.agent_builds -= self.agent_dels # in case agent deletes what it has built in the
            # current build-turn
            self.rend_state = new_state - self.agent_builds # separate state from agent's actions
            self.rend_state = torch.clamp(self.rend_state, 0, 1)
           #assert (self.rend_state >= 0).all() # was something built w/o being added to new state?
            # for separate rendering
            self.render(agent=True, n_frame=0)  # Render agent's action
        self.world.state = new_state
        if self.render_gui:
            self.render(agent=False, n_frame=1)  # Render resulting state
        self.update_ages()

    def update_ages(self):
        self.ages = self.ages + (self.world.state == 0) * (-self.ages)
        self.ages = self.ages + self.world.state

    def get_obs(self):
        ''' Combine scalar slices with world state to form observation. The
        agent sees only the target global parameters, leaving it to infer the
        current global properties of the map.'''
       #obs = torch.cat((self.scalar_obs, self.world.state.float()), dim=1)
        obs = self.world.state.float()

        return obs

    def action_idx_to_tensor(self, a):
        '''
        a: tensor of size (num_proc, 1) containing action indexes
        '''
        self.action_bin.fill_(0)
        action_bin, action_ixs = self.action_bin, self.action_ixs
        action_i = torch.cat((action_ixs, a), 1)
        action_bin[action_i[:, 0], action_i[:, 1]] = 1
        action = action_bin
        action = action.view(self.action_shape_2D)

        return action

    def reset(self):
        print('reset')
        self.init_ages()
        self.terminal = torch.zeros((self.num_proc, 1), dtype=bool)
        self.num_step = 0
        if self.compare_agents:
            # Compare agents on shared initial state
            if self.compare_idx == 0 or self.compare_idx == 3:
                # no agent
                self.rand_agent = False
                self.no_agent = True
            elif self.compare_idx == 1:
                # random agent
                self.rand_agent = True
                self.no_agent = False
            elif self.compare_idx == 2:
                # standard agent
                self.rand_agent = False
                self.no_agent = False
            if self.compare_idx == 3:
                # No agent
                self.init_state = self.world.repopulate_cells().clone()
                self.compare_idx = 0
            else:
                self.world.repopulate_cells(init_state=self.init_state.clone())
        else:
            self.world.repopulate_cells()
            if self.record:
                self.n_gif_ep += 1
        self.get_curr_param_vals()
       #self.world.prepopulate_neighbours()
        if hasattr(self, 'agent_builds'):
            self.agent_builds.fill_(0)

        if self.record:
            self.gif_ep_name = 'ep_{}'.format(self.n_gif_ep)
            if self.rand_agent:
                self.gif_ep_name += '_randAgent'
            if self.no_agent:
                self.gif_ep_name += '_noAgent'
            self.gif_ep_dir = os.path.join(self.gif_dir, self.gif_ep_name)
            if not os.path.isdir(self.gif_ep_dir):
                os.mkdir(self.gif_ep_dir)
            self.im_paths = []
            self.n_gif_frame = 0
        obs = self.get_obs()
        if self.render_gui:
            self.render(n_frame=0)

        return obs


    def render(self, mode=None, agent=True, n_frame=0):
        if not agent or self.num_step == 0:
            # No agent builds to render
            rend_state = self.world.state[self.render_idx].cpu().long()
            rend_state = np.vstack((rend_state * 1, rend_state * 1, rend_state * 1))
            rend_arr = rend_state

        elif agent and not self.num_step == 0:
            # Render agent build action
            rend_state = self.rend_state[self.render_idx].cpu()
            rend_dels = self.agent_dels[self.render_idx].cpu()
            rend_builds = self.agent_builds[self.render_idx].cpu()
            rend_state = np.vstack((rend_state * 1, rend_state * 1, rend_state * 1))
            rend_dels = np.vstack((rend_dels * 0, rend_dels * 0, rend_dels * 1))
            rend_builds = np.vstack((rend_builds * 0, rend_builds * 1, rend_builds * 0))
            rend_arr = rend_state + rend_dels + rend_builds
        rend_arr = rend_arr.transpose(2, 1, 0)
        rend_arr = rend_arr.astype(np.uint8)
        rend_arr = rend_arr * 255
        if self.render_gui:
            cv2.imshow("Game of Life", rend_arr)

        if self.record:
            if not self.gif_writer.done:
                im_dir = self.gif_ep_dir
                im_path = os.path.join(im_dir, 'frame-{}.png'.format(self.n_gif_frame))
                self.im_paths.append(im_path)
                cv2.imwrite(im_path, rend_arr)
                self.n_gif_frame += 1
        cv2.waitKey(1)


    def render_gif(self):
        gif_path = os.path.join(self.gif_dir, '{}.gif'.format(self.gif_ep_name))
        vid_path = os.path.join(self.gif_dir, '{}.mp4'.format(self.gif_ep_name))
        frames_to_gif(gif_path, self.im_paths)
        # self.gif_writer.create_gif(im_dir, gif_dir, 0, 0, 0)
        render_vid_cmd = 'ffmpeg -i {} -movflags faststart -vf ' \
                         '"scale=trunc(iw/2)*2:trunc(ih/2)*2" {}'.format(gif_path, vid_path)
        print(render_vid_cmd)
        os.system(render_vid_cmd)

    def player_build(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            a = int(self.actionsToInts[0][x][y])
            self.player_builds += [a]
            print('q\'d player build at: {}, {}'.format(x, y))
        elif event == cv2.EVENT_MBUTTONDOWN:
            a = int(self.actionsToInts[0][x][y])
            self.player_builds += [-a]
            print('q\'d player delete at: {}, {}'.format(x, y))


    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        seed2 = seeding.hash_seed(seed1 + 1) % 2**31
        np.random.seed(seed)
        self.world.seed(seed)

        return [seed1, seed2]

   #def delete(self, x, y):

   #   #self.world.build_cell(x, y, alive=False)


cv2.destroyAllWindows()

def main():
    env = GoLMultiEnv()
    env.configure(render=True)

    while True:
        env.step(0)
