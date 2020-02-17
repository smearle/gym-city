import torch
import gym
import numpy as np
import copy
from gym_city.wrappers import Extinguisher

class ExtinguisherMulti(Extinguisher):
    ''' Handle extinction events, for environments in which 1 env contains many (i.e. GoLMulti).
    '''
    def __init__(self, env, xt_type, xt_probs, xt_dels=15):
        super(ExtinguisherMulti, self).__init__(env, xt_type, xt_probs, xt_dels)
        dels = torch.zeros((self.num_proc, 1, self.map_width, self.map_width), dtype=torch.int8)
        self.dels = dels.to(self.device)
        n_curr_dels = torch.zeros((self.num_proc))
        self.n_curr_dels = n_curr_dels.to(self.device)

    def init_ages(self):
        self.unwrapped.init_ages()

    def localWipe(self):
        # assume square map
        print('LOCALWIPE')
        n_curr_dels = self.n_curr_dels.fill_(0)
        x = np.random.randint(0, self.map_width)
        y = np.random.randint(0, self.map_width)
        for r in range(0, max(x, abs(self.map_width - x), y, abs(self.map_width - y))):
            n_curr_dels += self.clear_border(x, y, r, n_curr_dels)
            if (n_curr_dels >= self.n_dels).all():
                break
        return n_curr_dels

    def clear_border(self, x, y, r, n_curr_dels):
        '''Clears the border r away (Manhattan distance) from a central point, one tile at a time.
        '''
        dels = self.dels.fill_(0)
        ages = self.ages
        for x_i in range(x - r, x + r):
            if x_i < 0 or x_i >= self.map_width:
                continue
            for y_i in range(y - r, y + r):
                if y_i < 0 or y_i >= self.map_width:
                    continue
                dels[(n_curr_dels < self.n_dels), 0, x_i, y_i] = 1
                dels = self.world.state * dels
                self.act_tensor(dels)
                n_curr_dels += torch.sum(dels, dim=[1, 2, 3])
                if (n_curr_dels >= self.n_dels).all():
                    return n_curr_dels
        return n_curr_dels

    def ranDemolish(self):
        # hack this to make it w/o replacement
        print('RANDEMOLISH')
        ages = self.ages.cpu().numpy()
        curr_dels = 0
        for i in range(self.n_dels):
           #ages = ages.flatten()
            builds = [np.where(ages[i, 0] > 0) for i in range(self.num_proc)]
            build_ix = [zip(builds[j][0], builds[j][1]) for j in range(self.num_proc)]
            build_ix = [list(build_ix[j]) for j in range(self.num_proc)]
           #if len(age_is) == 0:
           #    break
           #age_i = np.random.choice(np.where(ages_flat > -1))
            del_i = [np.random.choice(len(build_ix[j])) for j in range(self.num_proc)]
            del_coords = [build_ix[j][del_i[j]] for j in range(self.num_proc)]
           #result = self.micro.doBotTool(x, y, 'Clear', static_build=True)
            self.delete(del_coords)
            curr_dels += 1
        return curr_dels

    def delete(self, del_coords):
        dels = self.dels.fill_(0)
        for i, del_xy in enumerate(del_coords):
            dels[i, 0][tuple(del_xy)] = 1
        return self.act_tensor(dels)


    def elderCleanse(self):
        print('\n AGEIST VIOLENCE')
       #for i in range(20):
        curr_dels = 0
       #np.set_printoptions(threshold=sys.maxsize)
       #ages_arr = self.ages.cpu().numpy()
        for i in range(self.n_dels):
            ages = self.ages.clone()
            youngest = ages.max()
           #builds = [torch.where(ages[i, 0] > 0) for i in range(self.num_proc)]
           #if len(builds) == 0:
           #    break
           #ages += (ages < 0) * 2 * youngest
            ages = ages.view(self.num_proc, -1)
            age_i = torch.argmax(ages, dim=1)
           #x, y = np.unravel_index(age_i, self.micro.map.age_order.shape)
            age_i = age_i.unsqueeze(-1)
            x = age_i // self.ages.shape[-2]
            y = age_i % self.ages.shape[-1]
            xy = torch.cat((x, y), dim=1)
           #print('deleting {} {}'.format(x, y))
           #print('zone {}'.format(self.micro.map.zones[self.micro.map.zoneMap[-1, x, y]]))
           #result = self.micro.doBotTool(x, y, 'Clear', static_build=True)
            self.delete(xy)
           #self.render()
           #print('result {}'.format(result))
            curr_dels += 1
        # otherwise it's over!
       #self.micro.engine.setFunds(self.micro.init_funds)
        return curr_dels

    def reset(self):
        self.ages.fill_(0)
        obs = super().reset()
        return obs


class ParamRewMulti(gym.Wrapper):
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
        rew = self.get_reward()
        return ob, rew, done, info

    def get_reward(self):
        reward = torch.zeros(self.num_proc)
        reward = reward.to(self.device)
        for metric, trg in self.metric_trgs.items():
            last_val = self.last_metrics[metric].to(self.device)
            trg_change = trg - last_val
            val = self.metrics[metric]
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
       #print(reward)
        return reward
