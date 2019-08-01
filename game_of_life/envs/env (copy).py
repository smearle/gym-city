import gym
from gym import core, spaces
from .gol import utils
import argparse
import itertools

import cv2
import numpy as np
import torch
from torch import ByteTensor, Tensor
from torch.nn import Conv2d, Parameter
from torch.nn.init import zeros_



def step(state, get_neighbors):
    # Get neighbor counts of cells
    neighbors = get_neighbors(state)[0, ...]

    # Alive cell with less than two neighbors should die
    rule1 = (neighbors < 2).type(Tensor)
    
    rule1 = rule1.cuda()

    mask1 = (rule1 * state[0, ...]).type(ByteTensor)

    # Alive cell with more than two neighbors should die
    rule2 = (neighbors > 3).type(Tensor)
    rule2 = rule2.cuda()
    mask2 = (rule2 * state[0, ...]).type(ByteTensor)

    # Dead cell with exactly three neighbors should spawn
    rule3 = (neighbors == 3).type(Tensor)
    rule3 = rule3.cuda()
    mask3 = (rule3 * (1 - state[0, ...])).type(ByteTensor)

    # Update state
    state[0, mask1] = 0
    state[0, mask2] = 0
    state[0, mask3] = 1
    return state


def run_world(opts, device):
    step_count = 0
    size = opts.size
    prob = opts.prob
    tick_ratio = opts.tick_ratio
    record_entropy = opts.record_entropy

    entropies = []
    with torch.no_grad():
        combinations = 2 ** (3 * 3)
        channels = 1
        state = init_world(size, channels, prob).to(device)
        get_neighbors = get_neighbors_map(channels).to(device)
        structure_similarity = get_structure_similarity(combinations, channels)
        structure_similarity.to(device)
        i = 0
        cv2.namedWindow("Game of Life", cv2.WINDOW_NORMAL)
        while True:
            if should_step(step_count, tick_ratio):
                cv2.imshow("Game of Life", state.cpu().numpy())
                state.to(device)
                state = step(image2state(state), get_neighbors)

                if record_entropy:
                    entropy = get_entropy(
                        state, structure_similarity, combinations)
                    entropies.append(entropy)
                state = state2image(state)
            step_count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            i += 1
    cv2.destroyAllWindows()

    if record_entropy:
        entropies = np.array(entropies)
        utils.plot_entropies(entropies)




    return entropy


def should_step(step_count, tick_ratio):
    return step_count % tick_ratio == 0


def main():
    opts = argparse.ArgumentParser(description='Game of Life')
    opts.add_argument(
        '-s',
        '--size',
        help='Size of world grid',
        default=700)
    opts.add_argument(
        '-p',
        '--prob',
        help='Probability of life in the initial seed',
        default=.05)
    opts.add_argument(
        '-tr',
        '--tick_ratio',
        help='Ticks needed to update on time step in game',
        default=1)
    opts.add_argument(
        '-re',
        '--record_entropy',
        help='Should record entropy of configurations',
        default=True)
    opts = opts.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   #device = torch.device("cpu")
    run_world(opts, device)


if __name__ == "__main__":
    main()


class GameOfLifeEnv(core.Env):
    def __init__(self):
        self.num_tools = 2 # kill or grow
        pass
    def configure(self, render=False, map_width=16):
        self.size = size = map_width
        self.observation_space = spaces.Box(low=np.zeros((1, size, size)),
                high=np.ones((1, size, size)), dtype=int)
        self.action_space = spaces.Discrete(self.num_tools * size * size)

        self.step_count = 0
        self.prob = prob = 0.5
        self.tick_ratio = 1
        self.record_entropy = True
        self.state = None
       #self.device = device = torch.device("cpu")
        self.device = device = torch.device(
                 "cuda" if torch.cuda.is_available() else "cpu")
        self.render = render
        self.max_step = 100

        self.entropies = []

        with torch.no_grad():
            self.combinations = combinations = 2 ** (3 * 3)
            channels = 1
            self.state = init_world(size, channels, prob).to(device)
            self.get_neighbors = get_neighbors_map(channels).to(device)
            self.structure_similarity = get_structure_similarity(combinations, channels)
            self.structure_similarity.to(device)
            self.i = 0


        self.render = render
        self.size = map_width
        if self.render:
            cv2.namedWindow("Game of Life", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Predicted GoL", cv2.WINDOW_NORMAL)
        self.intsToActions = [[] for pixel in range(self.num_tools * self.size **2)]
        self.actionsToInts = np.zeros((self.num_tools, self.size, self.size))

        ''' Unrolls the action vector in the same order as the pytorch model
        on its forward pass.'''
        i = 0
        for z in range(self.num_tools):
            for x in range(self.size):
                for y in range(self.size):
                        self.intsToActions[i] = [z, x, y]
                        self.actionsToInts[z, x, y] = i
                        i += 1
        print('len of intsToActions: {}\n num tools: {}'.format(len(self.intsToActions), self.num_tools))
        self.bern_dist = None

    def init_world(self, size, channels, prob):
        if self.bern_dist is None:
            return torch.distributions.Bernoulli(
            Tensor([prob])).sample(torch.Size([size, size, channels])).squeeze(-1)
        else:
            return self.bern_dist(Tensor([prob]).sample(torch.Size([size, size, channels])).squeeze(-1))


    def image2state(image):
        return image.permute(2, 0, 1).unsqueeze(0)


    def state2image(state):
        return state.squeeze(0).permute(1, 2, 0)



    def reset(self):
        self.entropies = []
        self.step_count = 0
        size = self.size
        channels = 1
        self.state = self.init_world(size, channels, self.prob).to(self.device)
        self.get_neighbors = self.get_neighbors_map(channels).to(self.device)
        self.structure_similarity = self.get_structure_similarity(
                self.combinations, channels)
        return self.state.cpu().permute(2, 0, 1)

    def step(self, a):
        z, act_x, act_y = self.intsToActions[a]
        if self.render:
                cv2.imshow("Game of Life", self.state.cpu().numpy())
        self.state[act_x][act_y] = 1
        if self.render:
                cv2.imshow("Game of Life", self.state.cpu().numpy())
        self.state.to(self.device)
        self.state = image2state(self.state)
        for j in range(1):

            self.state = self.sim_step(self.state, self.get_neighbors)

            if self.record_entropy:
                entropy = get_entropy(
                    self.state, self.structure_similarity, self.combinations)
                self.entropies.append(entropy)
        self.state = state2image(self.state)
        terminal = self.step_count == self.max_step
        self.step_count += 1
        if self.render:
            cv2.waitKey(1)
        #   if cv2.waitKey(1) & 0xFF == ord('q'):
        #       break
            self.i += 1

       #if self.record_entropy:
       #    self.entropies = np.array(self.entropies)
       #    utils.plot_entropies(self.entropies)

        reward = 0
       #print(preds)
       #a = torch.Tensor(preds)
       #b = self.state.squeeze(2)
       #num_pix = self.size ** 2
        if self.entropies:
            reward = self.entropies[-1].numpy()
       #reward = (num_pix - abs((a - b).sum()).numpy()) * 100/(num_pix * self.max_step)
       #print(a.shape, b.shape)
       #inner_product = (a * b).sum()
       #a_norm = a.pow(2).sum().pow(0.5)
       #b_norm = b.pow(2).sum().pow(0.5)
       #print(b_norm)
       #print('a_norm {}, b_norm {}'.format(a_norm, b_norm))
       #cos = inner_product / (2 * a_norm * b_norm)
       #angle = torch.acos(cos).numpy()
       #reward = cos.numpy()
        infos = {}
        return (self.state.cpu().permute(2, 0, 1), reward, terminal, infos)
    def get_neighbors_map(channels):
        self.neighbors_filter = Conv2d(channels, channels, 3, padding=1)
        self.neighbors_filter.weight = Parameter(Tensor([[[[1, 1, 1],
                                                      [1, 0, 1],
                                                      [1, 1, 1]]]]),
                                            requires_grad=False)
        self.neighbors_filter.bias = zeros_(neighbors_filter.bias)
        return self.neighbors_filter


    def get_structure_similarity(combinations, channels):
        tensors = torch.zeros([combinations, 1, 3, 3])
        elems = list(map(Tensor, itertools.product([0, 1], repeat=9)))
        for i, elem in enumerate(elems):
            tensors[i] = elem.view(1, channels, 3, 3)
        structure_similarity = Conv2d(
            channels, combinations, 3, stride=3, groups=channels)
        structure_similarity.weight = Parameter(tensors, requires_grad=False)
        structure_similarity.bias = zeros_(structure_similarity.bias)
        return structure_similarity.cuda()


    def get_entropy(state, structure_similarity, combinations):
        configs = structure_similarity(state.cuda())
        match_weights = structure_similarity.weight.view(combinations, -1).sum(-1)
        distribution = torch.zeros([combinations])

        # Smooth distribution incase configuration doesn't exist
        distribution.fill_(utils.EPSILON)
        for i, weight in enumerate(match_weights):
            config = configs[0][i]
            mask = config == weight
            distribution[i] += config[mask].shape[0]
        distribution /= distribution.sum()

        entropy = utils.entropy(distribution, 2)
        info = "Max Event Probability: {} | Entropy: {}".format(
            distribution.max(), entropy)
       #print(info)

#cv2.destroyAllWindows()
