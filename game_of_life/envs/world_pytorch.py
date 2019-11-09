import numpy as np
import torch
from torch import nn

def pad_circular(x, pad):
    """

    :param x: shape [B, C, H, W]
    :param pad: int >= 0
    :return:
    """
    x = torch.cat([x, x[:, :, 0:pad, :]], dim=2)
    x = torch.cat([x, x[:, :, :, 0:pad]], dim=3)
    x = torch.cat([x[:, :, -2 * pad:-pad, :], x], dim=2)
    x = torch.cat([x[:, :, :, -2 * pad:-pad], x], dim=3)

    return x

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data)
   #bias_init(module.bias.data)
    return module


class World(nn.Module):
    def __init__(self, map_width=16, map_height=16, prob_life=20, cuda=False, 
                 num_proc=1, env=None):
        super(World, self).__init__()
        self.cuda = cuda
        self.map_width = map_width
        self.map_height = map_height
        self.prob_life = prob_life / 100
        self.num_proc = num_proc
        state_shape = (num_proc, 1, map_width, map_height)
        if self.cuda:
            self.y1 = torch.ones(state_shape).cuda()
            self.y0 = torch.zeros(state_shape).cuda()
        else:
            self.y1 = torch.ones(state_shape)
            self.y0 = torch.zeros(state_shape)
        device = torch.device("cuda:0" if cuda else "cpu")
        self.conv_init_ = lambda m: init(m,
            nn.init.dirac_, None,
            nn.init.calculate_gain('relu'))

        conv_weights = [[[[1, 1, 1],
                        [1, 9, 1],
                        [1, 1, 1]]]]
        self.transition_rule = nn.Conv2d(1, 1, 3, 1, 0, bias=False)
        self.conv_init_(self.transition_rule)
        self.transition_rule.to(device)
        self.populate_cells()
        if cuda:
            conv_weights = torch.cuda.FloatTensor(conv_weights)
        else:
            conv_weights = torch.FloatTensor(conv_weights)
        conv_weights = conv_weights
        self.transition_rule.weight = torch.nn.Parameter(conv_weights, requires_grad=False)
        self.to(device)

    def populate_cells(self):
        if self.cuda:
            self.state = torch.cuda.FloatTensor(size=
                    (self.num_proc, 1, self.map_width, self.map_height)).uniform_(0, 1)
            self.builds = torch.cuda.FloatTensor(size=
                    (self.num_proc, 1, self.map_width, self.map_height)).fill_(0)
            self.failed = torch.cuda.FloatTensor(size=
                    (self.num_proc, 1, self.map_width, self.map_height)).fill_(0)
        else:
            self.state = torch.FloatTensor(size=
                    (self.num_proc, 1, self.map_width, self.map_height)).uniform_(0, 1)
            self.builds = torch.FloatTensor(size=
                    (self.num_proc, 1, self.map_width, self.map_height)).fill_(0)
            self.failed = torch.FloatTensor(size=
                    (self.num_proc, 1, self.map_width, self.map_height)).fill_(0)
        self.state = torch.where(self.state < self.prob_life, self.y1, self.y0).float()

    def repopulate_cells(self):
        self.state.uniform_(0, 1)
        self.state = torch.where(self.state < self.prob_life, self.y1, self.y0).float()
        self.builds.fill_(0)
        self.failed.fill_(0)

    def build_cell(self, x, y, alive=True):
        if alive:
            self.state[0, 0, x, y] = 1
        else:
            self.state[0, 0, x, y] = 0

    def _tick(self):
        self.state = self.forward(self.state)
       #print(self.state[0][0])

    def forward(self, x):
        with torch.no_grad():
            x = x.float()
            if self.cuda:
                x = x.cuda()
            x = pad_circular(x, 1)
            x = self.transition_rule(x)
            return self.GoLu(x)

    def GoLu(self, x):
        '''
        Applies the Game of Life Unit activation function, element-wise:
        '''

#       return torch.where(2 < x, y1, y0)
        return torch.where((x == 3) | (x == 11) | (x == 12), self.y1, self.y0)

    def seed(self, seed=None):
        np.random.seed(seed)



def main():
    world = World()
    for j in range(4):
        world.repopulate_cells()
        for i in range(100):
            world._tick()
            print(world.state)

if __name__ == '__main__':
    main()

