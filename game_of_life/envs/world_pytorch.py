import numpy as np
import torch
from torch import nn
import copy

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
           #nn.init.calculate_gain('relu')
            )

        conv_weights = [[[[1, 1, 1],
                        [1, 9, 1],
                        [1, 1, 1]]]]
        self.transition_rule = nn.Conv2d(1, 1, 3, 1, 0, bias=False)
        self.conv_init_(self.transition_rule)
        self.transition_rule.to(device)
        self.populate_cells()
        conv_weights = torch.FloatTensor(conv_weights)
        if cuda:
            conv_weights.cuda()
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
        self.state.float().uniform_(0, 1)
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
            if self.cuda:
                x = x.cuda()
            x = pad_circular(x, 1)
            x = x.float()
           #print(x[0])
            x = self.transition_rule(x)
           #print(x[0])
            # Mysterious leakages appear here if we increase the batch size enough.
            x = x.round() # so we hack them back into shape
           #print(x[0])
            x = self.GoLu(x)
            return x

    def GoLu(self, x):
        '''
        Applies the Game of Life Unit activation function, element-wise:
                   _
        __/\______/ \_____
       0 2 4 6 8 0 2 4 6 8
        '''
        x_out = copy.deepcopy(x).fill_(0).float()
        ded_0 = (x >= 2).float()
        bth_0 = ded_0 * (x < 3).float()
        x_out = x_out + (bth_0 * (x - 2).float())
        ded_1 = (x >= 3).float()
        bth_1 = ded_1 * (x < 4).float()
        x_out = x_out + abs(bth_1 * (x - 4).float())
        alv_0 = (x >= 10).float()
        lif_0 = alv_0 * (x < 11).float()
        x_out = x_out + (lif_0 * (x - 10).float())
        alv_1 = (x >= 11).float()
        lif_1 = alv_1 * (x < 12).float()
        x_out = x_out + lif_1
        alv_2 = (x >= 12).float()
        lif_2 = alv_2 * (x < 13).float()
        x_out = x_out + abs(lif_2 * (x -13).float())
        assert (x_out >= 0).all() and (x_out <=1).all()
       #x_out = torch.clamp(x_out, 0, 1)
        return x_out

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

