import torch
import torch.nn as nn
import torch.nn.Functional as F
import copy

class _hardtanh(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        return F.hardtanh(x)

    def backward(self, y):
        raise NotImplementedError

class PowerConnector(nn.Module):
    ''' A Neural CA that creates the shortest path of wires connecting an unpowered zone to a
    powered tile.'''
    def __init__(self, map_width)
        self.map_width = map_width
        self.map_height = map_width
        self.conv = nn.Conv2d(2, 5, map_width, map_height)
        zero = np.zeros((3, 3))
        vn = np.array([ # von neumann neighborhood
                [0, 1, 0],
                [1, 1, 1],
                [0, 1, 0]])
        flow = copy.deepcopy(vn)
        flow[1][1] += 12 # once activation flowing from one source reaches a tile, it stays
        unflow = copy.deepcopy(vn)
        unflow[1][1] = 0
        unflow = vn * -3 # if a tile is adjacent to both flows, nothing will flow there
        weights = [
                  [
                      [flow],
                      [unflow],
                      [zero],
                      [zero],
                      [zero]
                  ],
                  [
                      [flow],
                      [unflow],
                      [zero],
                      [zero],
                      [zero]
                  ],
                  ]
        self.conv.init()
        self.conv.weight = torch.nn.Parameter(weights, required_grad=False)
        hardtanh = F.hardtanh(min_val=0)

    def forward(obs):
        '''obs:  2D image, suppose channel 0 indicates the poweredness of a tile,
                 and channel 1 indicates the presence of a zone.'''
        powered = obs[0]
        zone = obs[1]

        return hardtanh(self.conv(obs))


