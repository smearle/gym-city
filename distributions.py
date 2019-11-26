import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import AddBias, init, init_normc_

"""
Modify standard PyTorch distributions so they are compatible with this code.
"""

FixedCategorical = torch.distributions.Categorical

old_sample = FixedCategorical.sample
FixedCategorical.sample = lambda self: old_sample(self).unsqueeze(-1)

log_prob_cat = FixedCategorical.log_prob
FixedCategorical.log_probs = lambda self, actions: log_prob_cat(self, actions.squeeze(-1)).unsqueeze(-1)

FixedCategorical.mode = lambda self: self.probs.argmax(dim=1, keepdim=True)

FixedNormal = torch.distributions.Normal
log_prob_normal = FixedNormal.log_prob
FixedNormal.log_probs = lambda self, actions: log_prob_normal(self, actions).sum(-1, keepdim=True)

entropy = FixedNormal.entropy
FixedNormal.entropy = lambda self: entropy(self).sum(-1)

FixedNormal.mode = lambda self: self.mean

class CategoricalPaint(nn.Module):
    ''' Each channel of a map image is a categorical distribution. Handle
    reshaping.'''
    def __init__(self, num_actions=19):
        super(CategoricalPaint, self).__init__()
        self.obs_shape = None
        self.num_actions = num_actions

    def forward(self, x):
        # put channels last
       #assert (x>=0).all()
       #assert (x>0).all()
        n_chan = x.size(1)
        self.map_shape = x.shape[-2:]
        x = x.transpose(1, -1)
        # batch over all cells
        x = x.reshape(-1, self.num_actions)
       #self.dist = torch.distributions.Categorical(logits=x)
       #assert (x>0).all()
        self.dist = FixedCategorical(x)
       #assert min(self.dist.probs) > 0
        return self.dist

    def sample(self):
        s = self.dist.sample()
        s = s.view(-1, *self.map_shape)
        return s

    def log_probs(self, action):
        shape = action.shape
        n_batch = action.shape[0]
        action = action.view(-1, 1)
       #action = action.unsqueeze(-1)
        lp = self.dist.log_probs(action)
        lp = lp.view(shape)
        return lp

    def entropy(self):
        # don't need to reshape here atm
        e = self.dist.entropy()
        return e

    def __deepcopy__(self, memo):
        new_dist = type(self)()
        new_dist.__dict__.update(self.__dict__)
        return new_dist


class Categorical2D(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Categorical2D, self).__init__()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return FixedCategorical(logits=x)

class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()

        init_ = lambda m: init(m,
              nn.init.orthogonal_,
              lambda x: nn.init.constant_(x, 0),
              gain=0.01)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return FixedCategorical(logits=x)


class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DiagGaussian, self).__init__()

        init_ = lambda m: init(m,
              init_normc_,
              lambda x: nn.init.constant_(x, 0))

        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        self.logstd = AddBias(torch.zeros(num_outputs))

    def forward(self, x):
        action_mean = self.fc_mean(x)

        #  An ugly hack for my KFAC implementation.
        zeros = torch.zeros(action_mean.size())
        if x.is_cuda:
            zeros = zeros.cuda()

        action_logstd = self.logstd(zeros)
        return FixedNormal(action_mean, action_logstd.exp())
