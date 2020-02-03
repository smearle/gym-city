import torch
import torch.nn as nn
import torch.nn.functional as F

from distributions import Categorical, Categorical2D, CategoricalPaint, DiagGaussian
from utils import init, init_normc_
from torchsummary import summary
import math

import numpy as np
import copy

# from coord_conv_pytorch.coord_conv import nn.Conv2d, nn.Conv2dTranspose
from ConvLSTMCell import ConvLSTMCell
#from torchviz import make_dot

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, base_kwargs={}, curiosity=False, algo='A2C', model='MicropolisBase', args=None):
        super(Policy, self).__init__()
        self.action_bin = None
        self.obs_shape = obs_shape
        self.curiosity = curiosity
        self.args = args

        # TODO this info should come directly from the environment. Redundant code.

        if 'GameOfLife' in args.env_name:
            num_actions = 1
        elif 'Micropolis' in args.env_name:
            if args.power_puzzle:
                num_actions = 1
            else:
                num_actions = 19
        self.multi_env = False
        if 'GoLMultiEnv' in args.env_name:
            self.multi_env = True
            num_actions = 1
        self.num_actions = num_actions
        base_kwargs = {**base_kwargs, **{'num_actions': num_actions}}


        if len(obs_shape) == 3 or len(obs_shape) == 4: # latter being the GoLMultiEnv case
            if curiosity:
                self.base = MicropolisBase_ICM(obs_shape[0], **base_kwargs)
            else:
                if not args.model:
                    args.model = 'fixed'
                else:
                    base_model = globals()[args.model]
                if not args.model == 'FullyConv_linVal':
                    base_kwargs['val_kern'] = args.val_kern
                if args.model == 'FractalNet':
                    base_kwargs = {**base_kwargs, **{'n_recs': args.n_recs,
                            'intra_shr':args.intra_shr, 'inter_shr':args.inter_shr,
                            'rule':args.rule
                            }}
                self.base = base_model(**base_kwargs, n_chan=args.n_chan)
            print('BASE NETWORK: n', self.base)
            # if torch.cuda.is_available:
            #    print('device', torch.cuda.current_device())
            # else:
            #    print('device: cpu')

        elif len(obs_shape) == 1:
            self.base = MLPBase(**base_kwargs)
        else:
            print('unsupported environment observation shape: {}'.format(obs_shape))
            raise NotImplementedError

        if action_space.__class__.__name__ == "Discrete":
            if True:
                num_outputs = action_space.n
                self.dist = Categorical2D(self.base.output_size, num_outputs)
            else:
                num_outputs = action_space.n
                self.dist = Categorical2D(self.base.output_size, num_outputs)

        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape
            if self.args.env_name == 'MicropolisPaintEnv-v0':
                self.dist = CategoricalPaint(num_actions=self.num_actions)
            else:
                self.dist = DiagGaussian(self.base.output_size, self.num_actions)
    #           self.dist = Categorical2D(self.base.output_size, num_outputs)

        else:
            raise NotImplementedError

       #self.base.cuda()
       #summary(self.base, obs_shape)
       #self.base.cpu()

    def visualize_net(self):
        pass
       #x = torch.autograd.Variable(torch.zeros(size=(self.args.num_processes, *self.obs_shape)))
       #if False:
       #    x.cuda()
       #out = self.base(x)
       #out = out[0]
       #dot = make_dot(out.mean(), params=dict(self.base.named_parameters()))
       #dot.format = 'svg'
       #dot.filename = 'col_{}.gv'.format(self.base.active_column)
       #dot.render()

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False,
            player_act=None, icm_enabled=False):
        ''' assumes player actions can only occur on env rank 0'''
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
       #assert (actor_features >= 0).all()
        if 'paint' in self.args.env_name.lower():#or self.args.prebuild:
            smax = torch.nn.Softmax2d()
            actor_features = smax(actor_features)
       #    assert (actor_features >= 0).all()
       #    assert (actor_features > 0).all()
            # we sample over each channel, ending up with an action at each tile
            dist = self.dist(actor_features)
            action = self.dist.sample()
            action_log_probs = self.dist.log_probs(action).squeeze(0)

        else:
            dist = self.dist(actor_features)
            if player_act:
                # force the model to sample the player-selected action
                play_features = actor_features
                play_features = play_features.view(actor_features.size(0), -1)
                play_features.fill_(-99999)
                play_features[:1, player_act] = 99999
                play_features = play_features.view(actor_features.shape)
                play_dist = self.dist(play_features)
                action = play_dist.sample()
                # backprop is sent through the original distribution
                action_log_probs = dist.log_probs(action)

            else:

                if deterministic:
                    action = dist.mode()
                else:
                    action = dist.sample()
                action_log_probs = dist.log_probs(action)

        if icm_enabled:
            if self.action_bin is None:
                action_bin = torch.zeros(dist.probs.shape)
                # indexes of separate envs
                action_ixs = torch.LongTensor(list(range(dist.probs.size(0)))).unsqueeze(1)
                if self.args.cuda:
                    action_bin = action_bin.cuda()
                    action_ixs = action_ixs.cuda()
                self.action_bin, self.action_ixs = action_bin, action_ixs
            else:
                action_bin, action_ixs = self.action_bin, self.action_ixs
            action_i = torch.cat((action_ixs, action), 1)
            action_bin[action_i[:,0], action_i[:,1]] = 1
            if self.multi_env:
                action = action_bin
                action = action.view(actor_features.shape)

        return value, action, action_log_probs, rnn_hxs

    def icm_act(self, inputs):
        s1, pred_s1, pred_a = self.base(inputs, None, None, icm=True)
        return s1, pred_s1, self.dist(pred_a).probs

    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def evaluate_icm(self, inputs):
        s1, pred_s1, pred_a = self.base(inputs, None, None, icm=True)
        return s1, pred_s1, self.dist(pred_a).probs

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):

        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        if 'paint' in self.args.env_name.lower():
            dist = self.dist(actor_features)
            action = action.view(self.args.num_steps, -1)
            action_log_probs = self.dist.log_probs(action).squeeze(0)
            dist_entropy = self.dist.entropy().mean()
        else:
            dist = self.dist(actor_features)

            action_log_probs = dist.log_probs(action)
            dist_entropy = dist.entropy().mean()
        return value, action_log_probs, dist_entropy, rnn_hxs


class NNBase(nn.Module):

    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRUCell(recurrent_input_size, hidden_size)
            nn.init.orthogonal_(self.gru.weight_ih.data)
            nn.init.orthogonal_(self.gru.weight_hh.data)
            self.gru.bias_ih.data.fill_(0)
            self.gru.bias_hh.data.fill_(0)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x = hxs = self.gru(x, hxs * masks)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N, 1)

            outputs = []
            for i in range(T):
                hx = hxs = self.gru(x[i], hxs * masks[i])
                outputs.append(hx)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.stack(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)

        return x, hxs

class FullyConv_Atari(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=256,
            map_width=20, num_actions=1, in_w=1, in_h=1, out_w=1, out_h=1):
        super(FullyConv_Atari, self).__init__(recurrent, hidden_size, hidden_size)
        num_chan = 32
        num_actions = num_actions
        self.map_width = map_width
        init_ = lambda m: init(m,
            nn.init.dirac_,
            lambda x: nn.init.constant_(x, 0.1),
            nn.init.calculate_gain('relu'))
        self.embed = init_(nn.Conv2d(num_inputs, num_chan, 1, 1, 0))
       #self.k5 = init_(nn.Conv2d(num_chan, num_chan, 5, 1, 2))
       #self.k3 = init_(nn.Conv2d(num_chan, num_chan, 3, 1, 1))
        if in_w > out_w or in_h > out_h:
            self.n_sqz = int(math.log(max(in_w, in_h), 2) + 1)
            self.sqz = init_(nn.Conv2d(num_chan, num_chan, 3, 2, 1))
        init_ = lambda m: init(m,
            nn.init.dirac_,
            lambda x: nn.init.constant_(x, 0))
        self.val = init_(nn.Conv2d(num_chan, 1, 3, 1, 1))
        self.act = init_(nn.Conv2d(num_chan, num_actions, 1, 1, 0))

    def forward(self, x, rhxs=None, masks=None):

        x = F.relu(self.embed(x))
       #x = F.relu(self.k5(x))
       #x = F.relu(self.k3(x))
       #x_lin = torch.tanh(self.dense(x.view(x.shape[0], -1)))
       #val = self.val(x_lin)
        for i in range(self.n_sqz):
            x = F.hardtanh(self.sqz(x))
        act = self.act(x)
       #val = x
        val = self.val(x)
        return val.view(val.shape[0], -1), act, rhxs

class FullyConv(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=256,
            map_width=20, num_actions=1, in_w=1, in_h=1, out_w=1, out_h=1,
            n_chan=64, val_kern=3, prebuild=None):
        super(FullyConv, self).__init__(recurrent, hidden_size, hidden_size)
        num_chan = int(n_chan)
        num_actions = num_actions
        self.map_width = map_width
        init_ = lambda m: init(m,
            nn.init.dirac_,
            lambda x: nn.init.constant_(x, 0.1),
            nn.init.calculate_gain('relu'))
        self.embed = init_(nn.Conv2d(num_inputs, num_chan, 1, 1, 0))
        self.k5 = init_(nn.Conv2d(num_chan, num_chan, 5, 1, 2))
        self.k3 = init_(nn.Conv2d(num_chan, num_chan, 3, 1, 1))
        self.val_shrink = init_(nn.Conv2d(num_chan, num_chan, val_kern, 2, 1))
        init_ = lambda m: init(m,
            nn.init.dirac_,
            lambda x: nn.init.constant_(x, 0))
        self.val = init_(nn.Conv2d(num_chan, 1, 3, 1, 1))
        self.act = init_(nn.Conv2d(num_chan, num_actions, 1, 1, 0))

    def forward(self, x, rhxs=None, masks=None):

        x = F.relu(self.embed(x))
        x = F.relu(self.k5(x))
        x = F.relu(self.k3(x))
       #x = self.act_soft(self.k3(x))
        act = self.act(x)
       #assert (act > 0).all
        for i in range(int(math.log(self.map_width, 2))):
            x = F.relu(self.val_shrink(x))
        val = self.val(x)
       #assert (act > 0).all
        return val.view(val.shape[0], -1), act, rhxs

class FullyConv_linVal(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=256,
            map_width=20, num_actions=1, in_w=None, in_h=None, out_w=None, out_h=None, n_chan=32, prebuild=None):
        super(FullyConv_linVal, self).__init__(recurrent, hidden_size, hidden_size)
        num_chan = n_chan
        num_actions = num_actions
        self.map_width = map_width
        init_ = lambda m: init(m,
            nn.init.dirac_,
            lambda x: nn.init.constant_(x, 0.1),
            nn.init.calculate_gain('relu'))
        self.embed = init_(nn.Conv2d(num_inputs, num_chan, 1, 1, 0))
        self.k5 = init_(nn.Conv2d(num_chan, num_chan, 5, 1, 2))
        self.k3 = init_(nn.Conv2d(num_chan, num_chan, 3, 1, 1))
        state_size = map_width * map_width * num_chan
        linit_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0))
        self.dense = linit_(nn.Linear(state_size, hidden_size))
        self.val = linit_(nn.Linear(hidden_size, 1))
        init_ = lambda m: init(m,
            nn.init.dirac_,
            lambda x: nn.init.constant_(x, 0))
        self.act = init_(nn.Conv2d(num_chan, num_actions, 1, 1, 0))

    def forward(self, x, rhxs=None, masks=None):

        x = F.relu(self.embed(x))
        x = F.relu(self.k5(x))
        x = F.relu(self.k3(x))
        act = self.act(x)
        x_lin = torch.tanh(self.dense(x.view(x.shape[0], -1)))
        val = self.val(x_lin)
        return val.view(val.shape[0], -1), act, rhxs

class FullyConvLSTM(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=256,
            in_w=16, in_h=16, out_w=16, out_h=16, num_actions=18, prebuild=None,
            map_width=16, val_kern=3, n_chan=64):
        self.n_chan=n_chan
        self.hidden_size = (self.n_chan, map_width, map_width)
        super(FullyConvLSTM, self).__init__(recurrent, hidden_size, hidden_size)
        num_actions = num_actions
        self.map_width = map_width
        init_ = lambda m: init(m,
            nn.init.dirac_,
            lambda x: nn.init.constant_(x, 0.1),
            nn.init.calculate_gain('relu'))
        self.embed = init_(nn.Conv2d(num_inputs, self.n_chan, 1, 1, 0))
        self.k5 = init_(nn.Conv2d(self.n_chan, self.n_chan, 5, 1, 2))
        self.k3 = init_(ConvLSTMCell(self.n_chan, self.n_chan))
        self.val_shrink = init_(nn.Conv2d(self.n_chan, self.n_chan, 2, 2, 0))
       #state_size = map_width * map_width * self.n_chan
       #init_ = lambda m: init(m,
       #    nn.init.orthogonal_,
       #    lambda x: nn.init.constant_(x, 0))
       #self.dense = init_(nn.Linear(state_size, 256))
       #self.val = init_(nn.Linear(128, 1))
        self.val = init_(nn.Conv2d(self.n_chan, 1, 3, 1, 1))
        init_ = lambda m: init(m,
            nn.init.dirac_,
            lambda x: nn.init.constant_(x, 0))
        self.act = init_(nn.Conv2d(self.n_chan, num_actions, 1, 1, 0))

    def forward(self, x, rhxs=None, masks=None):

        x = F.relu(self.embed(x))
        x = F.relu(self.k5(x))
       #TODO: problem here when evaluating ConvLSTM
        x, rhxs = self.k3(x, rhxs)
        x = F.relu(x)
       #x_lin = torch.tanh(self.dense(x.view(x.shape[0], -1)))
       #val = self.val(x_lin)
        act = self.act(x)
        for i in range(int(math.log(self.map_width, 2))):
            x = F.relu(self.val_shrink(x))
       #val = x
        val = self.val(x)
        return val.view(val.shape[0], -1), act, rhxs

    def get_recurrent_state_size(self):
        return(self.n_chan, self.map_width, self.map_width)


class MicropolisBase_FullyConvRec(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=256,
            map_width=20, num_actions=18):

        super(MicropolisBase_FullyConvRec, self).__init__(recurrent, hidden_size, hidden_size)
        num_actions = num_actions
        self.map_width = map_width
        init_ = lambda m: init(m,
            nn.init.dirac_,
            lambda x: nn.init.constant_(x, 0.1),
            nn.init.calculate_gain('relu'))

        self.embed = init_(nn.Conv2d(num_inputs, 32, 1, 1, 0))
        self.k5 = init_(nn.Conv2d(32, 32, 5, 1, 2))
        self.k3 = init_(nn.Conv2d(32, 32, 3, 1, 1))
        self.val_shrink = init_(nn.Conv2d(32, 32, 2, 2, 0))
       #state_size = map_width * map_width * 32

       #init_ = lambda m: init(m,
       #    nn.init.orthogonal_,
       #    lambda x: nn.init.constant_(x, 0))

       #self.dense = init_(nn.Linear(state_size, 256))
       #self.val = init_(nn.Linear(128, 1))
        self.val = init_(nn.Conv2d(32, 1, 3, 1, 1))

        init_ = lambda m: init(m,
            nn.init.dirac_,
            lambda x: nn.init.constant_(x, 0))

        self.act = init_(nn.Conv2d(32, num_actions, 1, 1, 0))

    def forward(self, x, rhxs=None, masks=None):

        x = F.relu(self.embed(x))
        x = F.relu(self.k5(x))
        for i in range(10):
            x = F.relu(self.k3(x))
       #x_lin = torch.tanh(self.dense(x.view(x.shape[0], -1)))
       #val = self.val(x_lin)
        act = self.act(x)
        for i in range(int(math.log(self.map_width, 2))):
            x = F.relu(self.val_shrink(x))
       #val = x
        val = self.val(x)

        return val.view(val.shape[0], -1), act, rhxs


class FractalNet(NNBase):
    def __init__(self,num_inputs, recurrent=False, hidden_size=512,
                 map_width=16, n_conv_recs=2, n_recs=1,
                 intra_shr=False, inter_shr=False,
                 num_actions=19, rule='extend',
                 in_w=1, in_h=1, out_w=1, out_h=1, n_chan=64, prebuild=None,
                 val_kern=3):
        super(FractalNet, self).__init__(recurrent, hidden_size, hidden_size)
        self.map_width = map_width
       #self.bn = nn.BatchNorm2d(num_inputs)
        # We can stack multiple Fractal Blocks
       #self.block_chans = block_chans = [32, 32, 16]
        self.block_chans = block_chans = [n_chan]
        self.num_blocks = num_blocks = len(block_chans)
        self.conv_init_ = init_ = lambda m: init(m,
            nn.init.dirac_,
            lambda x: nn.init.constant_(x, 0.1),
            nn.init.calculate_gain('relu'))
        for i in range(num_blocks):
            setattr(self, 'block_{}'.format(i),
                    FractalBlock(n_chan_in=block_chans[i-1], n_chan=block_chans[i],
                                 num_inputs=num_inputs, intra_shr=intra_shr,
                                 inter_shr=inter_shr, recurrent=recurrent,
                                 n_recs=n_recs,
                                 num_actions=num_actions, rule=rule, base=self))
        # An assumption. Run drop path globally on all blocks of stack if applicable
        self.n_cols = self.block_0.n_cols

        n_out_chan = block_chans[-1]
        self.critic_dwn = init_(nn.Conv2d(n_out_chan, n_out_chan, val_kern, 2, 1))
        init_ = lambda m: init(m,
            nn.init.dirac_,
            lambda x: nn.init.constant_(x, 0))
        self.critic_out = init_(nn.Conv2d(n_out_chan, 1, 3, 1, 1))
        self.actor_out = init_(nn.Conv2d(n_out_chan, num_actions, 3, 1, 1))
        self.active_column = None

    def auto_expand(self):
        self.block_0.auto_expand() # assumption
        self.n_cols += 1

    def forward(self, x, rnn_hxs=None, masks=None):
       #x = self.bn(x)
        for i in range(self.num_blocks):
            block = getattr(self, 'block_{}'.format(i))
            x = F.relu(block(x, rnn_hxs, masks))
        actions = self.actor_out(x)
        values = x
        for i in range(int(math.log(self.map_width, 2))):
            values = F.relu(self.critic_dwn(values))
        values = self.critic_out(values)
        values = values.view(values.size(0), -1)
        return values, actions, rnn_hxs # no recurrent states

    def set_drop_path(self):
        for i in range(self.num_blocks):
            getattr(self, 'block_{}'.format(i)).set_drop_path()

    def set_active_column(self, a):
        self.active_column = a
        for i in range(self.num_blocks):
            getattr(self, 'block_{}'.format(i)).set_active_column(a)


class FractalBlock(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=512,
                 map_width=16, n_recs=5, intra_shr=False,
                 inter_shr=False, num_actions=19, rule='extend', n_chan=32,
                 n_chan_in=32, base=None):

        super(FractalBlock, self).__init__(
                recurrent, hidden_size, hidden_size)

        self.map_width = map_width
        self.n_chan = n_chan
        self.intracol_share = intra_shr # share weights between layers in a col.
        self.intercol_share = inter_shr # share weights between columns
        self.rule = rule # which fractal expansion rule to use
        # each rec is a call to a subfractal constructor, 1 rec = single-layered body
        self.n_recs = n_recs
        print("Fractal Block: expansion type: {}, {} recursions".format(
            self.rule, self.n_recs))

        self.SKIPSQUEEZE = rule == 'wide1' # actually we mean a fractal rule that grows linearly in max depth but exponentially in number of columns, rather than vice versa, with number of recursions #TODO: combine the two rules
        if self.rule == 'wide1':
            self.n_cols = 2 ** (self.n_recs - 1)
            print('{} cols'.format(self.n_cols))
        else:
            self.n_cols = self.n_recs
        self.COLUMNS = False # if true, we do not construct the network recursively, but as a row of concurrent columns
        # if true, weights are shared between recursions
        self.local_drop = False
        # at each join, which columns are taken as input (local drop as described in Fractal Net paper)
        self.global_drop = False
        self.active_column = None
        self.batch_norm = False
        self.c_init_ = init_ = lambda m: init(m,
            nn.init.dirac_,
            lambda x: nn.init.constant_(x, 0.1),
            nn.init.calculate_gain('relu'))
        self.embed_chan = nn.Conv2d(num_inputs, n_chan, 1, 1, 0)
        # TODO: right now, we initialize these only as placeholders to successfully load older models, get rid of these ASAP
        if False and self.intracol_share:
            # how many columns with distinct sets of layers?
            if self.intercol_share:
                n_unique_cols = 1
            else:
                n_unique_cols = self.n_recs
            for i in range(n_unique_cols):
                if self.intracol_share:
                    n_unique_layers = 1
                else:
                    n_unique_layers = 3
                setattr(self, 'fixed_{}'.format(i), init_(nn.Conv2d(
                    self.n_chan, self.n_chan, 3, 1, 1)))
                if n_unique_cols == 1 or i > 0:

                    setattr(self, 'join_{}'.format(i), init_(nn.Conv2d(
                        self.n_chan * 2, self.n_chan, 3, 1, 1)))
                    if self.rule == 'wide1' or self.rule == 'extend_sqz':
                        setattr(self, 'dwn_{}'.format(i), init_(nn.Conv2d(
                            self.n_chan, self.n_chan, 2, 2, 0)))
                        setattr(self, 'up_{}'.format(i), init_(nn.ConvTranspose2d(
                            self.n_chan, self.n_chan, 2, 2, 0)))
        f_c = None
        if self.rule == 'wide1':
            subfractal = SkipFractal
        elif self.rule == 'extend':
            if self.rule == 'extend_sqz':
                subfractal = SubFractal_squeeze
            else:
                subfractal = SubFractal
        n_recs = self.n_recs
        for i in range(n_recs):
            f_c = subfractal(self, f_c, n_rec=i, n_chan=self.n_chan)
        self.f_c = f_c
        self.subfractal = subfractal
        self.join_masks = self.f_c.join_masks


    def auto_expand(self):
        ''' Apply a fractal expansion without introducing new weight layers.
        For neuroevolution or inference.'''
        self.intracol_share = False
        self.f_c = self.subfractal(self, self.f_c, n_rec=self.n_recs, n_chan=self.n_chan)
        setattr(self, 'fixed_{}'.format(self.n_recs), None)
        self.f_c.copy_child_weights()
        self.f_c.fixed = copy.deepcopy(self.f_c.fixed)
        self.n_recs += 1
        self.n_cols += 1
        self.f_c.auto_expand()


    def forward(self, x, rnn_hxs=None, masks=None):
        x = self.embed_chan(x)
        depth = pow(2, self.n_recs - 1)
            # (column, join depth)
        if self.rule == 'wide1':
            net_coords = (0, self.n_recs - 1)
        else:
            net_coords = (self.n_recs - 1, depth - 1 )
        x = F.relu(self.f_c(x))
        return x

    def clear_join_masks(self):
        ''' Returns a set of join masks that will result in activation flowing
        through the entire fractal network.'''
        if self.rule == 'wide1':
            self.join_masks.fill(1)
            return
        i = 0
        for mask in self.join_masks:
            n_ins = len(mask)
            mask = [1]*n_ins
            self.join_masks[i] = mask
            i += 1

    def set_active_column(self, a):
        ''' Returns a set of join masks that will result in activation flowing
        through a (set of) sequential 'column(s)' of the network.
        - a: an integer, or list of integers, in which case multiple sequential
            columns are activated.'''
        self.global_drop = True
        self.local_drop = False
        if a == -1:
            self.f_c.reset_join_masks(True)
        else:
            self.f_c.reset_join_masks(False)
            self.f_c.set_active_column(a)
       #print('set active col to {}\n{}'.format(a, self.f_c.get_join_masks()))


    def set_local_drop(self):
        self.global_drop = False
        self.active_column = None
        reach = False # whether or not there is a path thru
        reach = self.f_c.set_local_drop(force=True)
       #print('local_drop\n {}'.format(self.get_join_masks()))
        assert reach


    def set_global_drop(self):
        a = np.random.randint(0, self.n_recs)
        self.set_active_column(a)

    def set_drop_path(self):
        if np.random.randint(0, 2) == 1:
            self.local_drop = self.set_local_drop()
        else:
            self.global_drop = self.set_global_drop()

    def get_join_masks(self):
        return self.f_c.get_join_masks()



class SubFractal(nn.Module):
    '''
    The recursive part of the network.
    '''
    def __init__(self, root, f_c, n_rec, n_chan):
        super(SubFractal, self).__init__()
        self.n_recs = root.n_recs
        self.n_rec = n_rec
        self.n_chan = n_chan
        self.join_layer = False
        init_ = root.c_init_
        if f_c is not None:
            self.f_c_A = f_c
            if root.intercol_share:
                self.copy_child_weights()
            self.f_c_B = f_c.mutate_copy(root)
            self.join_masks = {'body': True, 'skip': True}
        else:
            self.join_masks = {'body': False, 'skip': True}
        self.active_column = root.active_column
        if (not root.intercol_share) or self.n_rec == 0:
            self.fixed = init_(nn.Conv2d(self.n_chan, self.n_chan, 3, 1, 1))
            if self.join_layer and n_rec > 0:
                self.join = init_(nn.Conv2d(self.n_chan * 2, self.n_chan, 3, 1, 1))

                #if self.join_layer and n_rec > 0:
               #    self.join = getattr(root, 'join_{}'.format(j))

    def auto_expand(self):
        '''just increment n_recs'''
        self.n_recs += 1

    def mutate_copy(self, root):
        ''' Return a copy of myself to be used as my twin.'''
        if self.n_rec > 0:
            f_c = self.f_c_A.mutate_copy(root)
            twin = SubFractal(root, f_c, self.n_rec, n_chan=self.n_chan)
        else:
            twin = SubFractal(root, None, 0, n_chan=self.n_chan)
        if root.intracol_share:
            twin.fixed = self.fixed
        return twin


    def copy_child_weights(self):
        ''' Steal our child's weights to use as our own. Not deep (just refers to existing weights).'''
        if self.n_rec > 0:
            self.fixed = self.f_c_A.fixed
            if self.join_layer:
                self.join = self.f_c_A.join



    def reset_join_masks(self, val=True):
        self.join_masks['skip'] = val
        if self.n_rec > 0:
            self.join_masks['body'] = val
            self.f_c_A.reset_join_masks(val)
            self.f_c_B.reset_join_masks(val)
        else:
            self.join_masks['body'] = False # not needed


    def set_local_drop(self, force):
        ''' Returns True if path from source to target is yielded to self.join_masks.
                - force: a boolean, whether or not to force one path through.'''
        reach = False
        if self.n_rec == 0:
            self.set_child_drops(False, [0, 1])
            reach = True
        else:
            # try for natural path to target
            prob_body = 1 - (1/2) ** self.n_rec
            prob_skip = 1/2
            mask = (np.random.random_sample(2) > [prob_body, prob_skip]).astype(int)
            reach = self.set_child_drops(False, mask)
            if not reach and force: # then force one path down
                mask[1] = np.random.randint(0, 1) <= 1 / (self.n_recs - self.n_rec)
                mask[0] = (mask[1] + 1) % 2
                assert self.set_child_drops(True, mask) == True
                reach = True
        return reach


    def set_child_drops(self, force, mask):
        reach = False
        if force:
            assert 1 in mask
        if mask[1] == 1:
            self.join_masks['skip'] = True
            reach = True
        else:
            self.join_masks['skip'] = False
        self.join_masks['body'] = False
        if mask[0] == 1:
            reach_a = self.f_c_A.set_local_drop(force)
            if reach_a:
                reach_b = self.f_c_B.set_local_drop(force)
                if reach_b:
                    self.join_masks['body'] = True
                    reach = True
            else:
                assert not force
        if force:
            assert reach
        return reach


    def set_active_column(self, col_n):
        if col_n == self.n_rec:
            self.join_masks['skip'] = True
            self.join_masks['body'] = False
        else:
            self.join_masks['skip'] = False
            self.join_masks['body'] = True
            self.f_c_A.set_active_column(col_n)
            self.f_c_B.set_active_column(col_n)



    def get_join_masks(self):
        ''' for printing! '''
        stri = ''
        indent = ''
        for i in range(self.n_recs - self.n_rec):
            indent += '    '
        stri = stri + indent + str(self.join_masks)
        if self.n_rec != 0:
            stri = stri + '\n' + str(self.f_c_A.get_join_masks()) + '\n' + str(self.f_c_B.get_join_masks())
        return stri



    def forward(self, x):
        if x is None: return None
        x_c, x_c1 = x, x

        if self.join_masks['skip']:
            for i in range(1):
                x_c1 = F.relu(
                        #self.dropout_fixed
                        (self.fixed(x_c1)))
        if self.n_rec == 0:
            return x_c1
        if self.join_masks['body']:
            x_c = self.f_c_A(x_c)
            x_c =self.f_c_B(x_c)
        if x_c1 is None:
            return x_c
        if x_c is None:
            return x_c1
        if self.join_layer:
            x = F.relu(
                    #self.dropout_join
                    (self.join(torch.cat((x_c, x_c1), dim=1))))
        else:
            x = (x_c1 + x_c * (self.n_rec)) / (self.n_rec + 1)
        return x



class SubFractal_squeeze(nn.Module):
    def __init__(self, root, f_c, n_rec, net_coords=None):
        super(SubFractal_squeeze, self).__init__()
        self.map_width = root.map_width
        self.n_rec = n_rec
        root.n_recs += 1
        self.n_chan = root.n_chan
        self.join_masks = root.join_masks
        self.active_column = root.active_column
        self.num_down = min(int(math.log(self.map_width, 2)) - 1, n_rec)
        self.dense_nug = (self.num_down > 1)
        self.join_layer = False
        self.intracol_share = root.intracol_share
        self.init_ = init_ = lambda m: init(m,
            nn.init.dirac_,
            lambda x: nn.init.constant_(x, 0.1),
            nn.init.calculate_gain('relu'))

        if root.intercol_share:
            j = 0
        else:
            j = n_rec
        if self.intracol_share:
           #for i in range(self.num_down):
            for i in range(1):
                setattr(self, 'dwn_{}'.format(i),
                init_(nn.Conv2d(self.n_chan, self.n_chan,
                2, 2, 0)))
                setattr(self, 'up_{}'.format(i),
                    init_(nn.ConvTranspose2d(self.n_chan,
                        self.n_chan, 2, 2, 0)))
            self.fixed = init_(nn.Conv2d(self.n_chan,
                self.n_chan, 3, 1, 1))
        elif not self.intracol_share:
            if n_rec > 0:
                self.up = getattr(root, 'up_{}'.format(j))
                self.dwn = getattr(root, 'dwn_{}'.format(j))
                if self.join_layer:
                    self.join = getattr(root, 'join_{}'.format(j))
      # if self.dense_nug:
      #     squish_width = self.map_width / (2 ** self.num_down)
      #     hidden_size = int(squish_width * squish_width * self.n_chan)
      #     linit_ = lambda m: init(m,
      #         nn.init.orthogonal_,
      #         lambda x: nn.init.constant_(x, 0))
      #     self.dense = linit_(nn.Linear(hidden_size, hidden_size))
            self.fixed = getattr(root, 'fixed_{}'.format(j))
            if root.batch_norm:
                self.bn_join = nn.BatchNorm2d(self.n_chan)
                if self.num_down == 0:
                    setattr(self, 'bn_fixed_{}'.format(0), nn.BatchNorm2d(self.n_chan))
                for i in range(self.num_down):
                    setattr(self, 'bn_dwn_{}'.format(i), nn.BatchNorm2d(self.n_chan))
                    setattr(self, 'bn_fixed_{}'.format(i), nn.BatchNorm2d(self.n_chan))
                    setattr(self, 'bn_up_{}'.format(i),nn.BatchNorm2d(self.n_chan))
        self.f_c_A = f_c
        if f_c is not None:
            self.f_c_B = f_c.mutate_copy(root)
        else:
            self.f_c_B = f_c

    def mutate_copy(self, root):
        ''' '''
        if self.f_c_A is not None:
            f_c = self.f_c_A.mutate_copy(root)
            twin = SubFractal_squeeze(root, f_c, self.n_layer)
            return twin
        else:
            twin = SubFractal_squeeze(root, None, 0)
            ##win.body = nn.Sequential(twin.body, twin.body)
            return twin
    def forward(self, x, net_coords):
        if x is None:
            return x
        x_c, x_c1 = x, x
        col = net_coords[0]
        depth = net_coords[1]
        if self.n_rec > 0:
            x_c = self.f_c_A(x_c, (col - 1, depth - 2 ** (col - 1 )))
            x_c = self.f_c_B(x_c, (col - 1, depth))
            if  self.join_masks[depth][col]:
                for d in range(self.num_down):
                    #bn_dwn = getattr(self, 'bn_dwn_{}'.format(d))
                    dwn = getattr(self, 'dwn_{}'.format(0))
                    x_c1 = F.relu(#bn_dwn
                            (dwn(x_c1)))
                    # if self.dense_nug:
                    #     x_c1_shape = x_c1.shape
                    #     x_c1 = x_c1.view(x_c1.size(0), -1)
                    #     x_c1 = F.tanh(self.dense(x_c1))
                    #     x_c1 = x_c1.view(x_c1_shape)
                for f in range(1):
                    #bn_fixed= getattr(self, 'bn_fixed_{}'.format(f))
                    x_c1 = F.relu(#bn_fixed
                            (self.fixed(x_c1)))
                for u in range(self.num_down):
                    #bn_up = getattr(self, 'bn_up_{}'.format(u))
                    up = getattr(self, 'up_{}'.format(0))
                    x_c1 = F.relu(#bn_up
                            up(x_c1, output_size = (x_c1.size(0), x_c1.size(1),
                                x_c1.size(2) * 2, x_c1.size(3) * 2)))
        if x_c is None or col == 0:
            return x_c1
        if x_c1 is None:
            return x_c
        if self.join_layer:
            x = F.relu(#self.bn_join
                    (self.join(torch.cat((x_c, x_c1), dim=1))))
        else:
            x = (x_c1 + x_c * self.n_rec) / (self.n_rec + 1)
        return x

class SkipFractal(nn.Module):
    ''' Like fractal net, but where the longer columns compress more,
    and the shallowest column not at all.
    -skip_body - layer or sequence of layers, to be passed through Relu here'''
    def __init__(self, root, f_c, n_rec, skip_body=None):
        '''
        - root: The NN module containing the fractal structure. Has all unique layers.
        - f_c: the previous iteration of this fractal
        - n_rec: the depth of this fractal, 0 when base case
        '''
        super(SkipFractal, self).__init__()
        self.intracol_share = root.intracol_share
        self.n_rec = n_rec
        root.n_recs += 1
        root.n_recs += 1
        root.n_col = 2* root.n_col
        self.n_chan = 32
        self.f_c = f_c
        self.active_column = root.active_column
        self.join_masks = root.join_masks
        self.global_drop = root.global_drop

        if not self.intracol_share:
            self.fixed = init_(nn.Conv2D(self.n_chan, self.n_chan,
                3, 1, 1))
            if n_rec > 0:
                self.join = init_(nn.Conv2D(self.n_chan * 2, self.n_chan,
                    3, 1, 1))
                self.up = init_(nn.ConvTranspose2D(self.n_chan, self.n_chan,
                    2, 2, 0))
                self.dwn = init_(nn.ConvTranspose2D(self.n_chan, self.n_chan,
                    2, 2, 0))
        else:
            if root.SHARED:
                j = 0 # default index for shared layers
            else:
                j = n_rec # layer index = recursion index
            if n_rec == 0:
                self.fixed = getattr(root, 'fixed_{}'.format(j))
            if n_rec > 0:
                self.join = getattr(root, 'join_{}'.format(j))
                self.up = getattr(root, 'up_{}'.format(j))
                self.dwn = getattr(root, 'dwn_{}'.format(j))
        if f_c is not None:
            self.skip = f_c.mutate_copy(root)
            self.body = f_c

    def forward(self, x, net_coords=None):
       #print('entering {}'.format(net_coords))
        if x is None:
            return None
        col = net_coords[0]
        depth = net_coords[1]
        x_b, x_a = x, x
        if self.n_rec > 0:
            x_a = self.skip(x_a, (col + 2 ** (self.n_rec - 1), depth - 1))
        else:
            x_a = None
        if self.join_masks[depth][col]:
           #print('including body at: {}'.format(net_coords))
            if self.n_rec > 0:
                x_b = F.relu(self.dwn(x_b))
                x_b = self.body(x_b, (col, depth - 1))
               #print('x_b : \n' + str(x_b))
                if x_b is not None:
                    x_b = F.relu(self.up(x_b))
            else:
                x_b = self.body(x_b)
                return x_b
        else:
           #print('excluding body at: {}'.format(net_coords))
           #print(x_a, x_b)
            x_b = None
        if x_a is None:
            return x_b
        if x_b is None:
            return x_a
        x = F.relu(self.join(torch.cat((x_a, x_b), dim=1)))
       #x = x_a + x_b
        return x

    def mutate_copy(self, root):
        ''' In the skip-squeeze fractal, the previous iteration is duplicated and run in parallel.
        The left twin is to be sandwhiched between two new compressing/decompressing layers.
        This function creates the right twin and mutates it, recursively,
        replacing every application of the 'fixed' layer with two in sequence.
        - root: the fractal's owner
        '''
        if self.f_c is not None:
            f_c = self.f_c.mutate_copy(root)
            twin = SkipFractal(root, f_c, self.n_rec)
            return twin
        else:
            twin = SkipFractal(root, None, 0)
           ##win.body = nn.Sequential(twin.body, twin.body)
            return twin


class MicropolisBase_fixed(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=512, map_width=20, num_actions=19):
        super(MicropolisBase_fixed, self).__init__(recurrent, hidden_size, hidden_size)

        self.map_width = map_width
        self.RAND = False
        self.eval_recs = [1] + [i * 8 for i in range(1, int(map_width * 2 / 8) + 1)]

        self.num_recursions = map_width

        init_ = lambda m: init(m,
            nn.init.dirac_,
            lambda x: nn.init.constant_(x, 0.1),
            nn.init.calculate_gain('relu'))

        self.skip_compress = init_(nn.Conv2d(num_inputs, 15, 1, stride=1))

        self.conv_0 = init_(nn.Conv2d(num_inputs, 64, 1, 1, 0))
       #self.conv_1 = init_(nn.Conv2d(64, 64, 5, 1, 2))
        for i in range(1):
            setattr(self, 'conv_2_{}'.format(i), init_(nn.Conv2d(64, 64, 3, 1, 1)))
        self.critic_compress = init_(nn.Conv2d(79, 64, 3, 1, 1))
        for i in range(1):
            setattr(self, 'critic_downsize_{}'.format(i), init_(nn.Conv2d(64, 64, 2, 2, 0)))


        init_ = lambda m: init(m,
            nn.init.dirac_,
            lambda x: nn.init.constant_(x, 0))

        self.actor_compress = init_(nn.Conv2d(79, num_actions, 3, 1, 1))
        self.critic_conv = init_(nn.Conv2d(64, 1, 1, 1, 0))
        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs
        x = F.relu(self.conv_0(x))
        skip_input = F.relu(self.skip_compress(inputs))
       #x = F.relu(self.conv_1(x))
        conv_2 = getattr(self, 'conv_2_{}'.format(0))
        for i in range(self.num_recursions):
            x = F.relu(conv_2(x))
        x = torch.cat((x, skip_input), 1)
        values = F.relu(self.critic_compress(x))
        for i in range(4):
            critic_downsize = getattr(self, 'critic_downsize_{}'.format(0))
            values = F.relu(critic_downsize(values))
        values = self.critic_conv(values)
        values = values.view(values.size(0), -1)
        actions = self.actor_compress(x)

        return values, actions, rnn_hxs

class MicropolisBase_squeeze(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=512, map_width=20, num_actions=19):
        super(MicropolisBase_squeeze, self).__init__(recurrent, hidden_size, hidden_size)
        self.chunk_size = 2 # factor by which map dimensions are shrunk
        self.map_width = map_width
       #self.num_maps = 4
        self.num_maps = int(math.log(self.map_width, self.chunk_size)) - 1 # how many different sizes

        init_ = lambda m: init(m,
            nn.init.dirac_,
            lambda x: nn.init.constant_(x, -10),
            nn.init.calculate_gain('relu'))

        self.cmp_in = init_(nn.Conv2d(num_inputs, 64, 1, stride=1, padding=0))
        for i in range(self.num_maps):
            setattr(self, 'prj_life_obs_{}'.format(i), init_(nn.Conv2d(64, 64, 3, stride=1, padding=1)))
            setattr(self, 'cmp_life_obs_{}'.format(i), init_(nn.Conv2d(128, 64, 3, stride=1, padding=1)))
       #self.shrink_life = init_(nn.Conv2d(64, 64, 3, 3, 0))

       #self.conv_1 = init_(nn.Conv2d(64, 64, 3, 1, 1))
       #self.lin_0 = linit_(nn.Linear(1024, 1024))

        for i in range(self.num_maps):
            if i == 0:
                setattr(self, 'dwn_{}'.format(i), init_(nn.Conv2d(64, 64, 2, stride=2, padding=0)))
            setattr(self, 'expand_life_{}'.format(i), init_(nn.ConvTranspose2d(64 + 64, 64, 2, stride=2, padding=0)))
            setattr(self, 'prj_life_act_{}'.format(i), init_(nn.Conv2d(64, 64, 3, stride=1, padding=1)))
            setattr(self, 'cmp_life_act_{}'.format(i), init_(nn.Conv2d(128, 64, 3, stride=1, padding=1)))
            setattr(self, 'cmp_life_val_in_{}'.format(i), init_(nn.Conv2d(128, 64, 3, stride=1, padding=1)))
            setattr(self, 'dwn_val_{}'.format(i), init_(nn.Conv2d(64, 64, 2, stride=2, padding=0)))
            if i == self.num_maps - 1:
                setattr(self, 'prj_life_val_{}'.format(i), init_(nn.Conv2d(64, 64, 3, stride=1, padding=1)))
            else:
                setattr(self, 'prj_life_val_{}'.format(i), init_(nn.Conv2d(64, 64, 3, stride=1, padding=1)))

        self.cmp_act = init_(nn.Conv2d(128, 64, 3, stride=1, padding=1))


        init_ = lambda m: init(m,
            nn.init.dirac_,
            lambda x: nn.init.constant_(x, 0.1))

        self.act_tomap = init_(nn.Conv2d(64, 19, 5, stride=1, padding=2))
        self.cmp_val_out = init_(nn.Conv2d(64, 1, 1, stride=1, padding=0))
        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs
        x = F.relu(self.cmp_in(x))
        x_obs = []
        for i in range(self.num_maps): # shrink if not first, then run life sim
            if i != 0:
                shrink_life = getattr(self, 'dwn_{}'.format(0))
                x = F.relu(shrink_life(x))
            x_0 = x
            if i > -1:
                prj_life_obs = getattr(self, 'prj_life_obs_{}'.format(i))
                for c in range(3):
                    x = F.relu(prj_life_obs(x))
                x = torch.cat((x, x_0), 1)
                cmp_life_obs = getattr(self,'cmp_life_obs_{}'.format(i))
                x = F.relu(cmp_life_obs(x))
            x_obs += [x]


        for j in range(self.num_maps): # run life sim, then expand if not last
            if j != 0:
                x_i = x_obs[self.num_maps-1-j]
                x = torch.cat((x, x_i), 1)
                cmp_life_act = getattr(self, 'cmp_life_act_{}'.format(j))
                x =  F.relu(cmp_life_act(x))
            x_0 = x
            if j > -1:
                prj_life_act = getattr(self, 'prj_life_act_{}'.format(j))
                for c in range(1):
                    x = F.relu(prj_life_act(x))
                x = torch.cat((x, x_0), 1)
            if j < self.num_maps - 1:
                expand_life = getattr(self, 'expand_life_{}'.format(j))
                x = F.relu(expand_life(x))
        x = F.relu(self.cmp_act(x))
        acts = F.relu(self.act_tomap(x))

        for i in range(self.num_maps):
            dwn_val = getattr(self, 'dwn_val_{}'.format(0))
            prj_life_val = getattr(self, 'prj_life_val_{}'.format(i))
            cmp_life_val_in = getattr(self, 'cmp_life_val_in_{}'.format(i))
            x_i = x_obs[i]
            x = torch.cat((x, x_i), 1)
            x = F.relu(cmp_life_val_in(x))
            x = F.relu(dwn_val(x))
            x = F.relu(prj_life_val(x))
        vals = self.cmp_val_out(x)
        vals = vals.view(vals.size(0), -1)
        return  vals, acts, rnn_hxs

class MicropolisBase_ICM(MicropolisBase_fixed):
    def __init__(self, num_inputs, recurrent=False, hidden_size=512, num_actions=19):
        super(MicropolisBase_ICM, self).__init__(num_inputs, recurrent, hidden_size)

        ### ICM feature encoder

        init_ = lambda m: init(m,
            nn.init.dirac_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain('relu'))

        num_skip_inputs=15
        self.num_action_channels=19

        self.icm_state_in = init_(nn.Conv2d(num_inputs, 64, 3, 1, 1))
        self.icm_state_conv_0 = init_(nn.Conv2d(64, 64, 3, 1, 1))
        self.icm_state_out = init_(nn.Conv2d(64, 64, 3, 1, 1))

        self.icm_pred_a_in = init_(nn.Conv2d((num_inputs) * 2, 128, 3, 1, 1))
        self.icm_pred_a_conv_0 = init_(nn.Conv2d(128, 128, 3, 1, 1))

        self.icm_pred_s_in = init_(nn.Conv2d((num_inputs) + self.num_action_channels, 64, 1, 1, 0))
        self.icm_pred_s_conv_0 = init_(nn.Conv2d(64, 64, 3, 1, 1))
        self.icm_pred_s_conv_1 = init_(nn.Conv2d(64, 64, 3, 1, 1))
        self.icm_pred_s_conv_2 = init_(nn.Conv2d(64, 64, 3, 1, 1))
        self.icm_pred_s_conv_3 = init_(nn.Conv2d(64, 64, 3, 1, 1))
        self.icm_pred_s_conv_4 = init_(nn.Conv2d(64, 64, 3, 1, 1))
        self.icm_pred_s_conv_5 = init_(nn.Conv2d(64, 64, 3, 1, 1))
        self.icm_pred_s_conv_6 = init_(nn.Conv2d(64, 64, 3, 1, 1))
        self.icm_pred_s_conv_7 = init_(nn.Conv2d(64, 64, 3, 1, 1))
        self.icm_pred_s_conv_8 = init_(nn.Conv2d(64, 64, 3, 1, 1))
        self.icm_pred_s_conv_9 = init_(nn.Conv2d(64, 64, 3, 1, 1))
        self.icm_pred_s_conv_10 = init_(nn.Conv2d(64, 64, 3, 1, 1))

       #self.icm_skip_compress = init_(nn.Conv2d(num_inputs, 15, 1, stride=1))



        init_ = lambda m: init(m,
            nn.init.dirac_,
            lambda x: nn.init.constant_(x, 0))

        self.icm_pred_a_out = init_(nn.Conv2d(128, self.num_action_channels, 7, 1, 3))
        self.icm_pred_s_out = init_(nn.Conv2d(64 + 64, num_inputs, 1, 1, 0))

        self.train()

    def forward(self, inputs, rnn_hxs, masks, icm=False):
        if icm == False:
            return super().forward(inputs, rnn_hxs, masks)

        else:

            # Encode state feature-maps

            s0_in, s1_in, a1 = inputs
            a1 = a1.view(a1.size(0), self.num_action_channels, 20, 20)

            s0 = s0_in
          # s0 = F.relu(self.icm_state_in(s0))
          # for i in range(1):
          #     s0 = F.relu(self.icm_state_conv_0(s0))
          # s0 = F.relu(self.icm_state_out(s0))
          ##s0_skip = F.relu(self.icm_skip_compress(s0))

            s1 = s1_in
          # s1 = F.relu(self.icm_state_in(s1))
          # for i in range(1):
          #     s1 = F.relu(self.icm_state_conv_0(s1))
          # s1 = F.relu(self.icm_state_out(s1))
          ##s1_skip = F.relu(self.icm_skip_compress(s1_in))

            # Predict outcome state feature-map and action dist.
            if True:
                a1 = a1.cuda()
                s0 = s0.cuda()

            pred_s1 = pred_s1_0 = F.relu(self.icm_pred_s_in(torch.cat((s0, a1), 1)))
            for i in range(2):
                pred_s1 = F.relu(self.icm_pred_s_conv_0(pred_s1))
            for i in range(2):
                pred_s1 = F.relu(self.icm_pred_s_conv_1(pred_s1))
            for i in range(2):
                pred_s1 = F.relu(self.icm_pred_s_conv_2(pred_s1))
            for i in range(2):
                pred_s1 = F.relu(self.icm_pred_s_conv_3(pred_s1))
            for i in range(2):
                pred_s1 = F.relu(self.icm_pred_s_conv_4(pred_s1))
            for i in range(2):
                pred_s1 = F.relu(self.icm_pred_s_conv_5(pred_s1))
            for i in range(2):
                pred_s1 = F.relu(self.icm_pred_s_conv_6(pred_s1))
            for i in range(2):
                pred_s1 = F.relu(self.icm_pred_s_conv_7(pred_s1))
            for i in range(2):
                pred_s1 = F.relu(self.icm_pred_s_conv_8(pred_s1))
            for i in range(2):
                pred_s1 = F.relu(self.icm_pred_s_conv_9(pred_s1))
            for i in range(2):
                pred_s1 = F.relu(self.icm_pred_s_conv_10(pred_s1))

            pred_s1 = torch.cat((pred_s1, pred_s1_0), 1)
            pred_s1 = self.icm_pred_s_out(pred_s1)

            pred_a = F.relu(self.icm_pred_a_in(torch.cat((s0, s1), 1)))
            for i in range(1):
                pred_a = F.relu(self.icm_pred_a_conv_0(pred_a))
            pred_a = self.icm_pred_a_out(pred_a)
            pred_a = pred_a.view(pred_a.size(0), -1)

            return s1, pred_s1, pred_a

    def feature_state_size(self):
        return (32, 20, 20)


class MicropolisBase_acktr(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=512, num_actions=19):
        super(MicropolisBase, self).__init__(recurrent, hidden_size, hidden_size)

        init_ = lambda m: init(m,
            nn.init.dirac_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain('relu'))

        import sys

       #self.skip_compress = init_(nn.Conv2d(num_inputs, 15, 1, stride=1))

        self.conv_0 = nn.Conv2d(num_inputs, 64, 1, 1, 0)
        init_(self.conv_0)
        self.conv_1 = nn.Conv2d(64, 64, 5, 1, 2)
        init_(self.conv_1)
       #self.conv_2 = nn.Conv2d(64, 64, 3, 1, 0)
       #init_(self.conv_2)
       #self.conv_3 = nn.ConvTranspose2d(64, 64, 3, 1, 0)
       #init_(self.conv_3)
        self.actor_compress = init_(nn.Conv2d(64, 20, 3, 1, 1))

        self.critic_compress = init_(nn.Conv2d(64, 8, 1, 1, 1))

        init_ = lambda m: init(m,
            nn.init.dirac_,
            lambda x: nn.init.constant_(x, 0))

        self.critic_conv_1 = init_(nn.Conv2d(8, 1, 20, 20, 0))
        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs
        x = F.relu(self.conv_0(x))
       #skip_input = F.relu(self.skip_compress(inputs))
        x = F.relu(self.conv_1(x))
       #for i in range(5):
       #    x = F.relu(self.conv_2(x))
       #for j in range(5):
       #    x = F.relu(self.conv_3(x))
       #x = torch.cat((x, skip_input), 1)
        values = F.relu(self.critic_compress(x))
        values = self.critic_conv_1(values)
        values = values.view(values.size(0), -1)
        actions = F.relu(self.actor_compress(x))

        return values, actions, rnn_hxs


class MicropolisBase_1d(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=512, num_actions=19):
        super(MicropolisBase, self).__init__(recurrent, hidden_size, hidden_size)

        init_ = lambda m: init(m,
            nn.init.dirac_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain('relu'))

        import sys

        self.skip_compress = init_(nn.Conv2d(num_inputs, 15, 1, stride=1))

        self.conv_0 = nn.Conv2d(num_inputs, 64, 1, 1, 0)
        init_(self.conv_0)
        self.conv_1 = nn.Conv2d(64, 64, 5, 1, 2)
        init_(self.conv_1)
        self.conv_2 = nn.Conv2d(1, 1, 3, 1, 0)
        init_(self.conv_2)
        self.conv_2_chan = nn.ConvTranspose2d(1, 1, (1, 3), 1, 0)
        init_(self.conv_2_chan)
        self.conv_3 = nn.ConvTranspose2d(1, 1, 3, 1, 0)
        init_(self.conv_3)
        self.conv_3_chan = nn.Conv2d(1, 1, (1, 3), 1, 0)
        init_(self.conv_3_chan)

        self.actor_compress = init_(nn.Conv2d(79, 20, 3, 1, 1))

        self.critic_compress = init_(nn.Conv2d(79, 8, 1, 1, 1))

        init_ = lambda m: init(m,
            nn.init.dirac_,
            lambda x: nn.init.constant_(x, 0))

        self.critic_conv_1 = init_(nn.Conv2d(8, 1, 20, 20, 0))
        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs
        x = F.relu(self.conv_0(x))
        skip_input = F.relu(self.skip_compress(inputs))
        x = F.relu(self.conv_1(x))
        num_batch = x.size(0)
        for i in range(5):
            w, h = x.size(2), x.size(3)
            num_chan = x.size(1)
            x = x.view(num_batch * num_chan, 1, w, h)
            x = F.relu(self.conv_2(x))
            w, h = x.size(2), x.size(3)
            x = x.view(num_batch, num_chan, w, h)
            x = x.permute(0, 2, 3, 1)
            x = x.view(num_batch, 1, w * h, num_chan)
            x = F.relu(self.conv_2_chan(x))
            num_chan = x.size(3)
            x = x.view(num_batch, num_chan, w, h)
        for j in range(5):
            w, h = x.size(2), x.size(3)
            num_chan = x.size(1)
            x = x.view(num_batch * num_chan, 1, w, h)
            x = F.relu(self.conv_3(x))
            w, h = x.size(2), x.size(3)
            x = x.view(num_batch, num_chan, w, h)
            x = x.permute(0, 2, 3, 1)
            x = x.view(num_batch, 1, w * h, num_chan)
            x = F.relu(self.conv_3_chan(x))
            num_chan = x.size(3)
            x = x.view(num_batch, num_chan, w, h)
        x = torch.cat((x, skip_input), 1)
        values = F.relu(self.critic_compress(x))
        values = self.critic_conv_1(values)
        values = values.view(values.size(0), -1)
        actions = F.relu(self.actor_compress(x))

        return values, actions, rnn_hxs


class MicropolisBase_0(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=512, num_actions=19):
        super(MicropolisBase, self).__init__(recurrent, hidden_size, hidden_size)

        init_ = lambda m: init(m,
            nn.init.dirac_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain('relu'))

        import sys
        if sys.version[0] == '2':
            num_inputs=104
      # assert num_inputs / 4 == 25

        self.conv_A_0 = nn.Conv2d(num_inputs, 64, 5, 1, 2)
        init_(self.conv_A_0)
        self.conv_B_0 = nn.Conv2d(num_inputs, 64, 3, 1, 2)
        init_(self.conv_B_0)

        self.conv_A_1 = nn.Conv2d(64, 64, 5, 1, 2)
        init_(self.conv_A_1)
        self.conv_B_1 = nn.Conv2d(64, 64, 3, 1, 1)
        init_(self.conv_B_1)


        self.input_compress = nn.Conv2d(num_inputs, 15, 1, stride=1)
        init_(self.input_compress)
        self.actor_compress = nn.Conv2d(79, 18, 3, 1, 1)
        init_(self.actor_compress)


        self.critic_compress = init_(nn.Conv2d(79, 8, 1, 1, 0))
      # self.critic_conv_0 = init_(nn.Conv2d(16, 1, 20, 1, 0))

        init_ = lambda m: init(m,
            nn.init.dirac_,
            lambda x: nn.init.constant_(x, 0))

        self.critic_conv_1 = init_(nn.Conv2d(8, 1, 20, 1, 0))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
#       inputs = torch.Tensor(inputs)
#       inputs =inputs.view((1,) + inputs.shape)
        x = inputs
        x_A = self.conv_A_0(x)
        x_A = F.relu(x_A)
        x_B = self.conv_B_0(x)
        x_B = F.relu(x_B)
        for i in range(2):
#           x = torch.cat((x, inputs[:,-26:]), 1)
            x_A = F.relu(self.conv_A_1(x_A))
        for i in range(5):
            x_B = F.relu(self.conv_B_1(x_B))
        x = torch.mul(x_A, x_B)
        skip_input = F.relu(self.input_compress(inputs))
        x = torch.cat ((x, skip_input), 1)
        values = F.relu(self.critic_compress(x))
#       values = F.relu(self.critic_conv_0(values))
        values = self.critic_conv_1(values).view(values.size(0), -1)
        actions = F.relu(self.actor_compress(x))

        return values, actions, rnn_hxs

class CNNBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=512):
        super(CNNBase, self).__init__(recurrent, hidden_size, hidden_size)

        init_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain('relu'))

        self.main = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)),
            nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)),
            nn.ReLU(),
            Flatten(),
            init_(nn.Linear(32 * 7 * 7, hidden_size)),
            nn.ReLU()
        )

        init_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0))

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = self.main(inputs / 255.0)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs


class MicropolisBase_mlp(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=256, map_width=20, num_actions=19):
        super(MicropolisBase_mlp, self).__init__(recurrent, num_inputs, hidden_size)
        num_inputs = map_width * map_width * num_inputs
        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m,
            init_normc_,
            lambda x: nn.init.constant_(x, 0))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)),
            nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            init_(nn.Linear(hidden_size, map_width*map_width*num_actions)),
            nn.Tanh()
        )

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)),
            nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh()
        )

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs
        x=x.view(x.size(0), -1)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs


class MLPBase(NNBase):
    def __init__(self, recurrent=False, hidden_size=64,
                 map_width=16, num_inputs=1, num_actions=1,
                 in_w=1, in_h=1, out_w=1, out_h=1, n_chan=64,
                 prebuild=False, val_kern=None):
        super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)
        self.n_cols = 1
        num_inputs = in_w * in_h * num_inputs
        if recurrent:
            num_inputs = hidden_size
        print('num_inputs: {}, hidden_size: {}'.format(num_inputs, hidden_size))

        init_ = lambda m: init(m,
            init_normc_,
            lambda x: nn.init.constant_(x, 0))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)),
            nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            init_(nn.Linear(hidden_size, out_w * out_h * num_actions)),
            nn.Tanh()
        )

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)),
            nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh()
        )

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        inputs = inputs.float()
        x = inputs
        x=x.view(x.size(0), -1)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs
