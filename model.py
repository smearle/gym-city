import torch
import torch.nn as nn
import torch.nn.functional as F

from distributions import Categorical, Categorical2D
from utils import init, init_normc_
import math


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, base_kwargs=None, curiosity=False, algo='A2C', model='MicropolisBase', args=None):
        super(Policy, self).__init__()
        self.curiosity = curiosity
        self.args = args
        if base_kwargs is None:
            base_kwargs = {}

        if len(obs_shape) == 3:
            if curiosity:
                self.base = MicropolisBase_ICM(obs_shape[0], **base_kwargs)
            elif args.model == 'squeeze':
                self.base = MicropolisBase(obs_shape[0], **base_kwargs, map_width=args.map_width)
            elif args.model == 'mlp':
                self.base = MLPBase(obs_shape[0], **base_kwargs, map_width=args.map_width)
            else:
                self.base = MicropolisBase_fixedmap(obs_shape[0], **base_kwargs, map_width=args.map_width)

        elif len(obs_shape) == 1:
            self.base = MLPBase(obs_shape[0], **base_kwargs)
        else:
            raise NotImplementedError

        if action_space.__class__.__name__ == "Discrete":
            if True:
                num_outputs = action_space.n
                self.dist = Categorical2D(self.base.output_size, num_outputs)
            else:
                num_outputs = action_space.n
                self.dist = Categorical2D(self.base.output_size, num_outputs)

        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            if self.args.env_name == 'MicropolisPaintEnv-v0':
                self.dist = None
            else:
    #           self.dist = DiagGaussian(self.base.output_size, num_outputs)
                self.dist = Categorical2D(self.base.output_size, num_outputs)

        else:
            raise NotImplementedError

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
        action_bin = None
        if 'paint' in self.args.env_name.lower():
            dist = torch.distributions.binomial.Binomial(1, actor_features)
            action = dist.sample()
            action_log_probs = dist.log_prob(action)


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
                action_bin = torch.zeros(dist.probs.shape)
                action_ixs = torch.Tensor(list(range(dist.probs.size(0)))).unsqueeze(1).long()

                action_i = torch.cat((action_ixs.cuda(), action.cuda()), 1)   
                action_bin[action_i[:,0], action_i[:,1]] = 1
                if torch.cuda.current_device() > 0:
                    action_bin = action_bin.cuda()



        return value, action, action_log_probs, action_bin, rnn_hxs

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
            dist = torch.distributions.binomial.Binomial(1, actor_features)
            action_log_probs = dist.log_prob(action)
            dist_entropy = None
           #dist_entropy = (dist.logits * dist.probs).mean()
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

class MicropolisBase_fixedmap(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=512, map_width=20):
        super(MicropolisBase_fixedmap, self).__init__(recurrent, hidden_size, hidden_size)

        self.map_width = map_width

        init_ = lambda m: init(m,
            nn.init.dirac_,
            lambda x: nn.init.constant_(x, 0.0),
            nn.init.calculate_gain('relu'))


        self.skip_compress = init_(nn.Conv2d(num_inputs, 15, 1, stride=1))

        self.conv_0 = nn.Conv2d(num_inputs, 64, 1, 1, 0)
        init_(self.conv_0)
        self.conv_1 = nn.Conv2d(64, 64, 5, 1, 2)
        init_(self.conv_1)
        self.conv_2 = nn.Conv2d(64, 64, 3, 1, 1)
        init_(self.conv_2)
        self.critic_compress = init_(nn.Conv2d(79, 64, 3, 1, 1))
        self.critic_downsize = init_(nn.Conv2d(64, 64, 3, 3, 0))


        init_ = lambda m: init(m,
            nn.init.dirac_,
            lambda x: nn.init.constant_(x, 0))

        self.actor_compress = init_(nn.Conv2d(79, 19, 3, 1, 1))
        self.critic_conv = init_(nn.Conv2d(64, 1, 1, 1, 0))
#       self.critic_conv_2 = init_(nn.Conv2d(1, 1, 2, 1, 0)) # for 40x40 map
        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs
        x = F.relu(self.conv_0(x))
        skip_input = F.relu(self.skip_compress(inputs))
        x = F.relu(self.conv_1(x))
        for i in range(self.map_width):
           #print(self.conv_2.weight)
            x = F.relu(self.conv_2(x))
        x = torch.cat((x, skip_input), 1)
        values = F.relu(self.critic_compress(x))
        for i in range(self.num_maps):
            values = F.relu(self.critic_downsize(values))
        values = self.critic_conv(values)
        values = values.view(values.size(0), -1)
        actions = self.actor_compress(x)

        return values, actions, rnn_hxs

class MicropolisBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=512, map_width=20):
        super(MicropolisBase, self).__init__(recurrent, hidden_size, hidden_size)
        self.chunk_size = 3
        self.map_width = map_width
        self.num_maps = 3
       #self.num_maps = int(math.log(self.map_width, self.chunk_size)) # how many different sizes

        init_ = lambda m: init(m,
            nn.init.dirac_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain('relu'))
        linit_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0))

        self.cmp_in = init_(nn.Conv2d(num_inputs, 64, 1, 1, 0))
        for i in range(self.num_maps):
            setattr(self, 'prj_life_obs_{}'.format(i), init_(nn.Conv2d(64, 64, 3, 1, 1)))
            setattr(self, 'cmp_life_obs_{}'.format(i), init_(nn.Conv2d(128, 64, 3, 1, 1)))
       #self.shrink_life = init_(nn.Conv2d(64, 64, 3, 3, 0))

       #self.conv_1 = init_(nn.Conv2d(64, 64, 3, 1, 1))
       #self.lin_0 = linit_(nn.Linear(1024, 1024))

        for i in range(self.num_maps):
            setattr(self, 'dwn_{}'.format(i), init_(nn.Conv2d(64, 64, 3, 3, 0)))
            setattr(self, 'expand_life_{}'.format(i), init_(nn.ConvTranspose2d(64 + 64, 64, 3, 3, 0)))
            setattr(self, 'prj_life_act_{}'.format(i), init_(nn.Conv2d(64, 64, 3, 1, 1)))
            setattr(self, 'cmp_life_act_{}'.format(i), init_(nn.Conv2d(128, 64, 3, 1, 1)))
            setattr(self, 'cmp_life_val_in_{}'.format(i), init_(nn.Conv2d(128, 64, 3, 1, 1)))
            setattr(self, 'dwn_val_{}'.format(i), init_(nn.Conv2d(64, 64, 3, 3, 0)))
            setattr(self, 'prj_life_val_{}'.format(i), init_(nn.Conv2d(64, 64, 3, 1, 1)))

        self.cmp_act = init_(nn.Conv2d(128, 64, 3, 1, 1))


        init_ = lambda m: init(m,
            nn.init.dirac_,
            lambda x: nn.init.constant_(x, 0.1))

        self.act_tomap = init_(nn.Conv2d(64, 19, 5, 1, 2))
        self.cmp_val_out = init_(nn.Conv2d(64, 1, 1, 1, 0))
        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs
        x = F.relu(self.cmp_in(x))
        x_obs = [] 
        for i in range(self.num_maps): # shrink if not first, then run life sim
            if i != 0:
                shrink_life = getattr(self, 'dwn_{}'.format(i))
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
            dwn_val = getattr(self, 'dwn_val_{}'.format(i))
            prj_life_val = getattr(self, 'prj_life_val_{}'.format(i))
            cmp_life_val_in = getattr(self, 'cmp_life_val_in_{}'.format(i))
            x_i = x_obs[i]
            x = torch.cat((x, x_i), 1)
            x = F.relu(cmp_life_val_in(x))
            x = F.relu(dwn_val(x))
            x = F.relu(prj_life_val(x))
        vals = self.cmp_val_out(x) 
        return vals.view(vals.size(0), -1), acts, rnn_hxs

class MicropolisBase_ICM(MicropolisBase_fixedmap):
    def __init__(self, num_inputs, recurrent=False, hidden_size=512):
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
           #print(a1.is_cuda, s0.is_cuda)

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
    def __init__(self, num_inputs, recurrent=False, hidden_size=512):
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
    def __init__(self, num_inputs, recurrent=False, hidden_size=512):
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
    def __init__(self, num_inputs, recurrent=False, hidden_size=512):
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


class MLPBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=64, map_width=20):
        super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)
        num_inputs = map_width * map_width * 32
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
            init_(nn.Linear(hidden_size, map_width*map_width*19)),
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
