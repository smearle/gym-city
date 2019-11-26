import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])


class RolloutStorage(object):
    def __init__(self, num_steps, num_processes, obs_shape, action_space, recurrent_hidden_state_size, args=None):
        self.args=args
        if 'GoLMulti' in args.env_name: # since num_proc included in obs space
            self.obs = torch.zeros(num_steps + 1, *obs_shape)
        else:
            self.obs = torch.zeros(num_steps + 1, num_processes, *obs_shape)
        if type(recurrent_hidden_state_size) is tuple:
            self.recurrent_hidden_states = torch.zeros(num_steps + 1, 2, num_processes, *recurrent_hidden_state_size)
        else:
            self.recurrent_hidden_states = torch.zeros(num_steps + 1, num_processes, recurrent_hidden_state_size)
       #self.recurrent_hidden_states = [None for i in range(num_steps + 1)]
      # width = args.map_width
      # n_channels = 32 # make this an arg
      # for j in range(args.n_recs):
      #     reps = 2 ** (args.n_recs - j - 1) # number of times column segment repeats
      #     n_squish = min(j, self.num_maps)
      #     for r in range(reps):
      #         if j == 0:
      #             s = torch.cuda.FloatTensor(size=
      #                     (n_batch, n_channels, width, width)).fill_(0.0)
      #         for k in range(n_squish):
      #             width = int(width / 2)
      #             s = torch.cuda.FloatTensor(size=(
      #                     num_process, n_channels, width, width)).fill_(0.0)
      #             for l in range(k + 3):
      #                 rnn_hxs[j] += [(s, s)]
      #         rnn_hxs[j] += [(s, s)]
      #         for k in range(n_squish):
      #             for l in range(n_squish - k + 2):
      #                 width = width
      #                 s = torch.cuda.FloatTensor(size=(
      #                     n_batch, n_channels, width, width)).fill_(0.0)
      #                 rnn_hxs[j] += [(s, s)]
      #             width = int(width * 2)
      #     print([[t[0].shape for t in rnn_hxs[i]] for i in range(len(rnn_hxs))])
      #     rnn_hxs_i = rnn_hxs
      #     rnn_hxs = torch.cat(
      #         [torch.cat([u.squeeze(0) for u in rnn_hxs[v]], dim=0).squeeze(0) for v in len(rnn_hxs)], dim=0)
      #     rnn_hxs_i =

        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_steps, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        if args.env_name == 'MicropolisPaintEnv-v0':
            action_shape = action_space.shape[:]
            self.action_log_probs = torch.zeros(num_steps, num_processes, *action_shape)
            self.actions = torch.zeros(num_steps, num_processes, *action_shape)
        else:
            self.action_log_probs = torch.zeros(num_steps, num_processes, 1)
            if action_space.__class__.__name__ == 'Discrete':
                action_shape = 1
            else:
                action_shape = action_space.shape[0]
            self.actions = torch.zeros(num_steps, num_processes, action_shape)

        if action_space.__class__.__name__ == 'Discrete':
            self.actions = self.actions.long()
        self.masks = torch.ones(num_steps + 1, num_processes, 1)

        self.num_steps = num_steps
        self.step = 0

    def to(self, device):
        self.obs = self.obs.to(device)
        self.recurrent_hidden_states = self.recurrent_hidden_states.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        assert (self.actions >= 0).all()
        self.actions = self.actions.to(device)
        assert (self.actions >= 0).all()
        self.masks = self.masks.to(device)

    def insert(self, obs, recurrent_hidden_states, actions, action_log_probs, value_preds, rewards, masks):
        self.obs[self.step + 1].copy_(obs)
        self.recurrent_hidden_states[self.step + 1] = (recurrent_hidden_states)
        self.actions[self.step].copy_(actions)
        assert (actions >= 0).all()
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)

        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        self.obs[0].copy_(self.obs[-1])
        self.recurrent_hidden_states[0].copy_(self.recurrent_hidden_states[-1])
        self.masks[0].copy_(self.masks[-1])

    def compute_returns(self, next_value, use_gae, gamma, tau):
        if use_gae:
            self.value_preds[-1] = next_value
            gae = 0
            for step in reversed(range(self.rewards.size(0))):
                delta = self.rewards[step] + gamma * self.value_preds[step + 1] * self.masks[step + 1] - self.value_preds[step]
                gae = delta + gamma * tau * self.masks[step + 1] * gae
                self.returns[step] = gae + self.value_preds[step]
        else:
            self.returns[-1] = next_value
            for step in reversed(range(self.rewards.size(0))):
                self.returns[step] = self.returns[step + 1] * \
                    gamma * self.masks[step + 1] + self.rewards[step]


    def feed_forward_generator(self, advantages, num_mini_batch):
        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps
        assert batch_size >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "* number of steps ({}) = {} "
            "to be greater than or equal to the number of PPO mini batches ({})."
            "".format(num_processes, num_steps, num_processes * num_steps, num_mini_batch))
        mini_batch_size = batch_size // num_mini_batch
        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=False)
        for indices in sampler:
            obs_batch = self.obs[:-1].view(-1, *self.obs.size()[2:])[indices]
            recurrent_hidden_states_batch = self.recurrent_hidden_states[:-1].view(-1,
                self.recurrent_hidden_states.size(-1))[indices]
            actions_batch = self.actions.view(-1, self.actions.size(-1))[indices]
            value_preds_batch = self.value_preds[:-1].view(-1, 1)[indices]
            return_batch = self.returns[:-1].view(-1, 1)[indices]
            masks_batch = self.masks[:-1].view(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1, 1)[indices]
            adv_targ = advantages.view(-1, 1)[indices]

            yield obs_batch, recurrent_hidden_states_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ

    def recurrent_generator(self, advantages, num_mini_batch):
        num_processes = self.rewards.size(1)
        assert num_processes >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(num_processes, num_mini_batch))
        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes)
        for start_ind in range(0, num_processes, num_envs_per_batch):
            obs_batch = []
            recurrent_hidden_states_batch = []
            actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]
                obs_batch.append(self.obs[:-1, ind])
                recurrent_hidden_states_batch.append(self.recurrent_hidden_states[0:1, ind])
                actions_batch.append(self.actions[:, ind])
                value_preds_batch.append(self.value_preds[:-1, ind])
                return_batch.append(self.returns[:-1, ind])
                masks_batch.append(self.masks[:-1, ind])
                old_action_log_probs_batch.append(self.action_log_probs[:, ind])
                adv_targ.append(advantages[:, ind])

            T, N = self.num_steps, num_envs_per_batch
            # These are all tensors of size (T, N, -1)
            obs_batch = torch.stack(obs_batch, 1)
            actions_batch = torch.stack(actions_batch, 1)
            value_preds_batch = torch.stack(value_preds_batch, 1)
            return_batch = torch.stack(return_batch, 1)
            masks_batch = torch.stack(masks_batch, 1)
            old_action_log_probs_batch = torch.stack(old_action_log_probs_batch, 1)
            adv_targ = torch.stack(adv_targ, 1)

            # States is just a (N, -1) tensor
            recurrent_hidden_states_batch = torch.stack(recurrent_hidden_states_batch, 1).view(N, -1)

            # Flatten the (T, N, ...) tensors to (T * N, ...)
            obs_batch = _flatten_helper(T, N, obs_batch)
            actions_batch = _flatten_helper(T, N, actions_batch)
            value_preds_batch = _flatten_helper(T, N, value_preds_batch)
            return_batch = _flatten_helper(T, N, return_batch)
            masks_batch = _flatten_helper(T, N, masks_batch)
            old_action_log_probs_batch = _flatten_helper(T, N, \
                    old_action_log_probs_batch)
            adv_targ = _flatten_helper(T, N, adv_targ)

            yield obs_batch, recurrent_hidden_states_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ


class CuriosityRolloutStorage(RolloutStorage):

    def __init__(self, num_steps, num_processes, obs_shape, action_space, recurrent_hidden_state_size,
            state_feature_space #, curiosity_reccurent_hidden_state_size
            ):

        super().__init__(num_steps, num_processes, obs_shape, action_space, recurrent_hidden_state_size)
       #self.curiosity_recurrent_hidden_states = torch.zeros(num_steps + 1, num_processes, curiosity=_recurrent_hidden_state_size)

        self.action_bins = torch.zeros(num_steps, num_processes, action_space.n)
        self.action_dist_preds = torch.zeros(num_steps + 1, num_processes, action_space.n)
        self.feature_states = torch.zeros(num_steps, num_processes, *state_feature_space)
        self.feature_state_preds = torch.zeros(num_steps + 1, num_processes, *state_feature_space)

    def to(self, device):
        self.action_bins.to(device)
        self.action_dist_preds.to(device)
        self.feature_states.to(device)
        self.feature_state_preds.to(device)
        super().to(device)

    def insert(self, obs, recurrent_hidden_states, actions, action_log_probs, value_preds, rewards, masks,
            feature_state, feature_state_pred, action_bin, action_dist_pred):
        self.feature_states[self.step].copy_(feature_state)
        self.feature_state_preds[self.step + 1].copy_(feature_state_pred)
        self.action_bins[self.step].copy_(action_bin)
        self.action_dist_preds[self.step + 1].copy_(action_dist_pred)
        super().insert(obs, recurrent_hidden_states, actions, action_log_probs, value_preds, rewards, masks)


    def after_update(self):
        self.feature_state_preds[0].copy_(self.feature_state_preds[-1])
        self.action_dist_preds[0].copy_(self.action_dist_preds[-1])
       #self.curiosity_hidden_recurrent_states[0].copy_(self.curiosity_recurrent_hidden_states[-1])
        super().after_update()

    def feed_forward_generator(self):
        raise NotImplementedError

    def recurrent_generator(self):
        raise NotImplementedError
