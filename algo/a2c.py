import torch
import torch.nn as nn
import torch.optim as optim

import sys
if sys.version[0] == '3':
    from .kfac import KFACOptimizer


class A2C():
    def __init__(self,
                 actor_critic,
                 value_loss_coef,
                 entropy_coef,
                 lr=None,
                 eps=None,
                 alpha=None,
                 max_grad_norm=None,
                 acktr=False,
                 curiosity=False, args=None):

        self.args= args
        self.curiosity = curiosity
        self.actor_critic = actor_critic
        self.acktr = acktr

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm

        if acktr:
            self.optimizer = KFACOptimizer(actor_critic)
        else:
            self.optimizer = optim.RMSprop(
                actor_critic.parameters(), lr, eps=eps, alpha=alpha)
        self.paint = 'paint' in args.env_name.lower()

    def update(self, rollouts):
        obs_shape = rollouts.obs.size()[2:]
        if self.curiosity:
            act_shape = rollouts.action_bins.size()[2:]
        num_steps, num_processes, _ = rollouts.rewards.size()
        if 'paint' in self.args.env_name.lower():
            action_shape = rollouts.actions.size()[-3:]
            actions = rollouts.actions.view(-1, *action_shape)
        else:
            if isinstance(rollouts.actions, dict):
                action_shape = {}
                actions = {}
                for k in rollouts.actions:
                    action_shape[k] = rollouts.actions[k].size()[-1]
                    actions[k] = rollouts.actions[k].view(-1, action_shape[k])
            else:
                action_shape = rollouts.actions.size()[-1]
                actions = rollouts.actions.view(-1, action_shape)

        if 'LSTM' in self.args.model:
            rec_size = self.actor_critic.base.get_recurrent_state_size()
            rec_states = rollouts.recurrent_hidden_states[:-1].view(2, -1, *rec_size)
        else:
            rec_states = rollouts.recurrent_hidden_states[:-1].view(-1, 1)
        values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
        rollouts.obs[:-1].view(-1, *obs_shape),
        rec_states,
        rollouts.masks[:-1].view(-1, 1),
        actions)

        if self.curiosity:
           #action_dist_shape = rollouts.action_dists.size()[2:]
           #action_dists = rollouts.action_dists.view(-1, *action_dist_shape)
           #action_size = sum(action_dist_shape)
           #action_bin = torch.zeros((num_processes * num_steps, action_size)).cuda()
           #action_bin.fill_(0)
           #action_i = torch.cat((torch.Tensor(list(range(num_processes * num_steps))).cuda().unsqueeze(1).long(), actions), 1)
           #action_bin[action_i[:,0], action_i[:,1]] = 1

            feature_states, feature_state_preds, action_preds = self.actor_critic.evaluate_icm(
                    (rollouts.obs[:-1].view(-1, *obs_shape),
                     rollouts.obs[1:].view(-1, *obs_shape),
                     rollouts.action_bins.view(-1, *act_shape))
                    )
            forward_err = feature_states[:] - feature_state_preds[:]
            forward_loss = forward_err.pow(2).sum().cpu()

            inverse_loss = - (rollouts.action_bins.view(-1, *act_shape).cpu() * torch.log(action_preds + 1e-15).cpu()).sum()



        values = values.view(num_steps, num_processes, 1)
        if self.paint:
            action_log_probs = action_log_probs.view(num_steps, num_processes, -1)
        else:
            if isinstance(action_log_probs, dict):
                alps = []
                for k in action_log_probs:
                    alps.append(action_log_probs[k].view(num_steps, num_processes, 1))
                alps = tuple(alps)
                action_log_probs = torch.cat(alps, -1)
                action_log_probs = action_log_probs.mean(-1).unsqueeze(-1)
            else:
                action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

        returns =rollouts.returns
        returns = returns - returns.mean()
        returns = returns / returns.std()
        advantages = returns[:-1] - values
        value_loss = advantages.pow(2).mean()

        action_loss = -(advantages.detach() * action_log_probs).mean()

        if self.acktr and self.optimizer.steps % self.optimizer.Ts == 0:
            # Sampled fisher, see Martens 2014
            self.actor_critic.zero_grad()
            pg_fisher_loss = -action_log_probs.mean()

            value_noise = torch.randn(values.size())
            if values.is_cuda:
                value_noise = value_noise.cuda()

            sample_values = values + value_noise
            vf_fisher_loss = -(values - sample_values.detach()).pow(2).mean()

            fisher_loss = pg_fisher_loss + vf_fisher_loss
            self.optimizer.acc_stats = True
            fisher_loss.backward(retain_graph=True)
            self.optimizer.acc_stats = False

        self.optimizer.zero_grad()
        if not dist_entropy:
            (value_loss * self.value_loss_coef + action_loss).backward()
        else:
            (value_loss * self.value_loss_coef + action_loss -
            dist_entropy * self.entropy_coef).backward()

        if self.curiosity:
            icm = (1 - self.args.beta) * inverse_loss + self.args.beta * forward_loss
            icm.backward()

        if self.acktr == False:
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                     self.max_grad_norm)

        self.optimizer.step()

        if self.curiosity:
            return value_loss.item(), action_loss.item(), dist_entropy.item(),\
                forward_loss.item(), inverse_loss.item()
        else:
            if not dist_entropy:
                dist_item = None
            else:
                dist_item = dist_entropy.item()
            return value_loss.item(), action_loss.item(), dist_item
