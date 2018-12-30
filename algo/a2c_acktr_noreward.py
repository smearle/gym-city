import torch
import torch.nn as nn
import torch.optim as optim

import sys
if sys.version[0] == '3':
    from .kfac import KFACOptimizer


class A2C_ACKTR_NOREWARD():
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

    def update(self, rollouts):
        obs_shape = rollouts.obs.size()[2:]
        if self.curiosity:
            act_shape = rollouts.action_bins.size()[2:]
        num_steps, num_processes, _ = rollouts.rewards.size()
        if 'paint' in self.args.env_name.lower():
            action_shape = rollouts.actions.size()[-3:]
            actions = rollouts.actions.view(-1, *action_shape)
        else:
            action_shape = rollouts.actions.size()[-1]
            actions = rollouts.actions.view(-1, action_shape)

        values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
            rollouts.obs[:-1].view(-1, *obs_shape),
            rollouts.recurrent_hidden_states[0].view(-1, self.actor_critic.recurrent_hidden_state_size),
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
        if 'paint' in self.args.env_name.lower():
            action_log_probs = None
        else:
            action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

        returns =rollouts.returns
        returns = returns - returns.mean()
        returns = returns / returns.std()
        advantages = returns[:-1] - values
        value_loss = advantages.pow(2).mean()

        if 'paint' in self.args.env_name.lower():
            action_loss = -advantages.detach().mean()
        else:
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
