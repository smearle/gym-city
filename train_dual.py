import copy
import torch

from arguments import get_args
from model import Policy
from train import Trainer


def main():
    args = get_args()
    args.env_name = 'zeldaplay-wide-v0'
    trainer = DesignerPlayer(args)
    trainer.main()
  #player = Player()
  #player.main()

class Player(Trainer):
    ''' Train a player.
    '''
    def __init__(self, envs=None, args=None):
        if args is None:
            args = get_args()
        args.log_dir += '_dsgn'
        super().__init__(envs, args)
        self.actor_critic.to(self.device)

    def set_active_agent(self, n_agent):
        self.active_agent = self.actor_critic.base.active_agent = n_agent
        self.envs.set_active_agent(n_agent)

    def step(self):
        self.set_active_agent(1)
        obs, rew, done, infos = super().step()

       #if 'trg_agent' in infos[0]:
            # suppress environment's desire to switch modes
            # TODO: remove this desire?

        return obs, rew, done, infos

    def get_space_dims(self, envs, args):
        super().get_space_dims(envs, args)
        # TODO: get this from envs
        self.out_w, self.out_h = 1, 1
        n_player_actions = 4
        self.num_actions = n_player_actions

    def make_vec_envs(self, args):
        envs = super().make_vec_envs(args)

        return envs

    def init_policy(self, envs, args):
       #args.model = 'MLPBase'
        actor_critic = super().init_policy(envs, args)

        return actor_critic



class DesignerPlayer(Trainer):
    def __init__(self, args=None):
        super().__init__(args=args)
        design_args = copy.deepcopy(args)
        self.player = Player(self.envs, design_args)
        self.active_agent = 0
        # Assume player actions are discrete
       #self.envs.remotes[0].send(('get_player_action_space', None))
       #n_player_actions = self.envs.remotes[0].recv()
        self.playable_map = None

    def set_active_agent(self, n_agent):
        self.active_agent = self.actor_critic.base.active_agent = n_agent
        self.envs.set_active_agent(n_agent)

    def step(self):
        self.set_active_agent(0)
        obs, rew, done, infos = super().step()
        rew = 0

       #if 'trg_agent' in infos[0]:
            # suppress environment's desire to switch modes
            # TODO: remove this desire?

        # arbitrary env choice

        if 'playable_map' in infos[0] and done[0]:
            playable_map = infos[0]['playable_map']
            play_rew = self.train_player_epi(playable_map)
           #if play_rew > 0:
           #    play_rew = self.args.max_step - play_rew
            rew = rew + play_rew
            print(rew)
            #dummy_rews = torch.empty((self.args.num_processes), dtype=float).fill_(rew)
            rew = torch.Tensor([rew])
            self.rollouts.rewards[self.rollouts.step].copy_(rew)

        return obs, rew, done, infos

    def train(self):

        done, info = super().train()
        self.n_train += 1

        return info

    def train_player_epi(self, playable_map):
        ''' Trains a player for one episode on a given map. '''
       #self.envs.set_active_agent(1)
        epi_done = False
        self.player.set_active_agent(1)
        self.player.envs.set_map(playable_map)

        while not epi_done:
            done, info = self.player.train()
            self.player.n_train += 1
            epi_done = done[0]
        print(self.player.episode_rewards)
        epi_rew = self.player.episode_rewards[-1]
        self.player.visualize(self.player.plotter)
        print('epi reward', epi_rew)

        return epi_rew


if __name__ == "__main__":
    main()
