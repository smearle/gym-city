import torch

from model import Policy
from arguments import get_args
from train import Trainer


def main():
   #trainer = DualTrainer()
   #trainer.main()
   player = Player()
   player.main()

class Player(Trainer):
    ''' Train a player.
    '''
    def __init__(self):
        args = get_args()
        super().__init__(args)
        self.actor_critic.to(self.device)

    def set_active_agent(self, n_agent):
        self.active_agent = self.actor_critic.base.active_agent = n_agent
        self.envs.set_active_agent(n_agent)

    def step(self):
        obs, rew, done, infos = super().step()

        if 'trg_agent' in infos[0]:
            # suppress environment's desire to switch modes
            # TODO: remove this desire?
            self.set_active_agent(1)

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
        args.model = 'MLPBase'
        actor_critic = super().init_policy(envs, args)

        return actor_critic



class DesignerPlayer(Trainer):
    def __init__(self):
        super().__init__()
        self.player = Player(self.args)
        self.active_agent = 0
        # Assume player actions are discrete
       #self.envs.remotes[0].send(('get_player_action_space', None))
       #n_player_actions = self.envs.remotes[0].recv()
        self.actor_critic.base.add_player_head(n_player_actions)

    def set_active_agent(self, n_agent):
        self.active_agent = self.actor_critic.base.active_agent = n_agent
        self.envs.set_active_agent(n_agent)

    def step(self):
        obs, rew, done, infos = super().step()

        if 'trg_agent' in infos[0]:
            # suppress environment's desire to switch modes
            # TODO: remove this desire?
            self.set_active_agent(1)

        return obs, rew, done, infos

    def train(self):
        infos = super().train()

        if 'trg_agent' in infos[0]:
           #print(infos[0])
            self.set_active_agent(infos[0]['trg_agent'])
           #if infos[0]['trg_agent'] == 1:
           #    self.envs.set_map(infos[0]['playable_map'])

if __name__ == "__main__":
    main()
