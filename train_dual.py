import torch

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
        infos = super().self()

        if 'trg_agent' in infos[0]:
           #print(infos[0])
            self.set_active_agent(1)

        return infos

class DesignerPlayer(Trainer):
    def __init__(self):
        super().__init__()
        self.Player = None
        self.active_agent = 0
        # Assume player actions are discrete
       #self.envs.remotes[0].send(('get_player_action_space', None))
       #n_player_actions = self.envs.remotes[0].recv()
        self.actor_critic.base.add_player_head(n_player_actions)

    def set_active_agent(self, n_agent):
        self.active_agent = self.actor_critic.base.active_agent = n_agent
        self.envs.set_active_agent(n_agent)

    def train(self):
        infos = super().train()

        if 'trg_agent' in infos[0]:
           #print(infos[0])
            self.set_active_agent(infos[0]['trg_agent'])
           #if infos[0]['trg_agent'] == 1:
           #    self.envs.set_map(infos[0]['playable_map'])

if __name__ == "__main__":
    main()
