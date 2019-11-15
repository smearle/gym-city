from world_pytorch import World as MultiWorld
from world import World as TradWorld
from random import randint
import numpy as np

# It looks like the problem is in TradWorld

def main():
    map_width = 16
    prob_life = 20
    w0 = TradWorld(width=map_width, height=map_width, prob_life=prob_life)
    w1 = MultiWorld(map_width=map_width, prob_life=prob_life,
            cuda=False, num_proc=1)
   #w0.populate_cells()
    w1.populate_cells()
    while True:
        for i in range(map_width ** 2): #ensure same initial state
            x = i // map_width
            y = i % map_width
            alive = randint(0, 1)
            w0.build_cell(x, y, alive=alive)
            w1.state[0][0][x][y] = alive
        s_0 = w0.state[0]
        s_1 = w1.state[0][0].cpu().long().numpy()
        print(s_0, '\n')
       #print(s_1, '\n')
        assert np.array_equal(s_0, s_1)
        for i in range(1000):
            w0._tick()
            w1._tick()
            s_0 = w0.state[0]
            s_1 = w1.state[0][0].cpu().long().numpy()
            print(s_0, '\n')
           #print(s_1, '\n')
            if not np.array_equal(s_0, s_1):
                print(s_0 ^ s_1)
                raise Exception

