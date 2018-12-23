# gym-micropolis
An interface with micropolis for city-building agents, packaged as an OpenAI gym environment.


The gifs below show excerpts of the work of a a3c-trained models consisting of 5 convolutional lstms followed by fully-connected layers. At each step, the agents are rewarded by the change in total population from the last.

![14x14linAC population reward](https://github.com/smearle/gym-micropolis/blob/master/a3c/demo/14x14linAC.gif)
*14x14 map*

![10x10linAC population reward](https://github.com/smearle/gym-micropolis/blob/master/a3c/demo/10x10linAC.gif)
*10x10 map*

# Installation

Clone this repository, then 
```
cd micropolis/MicropolisCore/src
make
sudo make install
```

# Basic Use

From this directory, run:
```
from gym_micropolis.envs.corecontrol import MicropolisControl
m = MicropolisControl(MAP_W=50, MAP_H=50)
m.layGrid(4, 4)
```
# Training w/ RL

To use micropolis as a gym environment, install [gym](https://github.com/openai/gym).

To train an agent using ACKTR:

```
cd baselines-micropolis-pytorch
python3 main.py --log-dir trained_models/acktr --save-dir trained_models/acktr --algo acktr --num-process 24 --map-width 20
```

# Testing

Try initializing the bot with the included pre-trained weights (learn.py must be set to initialize with the same map dimensions and the same network architecture as the weights were trained on - if you plan on experimenting with training different models, it's a good idea to back up learn.py in the folders to which the training logs and weights are written):
```
python learn.py --mode=test --weights=12x12_step-1000000_weights.h5f
```
