# gym-micropolis
An interface with micropolis for city-building agents, packaged as an OpenAI gym environment.

# Installation

Clone this repository, then 
```
cd micropolis-4bots/MicropolisCore/src
make
sudo make install
```
![14x14linAC population reward](https://github.com/smearle/gym-micropolis/blob/master/a3c/demo/14x14linAC.gif) ![10x10linAC population reward](https://github.com/smearle/gym-micropolis/blob/master/a3c/demo/10x10linAC.gif)

# Usage

From this directory, we might run:
```
from gym_micropolis.envs.corecontrol import MicropolisControl
m = MicropolisControl(MAP_W=50, MAP_H=50)
m.layGrid(4, 4)
```
# Training

To use micropolis as a gym environment, install [gym](https://github.com/openai/gym).

## A3C

To use [dgriff777](https://github.com/dgriff777/rl_a3c_pytorch)'s pytorch implementation of A3C with gpu support (contained in 'a3c' with slight modifications), from the a3c directory we might run:
'''
python3 main.py --env MicropolisEnv-v0 --map-width 14 --design-head A3Cmicropolis14x14linAC --gpu-id 0 --workers 12 --log-dir logs/14x14_ --save-model-dir logs/14x14_
'''
with --gpu-id=-1 to run on cpu-only. This will use the version of MicropolisCore contained in micropolis-4bots-gtk3, which allows the game engine to run in python3, but will creash when trying to render the gui, since it has not been properly rewritten using gtk3. So to evalueate our model, we would run
'''
python2 gym-eval --env MicropolisEnv-v0 --design-head A3Cmiropolis14x14linAC --render R --load-model-dir logs/14x14_
'''
## DQN

To run the script `learn.py`, which trains a micropolis bots using deep-q learning, install [keras-rl](https://github.com/keras-rl/keras-rl), then run:
```
python learn.py --run-id='testrun'
```
Alternatively, you may install [AgentNet](https://github.com/yandexdataschool/AgentNet) and run
```
python learn_agentnet.py
```
(at the time of writing, however, the learn_agentnet script hasn't been modified to match the architecture of the keras-rl learn script, nor has it been tested with pyMicropolis)

# Testing

Try initializing the bot with the included pre-trained weights (learn.py must be set to initiazlize with the same map dimensions and the same network architecture as the weights were trained on - if you plan on experimenting with training different models, it's a good idea to back up learn.py in the folders to which the training logs and weights are written):
```
python learn.py --mode=test --weights=12x12_step-1000000_weights.h5f
```
