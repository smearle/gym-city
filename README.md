# gym-micropolis
An interface with micropolis for city-building agents, packaged as an OpenAI gym environment.

# Installation

Clone this repository, along with [micropolis-4bots](https://github.com/smearle/micropolis-4bots), into the same directory. Install the latter: 
```
cd micropolis-4bots/MicropolisCore/src
make
sudo make install
```
# Usage

'cd' back to this directory. Running python from here, we might do:
```
from gym_micropolis.envs.corecontrol import MicropolisControl
m = MicropolisControl(MAP_W=50, MAP_H=50)
m.layGrid(4, 4)
```
# Training

To use micropolis as a gym environment, install [gym](https://github.com/openai/gym).

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
