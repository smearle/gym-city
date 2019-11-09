The work in this repo was presented and demoed at the 2019 Experimental A.I. in Games (EXAG) workshop, an AIIDE workshop. You can read the paper here: http://www.exag.org/papers/EXAG_2019_paper_21.pdf


# gym-city
A Reinforcement Learning interface for variable-scale city-planing-type gym environments, including [Micropolis](https://github.com/simhacker/micropolis/) (open-source SimCity 1) and an interactive version of Conway's Game of Life.

# Example

![breathy](https://github.com/smearle/gym-micropolis/blob/master/gifs/breathy.gif)
 *Interacting with a trained agent*

# Installation

## Ubuntu

Make sure python >= 3.6 is installed.
For Micropolis, we need the python3 header files, gtk and cairo, and swig:
```
sudo apt-get install python3-dev libcairo2-dev python3-cairo-dev swig
```
We also need [pytorch](https://pytorch.org/get-started/locally/), since we'll be using it to build and train our agents, and [tensorflow](https://www.tensorflow.org/install) (since baselines depends on it).

Some additional python packages:
```
pip3 install gym baselines graphviz torchsummary imutils visdom sklearn
```

Clone this repository, then 
```
cd gym-city
```
To install dependencies, use either anaconda:
```
conda create --name gc --file requirements.txt
``` 
or pip:
```
pip install -r requirements.txt
```
Then build Micropolis:
```
make install
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

To train an agent using A2C:

```
python3 main.py --experiment test_0 --algo a2c --model FullyConv --num-process 24 --map-width 16 --render
```

To visualize reward, in a separate terminal run: ` python -m visdom.server`

To run inference using the agent we have just trained:

```
python3 enjoy.py --load-dir <directory/containing/MicropolisEnv-v0.tar> --map-width 16
```

Generally, agents quickly discover the power-plant + residential zone pairing that creates initial populations, then spend very long at a local minima consisting of a gameboard tiled by residential zones, surrounding a single power plant. On more successful runs, the agent will make the leap toward a smattering of lone road tiles place next to zones, at least one per zone to maximize population density. 

I'm interested in emergent transport networks, though it does not seem that the simulation is complex enough for population-optimizing builds to require large-scale transport networks. 

We can, however, reward traffic density just enough for continuous/traffic-producing roads to be drawn between zones, but not so much that they billow out into swaths of traffic-exploiting asphalt badlands.

# Interacting w/ the Bot

During training and inference, a micropolis gui will be rendered by default. During training, it is the environment of rank 0 if multiple games are being run in parallel. The gui is controlled by the learning algorithm, and thus can be laggy, but is fully interactive, so that a player may build or delete zones during training and inference.

Any action that is successfully performed by the player on the subsection of the larger game-map which corresponds to the bot's play-area, is registered by the bot as an action of its own, sampled from its own action distribution, so that the player can influence the agent's training data in real time.

