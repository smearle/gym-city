The work in this repo was presented and demoed at the 2019 Experimental A.I. in Games (EXAG) workshop, an AIIDE workshop. You can read the paper here: http://www.exag.org/papers/EXAG_2019_paper_21.pdf

Feel free to join the conversation surrounding this work via my [Twitter](https://twitter.com/Smearle_RH), and on [r/MachineLearning](https://www.reddit.com/r/MachineLearning/comments/d346e9/r_using_fractal_neural_networks_to_play_simcity_1/).

# gym-city

A Reinforcement Learning interface for variable-scale city-planing-type gym environments, including [Micropolis](https://github.com/simhacker/micropolis/) (open-source SimCity 1) and a 1-player version of Conway's Game of Life.

## Micropolis (SimCity 1)

The player builds places urban structures on a 2D map. In certain configurations, these structures invite population and vertical development. Reinforcement Learning agents are rewarded for this population.

![breathy](https://github.com/smearle/gym-micropolis/blob/master/gifs/breathy.gif)  

 *Helping the agent deal with demand.*

![lilBully](https://github.com/smearle/gym-city-notes/blob/master/gifs/lilBully.gif)  

*Inciting exploration of city-space via deletion of key features.*

![collab](https://github.com/smearle/gym-city-notes/blob/master/gifs/collab.gif)

*Working with a traffic-hungry agent.*

## 1-Player Game of Life

Like SimCity, but there are only two possible states for each tile (populated or not), and one engine tick corresponds to one application of the transition rules of Conway's Game of Life.

![entomb](https://github.com/smearle/gym-city-notes/blob/master/gifs/entomb.gif)

## Power Puzzle

Like SimCity, but the map spawns with one power plant, and several residential zones, all randomly placed, and the bot is restricted to building power lines.

![blindLonging](https://github.com/smearle/gym-micropolis/blob/master/gifs/blindLonging.gif) 

*An agent plays on a larger map than trained upon, with mixed results.*

![casual](https://github.com/smearle/gym-city-notes/blob/master/gifs/casual.gif) 

*Giving pointers to the agent.*

# Installation

## Ubuntu

Make sure python >= 3.6 is installed.

For Micropolis, we need the python3 header files, gtk and cairo, and swig:
```
sudo apt-get install python3-dev libcairo2-dev python3-cairo-dev libgirepository1.0-dev swig
pip install gobject PyGObject
```
We also need [pytorch](https://pytorch.org/get-started/locally/), since we'll be using it to build and train our agents, and [tensorflow](https://www.tensorflow.org/install) (since baselines depends on it).

Some additional python packages:
```
pip install cffi python3-mpi4py gym baselines graphviz torchsummary imutils visdom sklearn matplotlib torchsummary
```
Clone this repository, then 
```
cd gym-city
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

I'm interested in emergent transport networks, though I'm not sure the simulation is complex enough for population-optimizing builds to require large-scale transport networks. 

We can, however, reward traffic density just enough for continuous/traffic-producing roads to be drawn between zones, but not so much that they billow out into swaths of traffic-exploiting asphalt badlands.

# Interacting w/ the Bot

During training and inference, a micropolis gui will be rendered by default. During training, it is the environment of rank 0 if multiple games are being run in parallel. The gui is controlled by the learning algorithm, and thus can be laggy, but is fully interactive, so that a player may build or delete zones during training and inference.

Any action that is successfully performed by the player on the subsection of the larger game-map which corresponds to the bot's play-area, is registered by the bot as an action of its own, sampled from its own action distribution, so that the player can influence the agent's training data in real time.

