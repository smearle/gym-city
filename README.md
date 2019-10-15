This work was presented and demoed at the 2019 Experimental A.I. in Games (EXAG) workshop, an AIIDE workshop. You can read the paper here: http://www.exag.org/papers/EXAG_2019_paper_21.pdf

# gym-city
An interface with micropolis for city-building agents, packaged as an OpenAI gym environment.

We can render a training agent in real time and interact with its gui. An experimental hack makes builds taken by the player during training appear to the agent as samples from its own action probability distribution, so that it backpropogates over its own weights as a result of our actions, allowing us to expose the agent to handpicked behaviours so that it might learn from them, or, as is more often the case, so that they may violently disrupt the course of its training. It would be nice to establish a baseline of performance without intervention, and compare to performance of agents trained alongside various strategies of human intervention.

![breathy](https://github.com/smearle/gym-micropolis/blob/master/gifs/breathy.gif)

The agent above has a sequential brain of convolutions on a 3D feaure map, where the dimensions corresponding to height and width in terms of game tiles are fixed. One 3x3 kernel convolution is repeated 20 times to allow input activations on opposite ends of the map to affect one another. Only the critic compresses this feature map at its output layer to predict the reward value of an observed map-state. 

Currently I'm working on a model that uses repeated (de-)convolutions to compress and expand the feature map, combined with repeated convolutions on a fixed feature map, so that the agent might learn more abstract features of the map. 

I'm using acktr, but would like to develop and compare models using recurrent convolutions, which are helpful for establishing long-range dependencies not just over time, but also, like the recurrent convolutions mentioned above, over map-space.

# Installation

Clone this repository, then 
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

