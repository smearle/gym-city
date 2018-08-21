# gym-micropolis
An interface with micropolis for city-building agents, packaged as an OpenAI gym environment.

# Installation

Install [micropolis-4bots](https://github.com/smearle/micropolis-4bots), so that we can launch micropolis from the command line and from there call silent build functions and queries, to be used by our agent.

The files in envs are now accessible as python3 modules. E.g.:

```
from gym_micropolis.envs.control import MicropolisControl
m = MicropolisControl()
m.layGrid(4, 4)
```

To use micropolis as a gym environment, install [gym](https://github.com/openai/gym).

To run the script `micropolis_agent.py`, which trains a micropolis bots using deep-q learning, install [AgentNet], then run
```
python3 micropolis_agent.py
```
