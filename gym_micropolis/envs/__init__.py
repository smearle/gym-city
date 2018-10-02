import sys
if sys.version[0] == 2:
    from env import MicropolisEnv
    from arcadeenv import MicroArcadeEnv
    from walkcontrol import MicroWalkControl
else:
    from .env import MicropolisEnv
    from .arcadeenv import MicroArcadeEnv
    from .walkcontrol import MicroWalkControl
