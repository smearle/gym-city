import sys
if sys.version[0] == 2:
    from env import MicropolisEnv
    from paintenc import MicropolisPaintEnv
    from arcadeenv import MicroArcadeEnv
    from walkcontrol import MicroWalkControl
else:
    from .env import MicropolisEnv
    from .paintenv import MicropolisPaintEnv
    from .arcadeenv import MicroArcadeEnv
    from .walkcontrol import MicroWalkControl

MICROPOLIS_SRC_DIR = 'micropolis/MicropolisCore/src'
