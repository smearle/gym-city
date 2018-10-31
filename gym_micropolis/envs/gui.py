import os
import sys
import gtk
FILE_DIR = os.path.dirname(os.path.realpath(__file__))
GIT_DIR = os.path.abspath(os.path.join(FILE_DIR, os.pardir, os.pardir))
if sys.version_info[0] >= 3:
    raise NotImplementedError('the micropolis gtk frontend is not ported to python3, use python2')
else:
    MICROPOLISCORE_DIR = GIT_DIR + '/micropolis-4bots/MicropolisCore/src'
    sys.path.append(MICROPOLISCORE_DIR)
    from tilemap import TileMap
CURR_DIR = os.getcwd()
# we need to do this so the micropolisgenericengine can access images/micropolisEngine/dataColorMap.png
os.chdir(MICROPOLISCORE_DIR)   

from pyMicropolis.gtkFrontend import main

# renders and controls a parallel game of micropolis in python2
# TODO: track player moves
class MicropolisGUI():

    def __init__(self, MAP_W=20, MAP_H=20):
        print('initiated gui')
        engine, win1 = main.train()
        self.engine = engine
        self.engine.setGameLevel(2)
        self.engine.autoBulldoze = True
        win1.playCity()
        self.engine.setFunds(1000000)
        engine.setSpeed(3)
        engine.setPasses(500)


    def doSimToolInt(self, x, y, tool):
        self.render()
        print('gui step')
        return self.engine.toolDown(tool, x, y)

    def clearMap(self):
        self.engine.clearMap()

    def render(self):
        gtk.main_iteration()
        gtk.main_iteration()

