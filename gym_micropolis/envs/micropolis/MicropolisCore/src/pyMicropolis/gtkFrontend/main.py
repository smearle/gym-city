"""
@file main.py Main file for running Micropolis with the GTK frontend

@todo Implement run()
"""

import pyMicropolis.micropolisEngine
import gi
gi.require_version('Gtk', '3.0')
gi.require_version('PangoCairo', '1.0')
from pyMicropolis.micropolisEngine import micropolisengine, micropolisgtkengine, micropoliswindow, micropolisrobot

from gi.repository import Gtk as gtk
import random
import math


def run(builderBot=None):

    engine = micropolisgtkengine.CreateGTKEngine()

    engine.cityTax = 10
    engine.setPasses(200)
    setTile = engine.setTile

    if False:
        for i in range(0, 4):
            engine.addRobot(
                micropolisrobot.MicropolisRobot_PacBot(
                    x=(8 * 16) + 3 + (16 * 2 * i),
                    y=(7 * 16) + 3,
                    direction=0))

    if False:
        for i in range(0, 20):
            engine.addRobot(
                micropolisrobot.MicropolisRobot_PacBot(
                    x=random.randint(0, (micropolisengine.WORLD_W * 16) - 1),
                    y=random.randint(0, (micropolisengine.WORLD_H * 16) - 1),
                    direction = random.randint(0, 3) * math.pi / 2))

    if False:
        for y in range(0, micropolisengine.WORLD_H):
            for x in range(0, micropolisengine.WORLD_W):
                setTile(x, y, micropolisengine.RUBBLE | micropolisengine.BLBNBIT)

        for y in range(10, 15):
            for x in range(10, 15):
                setTile(x, y, micropolisengine.FIRE | micropolisengine.ANIMBIT)

    x = 0
    y = 0

    w = 800
    h = 600

    if True:
        win1 = micropoliswindow.MicropolisPanedWindow(engine=engine)
        win1.set_default_size(w, h)
        win1.set_size_request(w, h)
        win1.move(x, y)
        win1.show_all()

    gtk.main()


# for bots. Return the engine for training simulation
def train(env=None, rank=None, root_gtk=None, map_x=20, map_y=20, gui=False):

    kwargs = {'env': env, 'rank': rank, 'root_gtk': root_gtk}
    engine = micropolisgtkengine.CreateGTKEngine(**kwargs)

    engine.cityTax = 10

    x = 0
    y = 0

    w = 800
    h = 600

    win1 = micropoliswindow.MicropolisPanedWindow(engine=engine)
    win1.set_default_size(w, h)
    win1.set_size_request(w, h)
    win1.move(x, y)
    win1.show_all()

    return engine, win1
