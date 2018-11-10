from gym_micropolis.envs.env import MicropolisEnv
from gi.repository import Gtk

m = MicropolisEnv()
m.setMapSize(20)
m.reset()
go = True
while go:
    m.reset()
    for i in range(500):
        try:
            m.randomStep()
        except KeyError:
            print("KeyError")
            m.printMap()
            m.micro.printTileMap()
            m.render()
            Gtk.main()
            break
        except AssertionError:
            print("AssertionError")
            m.printMap()
            m.micro.printTileMap()
            m.render()
            Gtk.main()
            break

        m.printMap()
#       m.micro.printTileMap()
        m.render()
    go = True

Gtk.main()
