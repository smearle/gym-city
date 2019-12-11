import sys
if int(sys.version[0]) >= 3:
    from gi.repository import Gtk as gtk
else:
    import gtk

import corecontrol
m = corecontrol.MicropolisControl()
m.layGrid(4, 4)
for i in range(1000):
    gtk.mainiteration()
    m.layGrid(i % 12 + 1,i % 12 + 1)
    gtk.mainiteration()
    gtk.mainiteration()
gtk.main()
#import walkenv
#w = walkenv.MicroWalkEnv()
#w.setMapSize(50,50)
## w.micro.layGrid(10,10)
#print(w.micro.map.zoneMap[-1,:20,:20])
#print(w.micro.map.zoneMap[-2,-20:,-20:])
#for i in range(1000):
#    w.step(w.action_space.sample())
#print(w.micro.map.zoneMap[-1,:20,:20])
#print(w.micro.map.zoneMap[-2,-20:,-20:])
#gtk.main()
