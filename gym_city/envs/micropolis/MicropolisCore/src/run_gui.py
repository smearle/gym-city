#!/usr/bin/env python3.11
"""Launch the standalone Micropolis GTK GUI (ground-truth reference).

Usage:
    /opt/homebrew/bin/python3.11 run_gui.py

Requires the Homebrew python3.11 stack with PyGObject (gi), pycairo, gtk+3,
and the SWIG engine bindings built via `make` in this directory.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(HERE)  # so the engine finds images/, res/, dataColorMap.png, etc.

import gi
gi.require_version('Gtk', '3.0')

from pyMicropolis.gtkFrontend import main

main.run()
