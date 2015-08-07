#!/usr/bin/env python3
import numpy as np
from outsmart import return_copy
import outsmart as osmt
import os.path
import graphics as g
import ui
import pyglet

lvl_directory = os.path.dirname(os.path.abspath(__file__))

@return_copy
def step_1(s):
    """Remove the menu"""
    s.ui.active = ui.ALL_INACTIVE.copy()
    s.ui.active["lab_wild_quit"] = True
    return s

g.new_state()
g.STATE = step_1(g.STATE)
g.play(media_file=lvl_directory+"/intro.mp4")
