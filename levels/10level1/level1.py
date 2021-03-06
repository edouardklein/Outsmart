#!/usr/bin/env python3
import numpy as np
from outsmart import return_copy
import outsmart as osmt
import os.path
import graphics as g
import ui

lvl_directory = os.path.dirname(os.path.abspath(__file__))


@return_copy
def step_1(s):
    """Move the robot"""
    s.ui.filename = lvl_directory+"/level1"
    s = ui.load(s)
    s = ui.wild(s)
    s.ui.log_text = ""
    s.ui.story_text = """Neither bob nor the red robot can go through white rocks."""
    s.ui.obj_text = "Make the red robot go into the trap"
    return s


g.new_state()
g.STATE = step_1(g.STATE)
ui.checkpoint_now(g.STATE)

