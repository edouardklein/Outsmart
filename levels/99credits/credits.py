#!/usr/bin/env python3
import graphics
import ui


graphics.new_state()
graphics.STATE.ui.story_text="""Graphic assets:
Tiles CC0 by Kenney.nl
Robot CC0 by Johann-c

Programming:
Code AGPL by Denis Baheux and Edouard Klein
Usage of the Avbin library (LGPL)


Our thanks to the authors of Python, Pyglet, Emacs etc.
"""
graphics.STATE.ui.active["story_text"] = True
graphics.STATE.ui.active["lab_wild_quit"] = True

#graphics.STATE.ui.active.update(ui.EDITOR_ACTIVE)
