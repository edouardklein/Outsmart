#!/usr/bin/env python3
import graphics
import ui

from outsmart import return_copy
import graphics as g


g.new_state()
g.STATE.ui.active.update(ui.EDITOR_ACTIVE)



@return_copy
def wild(s):
    s = ui.wild(s)
    s.ui.active.update({k: True for k in ui.EDITOR_ACTIVE
                        if ui.EDITOR_ACTIVE[k]})
    s.ui.active["lab_go_wild"] = False
    return s

@return_copy
def lab(s):
    s = ui.lab(s)
    s.ui.active = EDITOR_ACTIVE.copy()
    return s

g.BUTTONS["lab_go_wild"][-1] = lambda: g._state(wild)
g.BUTTONS["wild_go_lab"][-1] = lambda: g._state(lab)
