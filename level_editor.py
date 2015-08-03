#!/usr/bin/env python3
import outsmart as osmt

from sys import argv
import os


if __name__ == "__main__":
    #FIXME filename, dirname, and directory hierarchy for the lvl componants (maps, script, ...)
    if len(argv)>1:
        osmt.STATE.filename = argv[1]
        if os.path.isfile(osmt.STATE.filename):
            osmt.load_cb()
    osmt.STATE.active_ui = {k:True for k in osmt.STATE.active_ui}
    osmt.STATE.active_ui["Lab"] = False
    osmt.STATE.level_editor = True
    osmt.pyglet.app.run()
