#!/usr/bin/env python3
import sys
import outsmart as osmt
import importlib.machinery
import argparse
import pyglet
import glob
import os.path

lvl_directory = "levels"


def import_lvl(name):
    # https://stackoverflow.com/questions/27381264/python-3-4-how-to-import-a-module-given-the-full-path
    fname = glob.glob(name+'/*.py')[0]
    importlib.machinery.SourceFileLoader(name, fname).load_module()

osmt.set_TTS_generate(True, "OSX-say")

y_offset = 0
for dirr in glob.glob(lvl_directory+"/*"):
    name = os.path.split(dirr)[1][2:]
    print("Dir : "+dirr+", name :"+name)
    img = pyglet.image.load(dirr+"/img.png")
    x = osmt.WINDOW.width//2-img.width//2
    y = osmt.WINDOW.height-img.height - y_offset
    y_offset += img.height+20
    osmt.STATE.buttons[name] = [[x, y, x+img.width, y+img.height],
                              "MainScreen", lambda img=img : img,
                              lambda dirr=dirr: import_lvl(dirr)]
osmt.STATE.active_ui = {k: False for k in osmt.STATE.active_ui}
osmt.STATE.active_ui["MainScreen"] = True
osmt.STATE.lab = None

osmt.pyglet.app.run()
