#!/usr/bin/env python3
import outsmart as osmt

from sys import argv
import os

import dill as pickle
#import pickle

import copy

def load_state(filename):
	with open("maps/%s"%filename, 'rb') as load_file:
		s = pickle.load(load_file)

	#FIXME : cleaner method? Maybe put these buttons in another window.
	if "Load Map" in osmt.STATE.buttons:
		load_button = osmt.STATE.buttons["Load Map"]
		s.buttons["Load Map"] = load_button
	if "Save Map" in osmt.STATE.buttons:
		save_button = osmt.STATE.buttons["Save Map"]
		s.buttons["Save Map"] = save_button
	return s

def save_state(s, filename):
	s_saved = copy.deepcopy(s)
	#FIXME: see load_state comments
	s_saved.buttons.pop("Save Map")
	s_saved.buttons.pop("Load Map")
	with open("maps/%s"%filename, 'wb') as save_file:
		pickle.dump(s_saved, save_file)

def init():
	osmt.STATE.lab[0, 0] = -1

if __name__ == "__main__":

	#FIXME filename, dirname, and directory hierarchy for the lvl componants (maps, script, ...)
	filename = "current_mapfile.map"
	init()

	def load_cb():
		osmt.STATE = load_state(filename)

	if len(argv)>1:
		filename = argv[1]
		if os.path.isfile("maps/%s"%filename):
			load_cb()


	osmt.create_load_button(load_cb, "Load Map")	
	osmt.create_save_button(lambda:save_state(osmt.STATE, filename), "Save Map")
	osmt.pyglet.app.run()
