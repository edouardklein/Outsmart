#!/usr/bin/env python3
import outsmart as osmt
from sys import argv
import dill as pickle
#import pickle
import os

def load_state(filename):
	with open("maps/%s"%filename, 'rb') as load_file:
		s = pickle.load(load_file)
	return s

def save_state(s, filename):
	with open("maps/%s"%filename, 'wb') as save_file:
		pickle.dump(s, save_file)

def init():
	osmt.STATE.lab[0, 0] = -1

if __name__ == "__main__":

	#FIXME
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
