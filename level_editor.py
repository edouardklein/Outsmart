#!/usr/bin/env python3
import outsmart as osmt
from sys import argv
import dill
import pickle

def load_state(filename):
	return pickle.load(open(filename, 'r'))

def save_state(s, filename):
	pickle.dump(s, open(filename, 'w'))

def init():
	osmt.STATE.lab[0, 0] = -1

if __name__ == "__main__":

	#FIXME
	FILENAME = "maps/mapfile.pickle"

	init()

	def load_cb():
		osmt.STATE = load_state(FILENAME)
	osmt.create_save_button(lambda:save_state(osmt.STATE, FILENAME))
	#osmt.create_load_button(lambda:osmt.STATE = load_state(FILENAME))
	osmt.create_load_button(load_cb)	
	
	if len(argv)>1:
		osmt.STATE = load_state(argv[1])
	osmt.pyglet.app.run()
