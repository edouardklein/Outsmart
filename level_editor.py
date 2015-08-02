#!/usr/bin/env python3
import outsmart as osmt
from sys import argv

if __name__ == "__main__":
	osmt.create_save_button()
	osmt.create_load_button()		
	
	if len(argv)==1:
		print("%s <map_file>", __name__)
	osmt.pyglet.app.run()
