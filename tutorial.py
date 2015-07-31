import outsmart as osmt
import numpy as np

osmt.TERRAIN[6,8] = 2
osmt.TERRAIN[0,0] = -1

osmt.story_text("""This is your lab.
The blue robot is a test robot.""")

osmt.objective_text("Click on the green patch to move the robot there.")


osmt.pyglet.app.run()

