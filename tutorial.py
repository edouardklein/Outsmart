import outsmart as osmt
import numpy as np

osmt.TERRAIN[6,8] = 2
osmt.TERRAIN[0,0] = -1

def step_2():
    osmt.script(s_text="VICTORY !")

osmt.script(s_text = """This is your lab.
The blue robot is a test robot.
His name is Bob.
You can control its position.""",
            o_text = "Click on the green patch to move the robot there.",
            objective_function = lambda m: m[6,8] == -2,
            next_step = step_2)

osmt.pyglet.app.run()

