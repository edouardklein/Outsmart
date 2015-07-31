import outsmart as osmt
import numpy as np

osmt.TERRAIN[6,8] = 2
osmt.TERRAIN[0,0] = -1


osmt.new_button(10, 100, "Test button", lambda: print("TEST BUTTON"))
osmt.new_button(900, 50, "Test button with long text", lambda: print("LONG BUTTON"))

def step_2():
    osmt.script(s_text="""Good Job !
You can train Bob with a state-of-the-art
Reinforcement Learning algorithm.
This will allow you to train him like you would a dog or a rat.""",
                o_text="Train bob by clicking on the train button.")

osmt.script(s_text = """This is your lab.
The blue robot is a test robot.
His name is Bob.
You can control its position.""",
            o_text = "Click on the green patch to move the robot there.",
            objective_function = lambda m: m[6,8] == -2,
            next_step = step_2)

osmt.pyglet.app.run()

