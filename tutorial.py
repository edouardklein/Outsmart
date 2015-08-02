import outsmart as osmt

osmt.STATE.lab[6, 8] = 2
osmt.STATE.lab[0, 0] = -1


def step_2():
    osmt.create_train_button()
    osmt.script(s_text="""Good Job !
You can train Bob with a state-of-the-art
Reinforcement Learning algorithm.
This will allow you to train him like you would a dog or a rat.""",
                o_text="Train bob by clicking on the train button.")

osmt.script(s_text="""This is your lab.
The blue robot is a test robot.
His name is Bob.
You can control its position.""",
            o_text="Click on the green patch to move the robot there.",
            objective_function=lambda m: m[6, 8] == -2,
            next_step=step_2)

osmt.pyglet.app.run()
