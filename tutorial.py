import outsmart as osmt
import level_editor as lvl

MAP_NAME = "tutorial.map"

lvl.init()
osmt.STATE = lvl.load_state(MAP_NAME)

def step_5():
    """Resetting the robot"""
    osmt.script(s_text="""Good Job !""")


def step_4():
    """Sucessful training"""
    osmt.script(s_text="""Good Job !
Now, training Bob will succeed.
Bob is wired to collect rocks and crystals. You can't change that.
You can, however, change the environment and use rocks and
crystals as a motivator to make Bob do what you want.
When dealing with animals food is used as a reward.
This is the same here, only with robots.""",
                o_text="Try again to train Bob by clicking on the"
                """"Train" button.""",
                objective_function=lambda s: s.log_text[0][1] == "Training"
                " successful !",
                next_step=step_5)


def step_3():
    """Rocks spawning"""
    def success(state):
        bob_on_patch = state.lab[6, 8] == -2
        rock_nearby = state.lab[7, 8] == 4
        rock_nearby = rock_nearby or state.lab[5, 8] == 4
        rock_nearby = rock_nearby or state.lab[6, 9] == 4
        rock_nearby = rock_nearby or state.lab[6, 7] == 4
        return bob_on_patch and rock_nearby
    osmt.script(s_text="""Uh-oh, something went wrong :
In this lab setting, there is no reward
that could let Bob know how it's doing.
Let's change that.""",
                o_text="""Spawn some rocks somewhere right next to Bob by right-clicking
multiple times on the appropriate tile.""",
                objective_function=success,
                next_step=step_4)


def step_2():
    """Frist training"""
    osmt.create_train_button()
    osmt.script(s_text="""Good Job !
You can train Bob with a state-of-the-art
Reinforcement Learning algorithm.
This will allow you to train him like you would a dog or a rat.""",
                o_text="Train bob by clicking on the train button.",
                objective_function=lambda s: s.log_text[0][1] == 'Error !',
                next_step=step_3)

osmt.script(s_text="""This is your lab.
The blue robot is a test robot.
His name is Bob.
You can control its position.""",
            o_text="Click on the green patch to move the robot there.",
            objective_function=lambda s: s.lab[6, 8] == -2,
            next_step=step_2)

osmt.pyglet.app.run()
