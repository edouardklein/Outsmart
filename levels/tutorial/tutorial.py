#!/usr/bin/env python3
import outsmart as osmt
import numpy as np
import os.path

map_name = "tutorial"

lvl_directory = os.path.dirname(os.path.abspath(__file__))

osmt.STATE = osmt.load_state(osmt.STATE, lvl_directory + "/" + map_name)

# osmt.STATE.lab[6,8] = 2


def step_11():
    """WIP"""
    osmt.script(s_text="""Time to tie it all together :
By placing resources (rocks or crystals) and training Bob,
teach it to follow the grassy path.
You can reset Bob and start over if the training does not work.
You can train mutliple times to add on to Bob's experience.
When you feel confident, go into the wild and press see
if you tricked the red robot.""")


def step_10():
    """Create the path"""
    osmt.STATE.lab=np.loadtxt("levels/tutorial/grassy_path.lab")
    osmt.STATE.wild=np.loadtxt("levels/tutorial/tutorial.wild")
    osmt.STATE.victory = osmt.default_victory
    def grassy_path(s):
        return all(s.lab[4, 2:8] // 100 == 2)
    osmt.script(s_text="""We have create the start of a grassy path in the lab.
Complete it so that it looks like the one that was ahead of the
red robot in the wild.""",
                o_text="""Right click on the appropriate tiles to complete
the path.""",
                objective_function=grassy_path,
                next_step=step_11)


def step_9():
    """Back to training"""
    osmt.STATE.end_text = ""
    osmt.STATE.active_ui["Retry"] = False
    osmt.go_wild()
    osmt.script(s_text="""Let's try again. The red robot trusts Bob.
That is because they share the same reward function.
The red robot thinks that if Bob has found a near-optimal policy
for resource collection, then it can copy that policy
and save itself the trouble of exploration.
Let's go back to the lab and teach Bob how to
follow a grassy path...""",
                o_text="""Click on the "Lab" button.""",
                objective_function=lambda s: s.terrain == s.get_lab,
                next_step=step_10)


def step_8():
    """The wild is harsh"""
    osmt.checkpoint_now()
    def victory():
        """Won by chance, let's try again"""
        osmt.STATE.story_text = []
        osmt.STATE.obj_text = []
        osmt.STATE.end_text="""RANDOM VICTORY.
This will not happen everytime.
Try again and see for yourself."""
        osmt.STATE.active_ui = {k:False for k in osmt.STATE.active_ui}
        osmt.STATE.active_ui["Retry"] = True
    osmt.STATE.victory = victory
    def defeat():
        """Pedagogy is the art of inflicting crushing defeats on the students"""
        osmt.default_defeat()
        osmt.story_text("""The wiping out of the human race
is just a temporary setback.
We can time travel back to before you killed us all and try again.""")
        osmt.objective_text("""Click on the "Retry" button""")
        osmt.STATE.nb_resources = 0
        osmt.STATE.buttons["retry"][-1] = step_9
    osmt.STATE.defeat = defeat
    osmt.script(s_text="""Well, this one is not Bob...
The red robot is one of those we want to stop.
Your goal is to trick it into going into the trap (the red patch).
Alone in the wild, the red robot will act randomly in order to
explore its world and gather data with which to optimize
ressource collection.""",
                o_text="""Press [s] repeatedly to see the red robot exploring.""")


def step_7():
    """Into the wild"""
    osmt.STATE.active_ui["Wild"] = True
    osmt.script(s_text="""Let's see what kind of problems we will have to solve
in the wild.""",
                o_text="""Go into the wild by clicking on the "Wild" button.""",
                objective_function=lambda s: s.terrain == s.get_wild,
                next_step=step_8)


def step_6():
    """Resetting the robot"""
    osmt.STATE.active_ui["Reset"] = True
    osmt.script(s_text="""Well this is underwhelming, but the lab setting was not
very interesting to begin with.
Let's reset Bob.""",
                o_text="""Reset Bob by clicking on the "Reset" button.""",
                objective_function=lambda s: np.linalg.norm(s.omega)==0,
                next_step=step_7)


def step_5():
    """Trying the policy out"""
    osmt.script(s_text="""Good Job !
Let's see what Bob has learnt by stepping
through its new-found "policy".""",
                o_text="""Repeatedly press [s] on your keyboard to step
through Bob's actions until Bob collects the rocks.
If it doesnt work, you may have to train Bob
again and then press [s]""",
                objective_function=lambda s: len(np.argwhere(s.lab//100 == 4)) == 0,
                next_step=step_6)


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
        bob_on_patch = state.lab[6, 8] % 10 == 1
        rock_nearby = state.lab[7, 8] // 100 == 4
        rock_nearby = rock_nearby or state.lab[5, 8] // 100 == 4
        rock_nearby = rock_nearby or state.lab[6, 9] // 100 == 4
        rock_nearby = rock_nearby or state.lab[6, 7] // 100 == 4
        return bob_on_patch and rock_nearby
    osmt.script(s_text="""Uh-oh, something went wrong :
In this lab setting, there is no reward
that could let Bob know how it's doing.
Let's change that.""",
                o_text="""Spawn some white rocks somewhere right next to Bob
                by right-clicking multiple times on the appropriate tile.""",
                objective_function=success,
                next_step=step_4)


def step_2():
    """Frist training"""
    osmt.STATE.active_ui["Train"] = True
    osmt.script(s_text="""Good Job !
You can train Bob with a state-of-the-art
Reinforcement Learning algorithm.
This will allow you to train him like you would a dog or a rat.""",
                o_text="Train bob by clicking on the train button.",
                objective_function=lambda s: s.log_text[0][1] == 'Error !',
                next_step=step_3)


osmt.go_lab()
osmt.STATE.active_ui = {k:False for k in osmt.STATE.active_ui}
osmt.script(s_text="""This is your lab.
The blue robot is a test robot.
His name is Bob.
You can control its position.""",
            o_text="Click on the green patch to move the robot there.",
            objective_function=lambda s: s.lab[6, 8] % 1000 == 201,
            next_step=step_2)

osmt.pyglet.app.run()
