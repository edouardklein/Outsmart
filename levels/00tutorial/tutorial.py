#!/usr/bin/env python3
import numpy as np
from outsmart import return_copy
import outsmart as osmt
import os.path
import graphics as g
import ui

lvl_directory = os.path.dirname(os.path.abspath(__file__))


@return_copy
def step_1(s):
    """Move the robot"""
    s.ui.filename = lvl_directory+"/tutorial"
    s = ui.load(s)
    s = ui.lab(s)
    s.ui.active = ui.ALL_INACTIVE
    s.ui.active["editor_wild_lab_terrain"] = True
    s.ui.active.update({k: True for k in ui.ALL_INACTIVE
                        if "_text" in k})
    s.ui.log_text = ""
    s.ui.story_text = """This is your lab.
The blue robot is a test robot.
His name is Bob.
You can control its position."""
    s.ui.obj_text = "Click on the green patch to move the robot there."
    s.obj_func = lambda s: s.lab[6, 8] % 1000 == 201
    s.next_func = step_2
    return s


@return_copy
def step_2(s):
    """Frist training"""
    print("Now in STEP 2")
    s.ui.active["lab_train"] = True
    s.ui.story_text = """Good Job !
You can train Bob with a state-of-the-art
Reinforcement Learning algorithm.
This will allow you to train him like you would a dog or a rat."""
    s.ui.obj_text = "Train bob by clicking on the train button."
    s.obj_func = lambda s: s.ui.log_text.startswith('Error !')
    s.next_func = step_3
    print("Exiting step2")
    return s

# FIXME: Ecrire les crédits quelque part


@return_copy
def step_3(s):
    """Rocks spawning"""
    def success(s):
        #HERE : success does not toggle
        robot_loc = np.argwhere(s.lab % 10 != 0)[0]
        i_list = [robot_loc[0]-1, robot_loc[0], robot_loc[0]+1]
        i_list = map(lambda i: i % (osmt.I_MAX+1), i_list)
        j_list = [robot_loc[1]-1, robot_loc[1], robot_loc[1]+1]
        j_list = map(lambda j: j % (osmt.J_MAX+1), j_list)
        rock_nearby = False
        for i in i_list:
            for j in j_list:
                rock_nearby = rock_nearby or s.lab[i, j] // 100 == 4
        return rock_nearby
    s.ui.story_text = """Uh-oh, something went wrong :
In this lab setting, there is no reward
that could let Bob know how it's doing.
Let's change that."""
    s.ui.obj_text = """Spawn some white rocks somewhere right next to Bob
by right-clicking multiple times on the appropriate tile."""
    s.ui.active["lab_next_tile"] = True
    s.ui.active["lab_prev_tile"] = True
    s.ui.active["lab_current_tile"] = True
    s.obj_func = success
    s.next_func = step_4
    return s


@return_copy
def step_4(s):
    """Sucessful training"""
    s.ui.story_text = """Good Job !
Now, training Bob will succeed.
Bob is wired to collect rocks and crystals. You can't change that.
You can, however, change the environment and use rocks and
crystals as a motivator to make Bob do what you want.
When dealing with animals food is used as a reward.
This is the same here, only with robots."""
    s.ui.obj_text = "Try again to train Bob by clicking on the"
    """"Train" button."""
    s.obj_func = lambda s: s.ui.log_text.startswith("Training")
    s.next_func = step_5
    return s


@return_copy
def step_5(s):
    """Trying the policy out"""
    s.ui.story_text = """Good Job !
Let's see what Bob has learnt by stepping
through its new-found "policy"."""
    s.ui.obj_text = """Repeatedly press [s] on your keyboard to step
through Bob's actions until Bob collects the rocks.
If it doesnt work, you may have to train Bob
again and then press [s]"""
    s.obj_func = lambda s: len(np.argwhere(s.lab//100 == 4)) == 0,
    s.next_func = step_6
    return s


@return_copy
def step_6(s):
    """Resetting the robot"""
    s.ui.active["lab_wild_reset"] = True
    s.ui.story_text = """Well this is underwhelming, but the lab setting was not
very interesting to begin with.
Let's reset Bob."""
    s.ui.obj_text = """Reset Bob by clicking on the "Reset" button."""
    s.obj_func = lambda s: np.linalg.norm(s.rl.omega) == 0,
    s.next_func = step_7
    return s


@return_copy
def step_7(s):
    """Into the wild"""
    s.ui.active['lab_go_wild'] = True
    s.ui.story_text = """Let's see what kind of problems we will have to solve
in the wild."""
    s.ui.obj_text = """Go into the wild by clicking on the "Wild" button."""
    s.obj_func = lambda s: s.ui.terrain == ui.get_wild,
    s.next_func = step_8
    return s


@return_copy
def step_8(s):
    """The wild is harsh"""
    ui.checkpoint_now(s)

    @return_copy
    def victory(s):
        """Won by chance, let's try again"""
        s = ui.victory()
        s.ui.end_text = """RANDOM VICTORY.
This will not happen everytime.
Try again and see for yourself."""
        s.ui.obj_text = """Click on the "Retry" button"""
        osmt.STATE.active_ui["retry"] = True
    s.ui.victory = victory

    @return_copy
    def defeat(s):
        """Pedagogy is the art of inflicting crushing defeats"""
        s = ui.defeat()
        s.ui.story_text = """The wiping out of the human race
is just a temporary setback.
We can time travel back to before you killed us all and try again."""
        s.ui.obj_text = """Click on the "Retry" button"""
        s.nb_resources = 0
        g.BUTTONS["retry"][-1] = lambda s: step_9(s)
    s.ui.defeat = defeat

    s.ui.story_text = """Well, this one is not Bob...
The red robot is one of those we want to stop.
Your goal is to trick it into going into the trap (the red patch).
Alone in the wild, the red robot will act randomly in order to
explore its world and gather data with which to optimize
ressource collection."""
    s.ui.obj_text = """Press [s] repeatedly to see the red robot exploring."""
    return s


@return_copy
def step_9(s):
    """Back to training"""
    s.ui.end_text = ""
    s.ui.active["retry"] = False
    s = ui.wild(s)
    s.ui.story_text = """Let's try again. The red robot trusts Bob.
That is because they share the same reward function.
The red robot thinks that if Bob has found a near-optimal policy
for resource collection, then it can copy that policy
and save itself the trouble of exploration.
Let's go back to the lab and teach Bob how to
follow a grassy path..."""
    s.ui.obj_text = """Click on the "Lab" button."""
    s.obj_func = lambda s: s.ui.terrain == ui.get_lab
    s.next_func = step_10
    return s


@return_copy
def step_10(s):
    """Create the path"""
    s.lab = np.loadtxt(lvl_directory + "/grassy_path.lab")
    s.wild = np.loadtxt(lvl_directory + "/tutorial.wild")
    s.ui.victory = ui.victory
    s.ui.defeat = ui.defeat

    def grassy_path(s):
        answer = all(s.lab[4, 2:8] // 100 == 2)
        answer = answer and all(s.lab[5, 1:9] // 100 == 1)
        answer = answer and all(s.lab[3, 1:9] // 100 == 1)
        answer = answer and s.lab[4, 1] // 100 == 1
        return answer

    s.ui.story_text = """We have created the start of a grassy path in the lab.
Complete it so that it looks like the one that was ahead of the
red robot in the wild.
Create only the grassy path, not the crystals around it.
You will use the crystals later to train Bob."""
    s.ui.obj_text = """Right click on the appropriate tiles to complete
the path."""
    s.obj_func = grassy_path
    s.next_func = step_11
    ui.checkpoint_now(s)
    return s


@return_copy
def step_11(s):
    """WIP"""
    s.ui.story_text = """Time to tie it all together :
By placing resources (rocks or crystals) and training Bob,
teach it to follow the grassy path.
You can reset Bob and start over if the training does not work.
You can train mutliple times to add on to Bob's experience.
When you feel confident, go into the wild and press [s] to see
if you tricked the red robot."""
    return s


g.STATE = step_1(g.STATE)
