import pyglet
from pyglet.gl import *
glEnable(GL_BLEND)
glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
from pyglet.window import key
import numpy as np
import numpy.random as nprand
from itertools import zip_longest
import copy
import math
import random
import glob
import functools
import itertools
import subprocess
import os.path

TILE_SIZE_X = 64
TILE_SIZE_Y = 32

I_MAX = 9
J_MAX = 9

X_MAX = 2*(I_MAX+1)*TILE_SIZE_X
Y_MAX = (2*(J_MAX+1)+2)*TILE_SIZE_Y

WINDOW = pyglet.window.Window(X_MAX, Y_MAX)
WINDOW.set_location(0, 0)

pyglet.font.add_file('img/kenvector_future_thin.ttf')
KenFuture = pyglet.font.load('KenVector Future Thin Regular')
# The display system :
# Each tile has a multi-digit number associated with it.
# Each position encodes a different information.
# The least significant digit is the presence of the robot:
#  - 0: No robot
#  - 1: Bob
#  - 2: Not-Bob
# The second digit is the presence of a trap:
#  - 0: No trap
#  - 1: Trap
# The third digit from the right is the type of terrain:
# - 0: No terrain (no use case, but I feel good knowing 0 means 'nothing')
# - 1: bare (e.g. Earth)
# - 2: No resources (e.g. grass)
# - 3: Resource type 1 (e.g. Crystals)
# - 4: Resouce type 2 (e.g. Rocks)
# The fourth digit is the set of tiles to use (to be implemented)
# - 0: Earth, grass, crystals, rocks
# - 1: (not implemented yet) e.g. grass, road, house, trees
# - 2: ...
# A tile is a superposition of the information at each position.
# Each position encodes things that can not be superimposed (e.g. it is not
# possible to have both crystals and rocks on the same tiles)
# On the other hand it is possible to have a Not-Bob on a trap on some rocks.
# This would tile 0412
# To draw tile 0412, either the file 0412.png exists and is displayed
# or we build it from files that sum to 0412.png, i.e. 0410 and 0002

IMAGES = {int(s[-8:-4]): pyglet.image.load(s)
          for s in glob.glob('img/'+'[0-9]'*4+'.png')}


@functools.lru_cache(None)
def nb2images(number):
    """Return the list of images one has to draw to represent the
    given number"""
    try:
        return [IMAGES[number]]
    except KeyError:
        all_comb = sum(map(list, [itertools.combinations(IMAGES.keys(), l)
                                  for l in [2, 3, 4]]), [])
        comb = [c for c in all_comb if sum(c) == number][0]
        return [IMAGES[n] for n in reversed(sorted(comb))]


def draw_sprite(number, x, y):
    """Draw at x,y the sprite(s) representing the given number"""
    for im in nb2images(number):
        sprite = pyglet.sprite.Sprite(im, x, y)
        sprite.draw()


BUTTON_LEFT = pyglet.image.load('img/button_left.png')
BUTTON_MID = pyglet.image.load('img/button_mid.png')
BUTTON_RIGHT = pyglet.image.load('img/button_right.png')

DEFEAT_SONG = pyglet.media.load('snd/dead.mp3')

ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT", "PICK"]


def default_victory():
    """FIXME: Send back to the main menu, or something"""
    STATE.story_text = []
    STATE.obj_text = []
    STATE.end_text = """VICTORY !
You got one robot, try the other levels !"""
    STATE.active_ui = {k: False for k in STATE.active_ui}
    STATE.active_ui["Retry"] = True


def victorious(s):
    """Return True if NotBob is on a trap"""
    return any((s.wild % 100 == 12).reshape(-1))


def default_defeat():
    """FIXME: Propose to try again"""
    global STATE
    play(DEFEAT_SONG)
    STATE.story_text = []
    STATE.obj_text = []
    STATE.end_text = """DEFEAT.
The robot grew in numbers and wiped out the human race."""
    STATE.active_ui = {k: False for k in STATE.active_ui}
    STATE.active_ui["Retry"] = True


def losing(s):
    """Return True if nb_resources goes past 0"""
    return s.nb_resources > 0


class State:
    def set_lab(self, m):
        self.lab = m

    def get_lab(self):
        return self.lab

    def set_wild(self, m):
        self.wild = m

    def get_wild(self):
        return self.wild

    def __init__(self):
        self.lab = np.ones((I_MAX+1, J_MAX+1))  # Terrain for the lab
        self.lab[0, 0] = -1

        self.wild = self.lab.copy()

        self.nb_resources = 0  # Amount of resources collected by Not-Bob

        self.terrain = self.get_lab
        self.set_terrain = self.set_lab

        self.obj_func = lambda s: False  # Return True when objective
        # is reached
        self.next_func = lambda: None  # Called when obj_func returns True

        self.obj_text = []  # Displayed in the upper right
        self.story_text = []  # Displayed in the upper left
        self.log_text = []  # Displayed below the train button
        self.end_text = ""  # Center of screen, big

        self.buttons = {}

        self.filename = "current_mapfile"

        self.omega = np.zeros(4*9*len(ACTIONS))  # Paramters for the Q-function
        self.active_ui = {"Train": True,
                          "Reset": True,
                          "Lab": False,
                          "Wild": True,
                          "Load": False,
                          "Save": False,
                          "TileSelector": False,
                          "Retry": False}

        self.level_editor = False

        self.victorious = victorious
        self.victory = default_victory
        self.losing = losing
        self.defeat = default_defeat
        self.selected_tile = 0

        self.player = pyglet.media.Player()
        self.on_the_fly_TTS_generate = False


STATE = State()


CURSORS = {}
for k in range(1, 5):
    CURSORS[k] = pyglet.window.ImageMouseCursor(nb2images(k*100)[0], 0, 0)
CURSORS[0] = WINDOW.CURSOR_DEFAULT


def load_state(s, filename):
    answer = copy.deepcopy(s)
    answer.lab = np.loadtxt(filename+'.lab')
    answer.lab.flags.writeable = True
    answer.wild = np.loadtxt(filename+'.wild')
    answer.wild.flags.writeable = True
    return answer


def load_cb():
    global STATE
    fn = STATE.filename
    STATE = load_state(STATE, STATE.filename)
    STATE.filename = fn


def save_state(s, filename):
    np.savetxt(filename+'.lab', s.lab, fmt="%d")
    np.savetxt(filename+'.wild', s.wild, fmt="%d")


def save_cb():
    save_state(STATE, STATE.filename)


def train():
    """Train the robot"""
    global STATE
    sars_list = []
    for i in range(10):
        ma = walk(STATE.terrain(), q_function(STATE.omega), 10, rand=.5)
        sars_list += sars(ma)
    try:
        STATE.omega = Q_learning(STATE.omega, sars_list)
    except AssertionError:
        STATE.log_text = [[[10, 75], "Error !"]]
    else:
        STATE.log_text = [[[10, 75], "Training successful !"]]
    print_omega(STATE.omega)


def reset():
    """Reset the Q-function of the robot"""
    global STATE
    STATE.omega = np.zeros(4*9*len(ACTIONS))
    STATE.log_text = [[[10, 75], "Bob has been reset !"]]


def load_checkpoint():
    global STATE
    STATE = copy.deepcopy(CHECKPOINT)


def go_wild():
    """Whoohoo GO WILD !"""
    global STATE
    STATE.terrain = STATE.get_wild
    STATE.set_terrain = STATE.set_wild
    STATE.active_ui["Wild"] = False
    STATE.active_ui["Lab"] = True
    STATE.active_ui["Train"] = False
    STATE.active_ui["Reset"] = False


def go_lab():
    """Go back to the lab"""
    global STATE
    STATE.terrain = STATE.get_lab
    STATE.set_terrain = STATE.set_lab
    STATE.active_ui["Wild"] = True
    STATE.active_ui["Lab"] = False
    STATE.active_ui["Train"] = True
    STATE.active_ui["Reset"] = True


def set_image_as_cursor(i):
    def fun():
        global STATE
        STATE.selected_tile = i
    return fun

for x, y, uid, group_ui, text, cb in [[10, 20, "reset", "Reset",
                                       "Reset", reset],
                                      [10, 100, "train", "Train",
                                       "Train", train],
                                      [10, 140, "load_map", "Load",
                                       "Load a map", load_cb],
                                      [10, 180, "save_map", "Save",
                                       "Save current map", save_cb],
                                      [900, 20, "laboratory", "Lab",
                                       "Lab", go_lab],
                                      [900, 20, "wilderness", "Wild",
                                       "Wild", go_wild],
                                      [10, 10,
                                       "retry", "Retry",
                                       "Retry", load_checkpoint]]:
    pixel_length = len(text)*10.5
    STATE.buttons[uid] = [[x, y,
                           x+12+math.ceil(pixel_length//16)*16,
                           y+26], group_ui, text, cb]

def checkpoint_now():
    global CHECKPOINT
    CHECKPOINT = copy.deepcopy(STATE)


for x, y, uid, group_ui, img, cb in [[1000, 100, "img_1", "TileSelector",
                                     nb2images(100)[0],
                                      set_image_as_cursor(1)],
                                     [1140, 100, "img_2", "TileSelector",
                                      nb2images(200)[0],
                                     set_image_as_cursor(2)],
                                     [1000, 0, "img_3", "TileSelector",
                                      nb2images(300)[0],
                                     set_image_as_cursor(3)],
                                     [1140, 0, "img_4", "TileSelector",
                                      nb2images(400)[0],
                                     set_image_as_cursor(4)]]:
    STATE.buttons[uid] = [[x, y, x+img.width, y+img.height], group_ui, img, cb]


def random_terrain():
    """Return a randomized terrain"""
    answer = np.ones((I_MAX+1, J_MAX+1))
    for i, j in zip(nprand.random_integers(0, I_MAX, 8),
                    nprand.random_integers(0, J_MAX, 8)):
        answer[i, j] = nprand.random_integers(2, 4)*100
    answer[0, 0] += 1
    return answer


def phi(s, a):
    """Feature vector on the state-action space"""
    s = s.copy().reshape(-1)
    s = s // 100  # Extracting third digit
    s = s % 1000
    answer = np.zeros(4*9*len(ACTIONS))
    start = 4*9*ACTIONS.index(a)
    answer[start:start+9] = (s == 1)*1.
    answer[start+9:start+18] = (s == 2)*1.
    answer[start+18:start+27] = (s == 3)*1.
    answer[start+27:start+36] = (s == 4)*1.
    return answer


def q_function(omega):
    """Returns the q function from the weight vectors"""
    def answer(s, a):
        return np.dot(omega, phi(s, a))
    return answer


def greedy(q, s):
    """Choose an action according to a random choice wheited by the Q-value"""
    # From https://docs.python.org/dev/library/random.html
    print(s)
    qsa = [q(s, a) for a in ACTIONS]
    if all([x == 0 for x in qsa]):
        a = random.choice(ACTIONS)
        print("All zero, radomly choosing "+a)
        return a
    for a in ACTIONS:
        print("q(s, %s) = %f" % (a, q(s, a)))
    a = ACTIONS[np.argmax(qsa)]
    print("Choosing "+a)
    return a


def ij2xy(m, i, j):
    """We use matrix-oriented coordinates i,j, but to display we need oriented
    abc/ord x, y instead, using isometric projection"""
    x = (m.shape[0] - i + j - 1)*TILE_SIZE_X
    y = (m.shape[1]+m.shape[0] - i - j - 2)*TILE_SIZE_Y
    return x, y


def draw_text(text_list):
    """Draw the given [[x, y], text] list"""
    for [x, y], t in text_list:
        label = pyglet.text.Label(t, x=x, y=y,
                                  font_name='KenVector Future Thin Regular')
        label.draw()


def draw_end_text(text):
    """Draw the text over the whole screen"""
    l = text.split("\n")
    label = pyglet.text.Label(l[0],
                              font_name='KenVector Future Thin Regular',
                              font_size=36,
                              x=WINDOW.width//2, y=WINDOW.height//2,
                              anchor_x='center', anchor_y='center')
    label.draw()
    for i, t in enumerate(l[1:]):
        label = pyglet.text.Label(t, x=WINDOW.width//2-200,
                                  y=WINDOW.height//2-100-i*25,
                                  font_name='KenVector Future Thin Regular')
        label.draw()


# https://docs.python.org/3.4/library/itertools.html
def grouper(iterable, n, fillvalue=" "):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return map(lambda t: ''.join(t), zip_longest(*args, fillvalue=fillvalue))


def draw_buttons(buttons):
    for _, [rect, _, content, _] in buttons.items():
        x, y, max_x, max_y = rect
        if type(content) == str:
            sprite = pyglet.sprite.Sprite(BUTTON_LEFT, x=x, y=y)
            sprite.draw()
            pixel_length = len(content)*10.5
            for i in range(0, math.ceil(pixel_length//16)):
                sprite = pyglet.sprite.Sprite(BUTTON_MID, x=x+6+i*16, y=y)
                sprite.draw()
            sprite = pyglet.sprite.Sprite(BUTTON_RIGHT, x=max_x-6,
                                          y=y)
            sprite.draw()
            lbl = pyglet.text.Label(content, x=x+6, y=y+9,
                                    font_name='KenVector Future Thin Regular')
            lbl.draw()
        else:  # image button
            sprite = pyglet.sprite.Sprite(content, x=x, y=y)
            sprite.draw()


def draw_assets(s):
    "Draw the game state"
    draw_buttons({k: s.buttons[k]
                  for k, [_, group_ui, _, _] in s.buttons.items()
                  if s.active_ui[group_ui]})
    for t in [STATE.obj_text, STATE.story_text, STATE.log_text]:
        draw_text(t)
    m = s.terrain()
    for i in range(0, m.shape[0]):
        for j in range(0, m.shape[1]):
            x, y = ij2xy(m, i, j)
            draw_sprite(m[i, j], x, y)
    if STATE.end_text:
        draw_end_text(STATE.end_text)


def xy_text(starting_xy, text):
    """Return the list of [[X, y], text] items that draw_text() will
    understand"""
    x, y = starting_xy
    return [[[x, y - i*15], line] for i, line in enumerate(text)]


def story_text(text):
    """Save the given text to be displayed in the upper left corner"""
    global STATE
    print("Story text now : "+text)
    STATE.story_text = xy_text([0, Y_MAX-10], text.split("\n"))
    play(text=text)


def objective_text(text):
    """Save the given text to be displayed in the upper right corner"""
    global STATE
    STATE.obj_text = xy_text([X_MAX-600, Y_MAX-10],
                             ["OBJECTIVES"]+text.split("\n"))


def script(s_text="", o_text="",
           objective_function=lambda s: False,
           next_step=lambda: None):
    """Script the user interface

    *_text variables are self explanatory

    at each redraw, objective_function will be called. If it returns True,
    next_step is called."""
    global STATE
    story_text(s_text)
    objective_text(o_text)
    STATE.obj_func = objective_function
    STATE.next_func = next_step


def play(media=None,media_file="", text=""):
    global STATE
    #we prefer on-the-fly generation if availabe, to be up to date
    if text and STATE.on_the_fly_TTS_generate:
        if not media_file:
            media_file="tmp_TTS.wav"
        cmd = STATE.TTS_command.format(out=media_file)
        cmd = cmd.split()
        p_cmd = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        p_cmd.communicate(input=text.encode())
        if p_cmd.returncode == 0:
            media = pyglet.media.load(media_file)
        
    if media:
        if not STATE.player.playing:
            STATE.player.queue(media)
            STATE.player.play()
        
def set_TTS_generate(activate = False, method="festival"):
    if activate:
        print("WARN: Activating TTS-OTF with %s."%method)
        if method=="festival":
            TTS_exe = "/usr/bin/text2wave"
            if not os.path.exists(TTS_exe):
                print("ERR: %s not found."%TTS_exe)
                return
            STATE.TTS_command = """%s -o {out}"""%TTS_exe
        elif method=="OSX-say":
            TTS_exe = "say"
            if not os.path.exists(TTS_exe):
                print("ERR: %s not found."%TTS_exe)
                return
            STATE.TTS_command = """%s -o {out}"""%TTS_exe
    STATE.on_the_fly_TTS_generate = activate

def robot_state(terrain):
    """Return the state visible to a robot"""
    t = terrain.copy()
    i, j = np.argwhere(t % 10 != 0)[0]
    if i-1 < 0:
        t = np.roll(t, 1, 0)
        i += 1
    elif i+1 > 9:
        t = np.roll(t, -1, 0)
        i -= 1
    if j-1 < 0:
        t = np.roll(t, 1, 1)
        j += 1
    elif j+1 > 9:
        t = np.roll(t, -1, 1)
        j -= 1
    return t[i-1:i+2, j-1:j+2]


def apply_action(m, action):
    """Return a copy of m after action has been applied to it"""
    if action == "RIGHT":
        m = move_robot(m, None, None, 1, 0)
    elif action == "LEFT":
        m = move_robot(m, None, None, -1, 0)
    elif action == "DOWN":
        m = move_robot(m, None, None, 0, 1)
    elif action == "UP":
        m = move_robot(m, None, None, 0, -1)
    elif action == "PICK":
        m = m.copy()
        robot_loc = np.argwhere(m % 10 != 0)[0]
        if m[tuple(robot_loc)] % 1000 // 100 == 4:  # ROCKS
            m[tuple(robot_loc)] -= 200  # GRASS
        elif m[tuple(robot_loc)] % 1000 // 100 == 3:  # CRYSTALS
            m[tuple(robot_loc)] -= 100  # GRASS
    return m


def walk(m, q, length, rand=0):
    """Apply the greedy policy with respect to the given q and return the list of
    state, action pairs"""
    answer = [m]
    for i in range(0, length):
        s = robot_state(m)
        a = greedy(q, s) if np.random.rand() > rand else random.choice(ACTIONS)
        m = apply_action(m, a)
        answer += [a, m]
    return answer


def display_traj(traj):
    global STATE
    print("Drawing traj of length %d" % len(traj))
    if len(traj) > 0:
        STATE.set_terrain(traj[0])
        pyglet.clock.schedule_once(lambda t: display_traj(traj[1:]), 0)


def reward(s1, a, s2):
    """Test reward function : we like to pick things"""
    if a == 'PICK' and s1[1, 1] % 1000 // 100 in [3, 4]:  # SHROOMS or CRYSTALS
        return 1
    else:
        return 0


def sars(ma):
    """Turns a list of terrain, action into a sars list suitable for
    Q-learning"""
    answer = []
    for m1, a, m2 in zip(ma[::2], ma[1::2], ma[2::2]):
        s1 = robot_state(m1)
        s2 = robot_state(m2)
        r = reward(s1, a, s2)
        answer.append([s1, a, r, s2])
    return answer


def Q_learning(Q, sars):
    answer = Q.copy()
    Q_func = q_function(answer)
    alpha = 1  # PARAMETER
    gamma = 0.9  # PARAMETER
    d = float('inf')
    q_iter = 0
    assert not all([r == 0 for _, _, r, _ in sars])
    while d > 1. and q_iter < 5:
        X = []
        Y = []
        for s1, a, r, s2 in sars:
            old = Q_func(s1, a)
            Vs2 = max([Q_func(s2, a) for a in ACTIONS])
            new = old + alpha*(r + gamma*Vs2 - old)
            X.append(phi(s1, a))
            Y.append(new)
        X = np.array(X)
        Y = np.array(Y)
        answer = np.dot(np.dot(np.linalg.pinv(np.dot(X.T, X)), X.T), Y)
        print("NORM of omega %f" % np.linalg.norm(answer))
        old_q = np.array([Q_func(s, a) for s, a, _, _ in sars])
        Q_func = q_function(answer)
        new_q = np.array([Q_func(s, a) for s, a, _, _ in sars])
        d = np.linalg.norm(old_q-new_q)
        print("Iteration %d, |Q(s,a) - Q'(s,a)| is %f" % (q_iter, d))
        q_iter += 1
    return answer


def print_omega(omega):
    terrains = ["EARTH", "GRASS", "CRYSTALS", "SHROOMS"]
    for a in range(len(ACTIONS)):
        print(ACTIONS[a])
        for t in range(len(terrains)):
            print(terrains[t])
            x = omega[4*9*a + 9*t:4*9*a + 9*t+9]
            print(x.reshape(3, 3))


@WINDOW.event
def on_draw():
    global STATE
    #print("Drawing main")
    WINDOW.clear()
    try:
        if STATE.victorious(STATE):
            STATE.victory()
        elif STATE.losing(STATE):
            STATE.defeat()
        elif STATE.obj_func(STATE):
            STATE.next_func()
    except Exception as e:
        print(e)
    draw_assets(STATE)


@WINDOW.event
def on_mouse_motion(x, y, dx, dy):
    WINDOW.set_mouse_cursor(CURSORS[STATE.selected_tile])


def available_resources(m):
    """Return the number of available resources on the given map"""
    print(m % 1000 // 100 >= 3)
    return sum((m % 1000 // 100 >= 3).reshape(-1))


@WINDOW.event
def on_key_press(symbol, modifiers):
    print(symbol)
    global STATE
    if symbol == key.S:  # Step
        print("Stepping")
        a = greedy(q_function(STATE.omega), robot_state(STATE.terrain()))
        nb_old = available_resources(STATE.terrain())
        print(a)
        STATE.set_terrain(apply_action(STATE.terrain(), a))
        if STATE.terrain == STATE.get_wild:
            STATE.nb_resources += nb_old - available_resources(STATE.terrain())
    elif symbol == key.Q:  # Quit
        print("Quitting")
        pyglet.app.exit()
    elif symbol == key.W:  # Wild
        go_wild()
    if STATE.terrain == STATE.get_wild and not STATE.level_editor:
        return  # Deactivate the next keys when in the wild
    if symbol == key.RIGHT:
        STATE.set_terrain(apply_action(STATE.terrain(), "RIGHT"))
    elif symbol == key.LEFT:
        STATE.set_terrain(apply_action(STATE.terrain(), "LEFT"))
    elif symbol == key.DOWN:
        STATE.set_terrain(apply_action(STATE.terrain(), "DOWN"))
    elif symbol == key.UP:
        STATE.set_terrain(apply_action(STATE.terrain(), "UP"))
    elif symbol == key.SPACE:
        STATE.set_terrain(apply_action(STATE.terrain(), "PICK"))
    elif symbol == key.R:  # Randomize
        STATE.set_terrain(random_terrain())
    elif symbol == key.T:  # Train
        print('Walking')
        train()


def move_robot(m, i, j, di=0, dj=0):
    """Return the m matrix after the robot has been moved to i,j"""
    m = m.copy()
    robot_loc = np.argwhere(m % 10 != 0)[0]
    if i is None:
        i = robot_loc[0]
    if j is None:
        j = robot_loc[1]
    robot = m[tuple(robot_loc)] % 10
    m[tuple(robot_loc)] = m[tuple(robot_loc)] // 10 * 10
    m[(i + di) % (I_MAX+1), (j + dj) % (J_MAX+1)] += robot
    return m


def check_buttons(x, y, button, modifiers):
    if button == pyglet.window.mouse.LEFT:
        for rect, _, _, cb in [STATE.buttons[k] for k in STATE.buttons
                               if STATE.active_ui[STATE.buttons[k][1]]]:
            if x >= rect[0] and y >= rect[1] and x <= rect[2] and y <= rect[3]:
                print("Callback ! ")
                cb()
                return True
    return False


def xy2ij(x, y):
    ix = x
    iy = ((2*(J_MAX+1)+2)*TILE_SIZE_Y-y)
    ix = ix / TILE_SIZE_X / 2
    iy = iy / TILE_SIZE_Y / 2 - .2
    i = round(iy-ix)+4
    j = round(ix+iy)-6  # TGCM!
    return i, j


def on_mouse_press_lvl_editor(*args):
    global STATE
    x, y, button, modifiers = args
    if check_buttons(*args):
        return
    i, j = xy2ij(x, y)
    if i not in range(10) or j not in range(10):
        return
    if button == pyglet.window.mouse.LEFT:
        if STATE.selected_tile:
            STATE.terrain()[i, j] = STATE.selected_tile
            if STATE.terrain()[i, j] < 0:
                STATE.terrain()[i, j] = - STATE.terrain()[i, j]
        else:
            STATE.set_terrain(move_robot(STATE.terrain(), i, j))
    if button == pyglet.window.mouse.RIGHT and STATE.selected_tile:
        STATE.selected_tile = 0


def on_mouse_press_game(*args):
    global STATE
    x, y, button, modifiers = args
    if check_buttons(*args):
        return
    if STATE.terrain == STATE.get_wild:
        return  # Deactivate terrain modifs when in the wild
    i, j = xy2ij(x, y)
    if i not in range(10) or j not in range(10):
        return
    if button == pyglet.window.mouse.LEFT:
        STATE.set_terrain(move_robot(STATE.terrain(), i, j))
    elif button == pyglet.window.mouse.RIGHT:
        STATE.terrain()[i, j] = (STATE.terrain()[i, j] + 100) % 500
        if STATE.terrain()[i, j] < 100:
            STATE.terrain()[i, j] += 100


@WINDOW.event
def on_mouse_press(*args):
    if STATE.level_editor:
        on_mouse_press_lvl_editor(*args)
    else:
        on_mouse_press_game(*args)
