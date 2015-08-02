import pyglet
from pyglet.gl import *
glEnable(GL_BLEND)
glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
from pyglet.window import key
import numpy as np
import numpy.random as nprand
from itertools import zip_longest
import math
import random

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
EARTH = pyglet.image.load('img/earth.png')
GRASS = pyglet.image.load('img/grass.png')
CRYSTALS = pyglet.image.load('img/crystals.png')
ROCKS = pyglet.image.load('img/rocks.png')
ROBOT = pyglet.image.load('img/robot_blue_right.png')
BUTTON_LEFT = pyglet.image.load('img/button_left.png')
BUTTON_MID = pyglet.image.load('img/button_mid.png')
BUTTON_RIGHT = pyglet.image.load('img/button_right.png')
IMAGES = {1: EARTH,
          2: GRASS,
          3: CRYSTALS,
          4: ROCKS,
          -1: ROBOT}  # -2 robot & shrooms, -2 robots and berries, etc.

ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT", "PICK"]


class State:
    def __init__(self):
        self.lab = np.ones((I_MAX+1, J_MAX+1))  # Terrain for the lab

        self.obj_func = lambda: False  # Return True when objective is reached
        self.next_func = lambda: None  # Called when obj_func returns True

        self.obj_texts = []  # Displayed in the upper right
        self.story_text = []  # Displayed in the upper left
        self.log_text = []  # Displayed below the train button

        self.buttons = {}

        self.omega = np.zeros(4*9*len(ACTIONS))  # Paramters for the Q-function


STATE = State()


def random_terrain():
    """Return a randomized terrain"""
    answer = np.ones((I_MAX+1, J_MAX+1))
    for i, j in zip(nprand.random_integers(0, I_MAX, 8),
                    nprand.random_integers(0, J_MAX, 8)):
        answer[i, j] = nprand.random_integers(2, 4)
    answer[0, 0] = -answer[0, 0]
    return answer


def phi(s, a):
    """Feature vector on the state-action space"""
    s = abs(s).reshape(-1)
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


# https://docs.python.org/3.4/library/itertools.html
def grouper(iterable, n, fillvalue=" "):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return map(lambda t: ''.join(t), zip_longest(*args, fillvalue=fillvalue))


def draw_buttons(buttons):
    for text, [rect, _] in buttons.items():
        x, y, max_x, max_y = rect
        sprite = pyglet.sprite.Sprite(BUTTON_LEFT, x=x, y=y)
        sprite.draw()
        pixel_length = len(text)*10.5
        for i in range(0, math.ceil(pixel_length//16)):
            sprite = pyglet.sprite.Sprite(BUTTON_MID, x=x+6+i*16, y=y)
            sprite.draw()
        sprite = pyglet.sprite.Sprite(BUTTON_RIGHT, x=max_x-6,
                                      y=y)
        sprite.draw()
        label = pyglet.text.Label(text, x=x+6, y=y+9,
                                  font_name='KenVector Future Thin Regular')
        label.draw()


def new_button(x, y, text, callback):
    global STATE
    pixel_length = len(text)*10.5
    STATE.buttons[text] = [[x, y, x+12+math.ceil(pixel_length//16)*16, y+26],
                           callback]


def draw_assets(s):
    "Draw the game state"
    if s.obj_func(STATE.lab):
        s.next_func()
    draw_buttons(s.buttons)
    for t in [STATE.obj_text, STATE.story_text, STATE.log_text]:
        draw_text(t)
    m = s.lab
    for i in range(0, m.shape[0]):
        for j in range(0, m.shape[1]):
            x, y = ij2xy(m, i, j)
            sprite = pyglet.sprite.Sprite(IMAGES[1], x=x, y=y)
            sprite.draw()
            if abs(m[i, j]) > 1:
                sprite = pyglet.sprite.Sprite(IMAGES[abs(m[i, j])], x=x, y=y)
                sprite.draw()
            if m[i, j] < 0:
                sprite = pyglet.sprite.Sprite(IMAGES[-1], x=x, y=y)
                sprite.draw()


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


def robot_state(terrain):
    """Return the state visible to a robot"""
    t = terrain.copy()
    i, j = np.argwhere(t < 0)[0]
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
    """Return m after action has been applied to it"""
    robot_loc = np.argwhere(m < 0)[0]
    m = m.copy()
    m[tuple(robot_loc)] = -m[tuple(robot_loc)]
    if action == "RIGHT":
        robot_loc[1] = (robot_loc[1]+1) % (J_MAX+1)
    elif action == "LEFT":
        robot_loc[1] = (robot_loc[1]-1) % (J_MAX+1)
    elif action == "DOWN":
        robot_loc[0] = (robot_loc[0]+1) % (I_MAX+1)
    elif action == "UP":
        robot_loc[0] = (robot_loc[0]-1) % (I_MAX+1)
    elif action == "PICK":
        if m[tuple(robot_loc)] == 4:  # SHROOMS
            m[tuple(robot_loc)] = 1  # EARTH
        elif m[tuple(robot_loc)] == 3:  # CRYSTALS
            m[tuple(robot_loc)] = 2  # BUSHES
    m[tuple(robot_loc)] = -m[tuple(robot_loc)]
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
        STATE.lab = traj[0]
        pyglet.clock.schedule_once(lambda t: display_traj(traj[1:]), 0)


def reward(s1, a, s2):
    """Test reward function : we like to pick things"""
    if a == 'PICK' and s1[1, 1] in [3, 4]:  # SHROOMS or CRYSTALS
        return 1
    else:
        return 0


def sars(ma):
    """Turns a list of terrain, action into a sars list suitable for
    Q-learning"""
    answer = []
    for m1, a, m2 in zip(ma[::2], ma[1::2], ma[2::2]):
        s1 = abs(robot_state(m1))
        s2 = abs(robot_state(m2))
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
    while d > 1. or q_iter < 5:
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
    print("Drawing main")
    WINDOW.clear()
    draw_assets(STATE)


def train():
    """Train the robot"""
    global STATE
    sars_list = []
    for i in range(10):
        ma = walk(STATE.lab, q_function(STATE.omega), 10, rand=.5)
        sars_list += sars(ma)
    try:
        STATE.omega = Q_learning(STATE.omega, sars_list)
    except AssertionError:
        STATE.log_text = [[[10, 75], "Error !"]]
    else:
        STATE.log_text = [[[10, 75], "Training successful !"]]
    print_omega(STATE.omega)


def create_train_button():
    new_button(10, 100, "Train", train)


@WINDOW.event
def on_key_press(symbol, modifiers):
    print(symbol)
    global STATE
    if symbol == key.RIGHT:
        STATE.lab = apply_action(STATE.lab, "RIGHT")
    elif symbol == key.LEFT:
        STATE.lab = apply_action(STATE.lab, "LEFT")
    elif symbol == key.DOWN:
        STATE.lab = apply_action(STATE.lab, "DOWN")
    elif symbol == key.UP:
        STATE.lab = apply_action(STATE.lab, "UP")
    elif symbol == key.SPACE:
        STATE.lab = apply_action(STATE.lab, "PICK")
    elif symbol == key.R:  # Randomize
        STATE.lab = random_terrain()
    elif symbol == key.T:  # Train
        print('Walking')
        train()
    elif symbol == key.S:  # Step
        print("Stepping")
        a = greedy(q_function(omega), robot_state(STATE.lab))
        print(a)
        STATE.lab = apply_action(STATE.lab, a)
    elif symbol == key.Q:  # Quit
        print("Quitting")
        pyglet.app.exit()


@WINDOW.event
def on_mouse_press(x, y, button, modifiers):
    global STATE
    if button == pyglet.window.mouse.LEFT:
        for rect, cb in STATE.buttons.values():
            if x >= rect[0] and y >= rect[1] and x <= rect[2] and y <= rect[3]:
                cb()
                return
    ix = x
    iy = ((2*(J_MAX+1)+2)*TILE_SIZE_Y-y)
    ix = ix / TILE_SIZE_X / 2
    iy = iy / TILE_SIZE_Y / 2 - .2
    i = round(iy-ix)+4
    j = round(ix+iy)-6
    if button == pyglet.window.mouse.LEFT:
        robot_loc = np.argwhere(STATE.lab < 0)[0]
        STATE.lab[i, j] = -STATE.lab[i, j]
        STATE.lab[tuple(robot_loc)] = -STATE.lab[tuple(robot_loc)]
    elif button == pyglet.window.mouse.RIGHT:
        STATE.lab[i, j] = STATE.lab[i, j] + 1 if STATE.lab[i, j] != 4 else 1
