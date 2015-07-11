import pyglet
from pyglet.gl import *
glEnable(GL_BLEND)
glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
from pyglet.window import key
import numpy as np
import numpy.random as nprand
import itertools
import bisect
import random

I_MAX = 9
J_MAX = 9

WINDOW = pyglet.window.Window((I_MAX+1)*32, (J_MAX+1)*32)
WINDOW.set_location(0,0)
ROBOT_WINDOW = pyglet.window.Window(3*32, 3*32)
ROBOT_WINDOW.set_location((I_MAX+1)*32,0)
EARTH = pyglet.image.load('EARTH.png')
BUSH = pyglet.image.load('BUSH.png')
BERRIES = pyglet.image.load('BERRIES.png')
SHROOMS = pyglet.image.load('SHROOMS.png')
ROBOT = pyglet.image.load('ROBOT.png')
IMAGES = {1: EARTH,
          2: BUSH ,
          3: BERRIES,
          4: SHROOMS,
          -1: ROBOT} #-2 robot & shrooms, -2 robots and berries, etc.

def random_terrain():
    """Return a randomized terrain"""
    answer = np.ones((I_MAX+1,J_MAX+1))
    for i,j in zip(nprand.random_integers(0, I_MAX, 60),
                   nprand.random_integers(0, J_MAX, 60)):
        answer[i,j] = nprand.random_integers(2, 4)
    answer[0,0] = -1 if answer[0,0] == 0 else -answer[0,0]
    return answer

TERRAIN = random_terrain()

ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT", "PICK"]

Q0 = {k:np.array([0. for a in ACTIONS])
      for k in itertools.product(*[range(1,5) for i in range(0,9)])}
Q = Q0

def q_function(q):
    """Handle the nitty gritty BS of querying the dict"""
    def answer(s):
        return q[tuple(abs(s).reshape(-1))]
    return answer

def random_greedy(q, s):
    """Choose an action according to a random choice wheited by the Q-value"""
    # From https://docs.python.org/dev/library/random.html
    weights = q(s)
    if sum(weights) == 0:
        return random.choice(ACTIONS)
    cumdist = list(itertools.accumulate(weights))
    x = random.random() * cumdist[-1]
    return ACTIONS[bisect.bisect(cumdist, x)]

def ij2xy(m, i, j):
    "We use matrix-oriented coordinates i,j, but to display we need oriented abc/ord x, y instead"
    x = j*32
    y = (m.shape[1] - 1 - i)*32
    return x,y

def draw_assets(m, IMAGES):
    "Dranw fancy drawings of shrooms, etc."
    for i in range(0, m.shape[0]):
        for j in range(0, m.shape[1]):
            x,y = ij2xy(m, i, j)
            sprite = pyglet.sprite.Sprite(IMAGES[1], x=x, y=y)
            sprite.draw()
            if abs(m[i,j]) > 1:
                sprite = pyglet.sprite.Sprite(IMAGES[abs(m[i,j])], x=x, y=y)
                sprite.draw()
            if m[i,j] < 0:
                sprite = pyglet.sprite.Sprite(IMAGES[-1], x=x, y=y)
                sprite.draw()

def robot_state(TERRAIN):
    """Return the state visible to a robot"""
    t = TERRAIN
    i,j = np.argwhere(t<0)[0]
    if i-1 < 0:
        t = np.roll(t, 1, 0)
        i+=1
    elif i+1 > 9:
        t = np.roll(t, -1, 0)
        i-=1
    if j-1 < 0:
        t = np.roll(t, 1, 1)
        j+=1
    elif j+1 > 9:
        t = np.roll(t, -1, 1)
        j-=1
    return t[i-1:i+2, j-1:j+2]

def apply_action(m, action):
    """Return m after action has been applied to it"""
    robot_loc = np.argwhere(m<0)[0]
    m = m.copy()
    m[tuple(robot_loc)] = -m[tuple(robot_loc)]
    if action == "RIGHT":
        robot_loc[1] = (robot_loc[1]+1) % (J_MAX+1)
    elif action == "LEFT":
        robot_loc[1] = (robot_loc[1]-1) % (J_MAX+1)
    elif action == "DOWN":
        robot_loc[0] =( robot_loc[0]+1) % (I_MAX+1)
    elif action == "UP":
        robot_loc[0] = (robot_loc[0]-1) % (I_MAX+1)
    elif action == "PICK":
        if m[tuple(robot_loc)] == 4:  # SHROOMS
            m[tuple(robot_loc)] = 1  # EARTH
        elif m[tuple(robot_loc)] == 3:  # BERRIES
            m[tuple(robot_loc)] = 2  # BUSHES
    m[tuple(robot_loc)] = -m[tuple(robot_loc)]
    return m

def walk(m, q, length):
    """Apply the greedy policy with respect to the given q and return the list of
    state, action pairs"""
    answer = [m]
    for i in range(0, length):
        s = robot_state(m)
        a = random_greedy(q, s)
        m = apply_action(m, a)
        answer += [a, m]
    return answer

def display_traj(traj):
    global TERRAIN
    print("Drawing traj of length %d"%len(traj))
    if len(traj) > 0:
        TERRAIN = traj[0]
        #ROBOT_WINDOW.dispatch_event('on_draw')
        #WINDOW.dispatch_event('on_draw')
        pyglet.clock.schedule_once(lambda t: display_traj(traj[1:]), 0)

def reward(s1, a, s2):
    """Test reward function : we like to pick things"""
    if a == 'PICK' and s1[4] in [3,4]:  # SHROOMS or BERRIES
        return 1
    else:
        return 0

def sars(ma):
    """Turns a list of terrain, action into a sars list suitable for Q-learning"""
    answer = []
    for m1,a,m2 in zip(ma[::2], ma[1::2], ma[2::2]):
        s1 = tuple(abs(robot_state(m1)).reshape(-1))
        s2 = tuple(abs(robot_state(m2)).reshape(-1))
        r = reward(s1, a, s2)
        answer.append([s1, a, r, s2])
    return answer

def Q_learning(Q, sars):
    answer = Q.copy()
    alpha = 0.1  # PARAMETER
    gamma = 0.9  # PARAMETER
    for s1, a, r, s2 in sars:
        old = answer[s1][ACTIONS.index(a)]
        Vs2 = max(Q[s2])
        new = old + alpha*(r + gamma*Vs2 - old)
        answer[s1][ACTIONS.index(a)] = new
    return answer


@WINDOW.event
def on_draw():
    global TERRAIN
    print("Drawing main")
    WINDOW.clear()
    draw_assets(TERRAIN, IMAGES)

@ROBOT_WINDOW.event
def on_draw():
    global TERRAIN
    print("Drawing state")
    ROBOT_WINDOW.clear()
    draw_assets(robot_state(TERRAIN), IMAGES)

@WINDOW.event
def on_key_press(symbol, modifiers):
    print(symbol)
    global TERRAIN
    global Q
    if symbol == key.RIGHT:
        TERRAIN = apply_action(TERRAIN, "RIGHT")
    elif symbol == key.LEFT:
        TERRAIN = apply_action(TERRAIN, "LEFT")
    elif symbol == key.DOWN:
        TERRAIN = apply_action(TERRAIN, "DOWN")
    elif symbol == key.UP:
        TERRAIN = apply_action(TERRAIN, "UP")
    elif symbol == key.SPACE:
        TERRAIN = apply_action(TERRAIN, "PICK")
    elif symbol == key.R:  # Randomize
        TERRAIN = random_terrain()
    elif symbol == key.W:  # Walk
        print('Walking')
        ma = walk(TERRAIN, q_function(Q), 100)
        sars_list = sars(ma)
        Q = Q_learning(Q, sars_list)
        pyglet.clock.schedule_once(lambda t: display_traj(ma[::2]), 0)
        TERRAIN = ma[-1]
        print("Mean value is now %f"%np.mean([max(v) for v in Q.values()]))
    ROBOT_WINDOW.dispatch_event('on_draw')

pyglet.app.run()
