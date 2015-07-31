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

TILE_SIZE_X = 64
TILE_SIZE_Y = 32

I_MAX = 9
J_MAX = 9

X_MAX = 2*(I_MAX+1)*TILE_SIZE_X
Y_MAX = (2*(J_MAX+1)+2)*TILE_SIZE_Y

#ROBOT_WINDOW = pyglet.window.Window(2*3*TILE_SIZE_X, 2*3*TILE_SIZE_Y)
#ROBOT_WINDOW.set_location((I_MAX+1)*TILE_SIZE_X,0)

TERRAIN = np.ones((10,10))
WINDOW = pyglet.window.Window(X_MAX, Y_MAX)
WINDOW.set_location(0,0)
EARTH = pyglet.image.load('img/earth.png')
GRASS = pyglet.image.load('img/grass.png')
CRYSTALS = pyglet.image.load('img/crystals.png')
ROCKS = pyglet.image.load('img/rocks.png')
ROBOT = pyglet.image.load('img/robot_blue_right.png')
IMAGES = {1: EARTH,
          2: GRASS ,
          3: CRYSTALS,
          4: ROCKS,
          -1: ROBOT} #-2 robot & shrooms, -2 robots and berries, etc.

TEXT = [[[10, 10], "TEST 10 10"], [[20, 30], "Further test"]]

print("FIRST SET\n",TEXT)

def random_terrain():
    """Return a randomized terrain"""
    answer = np.ones((I_MAX+1,J_MAX+1))
    #for i,j in zip(nprand.random_integers(0, I_MAX, 3),
    #               nprand.random_integers(0, J_MAX, 3)):
    #    answer[i,j] = nprand.random_integers(2, 4)
    answer[0,0] = -answer[0,0]
    answer[0,0] = -3  # DEBUG
    return answer

#TERRAIN = random_terrain()

ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT", "PICK"]

def phi(s, a):
    """Feature vector on the state-action space"""
    s = abs(s).reshape(-1)
    answer = np.zeros(4*9*len(ACTIONS))
    start = 4*9*ACTIONS.index(a)
    answer[start:start+9] = (s==1)*1.
    answer[start+9:start+18] = (s==2)*1.
    answer[start+18:start+27] = (s==3)*1.
    answer[start+27:start+36] = (s==4)*1.
    return answer

omega_0 = np.zeros(4*9*len(ACTIONS))
omega = omega_0

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
    if all([x==0 for x in qsa]):
        a = random.choice(ACTIONS)
        print("All zero, radomly choosing "+a)
        return a
    for a in ACTIONS:
        print("q(s, %s) = %f"%(a, q(s,a)))
    a = ACTIONS[np.argmax(qsa)]
    print("Choosing "+a)
    return a
    #weights += min(weights) + 0.00001  # All weights > 0, and adding some weight to all options introduces some randomness
    #assert len(weights) == len(ACTIONS)
    #assert all(weights >= 0)
    #cumdist = list(itertools.accumulate(weights))
    #x = random.random() * cumdist[-1]
    #try:
    #    return ACTIONS[bisect.bisect(cumdist, x)]
    #except IndexError:
    #    print("BBBBUUUUUUUUUUUG")
    #    print("x is %f"%x)
    #    print("Cumul dist is ")
    #    print(cumdist)
    #    return random.choice(ACTIONS)

def ij2xy(m, i, j):
    """We use matrix-oriented coordinates i,j, but to display we need oriented abc/ord
    x, y instead, using isometric projection"""
    x = (m.shape[0] - i + j - 1)*TILE_SIZE_X
    #    x = j*TILE_SIZE
    #y = (m.shape[1] - 1 - i)*TILE_SIZE
    y = (m.shape[1]+m.shape[0] -i - j - 2)*TILE_SIZE_Y
    return x,y

def draw_text(text_list):
    """Draw the given [[x, y], text] list"""
    for [x,y], t in text_list:
        label = pyglet.text.Label(t, x=x, y=y,
                                  font_name='KenVector Future Thin Regular')
        label.draw()


def draw_assets(m, text, IMAGES):
    "Dranw fancy drawings of shrooms, etc."
    draw_text(STORY_TEXT)
    draw_text(OBJ_TEXT)
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

def xy_text(starting_xy, text):
    """Return the list of [[X, y], text] items that draw_text() will understand"""
    x, y = starting_xy
    return [[[x, y - i*15], line] for i, line in enumerate(text)]

def story_text(text):
    """Save the given text to be displayed in the upper left corner"""
    global STORY_TEXT
    STORY_TEXT = xy_text([0, Y_MAX-10], text.split("\n"))

def objective_text(text):
    """Save the given text to be displayed in the upper right corner"""
    global OBJ_TEXT
    OBJ_TEXT = xy_text([X_MAX-600, Y_MAX-10], ["OBJECTIVES"]+text.split("\n"))

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
    return t[i-1:i+2, j-1:j+2].copy()

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
    global TERRAIN
    print("Drawing traj of length %d"%len(traj))
    if len(traj) > 0:
        TERRAIN = traj[0]
        #ROBOT_WINDOW.dispatch_event('on_draw')
        #WINDOW.dispatch_event('on_draw')
        pyglet.clock.schedule_once(lambda t: display_traj(traj[1:]), 0)

def reward(s1, a, s2):
    """Test reward function : we like to pick things"""
    if a == 'PICK' and s1[1,1] in [3,4]:  # SHROOMS or CRYSTALS
        return 1
    else:
        return 0

def sars(ma):
    """Turns a list of terrain, action into a sars list suitable for Q-learning"""
    answer = []
    for m1,a,m2 in zip(ma[::2], ma[1::2], ma[2::2]):
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
    assert not all([r==0 for _,_,r,_ in sars])
    while d>1. or q_iter<5:
        old_answer = answer.copy()
        X = []
        Y = []
        for s1, a, r, s2 in sars:
            old = Q_func(s1, a)
            Vs2 = max([Q_func(s2,a) for a in ACTIONS])
            new = old + alpha*(r + gamma*Vs2 - old)
            X.append(phi(s1, a))
            Y.append(new)
        X = np.array(X)
        Y = np.array(Y)
        #answer,_,_,_ = np.linalg.lstsq(np.array(X),
        #                                  np.array(Y))
        answer = np.dot(np.dot(np.linalg.pinv(np.dot(X.T,X)), X.T), Y)
        print("NORM of omega %f"%np.linalg.norm(answer))
        old_q = np.array([Q_func(s,a) for s,a,_,_ in sars])
        Q_func = q_function(answer)
        new_q = np.array([Q_func(s,a) for s,a,_,_ in sars])
        d = np.linalg.norm(old_q-new_q)
        old_answer = answer.copy()
        print("Iteration %d, |Q(s,a) - Q'(s,a)| is %f"%(q_iter,d))
        q_iter+=1
    return answer

def print_omega(omega):
    terrains = ["EARTH", "GRASS", "CRYSTALS", "SHROOMS"]
    for a in range(len(ACTIONS)):
        print(ACTIONS[a])
        for t in range(len(terrains)):
            print(terrains[t])
            x = omega[4*9*a + 9*t:4*9*a + 9*t+9]
            print(x.reshape(3,3))



@WINDOW.event
def on_draw():
    global TERRAIN
    global TEXT
    print("Drawing main")
    print(TEXT)
    WINDOW.clear()
    draw_assets(TERRAIN, TEXT, IMAGES)

#@ROBOT_WINDOW.event
#def on_draw():
#    global TERRAIN
#    print("Drawing state")
#    ROBOT_WINDOW.clear()
#    draw_assets(robot_state(TERRAIN), IMAGES)

@WINDOW.event
def on_key_press(symbol, modifiers):
    print(symbol)
    global TERRAIN
    global omega
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
    elif symbol == key.T:  # Train
        print('Walking')
        sars_list = []
        for i in range(10):
            ma = walk(TERRAIN, q_function(omega), 10, rand=.5)
            sars_list += sars(ma)
        omega = Q_learning(omega, sars_list)
        #pyglet.clock.schedule_once(lambda t: display_traj(ma[::2]), 0)
        TERRAIN = ma[-1]
        print_omega(omega)
    elif symbol == key.S:  # Step
        print("Stepping")
        a = greedy(q_function(omega), robot_state(TERRAIN))
        print(a)
        TERRAIN = apply_action(TERRAIN, a)
    elif symbol == key.Q:  # Quit
        print("Quitting")
        pyglet.app.exit()
    #ROBOT_WINDOW.dispatch_event('on_draw')

@WINDOW.event
def on_mouse_press(x, y, button, modifiers):
    global TERRAIN
    ix = x
    iy = ((2*(J_MAX+1)+2)*TILE_SIZE_Y-y)
    ix = ix / TILE_SIZE_X / 2
    iy = iy / TILE_SIZE_Y / 2 - .2
    i = round(iy-ix)+4
    j = round(ix+iy)-6
    if button == pyglet.window.mouse.LEFT:
        robot_loc = np.argwhere(TERRAIN<0)[0]
        TERRAIN[tuple(robot_loc)] = -TERRAIN[tuple(robot_loc)]
        TERRAIN[i,j] = -TERRAIN[i, j]
    elif button == pyglet.window.mouse.RIGHT:
        TERRAIN[i,j] = TERRAIN[i,j] + 1 if TERRAIN[i,j] != 4 else 1

pyglet.font.add_file('img/kenvector_future_thin.ttf')
KenFuture = pyglet.font.load('KenVector Future Thin Regular')

#pyglet.app.run()

