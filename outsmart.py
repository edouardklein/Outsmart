import random
import copy as cp
import numpy as np
import numpy.random as nprand
import rl


############################################
# Util
############################################
def copy(s):
    return cp.deepcopy(s)


def return_copy(f, *args):
    """Decorator for the pattern answer = s.copy()...return answer"""
    def func(s, *args):
        answer = copy(s)
        answer = f(answer, *args)
        return answer
    return func


@return_copy
def reset(s):
    """Reset the learned training"""
    s.rl = rl.RL(4*9*len(ACTIONS), ACTIONS, phi)
    return s


@return_copy
def load_state(s, filename):
    s.lab = np.loadtxt(filename+'.lab')
    s.wild = np.loadtxt(filename+'.wild')
    return s


def save_state(s, filename):
    np.savetxt(filename+'.lab', s.lab, fmt="%d")
    np.savetxt(filename+'.wild', s.wild, fmt="%d")


def random_terrain():
    """Return a randomized terrain"""
    answer = np.ones((I_MAX+1, J_MAX+1))*100
    for i, j in zip(nprand.random_integers(0, I_MAX, 8),
                    nprand.random_integers(0, J_MAX, 8)):
        answer[i, j] = nprand.random_integers(2, 4)*100
    answer[0, 0] += 1
    return answer


############################################
# Game dynamics
############################################
I_MAX = 9
J_MAX = 9

ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT", "PICK"]


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
        tile = m[tuple(robot_loc)] % 1000 // 100
        if (tile == 3 or tile == 4):
            m[tuple(robot_loc)] -= (tile - 2)*100
    return m


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


class State:
    def victorious(self, s):
        """Return True if NotBob is on a trap"""
        return any((s.wild % 100 == 12).reshape(-1))

    def losing(self, s):
        """Return True if nb_resources goes past 0"""
        return s.nb_resources > 0

    def __init__(self):
        self.lab = 100*np.ones((I_MAX+1, J_MAX+1))  # Terrain for the lab
        self.lab[0, 0] = 101
        self.wild = self.lab.copy()

        self.nb_resources = 0  # Amount of resources collected by Not-Bob

        self.obj_func = lambda s: False  # Return True when objective
        # is reached
        self.next_func = lambda: None  # Called when obj_func returns True

        self.rl = rl.RL(4*9*len(ACTIONS), ACTIONS, phi)  # Q-learning


############################################
# A.I.
############################################
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


def train(m, _rl):
    """Generate sars data and run the learning algorithm on the the given
    matrix"""
    _rl = rl.copy(_rl)
    sars_list = []
    for i in range(10):
        ma = _walk(m, _rl.policy, 10, rand=.5)
        sars_list += sars(ma)
    return rl.q_learning(_rl, sars_list)


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


@return_copy
def walk_lab(s, length, rand=0):
    """Apply the given state's policy to its lab matrix"""
    ma = _walk(s.lab, s.rl.policy, length, rand=rand)
    s.ui.log_text = "Stepping : chosen action is "+ma[-2]+"."  # Possible
    # abstraction leak ? We shouldn't touch ui in this module...
    s.lab = ma[-1]
    return s


@return_copy
def walk_wild(s, length, rand=0):
    """Apply the given state's policy to its wild matrix"""
    ma = _walk(s.wild, s.rl.policy, length, rand=rand)
    sars_list = sars(ma)
    s.nb_resources += sum([r for _, _, r, _ in sars_list])
    s.wild = ma[-1]
    s.ui.log_text = "Stepping : chosen action is "+ma[-2]+"."  # Possible
    # abstraction leak ? We shouldn't touch ui in this module...
    return s


def _walk(m, pi, length, rand=0):
    """Apply the given policy with and return the list of matrix action..."""
    answer = [m]
    for i in range(0, length):
        s = robot_state(m)
        a = pi(s) if np.random.rand() > rand else random.choice(ACTIONS)
        m = apply_action(m, a)
        answer += [a, m]
    return answer


def reward(s1, a, s2):
    """Test reward function : we like to pick things"""
    if a == 'PICK' and s1[1, 1] % 1000 // 100 in [3, 4]:  # ROCKS or CRYSTALS
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
