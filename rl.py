"""Implementation of q_learning with a linear approximator for the Q-value"""
import random
import numpy as np


class RL:
    """Simple object store"""
    def __init__(self, dim, actions, phi):
        self.omega = np.zeros(dim)
        self.policy = lambda: random.choice(actions)
        self.phi = phi
        self.actions = actions


def q_function(omega, phi):
    """Returns the q function from the weight vectors"""
    def answer(s, a):
        return np.dot(omega, phi(s, a))
    return answer


def greedy(q, s, actions):
    """Choose an action according to a random choice wheited by the Q-value"""
    # From https://docs.python.org/dev/library/random.html
    qsa = [q(s, a) for a in actions]
    return actions[np.argmax(qsa)]


def q_learning(rl, sars):
    """Return an updated rl after training on sars"""
    answer = RL(len(rl.omega), rl.actions, rl.phi)
    q_func = q_function(answer.omega, answer.phi)
    alpha = 1  # PARAMETER
    gamma = 0.9  # PARAMETER
    d = float('inf')
    q_iter = 0
    assert not all([r == 0 for _, _, r, _ in sars])
    while d > 1. and q_iter < 5:
        X = []
        Y = []
        for s1, a, r, s2 in sars:
            old = q_func(s1, a)
            Vs2 = max([q_func(s2, a) for a in answer.actions])
            new = old + alpha*(r + gamma*Vs2 - old)
            X.append(answer.phi(s1, a))
            Y.append(new)
        X = np.array(X)
        Y = np.array(Y)
        answer = np.dot(np.dot(np.linalg.pinv(np.dot(X.T, X)), X.T), Y)
        old_q = np.array([q_func(s, a) for s, a, _, _ in sars])
        Q_func = q_function(answer)
        new_q = np.array([Q_func(s, a) for s, a, _, _ in sars])
        d = np.linalg.norm(old_q-new_q)
        print("Iteration %d, |Q(s,a) - Q'(s,a)| is %f" % (q_iter, d))
        q_iter += 1
    answer.policy = lambda s: greedy(q_func, s, answer.actions)
    return answer
