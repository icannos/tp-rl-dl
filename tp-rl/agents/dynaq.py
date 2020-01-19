import gridworld
import gym
from gym import wrappers
import numpy as np
import random
from scipy.special import softmax


class dynaQAgent():
    def __init__(self, state_dict, states, actions, exploration_min=0, exploration_rate=1, exploration_decay=0.99, k=1):
        self.exploration_min = exploration_min
        self.k = k
        self.exploration_decay = exploration_decay
        self.exploration_rate = exploration_rate
        self.state_dict = state_dict
        self.states = states
        self.actions = actions

        self.Q = np.random.uniform(0, 2, (max(self.states) + 1, self.actions))
        self.R = np.random.uniform(-1, 1, (max(self.states) + 1, self.actions, max(self.states) + 1))

        random_vect = np.random.uniform(0, 1, (max(self.states) + 1, max(self.states) + 1, self.actions))

        self.P = softmax(random_vect)

    def updateQ(self, st, at, rt, stp, alpha=0.9, gamma=0.9, alphar=0.8):

        st = self.state_dict[st.dumps()]
        stp = self.state_dict[stp.dumps()]
        self.Q[st, at] = (1 - alpha) * self.Q[st, at] + alpha * (rt + gamma * np.max(self.Q[stp, :]))

        self.R[st, at, stp] = (1 - alphar) * self.R[st, at, stp] + alphar * rt
        self.P[stp, st, at] = (1 - alphar) * self.P[stp, st, at] + alphar

        for s in range(max(self.states) + 1):
            if s != stp:
                self.P[s, st, at] = (1 - alphar) * self.P[s, st, at]

        self.P = softmax(self.P)

        states = list(self.state_dict.values())

        s = list(np.random.choice(states, self.k))
        a = list(np.random.choice(self.actions, self.k))

        for i in range(self.k):
            self.Q[s[i], a[i]] = (1 - alpha) * self.Q[s[i], a[i]] + alpha * \
                                 (sum([self.P[sp, s[i], a[i]] * (self.R[s[i], a[i], sp] +
                                                                 gamma * np.max(self.Q[sp, :]))
                                       for sp in s]))

    def exploration_policy(self, s):
        if np.fmax(np.random.uniform(0, 1), self.exploration_min) < self.exploration_rate:
            return np.random.randint(0, self.actions)
        else:
            return np.argmax(self.Q[self.state_dict[s.dumps()], :])

    def act(self, s):
        return np.argmax(self.Q[s, :])



