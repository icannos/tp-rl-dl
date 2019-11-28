
import gridworld
import gym
from gym import wrappers
import numpy as np


class qlearningAgent():
    def __init__(self,state_dict, states, actions, exploration_rate=1, exploration_decay=0.99):
        self.exploration_decay = exploration_decay
        self.exploration_rate = exploration_rate
        self.state_dict = state_dict
        self.states = states
        self.actions = actions

        self.Q = np.random.uniform(0, 2, (max(self.states) + 1, self.actions))

    def updateQ(self, st, at, rt, stp, alpha=0.9, gamma=0.8):
        self.Q[self.state_dict[st.dumps()], at] = (1-alpha) * self.Q[self.state_dict[st.dumps()], at]  \
                                                  + alpha * (rt + gamma * np.max(self.Q[self.state_dict[stp.dumps()], :]))

    def exploration_policy(self, s):
        if np.random.uniform(0, 1) < self.exploration_rate:
            return np.random.randint(0, self.actions)
        else:
            return np.argmax(self.Q[self.state_dict[s.dumps()], :])

    def act(self, s):
        return np.argmax(self.Q[s, :])










