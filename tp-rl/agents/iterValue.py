
import gridworld
import gym
from gym import wrappers
import numpy as np



class IterValAgent():
    def __init__(self, state_dict):
        self.V = None
        self.Q = None
        self.state_dict = state_dict

    def train(self, states, actions, P, gamma=0.9, max_iter=1000, eps=1E-7):
        V = np.random.uniform(-5, 5, max(states)+1)
        Q = np.zeros((max(states)+1, actions))

        V_prev = np.copy(V)

        for i in range(max_iter):
            for s in states:
                V_prev = np.copy(V)
                for a in range(0,actions):
                    Q[s, a] = np.sum([p * (r + gamma * V_prev[self.state_dict[sp]]) for p, sp, r, done in P[s][a]])

                V[s] = np.max(Q[s, :])

            if np.sum(np.fabs(V_prev - V)) <= eps:
                self.Q = Q
                self.V = V
                return Q, V, True, i

        self.Q = Q
        self.V = V

        return Q, V, False, max_iter

    def act(self, s):
        return np.argmax(self.Q[self.state_dict[s.dumps()], :])

class IterPolicyAgent():
    def __init__(self,state_dict, states, actions, P):
        self.P = P
        self.state_dict = state_dict
        self.states = states
        self.actions = actions

        self.policy = None


    def train(self, eps, gamma=0.8):

        policy = np.zeros(max(self.states) + 1)
        self.policy = np.zeros(max(self.states) + 1)

        while True:
            V = np.random.uniform(-5, 5, max(self.states) + 1)

            while True:
                Vprev = np.copy(V)
                for s in self.states:
                    V[s] = sum([p * (r + gamma * Vprev[self.state_dict[sp]]) for p, sp, r, done in self.P[s][self.policy[s]]])

                if np.sum(np.fabs(Vprev - V)) <= eps:
                    V = Vprev
                    break
                V = Vprev

            for s in self.states:
                policy[s] = np.argmax([sum([p * (r + gamma * V[self.state_dict[sp]])
                                                for p, sp, r, done in self.P[s][a]])
                                            for a in range(self.actions)])

            if policy == self.policy:
                self.policy = policy
                break

            self.policy = policy

        return policy


    def act(self, obs):
        return self.policy[self.state_dict[obs.dumps()]]







