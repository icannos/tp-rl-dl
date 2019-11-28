import numpy as np


class IterPolicyAgent():
    def __init__(self,state_dict, states, actions, P):
        self.P = P
        self.state_dict = state_dict
        self.states = states
        self.actions = actions

        self.policy = None


    def train(self, eps, gamma=0.8):
        policy = np.zeros(max(self.states) + 1)
        self.policy = np.random.randint(0, self.actions, max(self.states) + 1)

        while True:
            V = np.random.uniform(-5, 5, max(self.states) + 1)

            while True:
                Vprev = np.copy(V)
                for s in self.states:
                    V[s] = sum([p * (r + gamma * Vprev[self.state_dict[sp]]) for p, sp, r, done in self.P[s][self.policy[s]]])

                err = np.sum(np.fabs(Vprev - V))
                print(err)
                if err <= eps:
                    break

            for s in self.states:
                policy[s] = np.argmax([sum([p * (r + gamma * V[self.state_dict[sp]])
                                                for p, sp, r, done in self.P[s][a]])
                                            for a in range(self.actions)])

            if all(policy == self.policy):
                self.policy = policy
                break

            self.policy = policy

        return policy


    def act(self, obs):
        return self.policy[self.state_dict[obs.dumps()]]










