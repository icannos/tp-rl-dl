import torch.nn as nn
import torch
from torch.optim import sgd, Adam
from collections import deque
import numpy as np
import random
from copy import copy

class ValueNN(nn.Module):
    def __init__(self, input_dim, output_dim, layers=[]):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim

        self.layers = nn.ModuleList([])
        temp_in = self.input_dim[0]
        for x in layers:
            self.layers.append(nn.Linear(temp_in, x))
            temp_in = x

        self.layers.append(nn.Linear(temp_in, self.output_dim))

    def forward(self, x):
        x = self.layers[0](torch.Tensor(x))
        for i in range(1, len(self.layers)):
            x = torch.nn.functional.tanh(x)
            x = self.layers[i](x)

        return x


class stochasticPolicyNN(nn.Module):
    def __init__(self, input_dim, output_dim, layers=[]):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim

        self.layers = nn.ModuleList([])
        temp_in = self.input_dim[0]
        for x in layers:
            self.layers.append(nn.Linear(temp_in, x))
            temp_in = x

        self.layers.append(nn.Linear(temp_in, self.output_dim))

    def forward(self, x):
        x = self.layers[0](torch.Tensor(x))
        for i in range(1, len(self.layers)):
            x = torch.nn.functional.relu(x)
            x = self.layers[i](x)

        return nn.functional.softmax(x)


class a2cAgent:
    def __init__(self, input_space, action_space, gamma=0.99, alpha=1, learning_rate=0.001):
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.gamma = gamma
        self.input_space = input_space
        self.action_space = action_space

        self._value_loss = nn.MSELoss(reduction="mean")

        self.V = ValueNN(input_dim=input_space, output_dim=1, layers=[16, 16])
        self.policy = stochasticPolicyNN(input_dim=input_space, output_dim=action_space, layers=[16, 16])


        self.V_optimizer = Adam(params=self.V.parameters(), lr=self.learning_rate)
        self.policy_optimizer = Adam(params=self.policy.parameters(), lr=0.001)

    def act(self, obs):
        return np.random.choice(self.action_space, p=self.policy(obs).detach().numpy())

    def training_step(self, state, action, reward, next_state, done):
        if not done:
            y = (1-self.alpha)*self.V(state).detach().numpy() + self.alpha*(reward + self.gamma * self.V(next_state).detach().numpy())
        else:
            y = [(1-self.alpha)*self.V(state).detach().numpy() + self.alpha*reward]

        self.update_value_function(state, y)

        self.update_policy(state, action, reward, next_state)

    def batch_training(self, trajectory):
        Y = []
        X = []
        cumulative_reward = 0

        traj = copy(trajectory)
        traj.reverse()

        for state, action, reward, next_state, done in traj:
            if not done:
                cumulative_reward = reward + self.gamma * cumulative_reward
            else:
                cumulative_reward = reward

            y = (1-self.alpha) * self.V(state).detach().numpy() + self.alpha * cumulative_reward

            X.append(state)
            Y.append(y)

        self.update_value_function(X, Y)

        self.policy.zero_grad()

        logpi = - sum(torch.log(self.policy(state)[action]) * self.advantage_function(reward, state, next_state)
                    for state, action, reward, next_state, done in traj) / len(traj)

        logpi.backward()

        self.policy_optimizer.step()
        self.policy_optimizer.zero_grad()
        self.policy.zero_grad()

    def update_policy(self, state, action, reward, next_state):
        self.policy.zero_grad()

        logpi = torch.log(self.policy(state)[action])
        logpi.backward()

        A = self.advantage_function(reward, state, next_state)
        for p in self.policy.parameters():
            p.grad *= - A


        self.policy_optimizer.step()
        self.policy_optimizer.zero_grad()
        self.policy.zero_grad()

    def advantage_function(self, r, state, next_state):
        return r + self.gamma*self.V(next_state) - self.V(state)

    def update_value_function(self, X, Y):
        l = self._value_loss(self.V(X), torch.Tensor(Y))
        l.backward()

        self.V_optimizer.step()
        self.V_optimizer.zero_grad()
        self.V.zero_grad()



