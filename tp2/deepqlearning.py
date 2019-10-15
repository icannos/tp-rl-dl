import torch.nn as nn
import torch
from torch.optim import sgd
from collections import deque
import numpy as np
import random


class NN(nn.Module):
    def __init__(self, input_dim, output_dim, layers=[]):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim

        self.layers = nn.ModuleList([])

        temp_in = self.input_dim
        for x in layers:
            self.layers.append(nn.Linear(temp_in, x))
            temp_in = x

        self.layers.append(nn.Linear(temp_in, self.output_dim))

    def forward(self, x):
        x = self.layers[0](x)
        for i in range(1, len(self.layers)):
            x = torch.nn.functional.leaky_relu(x)
            x = self.layers[i](x)

        return x


class DeepQAgent:
    def __init__(self, state_dim, action_space, exploration_rate=1, exploration_decay=0.99, exploration_min=0.05,
                 learning_rate=0.01, batch_size=256, gamma=0.9):

        self.gamma = gamma
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.action_space = action_space
        self.state_dim = state_dim
        self.exploration_min = exploration_min
        self.exploration_decay = exploration_decay
        self.exploration_rate = exploration_rate
        self._memory = deque([], maxlen=5000)

        self.Q = NN(self.state_dim, self.action_space, [16, 16])
        self.Q_target = NN(self.state_dim, self.action_space, [16, 16])

        self.Q_optimizer = sgd.SGD(params=list(self.Q.parameters()))
        self.Qtarget_optimizer = sgd.SGD(params=list(self.Q_target.parameters()))

    def store(self, state, action, reward, next_state):
        self._memory.append((state, action, reward, next_state))

    def act(self, obs, exploration=False):
        if exploration:
            if np.random.uniform(0, 1) <= self.exploration_rate:
                return np.random.randint(0, self.action_space)

        return np.argmax(self.Q(obs))

    def optimization_step(self, X, Y):
        X = np.array(X)
        Y = np.array(Y)

        X = torch.tensor(X)
        Y = torch.tensor(Y)

        l = nn.functional.smooth_l1_loss(X, Y)
        l.backward()

        self.Q_optimizer.step()
        self.Q_optimizer.zero_grad()
        self.Q.zero_grad()

    def experience_replay(self):
        batch = random.sample(self._memory, self.batch_size)

        X = []
        Y = []

        for state, action, reward, next_sate in batch:
            y = self.Q(state)
            y[action] += reward + self.gamma * np.max(self.Q(next_sate))

            X.append(state)
            Y.append(y)

        self.optimization_step(X, Y)
