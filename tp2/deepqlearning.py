import torch.nn as nn
import torch
from torch.optim import sgd, Adam
from collections import deque
import numpy as np
import random


class NN(nn.Module):
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

        return x

class DeepQAgent:
    def __init__(self, state_dim, action_space, exploration_rate=1, exploration_decay=0.999, exploration_min=0.05,
                 learning_rate=0.01, batch_size=64, gamma=0.9):

        self.gamma = gamma
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.action_space = action_space.n
        self.state_dim = state_dim
        self.exploration_min = exploration_min
        self.exploration_decay = exploration_decay
        self.exploration_rate = exploration_rate
        self._memory = deque([], maxlen=1000)

        self.Q = NN(self.state_dim, self.action_space, [16,16, 16])
        self.Q_target = NN(self.state_dim, self.action_space, [64, 16])

        self.Q_optimizer = Adam(params=list(self.Q.parameters()), lr=0.001)
        self.Qtarget_optimizer = Adam(params=list(self.Q_target.parameters()))

    def store(self, state, action, reward, next_state, done):
        self._memory.append((state, action, reward, next_state, done))

    def act(self, obs, exploration=False):
        if exploration:
            if np.random.uniform(0, 1) <= self.exploration_rate:
                return np.random.randint(0, self.action_space)

        return np.argmax(self.Q(obs).detach().numpy())

    def optimization_step(self, X, Y):
        l = nn.functional.smooth_l1_loss(self.Q(X), Y, reduction="mean")
        l.backward()

        self.Q_optimizer.step()
        self.Q_optimizer.zero_grad()
        self.Q.zero_grad()

    def build_target(self, batch):
        X = []
        Y = []

        for state, action, reward, next_sate, done in batch:
            y = self.Q(state).detach().numpy()
            if not done:
                y[action] = y[action] * 0.8 + 0.2 * (reward + self.gamma * np.max(self.Q(next_sate).detach().numpy()))
            else:
                y[action] = y[action] * 0.5 + 0.5 * reward

            X.append(state)
            Y.append(y)

        return X,Y

    def experience_replay(self, epoch=5):
        for _ in range(epoch):
            batch = random.sample(self._memory, self.batch_size)
            X, Y = self.build_target(batch)

            self.optimization_step(torch.Tensor(X), torch.Tensor(Y))
        self.exploration_rate = max(self.exploration_decay*self.exploration_rate, self.exploration_min)