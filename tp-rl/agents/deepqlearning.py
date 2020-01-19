import torch.nn as nn
import torch
from torch.optim import sgd, Adam
from collections import deque
import numpy as np
import random
from copy import copy

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

def weights_init_uniform_rule(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # get the number of the inputs
        n = m.in_features
        y = 1.0 / np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0)


class DeepQAgent:
    def __init__(self, state_dim, action_space, exploration_rate=1, exploration_decay=0.99, exploration_min=0.05,
                 learning_rate=0.01, batch_size=64, gamma=0.9, tau=False, alpha=0.2):


        self.alpha = alpha
        self.tau = tau
        self.gamma = gamma
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.action_space = action_space.n
        self.state_dim = state_dim
        self.exploration_min = exploration_min
        self.exploration_decay = exploration_decay
        self.exploration_rate = exploration_rate
        self._memory = deque([], maxlen=500)

        self.Q = NN(self.state_dim, self.action_space, [16, 16])
        self.Q_target = NN(self.state_dim, self.action_space, [16, 16])

        self.Q.apply(weights_init_uniform_rule)
        self.Q_target.apply(weights_init_uniform_rule)

        self.Q_optimizer = Adam(params=list(self.Q.parameters()), lr=learning_rate)

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

        if self.tau:
            Q = self.Q_target
        else:
            Q = self.Q

        for state, action, reward, next_sate, done in batch:
            y = self.Q(state).detach().numpy()
            if not done:
                y[action] = y[action] * self.alpha + (1-self.alpha) * (reward + self.gamma * np.max(Q(next_sate).detach().numpy()))
            else:
                y[action] = y[action] * self.alpha + (1-self.alpha) * reward

            X.append(state)
            Y.append(y)

        return X, Y

    def updateQ(self, obs, action, reward, next_obs, done, update_target=False):
        batch = [(obs, action, reward, next_obs, done)]
        X, Y = self.build_target(batch)

        self.optimization_step(torch.Tensor(X), torch.Tensor(Y))

        if update_target and self.tau:
            self.update_targets()

        self.exploration_rate = max(self.exploration_decay * self.exploration_rate, self.exploration_min)

    def experience_replay(self, epoch=5, update_target = True):
        for _ in range(epoch):
            batch = random.sample(self._memory, self.batch_size)
            X, Y = self.build_target(batch)

            self.optimization_step(torch.Tensor(X), torch.Tensor(Y))

        if update_target and self.tau:
            self.update_targets()

        self.exploration_rate = max(self.exploration_decay * self.exploration_rate, self.exploration_min)

    def update_target(self, curr, target):
        for k, v in target.items():
            curr[k] = (1 - self.tau) * curr[k] + self.tau * target[k]

        return curr

    def equalize_networks(self):
        self.Q.load_state_dict(self.Q_target.state_dict())


    def update_targets(self):
        curr = copy(self.Q.state_dict())
        T = copy(self.Q_target.state_dict())

        updated_V = self.update_target(curr, T)

        self.Q_target.load_state_dict(updated_V)

