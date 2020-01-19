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
        temp_in = self.input_dim
        for x in layers:
            self.layers.append(nn.Linear(temp_in, x))
            temp_in = x

        self.layers.append(nn.Linear(temp_in, self.output_dim))

    def forward(self, x, a):
        x = torch.cat([x, a], dim=1)
        x = self.layers[0](torch.Tensor(x))
        for i in range(1, len(self.layers)):
            x = torch.nn.functional.tanh(x)
            x = self.layers[i](x)

        return x

    def predict(self, x, a):
        return self.forward(torch.Tensor(x), torch.Tensor(a)).cpu().detach().numpy()


class continuousPolicyNN(nn.Module):
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

        return torch.tanh(x)

    def getAction(self, x):
        return self.forward(torch.Tensor(x)).cpu().detach().numpy()


class ddpgAgent:
    def __init__(self, input_space, action_space, gamma=0.999, alpha=0.9, tau=0.1, memsize=1000):
        self.memsize = memsize
        self.tau = tau
        self.alpha = alpha
        self.gamma = gamma
        self.input_space = input_space
        self.action_space = action_space

        self._memory = deque(maxlen=self.memsize)

        self._value_loss = nn.SmoothL1Loss(reduction="mean")

        self.V = ValueNN(input_dim=input_space[0] + action_space, output_dim=1, layers=[128, 128])
        self.V_target = ValueNN(input_dim=input_space[0] + action_space, output_dim=1, layers=[128, 128])

        self.policy = continuousPolicyNN(input_dim=input_space, output_dim=action_space, layers=[128, 128])
        self.policy_target = continuousPolicyNN(input_dim=input_space, output_dim=action_space, layers=[128, 128])

        self.V_optimizer = Adam(params=self.V.parameters(), lr=0.01)
        self.policy_optimizer = Adam(params=self.policy.parameters(), lr=0.001)

    def store(self, trajectory):
        self._memory.append(trajectory)

    def act(self, state, exploration=False):
        return np.clip(np.random.normal(self.policy.getAction(state), scale=exploration), -1,1) if exploration \
            else self.policy.getAction(state)

    def train_value_network(self, batch):
        X_s = []
        X_a = []
        Y = []

        self.V_optimizer.zero_grad()

        for t in batch:
            if not t:
                continue
            rsum = np.zeros(1)

            t.reverse()

            state, action, reward, next_state, done = t[0]

            X_s.append(state)
            X_a.append(action)

            if done:
                rsum += reward
            else:
                rsum += reward + self.gamma*self.V_target.predict([state],
                                                                  [self.policy_target.getAction(state)])[0]

            Y.append(rsum)

            for state, action, reward, next_state, done in t[1:]:
                X_s.append(state)
                X_a.append(action)
                rsum = reward + self.gamma * rsum

                y = rsum
                #print(y)

                Y.append(y)
        Y = torch.Tensor(Y)
        X_s = torch.Tensor(X_s)
        X_a = torch.Tensor(X_a)

        Ychap = self.V(X_s, X_a)

        loss = self._value_loss(Ychap, Y)

        loss.backward()
        self.V_optimizer.step()
        self.V_optimizer.zero_grad()

    def train_policy(self, batch):
        self.policy_optimizer.zero_grad()

        loss = 0
        B = 0

        for t in batch:
            for state, action, reward, next_state, done in t:
                state = torch.Tensor([state])
                loss += self.V(state, self.policy(state))
                B += 1

        loss = - loss
        loss /= B

        loss.backward()
        self.policy_optimizer.step()
        self.policy_optimizer.zero_grad()

    def update(self, batch_size):
        batch = random.sample(self._memory, batch_size)
        self.train_value_network(batch)
        self.train_policy(batch)

    def update_target(self, curr, target):
        for k, v in target.items():
            curr[k] = (1 - self.tau) * curr[k] + self.tau * target[k]

        return curr

    def equalize_networks(self):
        self.V.load_state_dict(self.V_target.state_dict())
        self.policy.load_state_dict(self.policy_target.state_dict())

    def update_targets(self):
        curr = copy(self.V.state_dict())
        T = copy(self.V_target.state_dict())

        updated_V = self.update_target(curr, T)

        self.V_target.load_state_dict(updated_V)

        curr = copy(self.policy.state_dict())
        T = copy(self.policy_target.state_dict())

        updated_pol = self.update_target(curr, T)

        self.policy_target.load_state_dict(updated_pol)
