
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
            x = torch.nn.functional.tanh(x)
            x = self.layers[i](x)

        return torch.nn.functional.tanh(x)

    def getAction(self, x):
        return self.forward(torch.Tensor(x)).cpu().detach().numpy()


class maddpgAgent:
    def __init__(self, observation_dim, action_dim, N_collaborators, alpha=0.9, gamma=0.9, tau=0.5):
        self.N_collaborators = N_collaborators
        self.tau = tau
        self.alpha = alpha
        self.gamma = gamma
        self.input_space = observation_dim
        self.action_space = action_dim

        self._memory = deque(maxlen=10)

        self._value_loss = nn.MSELoss(reduction="mean")

        self.V = ValueNN(input_dim=self.input_space[0]*N_collaborators+self.action_space, output_dim=1, layers=[16, 16])
        self.V_target = ValueNN(input_dim=self.input_space[0]*N_collaborators+self.action_space, output_dim=1, layers=[16, 16])

        self.policy = continuousPolicyNN(input_dim=input_space, output_dim=action_space, layers=[16, 16])
        self.policy_target = continuousPolicyNN(input_dim=input_space, output_dim=action_space, layers=[16, 16])

        self.V_optimizer = Adam(params=self.V.parameters())
        self.policy_optimizer = Adam(params=self.policy.parameters())


class maddpgTrainer:
    pass