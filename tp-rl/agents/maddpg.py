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
        temp_in = self.input_dim
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

    def predict(self, x):
        return self.forward(torch.Tensor(x)).cpu().detach().numpy()


class maddpgAgent:
    def __init__(self, agent_id, observation_dim, action_space, N_collaborators, tau=0.5):
        self.agent_id = agent_id
        self.N_collaborators = N_collaborators
        self.tau = tau
        self.input_space = observation_dim
        self.action_space = int(action_space.n)

        self._value_loss = nn.SmoothL1Loss(reduction="mean")

        self.V = ValueNN(input_dim=self.input_space * N_collaborators + self.action_space * N_collaborators,
                         output_dim=1, layers=[128, 128, 128])
        self.V_target = ValueNN(input_dim=self.input_space * N_collaborators + self.action_space, output_dim=1,
                                layers=[128, 128, 128])

        self.policy = continuousPolicyNN(input_dim=observation_dim, output_dim=self.action_space, layers=[128, 128])
        self.policy_target = continuousPolicyNN(input_dim=observation_dim, output_dim=self.action_space, layers=[128, 128])

        self.V_optimizer = Adam(params=self.V.parameters())
        self.policy_optimizer = Adam(params=self.policy.parameters())

    def act(self, state, exploration=False):
        return np.clip(np.random.normal(self.policy.getAction(state), scale=exploration), -1, 1) if exploration \
            else self.policy.getAction(state)

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


class maddpgTrainer:
    def __init__(self, agents, gamma):

        self.gamma = gamma
        self._memory = deque(maxlen=1000000)
        self.agents = agents

        self._value_loss = nn.MSELoss(reduction="mean")

    def store(self, observations, actions, rewards, next_observations, dones):
        self._memory.append((observations, actions, rewards, next_observations, dones))

    def training_step(self, batch_size):
        batch = random.sample(self._memory, batch_size)

        self.train_value_network(batch)

        for i in range(len(self.agents)):
            self.train_policy(i, batch)

    def predict_value(self, agent_id, observations, actions, target=True):
        obs = np.array(observations).flatten()
        act = np.array(actions).flatten()

        if target:
            return self.agents[agent_id].V_target.predict([obs], [act])[0]
        else:
            return self.agents[agent_id].V.predict([obs], [act])[0]

    def predict_action(self, agent_id, observations, target=True):
        obs = observations[agent_id]

        if target:
            return self.agents[agent_id].policy_target.predict([obs])[0]
        else:
            return self.agents[agent_id].policy.predict([obs])[0]

    def predict_actions(self, observations):
        return [self.predict_action(i, observations) for i in range(self.agents)]

    def _mk_targets(self, batch):
        y = [[] for i in range(len(self.agents))]
        X_s = [[] for i in range(len(self.agents))]
        X_a = [[] for i in range(len(self.agents))]

        for observations, actions, rewards, next_observations, dones in batch:
            for i in range(len(self.agents)):
                acts = [self.predict_action(i, next_observations) for i in range(len(self.agents))]
                X_a[i].append(actions)
                X_s[i].append(observations)

                y[i].append(rewards[i] + self.gamma * self.predict_value(i, next_observations, acts))

        return X_a, X_s, y

    def train_policy(self, agent_id, batch):
        loss = 0
        for observations, actions, rewards, next_observations, dones in batch:
            actions[agent_id] = self.agents[agent_id].policy(observations[agent_id])

            loss += self.agents[agent_id].V(torch.Tensor(observations).flatten(), torch.Tensor(actions).flatten())

        loss = loss / len(batch)
        self.agents[agent_id].policy_optimizer.zero_grad()
        (-loss).backward()
        self.agents[agent_id].policy_optimizer.step()
        self.agents[agent_id].policy_optimizer.zero_grad()

    def train_value_network(self, batch):
        X_a, X_s, y = self._mk_targets(batch)

        for i in range(len(self.agents)):
            l = self._value_loss(y, self.agents[i].V(torch.Tensor(X_s[i]).flatten()), torch.Tensor(X_a[i]).flatten())

            self.agents[i].V_optimizer.zero_grad()
            l.backward()
            self.agents[i].V_optimizer.step()
            self.agents[i].V_optimizer.zero_grad()

    def update_targets(self):
        for i in range(len(self.agents)):
            self.agents[i].update_targets()
