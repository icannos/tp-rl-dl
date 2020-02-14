import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import random
from copy import deepcopy, copy


class Memory(object):
    """
    Save state,action,reward,mask to memory,
    Sample from memory with .sample()
    """

    def __init__(self):
        self.memory = []

    def push(self, state, action, mask, reward):
        """Saves a transition."""
        self.memory.append(state, action, mask, reward)

    def sample(self):
        return zip(*self.memory)

    def __len__(self):
        return len(self.memory)


class Actor(nn.Module):
    """
    Actor Critic network
    """

    def __init__(self, n_inp, n_hidden, n_output):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(n_inp, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_output)

        # nn.init.normal_(self.fc1.weight, mean=0.0, std=1.0)
        # nn.init.normal_(self.fc2.weight, mean=0.0, std=1.0)

        # self.log_stddev = nn.Parameter(torch.zeros(1))

        self.module_list = [self.fc1, self.fc2]
        self.module_list_old = [None] * 2

        # required so that start of episode does not throw error
        n = n_inp
        y = 1.0 / np.sqrt(n)

        fc1_old = self.fc1
        fc2_old = self.fc2

        # Weights init
        fc1_old.weight.data.uniform_(-y, y)
        fc2_old.bias.data.fill_(0)

        self.module_list_old[0], self.module_list_old[1] = fc1_old, fc2_old

    def backup(self):
        for i in range(len(self.module_list)):
            self.module_list_old[i] = deepcopy(self.module_list[i])

    def forward(self, x, old=False):

        if not old:
            x = F.tanh(self.fc1(x))

            x = F.softmax(self.fc2(x), dim=-1)
            # mu = self.mean(x)
            # log_stddev = self.log_stddev.expand_as(mu)
        else:
            x = F.tanh(self.module_list_old[0](x))
            x = F.softmax(self.module_list_old[1](x), dim=-1)

        return x


class Critic(nn.Module):
    def __init__(self, n_inp, n_hidden):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(n_inp, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)

        # Weights init
        self.state_val = nn.Linear(n_hidden, 1)
        self.state_val.weight.data.mul_(0.1)
        self.state_val.bias.data.mul_(0.0)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        state_val = self.state_val(x)
        return state_val


class PPO:
    def __init__(self, env, actor, critic, gamma=0.01, tau=0.01, KL=False, Clip=True, A_LEARNING_RATE=None,
                 C_LEARNING_RATE=None):
        self.env = env
        self.actor = actor
        self.critic = critic
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=A_LEARNING_RATE)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=C_LEARNING_RATE)
        self.kl = KL
        self.clip = Clip
        self.num_outputs = env.n_actions

    def act(self, S, countinuous=False):
        if countinuous:
            S = torch.FloatTensor(S)
            mu, log_sigma = self.actor(torch.Tensor(S))
            action = torch.normal(mu, torch.exp(log_sigma))
            return action
        else:
            S = torch.Tensor(torch.FloatTensor(S))
            dist = self.actor(S)
            action = np.random.choice(self.num_outputs, p=np.squeeze(dist.detach().numpy()))
            return action

    def compute_advantage(self, values, batch_R, batch_mask):
        batch_size = len(batch_R)

        v_target = torch.FloatTensor(batch_size)
        advantages = torch.FloatTensor(batch_size)

        prev_v_target = 0
        prev_v = 0
        prev_A = 0

        for i in reversed(range(batch_size)):
            v_target[i] = batch_R[i] + self.GAMMA * prev_v_target * batch_mask[i]
            delta = batch_R[i] + self.GAMMA * prev_v * batch_mask[i] - values.data[i]
            advantages[i] = delta + self.GAMMA * self.TAU * prev_A * batch_mask[i]

            prev_v_target = v_target[i]
            prev_v = values.data[i]
            prev_A = advantages[i]

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-4)
        return advantages, v_target

    def update_params(self, batch, i, eps=0.1, beta=0.5):
        S = torch.FloatTensor(batch.state)
        masks = torch.FloatTensor(batch.mask)
        A = torch.FloatTensor(np.concatenate(batch.action, 0))
        R = torch.FloatTensor(batch.reward)

        V_S = self.critic(torch.Tensor(S))
        advantages, v_target = self.compute_advantage(V_S, R, masks)

        # loss function for value net
        L_vf = torch.mean(torch.pow(V_S - torch.Tensor(v_target), 2))

        # optimize the critic net
        self.critic_optimizer.zero_grad()
        L_vf.backward()
        self.critic_optimizer.step()

        # cast into variable
        # A = Variable(A)
        A = torch.tensor(A, dtype=torch.int8).numpy()
        # new log probability of the actions
        # means, log_stddevs = self.actor(Variable(S))
        # new_log_prob = get_gaussian_log(A, means, log_stddevs)
        new_prob = self.actor(torch.Tensor(S))
        new_log_prob = torch.log(new_prob[np.arange(0, len(new_prob)), A])

        # old log probability of the actions
        with torch.no_grad():
            # old_means, old_log_stddevs = self.actor(Variable(S), old=True)
            # old_log_prob = get_gaussian_log(A, old_means, old_log_stddevs)
            old_prob = self.actor(torch.Tensor(S), old=True)
            old_log_prob = torch.log(old_prob[np.arange(0, len(old_prob)), A])

        # save the old actor

        self.actor.backup()

        # Adding an entropy term to critic loss, not sure it's a good idea
        entropy = - (torch.sum((new_prob) * torch.log(new_prob), dim=1)).mean()

        # ratio of new and old policies
        ratio = torch.exp(new_log_prob - old_log_prob)

        # find clipped loss
        if self.clip:
            advantages = torch.Tensor(advantages)
            L_cpi = ratio * advantages
            clip_factor = torch.clamp(ratio, 1 - eps, 1 + eps) * advantages
            L_clip = -torch.mean(torch.min(L_cpi, clip_factor))
            actor_loss = L_clip + beta * entropy

            # It is too little otherwise
            actor_loss *= 10

        elif self.kl:
            # To code
            pass

        else:
            actor_loss = -torch.mean(ratio * torch.Tensor(advantages)) + beta * entropy
            actor_loss *= 10

        # optimize actor network
        self.actor_optimizer.zero_grad()
        actor_loss.backward()

        if self.clip:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 100)

        self.actor_optimizer.step()

        return actor_loss, L_vf
