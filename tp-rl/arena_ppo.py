import os
import pickle

import gym

import random
import numpy as np

from utils import plot_data

from agents.ppo import PPO, Actor, Critic, Memory

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from copy import deepcopy
from collections import namedtuple, deque

import matplotlib.pyplot as plt

# %%

# CONSTANTS

ENV = 'CartPole-v0'
# Upgrade values
BETA = 0.2
DELTA = 1
GAMMA = 0.995
EPSILON = 0.3
TAU = 0.97

# Memory and number of runs
N_EPISODES = 2
BATCH_SIZE = 1000

# Model hyperparameters
N_HIDDEN = 64
A_LEARNING_RATE = 0.001
C_LEARNING_RATE = 0.0001  # Current best : 0.001 & 0.001, mean reward ~ 22 with cartpole

# Unterval of steps after which statistics should be printed
LOG_STEPS = 1
SAVE_STEPS = 20


# ACTOR_SAVE_PATH = "saved_models/actor_ppo.pth"
# CRITIC_SAVE_PATH = "saved_models/critic_ppo.pth"

class Game:
    """
    Running environment, taking one step,...
    """

    def __init__(self):
        self.env = gym.make(ENV)
        self.state_dim = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.n
        # typecast since problems with multiplication if type(limit) = np.float32
        # self.limit = float(self.env.action_space.high[0])

    def reset(self):
        start = self.env.reset()
        return start

    def take_one_step(self, action):
        # new_frame, reward, is_done, _ = self.env.step([action])
        new_frame, reward, is_done, _ = self.env.step(action)
        return new_frame, reward, is_done

    def sample_action(self):
        return self.env.action_space.sample()


env = Game()
n_input = env.state_dim
num_outputs = env.n_actions

actor = Actor(n_input, N_HIDDEN, num_outputs)
critic = Critic(n_input, N_HIDDEN)

ppo_agent = PPO(env, actor, critic, KL=False, Clip=True)

# running_state = ZFilter((2,), clip=5)

statistics = {
    'reward': [],
    'val_loss': [],
    'policy_loss': [],
}

N_EPISODES = 200
MEM_BATCH_SIZE = 1000

best_reward = 0
for i in range(0, N_EPISODES):
    memory = Memory()
    num_steps = 0
    num_ep = 0
    reward_batch = 0

    while num_steps < MEM_BATCH_SIZE:
        S = env.reset()
        # S = running_state(S)
        t = 0
        reward_sum = 0

        while True:
            t += 1

            A = ppo_agent.select_best_action(S)
            S_prime, R, is_done = env.take_one_step(A)  # a.item()

            reward_sum += R
            mask = 1 - int(is_done)

            memory.push(S, np.array([A]), mask, R)  # a.item()

            if is_done:
                break

            # S = running_state(S_prime)
            # S = S_prime

        num_steps += t
        num_ep += 1
        reward_batch += reward_sum

    reward_batch /= num_ep

    # The memory is now full of rollouts. Sample from memory and optimize
    batch = memory.sample()
    policy_loss, val_loss = ppo_agent.update_params(batch, i)

    # log data onto stdout
    if i == 0 or i % LOG_STEPS == 0:
        # save statistics
        statistics['reward'].append(reward_batch)
        statistics['val_loss'].append(val_loss.item())
        statistics['policy_loss'].append(policy_loss.item())

        if i % 20 == 0:
            print("Episode: %d, Mean Reward: %.8f, Mean Value loss: [%.8f],Mean Policy Loss: [%.8f]" % (
                i, np.asarray(statistics['reward'][-20:]).mean(),
                np.asarray(statistics['val_loss'][-20:]).mean(),
                np.asarray(statistics['policy_loss'][-20:]).mean()))

plot_data(statistics, N_EPISODES, LOG_STEPS)
