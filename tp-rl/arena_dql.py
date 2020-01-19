import matplotlib

matplotlib.use("TkAgg")
import gym
import gridworld
from gym import wrappers, logger
import numpy as np
import copy
from utils import running_mean
from matplotlib import pyplot as plt
import torch

game = "cartpole-0"

plan = "plan0"
episode_count = 1000

alpha = 0
gamma = 0.99
name = "dqlcartpole"
gammas = [0.1, 0.99]

tau = 0.8
experience_replay = True

parameters = [(0.9, False, 10000), (None, False, 10000), (0.9, True, 10000), (None, True, 10000)]
#parameters = [(0.9, True)]


from agents.deepqlearning import DeepQAgent

if __name__ == '__main__':
    rewards = []
    for tau, experience_replay, episode_count in parameters:
        np.random.seed(5)
        torch.manual_seed(10)
        env = gym.make('CartPole-v1')

        # Enregistrement de l'Agent
        agent = DeepQAgent(env.observation_space.shape, env.action_space, tau=tau, gamma=gamma, alpha=alpha)
        agent.equalize_networks()

        outdir = f'{game}/dql-{game}-{"target" if tau else "notarget"}'
        envm = wrappers.Monitor(env, directory=outdir, force=True, video_callable=False)
        env.seed(0)

        reward = 0
        done = False
        env.verbose = True

        rsum = 0

        for i in range(episode_count):
            obs = envm.reset()
            env.verbose = False  # afficher 1 episode sur 100
            if env.verbose:
                env.render()
            j = 0
            rsum = 0
            while True:
                action = agent.act(obs, exploration=True)
                next_obs, reward, done, _ = envm.step(action)

                if experience_replay:
                    agent.store(obs, action, reward, next_obs, done)
                else:
                    agent.updateQ(obs, action, reward, next_obs, done, update_target=j%20==0)
                obs = next_obs
                rsum += reward
                j += 1
                if env.verbose:
                    env.render()
                if done:
                    print("Episode : " + str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions")
                    break

            if experience_replay and len(agent._memory) > agent.batch_size:
                agent.experience_replay(epoch=10, update_target=True)

        rewards.append(envm.get_episode_rewards())

        env.close()

    import pickle as pk

    pk.dump((rewards, parameters), open("dqldata.dat", "wb"))
    for i, r in enumerate(rewards):
        reward = running_mean(r, 50)

        plt.plot([i for i in range(len(reward))], reward, label=f'tau={parameters[i][0]} '
                                                                f' er={parameters[i][1]}')

    plt.legend(loc="upper left")

    plt.savefig(f'img/{name}gamma{gamma}alpha{alpha}.png')