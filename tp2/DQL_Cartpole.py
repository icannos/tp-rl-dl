import matplotlib

matplotlib.use("TkAgg")
import gym
import gridworld
from gym import wrappers, logger
import numpy as np
import copy

from deepqlearning import DeepQAgent

if __name__ == '__main__':
    env = gym.make('CartPole-v1')

    # Enregistrement de l'Agent
    agent = DeepQAgent(env.observation_space.shape, env.action_space)

    outdir = 'cartpole-v0/random-agent-results'
    envm = wrappers.Monitor(env, directory=outdir, force=True, video_callable=False)
    env.seed(0)

    episode_count = 1000000
    reward = 0
    done = False
    env.verbose = True
    np.random.seed(5)
    rsum = 0

    for i in range(episode_count):
        obs = envm.reset()
        env.verbose = (i % 1000 == 0 and i > 0)  # afficher 1 episode sur 100
        if env.verbose:
            env.render()
        j = 0
        rsum = 0
        while True:
            action = agent.act(obs, exploration=True)
            next_obs, reward, done, _ = envm.step(action)
            agent.store(obs, action, reward, next_obs, done)
            obs = next_obs
            rsum += reward
            j += 1
            if env.verbose:
                env.render()
            if done:
                print("Episode : " + str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions")
                break

        if i >= 20:
            agent.experience_replay(epoch=5)

    print("done")
    env.close()