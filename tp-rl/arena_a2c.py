import matplotlib

matplotlib.use("TkAgg")
import gym
import gridworld
from gym import wrappers, logger
import numpy as np
import copy

batch = True
#game = "CartPole-v1"
game = "LunarLander-v2"

episode_count = 1000000


from agents.a2c import a2cAgent

if __name__ == '__main__':
    env = gym.make(game)

    # Enregistrement de l'Agent
    agent = a2cAgent(env.observation_space.shape, env.action_space.n)

    outdir = f'{game}/ac2-{game}-{"batch" if batch else "single"}'
    envm = wrappers.Monitor(env, directory=outdir, force=True, video_callable=False)
    env.seed(0)

    reward = 0
    done = False
    env.verbose = True
    np.random.seed(5)
    rsum = 0

    for i in range(episode_count):
        obs = envm.reset()
        env.verbose = (i % 20 == 0 and i > 0)  # afficher 1 episode sur 100
        if env.verbose:
            env.render()
        j = 0
        rsum = 0
        trajectory = []
        while True:
            action = agent.act(obs)
            next_obs, reward, done, _ = envm.step(action)
            if not batch:
                agent.training_step(obs, action, reward, next_obs, done)
            else:
                trajectory.append((obs, action, reward, next_obs, done))

            obs = next_obs
            rsum += reward
            j += 1
            if env.verbose:
                env.render()
            if done:
                print("Episode : " + str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions")
                break

        if batch:
            agent.batch_training(trajectory)

    print("done")
    env.close()