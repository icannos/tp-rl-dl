import matplotlib

matplotlib.use("TkAgg")
import gym
import gridworld
from gym import wrappers, logger
import numpy as np
import copy
import torch

batch = True

game = "LunarLanderContinuous-v2"
#game = "Pendulum-v0"
#game = "MountainCarContinuous-v0"

episode_count = 1000000

batch_size = 5
steps = 2
traj_length = 100

exploration_rate = 0.1

from agents.ddpg import ddpgAgent

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

if __name__ == '__main__':
    env = gym.make(game)

    # Enregistrement de l'Agent"
    agent = ddpgAgent(env.observation_space.shape, env.action_space.shape[0],
                      gamma=0.9, alpha=0.9, tau=0, memsize=batch_size+100)

    agent.equalize_networks()

    outdir = f'{game}/ac2-{game}-{"batch" if batch else "single"}'
    envm = wrappers.Monitor(env, directory=outdir, force=True, video_callable=False)
    env.seed(0)

    reward = 0
    done = False
    env.verbose = False
    np.random.seed(5)
    rsum = 0

    c = 0

    agent.equalize_networks()

    for i in range(episode_count):
        obs = envm.reset()
        #env.verbose = (i % 500 == 0 and i > 0)  # afficher 1 episode sur 100
        if env.verbose:
            env.render()
        j = 0
        rsum = 0

        trajectory = []

        while True:
            action = agent.act(obs, exploration=exploration_rate)
            next_obs, reward, done, _ = envm.step(action)
            trajectory.append((obs, action, reward, next_obs, done))

            if len(trajectory) >= traj_length or done:
                agent.store(trajectory)
                trajectory = []

            obs = next_obs
            rsum += reward
            j += 1
            c+=1

            #if env.verbose:
                #env.render()
            if done:
                print("Episode : " + str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions")
                break

        if len(agent._memory) > batch_size:
            for i in range(steps):
                agent.update(batch_size)
            agent.update_targets()

    print("done")
    env.close()