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

batch_size = 1000
steps = 1


exploration_rate = 0.1

from agents.ddpg2 import ddpgAgent


torch.set_default_tensor_type('torch.cuda.FloatTensor')

if __name__ == '__main__':
    env = gym.make(game)

    # Enregistrement de l'Agent
    agent = ddpgAgent(env.observation_space.shape, env.action_space.shape[0],
                      gamma=0.95, tau=0.01, memsize=batch_size+1000000)

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

    agent.update_targets()

    for i in range(1, episode_count+1):
        obs = envm.reset()
        #env.verbose = (i % 500 == 0 and i > 0)  # afficher 1 episode sur 100
        if env.verbose:
            env.render()
        j = 0
        rsum = 0

        #print(f'{3*np.log(i)=}')
        while True:

            action = agent.act(obs, exploration= exploration_rate / np.log(10+i))
            #print(action)
            next_obs, reward, done, _ = envm.step(action)
            agent.store(obs, action, reward, next_obs, done)

            obs = next_obs
            rsum += reward
            j += 1
            c+=1

            if done:
                print("Episode : " + str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions")
                break

            if len(agent._memory) > batch_size:
                for _ in range(steps):
                    agent.update(batch_size)
                    agent.update_targets()

        obs = envm.reset()
        rsum = 0
        j=0

        if i > 100 and i % 50 == 0:
            while True:

                action = agent.act(obs, exploration= False)
                #print(action)
                next_obs, reward, done, _ = envm.step(action)
                #agent.store(obs, action, reward, next_obs, done)

                obs = next_obs
                rsum += reward
                j += 1
                c+=1

                if done:
                    print("Episode Test : " + str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions")
                    break

    print("done")
    env.close()