import gridworld
import gym
from gym import wrappers
import numpy as np
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("game", type=str, help="Name of the environnement to use.")
    parser.add_argument("--algorithm", '-a', type=str, help="Algorithm to use.")

    args = parser.parse_args()

    # Simple execution
    env = gym.make(args.game)
    env.seed(0)  # Initialise le seed du pseudo-random

    env.verbose = False
    env.render()  # permet de visualiser la grille du jeu (si verbose = True)

    statedic, mdp = env.getMDP()

    clean_mdp = {statedic[s]: v for s, v in mdp.items()}

    agent = dynaQAgent(statedic, [statedic[s] for s, t in list(mdp.items())], env.action_space.n)

    # Faire un fichier de log sur plusieurs scenarios
    outdir = f'{args.game}/daynq-learning'
    envm = wrappers.Monitor(env, directory=outdir, force=True, video_callable=False)
    env.setPlan("gridworldPlans/plan0.txt", {0: -0.001, 3: 1, 4: 1, 5: -1, 6: -1})
    env.seed()  # Initialiser le pseudo aleatoire
    episode_count = 100000
    reward = 0
    done = False
    rsum = 0
    for i in range(episode_count):
        obs = envm.reset()
        env.verbose = (i % 1000 == 0 and i > 0)  # afficher 1 episode sur 100

        env.render(0.1)
        j = 0
        rsum = 0
        while True:
            action = agent.exploration_policy(obs)
            prev_obs = np.copy(obs)
            obs, reward, done, _ = envm.step(action)

            agent.updateQ(prev_obs, action, reward, obs)

            rsum += reward
            j += 1
            env.render()
            if done:
                print("Episode : " + str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions")
                break

        agent.exploration_rate *= agent.exploration_decay

    print("done")
    env.close()