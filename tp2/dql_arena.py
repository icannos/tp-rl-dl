import gym
from gym import wrappers
from deepqlearning import DeepQAgent

if __name__ == '__main__':

    # Simple execution
    env = gym.make("gridworld-v0")
    env.seed(0)  # Initialise le seed du pseudo-random

    env.verbose = False
    env.render()  # permet de visualiser la grille du jeu (si verbose = True)

    statedic, mdp = env.getMDP()

    clean_mdp = {statedic[s]: v for s, v in mdp.items()}

    agent = DeepQAgent(state_dim=env.observation_space.n, action_space=env.action_space.n)


    # Faire un fichier de log sur plusieurs scenarios
    outdir = 'gridworld-v0/naive-qlearning'
    envm = wrappers.Monitor(env, directory=outdir, force=True, video_callable=False)

    env.seed()  # Initialiser le pseudo aleatoire
    episode_count = 1000
    reward = 0
    done = False
    rsum = 0
    for i in range(episode_count):
        obs = envm.reset()
        env.verbose = (i % 100 == 0 and i > 0)  # afficher 1 episode sur 100

        env.render(0.1)
        j = 0
        rsum = 0
        while True:
            action = agent.exploration_policy(obs)
            prev_obs = np.copy(obs)
            obs, reward, done, _ = envm.step(action)

            agent.updateQ(prev_obs, action, reward, obs, alpha= np.fmax(0.1, 10/(i+1)), gamma=0.99)

            rsum += reward
            j += 1
            env.render()
            if done:
                print("Episode : " + str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions")
                break

        agent.exploration_rate *= agent.exploration_decay

    print("done")
    env.close()