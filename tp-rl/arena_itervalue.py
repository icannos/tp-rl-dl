import gym
from gym import wrappers
from agents.iterValue import IterValAgent

plan = "plan0"
episode_count = 1000

if __name__ == '__main__':

    # Simple execution
    env = gym.make("gridworld-v0")
    env.seed(0)  # Initialise le seed du pseudo-random

    env.verbose = False
    env.render()  # permet de visualiser la grille du jeu (si verbose = True)

    statedic, mdp = env.getMDP()

    clean_mdp = {statedic[s]: v for s, v in mdp.items()}

    agent = IterValAgent(statedic)
    agent.train([statedic[s] for s, t in list(mdp.items())], env.action_space.n, clean_mdp)


    # Faire un fichier de log sur plusieurs scenarios
    outdir = f'gridworld-v0/{plan}iterpolicy-agent-results'
    envm = wrappers.Monitor(env, directory=outdir, force=True, video_callable=False)
    env.setPlan(f"gridworldPlans/{plan}.txt", {0: -0.001, 3: 1, 4: 1, 5: -1, 6: -1})
    env.seed()  # Initialiser le pseudo aleatoire

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
            action = agent.act(obs)
            obs, reward, done, _ = envm.step(action)
            rsum += reward
            j += 1
            env.render()
            if done:
                print("Episode : " + str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions")
                break

    print("done")
    env.close()