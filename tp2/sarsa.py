
import gridworld
import gym
from gym import wrappers
import numpy as np


class sarsaAgent():
    def __init__(self,state_dict, states, actions, exploration_rate=1, exploration_decay=0.99):
        self.exploration_decay = exploration_decay
        self.exploration_rate = exploration_rate
        self.state_dict = state_dict
        self.states = states
        self.actions = actions

        self.Q = np.random.uniform(0, 2, (max(self.states) + 1, self.actions))

    def updateQ(self, st, at, rt, stp, atp, alpha=0.9, gamma=0.8):
        self.Q[self.state_dict[st.dumps()], at] = (1-alpha) * self.Q[self.state_dict[st.dumps()], at]  \
                                                  + alpha * (rt + gamma * self.Q[self.state_dict[stp.dumps()], atp])

    def exploration_policy(self, s):
        if np.random.uniform(0, 1) < self.exploration_rate:
            return np.random.randint(0, self.actions)
        else:
            return np.argmax(self.Q[self.state_dict[s.dumps()], :])

    def act(self, s):
        return np.argmax(self.Q[s, :])


if __name__ == '__main__':

    # Simple execution
    env = gym.make("gridworld-v0")
    env.seed(0)  # Initialise le seed du pseudo-random

    env.verbose = False
    env.render()  # permet de visualiser la grille du jeu (si verbose = True)

    statedic, mdp = env.getMDP()

    clean_mdp = {statedic[s]: v for s, v in mdp.items()}

    agent = sarsaAgent(statedic, [statedic[s] for s, t in list(mdp.items())], env.action_space.n)


    # Faire un fichier de log sur plusieurs scenarios
    outdir = 'gridworld-v0/sarsa'
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
        action = agent.exploration_policy(obs)

        while True:
            prev_obs = np.copy(obs)
            obs, reward, done, _ = envm.step(action)

            next_action = agent.exploration_policy(obs)

            agent.updateQ(prev_obs, action, reward, obs, next_action, alpha= np.fmax(0.1, 10/(i+1)), gamma=0.99)

            action = next_action

            rsum += reward
            j += 1
            env.render()
            if done:
                print("Episode : " + str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions")
                break

        agent.exploration_rate *= agent.exploration_decay

    print("done")
    env.close()







