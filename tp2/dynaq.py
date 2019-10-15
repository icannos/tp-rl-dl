import gridworld
import gym
from gym import wrappers
import numpy as np
import random
from scipy.special import softmax


class dynaQAgent():
    def __init__(self, state_dict, states, actions, exploration_min=0.05, exploration_rate=1, exploration_decay=0.99, k=4):
        self.exploration_min = exploration_min
        self.k = k
        self.exploration_decay = exploration_decay
        self.exploration_rate = exploration_rate
        self.state_dict = state_dict
        self.states = states
        self.actions = actions

        self.Q = np.random.uniform(0, 2, (max(self.states) + 1, self.actions))
        self.R = np.random.uniform(-1, 1, (max(self.states) + 1, self.actions, max(self.states) + 1))

        random_vect = np.random.uniform(0, 1, (max(self.states) + 1, max(self.states) + 1, self.actions))

        self.P = softmax(random_vect)

    def updateQ(self, st, at, rt, stp, alpha=0.9, gamma=0.9, alphar=0.8):

        st = self.state_dict[st.dumps()]
        stp = self.state_dict[stp.dumps()]
        self.Q[st, at] = (1 - alpha) * self.Q[st, at] + alpha * (rt + gamma * np.max(self.Q[stp, :]))

        self.R[st, at, stp] = (1 - alphar) * self.R[st, at, stp] + alphar * rt
        self.P[stp, st, at] = (1 - alphar) * self.P[stp, st, at] + alphar

        for s in range(max(self.states) + 1):
            if s != stp:
                self.P[s, st, at] = (1 - alphar) * self.P[s, st, at]

        self.P = softmax(self.P)

        states = list(statedic.values())

        s = list(np.random.choice(states, self.k))
        a = list(np.random.choice(self.actions, self.k))

        for i in range(self.k):
            self.Q[s[i], a[i]] = (1 - alpha) * self.Q[s[i], a[i]] + alpha * \
                                 (sum([self.P[sp, s[i], a[i]] * (self.R[s[i], a[i], sp] +
                                                                 gamma * np.max(self.Q[sp, :]))
                                       for sp in s]))

    def exploration_policy(self, s):
        if np.fmax(np.random.uniform(0, 1), self.exploration_min) < self.exploration_rate:
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

    agent = dynaQAgent(statedic, [statedic[s] for s, t in list(mdp.items())], env.action_space.n)

    # Faire un fichier de log sur plusieurs scenarios
    outdir = 'gridworld-v0/daynq-learning'
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
