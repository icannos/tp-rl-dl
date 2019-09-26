
import gridworld
import gym
from gym import wrappers
import numpy as np


class IterPolicyAgent():
    def __init__(self,state_dict, states, actions, P):
        self.P = P
        self.state_dict = state_dict
        self.states = states
        self.actions = actions

        self.policy = None


    def train(self, eps, gamma=0.8):

        policy = np.zeros(max(self.states) + 1)
        self.policy = np.random.randint(0, self.actions, max(self.states) + 1)

        while True:
            V = np.random.uniform(-5, 5, max(self.states) + 1)

            while True:
                Vprev = np.copy(V)
                for s in self.states:
                    V[s] = sum([p * (r + gamma * Vprev[self.state_dict[sp]]) for p, sp, r, done in self.P[s][self.policy[s]]])

                err = np.sum(np.fabs(Vprev - V))
                print(err)
                if err <= eps:
                    break

            for s in self.states:
                policy[s] = np.argmax([sum([p * (r + gamma * V[self.state_dict[sp]])
                                                for p, sp, r, done in self.P[s][a]])
                                            for a in range(self.actions)])

            if all(policy == self.policy):
                self.policy = policy
                break

            self.policy = policy

        return policy


    def act(self, obs):
        return self.policy[self.state_dict[obs.dumps()]]


if __name__ == '__main__':

    # Simple execution
    env = gym.make("gridworld-v0")
    env.seed(0)  # Initialise le seed du pseudo-random

    env.verbose = False
    env.render()  # permet de visualiser la grille du jeu (si verbose = True)

    statedic, mdp = env.getMDP()

    clean_mdp = {statedic[s]: v for s, v in mdp.items()}

    agent = IterPolicyAgent(statedic, [statedic[s] for s, t in list(mdp.items())], env.action_space.n, clean_mdp)
    agent.train(eps=1E-3)


    # Faire un fichier de log sur plusieurs scenarios
    outdir = 'gridworld-v0/random-agent-results'
    envm = wrappers.Monitor(env, directory=outdir, force=True, video_callable=False)
    env.setPlan("gridworldPlans/plan0.txt", {0: -0.001, 3: 1, 4: 1, 5: -1, 6: -1})
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







