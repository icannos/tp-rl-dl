import numpy as np
from matplotlib import pyplot as plt

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)



def plot_data(statistics, N_EPISODES, LOG_STEPS):
    '''
    plots reward and loss graph for entire training
    '''
    x_axis = np.linspace(0, N_EPISODES, N_EPISODES // LOG_STEPS)
    plt.plot(x_axis, statistics["reward"])
    plt.title("Variation of mean rewards")
    plt.show()
    plt.clf()
    plt.savefig("img/mean_rew_ppo.png")

    plt.plot(x_axis, statistics["val_loss"])
    plt.title("Variation of Critic Loss")
    plt.show()

    plt.clf()
    plt.savefig("img/mean_critloss_ppo.png")

    plt.plot(x_axis, statistics["policy_loss"])
    plt.title("Variation of Actor loss")
    plt.show()

    plt.clf()
    plt.savefig("img/mean_actorloss_ppo.png")