import json
from matplotlib import pyplot as plt
import numpy as np
def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

with open('gridworld-v0/plan0naive-qlearning/openaigym.episode_batch.0.25731.stats.json') as json_file:
    data = json.load(json_file)

rewards = data["episode_rewards"]
rewards = running_mean(rewards, 100)

plt.plot([i for i in range(len(rewards))], rewards, label="test")
plt.legend(loc="lower right")
plt.show()

