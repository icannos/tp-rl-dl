import numpy as np
import matplotlib.pyplot as plt


def read_data(path):
    f = open(path, "r")

    articles = []
    click_rates = []

    for l in f.readlines():
        id_article, str_repr, click_rate_str = l.split(":")
        article = str_repr.split(";")
        click_rate = click_rate_str.split(";")

        articles.append(np.array(list(map(float, article))))
        click_rates.append(np.array(list(map(float, click_rate))))

    return articles, click_rates


def random_policy():
    return [np.random.randint(0, 10) for i in range(5000)]


def staticbest_policy(click_rates):
    a = np.argmax(np.sum(click_rates, axis=0))
    return [a for i in range(5000)]


def opt_policy(click_rates):
    return np.argmax(click_rates, axis=1)


def upper_bound(t, N, mu):
    return mu + np.sqrt(2 * np.log(t) / N)


def ucb_policy(click_rates):
    histo = np.array([click_rates[i][i] for i in range(10)])
    counter = [1 for i in range(10)]

    action_list = [i for i in range(10)]

    for t in range(10, 5000):
        action = np.argmax([upper_bound(t, counter[i], histo[i] / counter[i]) for i in range(10)])
        counter[action] += 1
        histo[action] += click_rates[t][action]

        action_list.append(action)

    return action_list


def linucb_policy(alpha, articles, click_rates):
    A = [np.identity(5, dtype=float) for i in range(10)]
    b = [np.zeros((5, 1), dtype=float) for i in range(10)]

    theta = [None for i in range(10)]
    pt = [None for i in range(10)]

    actions_list = []

    for t in range(0, 5000):
        for i in range(10):
            theta[i] = np.dot(np.linalg.inv(A[i]), b[i])

            pt[i] = (np.dot(np.transpose(theta[i]),  articles[t]) + alpha * np.sqrt(
                np.dot(
                    np.dot(np.transpose(articles[t]),
                           np.linalg.inv(A[i])),
                    articles[t])))[0]

        at = np.argmax(pt)
        rt = click_rates[t][at]

        A[at] = A[at] + np.dot(np.transpose(articles[t]), articles[t])
        b[at] = b[at] + rt * articles[t]

        actions_list.append(at)

    return actions_list

def cumulative_reward(actions, click_rates):
    cum = [0]

    for t in range(5000):
        cum.append(cum[t] + click_rates[t][actions[t]])

    return cum


articles, click_rates = read_data("CTR.txt")

random_actions = random_policy()
staticbest_actions = staticbest_policy(click_rates)
opt_actions = opt_policy(click_rates)
ucb_actions = ucb_policy(click_rates)
lin_ucb_actions = linucb_policy(0.3, articles, click_rates)

c_random = cumulative_reward(random_actions, click_rates)
c_static = cumulative_reward(staticbest_actions, click_rates)
c_opt = cumulative_reward(opt_actions, click_rates)
c_ucb = cumulative_reward(ucb_actions, click_rates)


X = [i for i in range(5001)]

c_linucb_1 = cumulative_reward(linucb_policy(0.01, articles, click_rates), click_rates)
c_linucb_2 = cumulative_reward(linucb_policy(0.9, articles, click_rates), click_rates)
c_linucb_3 = cumulative_reward(linucb_policy(0.5, articles, click_rates), click_rates)
c_linucb_4 = cumulative_reward(linucb_policy(0.2, articles, click_rates), click_rates)



plt.plot(X, c_random, label="random")
plt.plot(X, c_static, label="static best")
#plt.plot(X, c_opt, label="Opt policy")
plt.plot(X, c_ucb, label="UCB")
#plt.plot(X, c_linucb_1, label="LinUCB - 0.01")
#plt.plot(X, c_linucb_2, label="LinUCB - 0.9")
plt.plot(X, c_linucb_3, label="LinUCB - 0.5")
plt.legend()
plt.show()
