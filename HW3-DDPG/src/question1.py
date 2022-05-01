#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from agent import PolicyGradientAgent
from env import GBMStock



def train_episode(agent, env, n_episode):

    mat_S, mat_V = env.sample(n_episode)
    cum_reward = []

    for episode in range(n_episode):
        # generate 1 episode
        t_list = range(n_step)
        S_list = mat_S[:, episode]
        V_list = mat_V[:, episode]
        act_list = np.zeros(n_step)
        reward_list = np.zeros(n_step)
        for t in t_list:
            state = [t_list[t], S_list[t], V_list[t]]
            act_list[t] = agent.predict(state)
            if t > 0:
                reward_list[t] = -np.abs(V_list[t] - act_list[t-1] * S_list[t])

        reward_total = np.zeros(n_step)
        for t in t_list[n_step-2::-1]:
            reward_total[t] = reward_list[t+1] + gamma * reward_total[t+1]

        for t in t_list:
            state = [t_list[t], S_list[t], V_list[t]]
            agent.learn(state, act_list[t], reward_total[t])

        cum_reward.append(reward_total[0])

        if episode % 1000 == 0:
            print('Finish training %d%% episodes.' % int(episode/n_episode*100))


    return cum_reward


action_bound = [0, 1]
learning_rate = 0.1
gamma = 0.9
T = 1
env = GBMStock(50, 0.05, 0.3, T)
n_step = env.n_step

agent = PolicyGradientAgent(n_step, action_bound, learning_rate, gamma)
n_episode = 10000

cum_reward = train_episode(agent, env, n_episode)

plt.plot(range(n_episode), cum_reward)
plt.show()
# %%
