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
    mat_act = np.zeros((n_episode, n_step))
    rf = 0.05
    dt = 1 / 365

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
            if t > 1:
                pv = V_list[t] - act_list[t-1] * S_list[t]
                pre_pv = V_list[t-1] - act_list[t-2] * S_list[t-1]
                reward_list[t] = -np.abs(pv * np.exp(-rf * dt) - pre_pv)
                # reward_list[t] = -np.abs(V_list[t] - act_list[t-1] * S_list[t])

        reward_total = np.zeros(n_step)
        for t in t_list[::-1]:
            if t < n_step-1:
                reward_total[t] = reward_list[t+1] + gamma * reward_total[t+1]

        for t in t_list:
            if t < n_step-1:
                state = [t_list[t], S_list[t], V_list[t]]
                agent.learn(state, act_list[t], reward_total[t])

        cum_reward.append(reward_total[0])
        mat_act[episode] = act_list
        if episode/n_episode*100 % 10 == 0:
            print('Finish training %d%% episodes.' % int(episode/n_episode*100))


    return cum_reward, mat_act


action_bound = [0, 1]
learning_rate = 0.01
gamma = 0.9
n_days = 15
env = GBMStock(50, 0.05, 0.3, n_days)
n_step = env.n_step

agent = PolicyGradientAgent(n_step, action_bound, learning_rate, gamma)
n_episode = 10000

cum_reward, mat_act = train_episode(agent, env, n_episode)

plt.plot(range(n_episode), cum_reward)
plt.show()

test = agent.mat_theta
# %%
